"""Metric family 4 — intrinsic quality of a rubric, judged without a response.

Four properties, none of which need a candidate answer (except self-agreement,
which needs one fixed answer and is really a property of the *judge*):

.. note::
   **General within-question redundancy is deliberately not measured here.**
   An earlier version reported ``redundancy_tfidf`` and ``redundancy_llm``. The
   forensics pass then disproved the premise on the full corpus: naive TF-IDF
   flags a high-similarity criterion pair in 22% (science) / 58% (medicine) of
   questions, but almost all of those are *parallel and distinct* items — "follow
   up every 3 months" versus "every 6 months", "identifies Pimozide" versus
   "identifies Penfluridol". The default tokeniser drops the numbers and rare
   entities that carry the entire difference. Requiring the discriminative tokens
   to match as well puts real redundancy at 0.8% / 7.9%
   (``docs/01_data_forensics.md`` §5). Reporting the naive number would overstate
   the flaw by an order of magnitude, and paying for an LLM pass to measure
   something that occurs in under 8% of questions is a poor use of budget.

   The one redundancy form that *is* real — a Pitfall that merely restates an
   Essential item with the polarity flipped, so one fact scores twice (F7, 16.2%
   of medicine questions) — is measured in :mod:`harness.eval.lint` as
   ``pitfall_mirror_rate``.

**Verifiability.** ``objective_fraction`` — can a criterion be settled yes/no
from the response alone, or does it need taste (``subjective``) or knowledge the
criterion withholds (``underspecified``)?

**Self-agreement.** Judge stability, not rubric content: the same rubric is
scored against the same gold response ``config.self_agreement_repeats`` times
with different salts, which also re-randomises criterion order. A rubric whose
criteria the judge answers differently depending on where they appear in the
list is a noisy reward signal regardless of how good it looks. Reported as raw
agreement and Cohen's kappa (kappa is undefined when a repeat is constant — see
:func:`~harness.eval.stats.cohens_kappa` — so NaN is returned and counted).

**Atomicity.** ``mean_checks_per_criterion`` — how many independent assertions a
criterion bundles. Values above 1 mean a response can satisfy a criterion
halfway and still score 0, which makes the reward coarse. Closer to 1 is better.

**Weight distribution.** Shannon entropy of the normalised weight magnitudes plus
the category mix. Low entropy means the rubric concentrates its score on a few
items; this is descriptive, not a quality claim, and exists so a reviewer can
check that no source wins simply by weighting differently.

Cost per (question, source): 2 batched calls (verifiability, atomicity) plus
``self_agreement_repeats`` judge calls.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
import math
import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..config import EvalConfig
from ..llm import JSONParseError, LLMEngine
from ..prompts.eval_prompts import (
    ATOMICITY_SYSTEM,
    VERIFIABILITY_SYSTEM,
    build_atomicity_prompt,
    build_verifiability_prompt,
)
from ..schema import Category, Criterion, Example, Rubric
from ..tracing import RunDir
from .judge import JudgeResult, judge_rubric
from .stats import cohens_kappa, mean_ci, safe_mean

logger = logging.getLogger(__name__)

__all__ = [
    "IntrinsicResults",
    "TFIDF_REDUNDANCY_THRESHOLD",
    "run_intrinsic",
    "tfidf_redundancy",
    "weight_profile",
]

NAN = float("nan")
TFIDF_REDUNDANCY_THRESHOLD = 0.5
VERIFIABILITY_LABELS = ("objective", "subjective", "underspecified")
GOLD_RESPONSE_ID = "gold"

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# Rubric prose is formulaic ("the response mentions that ..."); leaving these in
# would make every pair of criteria look similar and wash out real overlap.
_STOPWORDS = frozenset(
    """a an the and or but if then than that this these those of to in on at by for with
    from as is are was were be been being it its they them their there here which who whom
    whose what when where why how all any both each few more most other some such no nor
    not only own same so too very can will just should now must may might could would does
    do did doing done have has had having also into about over under between within without
    across during before after above below response answer answers criteria criterion
    mention mentions mentioned state states stated explain explains explained include
    includes included provide provides provided correctly correct clearly clear identify
    identifies identified describe describes described note notes noted specify specifies
    given uses use used one two
    """.split()
)


# ---------------------------------------------------------------------------
# Pure-numpy redundancy
# ---------------------------------------------------------------------------


def _tokens(criterion: Criterion) -> list[str]:
    # ``description`` already has the category prefix stripped by Criterion, so
    # criteria do not share tokens merely by sharing a category.
    text = f"{criterion.title} {criterion.description}".lower()
    return [t for t in _TOKEN_RE.findall(text) if len(t) > 2 and t not in _STOPWORDS]


def tfidf_redundancy(
    criteria: Sequence[Criterion], *, threshold: float = TFIDF_REDUNDANCY_THRESHOLD
) -> dict[str, Any]:
    """Lexical overlap among criteria of one rubric — pure numpy, no sklearn.

    Each criterion is a document; the vocabulary and the IDF are built *within
    the rubric*, so the score answers "do these items repeat each other" rather
    than "are these items unusual for the corpus". Weighting is sublinear TF
    (``1 + log tf``) times smoothed IDF (``log((1+N)/(1+df)) + 1``), L2
    normalised, exactly the standard formulation.

    Returns
    -------
    dict with keys ``redundancy_tfidf`` (fraction of criterion pairs above
    ``threshold``), ``max_pair_cosine``, ``mean_pair_cosine``, ``n_pairs``,
    ``n_redundant_pairs`` and ``redundant_pairs`` (index pairs, 0-based).
    All are NaN / empty for rubrics with fewer than two criteria.
    """
    n = len(criteria)
    empty = {
        "redundancy_tfidf": NAN,
        "max_pair_cosine": NAN,
        "mean_pair_cosine": NAN,
        "n_pairs": 0,
        "n_redundant_pairs": 0,
        "redundant_pairs": [],
    }
    if n < 2:
        return empty

    docs = [_tokens(c) for c in criteria]
    vocab: dict[str, int] = {}
    for doc in docs:
        for token in doc:
            vocab.setdefault(token, len(vocab))
    if not vocab:
        return empty

    counts = np.zeros((n, len(vocab)), dtype=float)
    for i, doc in enumerate(docs):
        for token in doc:
            counts[i, vocab[token]] += 1.0

    tf = np.where(counts > 0, 1.0 + np.log(np.maximum(counts, 1.0)), 0.0)
    df = np.count_nonzero(counts, axis=0).astype(float)
    idf = np.log((1.0 + n) / (1.0 + df)) + 1.0
    matrix = tf * idf
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    matrix = matrix / np.where(norms > 0, norms, 1.0)

    cosine = matrix @ matrix.T
    iu = np.triu_indices(n, k=1)
    pair_values = cosine[iu]
    if pair_values.size == 0:
        return empty
    flagged = pair_values > threshold
    return {
        "redundancy_tfidf": float(flagged.mean()),
        "max_pair_cosine": float(pair_values.max()),
        "mean_pair_cosine": float(pair_values.mean()),
        "n_pairs": int(pair_values.size),
        "n_redundant_pairs": int(flagged.sum()),
        "redundant_pairs": [
            [int(a), int(b)] for a, b in zip(iu[0][flagged], iu[1][flagged])
        ],
    }


def weight_profile(rubric: Rubric) -> dict[str, Any]:
    """Shannon entropy of the weight distribution plus the category mix.

    Returns
    -------
    dict with keys

    ``n_items``
        Number of criteria.
    ``weight_entropy``
        Entropy in **nats** of the weight magnitudes normalised to a probability
        vector. 0 when one criterion carries all the weight.
    ``weight_entropy_norm``
        ``weight_entropy / log(n_items)`` in ``[0, 1]``; 1 means perfectly flat
        weighting. NaN for a single-item rubric.
    ``mean_weight_magnitude`` / ``max_weight_share``
        Mean ``|weight|``, and the largest single share of total weight.
    ``frac_essential`` / ``frac_important`` / ``frac_optional`` / ``frac_pitfall``
        Category mix.
    """
    items = list(rubric.items)
    n = len(items)
    out: dict[str, Any] = {
        "n_items": n,
        "weight_entropy": NAN,
        "weight_entropy_norm": NAN,
        "mean_weight_magnitude": NAN,
        "max_weight_share": NAN,
        "frac_essential": NAN,
        "frac_important": NAN,
        "frac_optional": NAN,
        "frac_pitfall": NAN,
    }
    if n == 0:
        return out

    magnitudes = np.array([float(c.magnitude) for c in items], dtype=float)
    total = float(magnitudes.sum())
    if total > 0:
        p = magnitudes / total
        nonzero = p[p > 0]
        entropy = float(-(nonzero * np.log(nonzero)).sum())
        out["weight_entropy"] = entropy
        out["weight_entropy_norm"] = entropy / math.log(n) if n > 1 else NAN
        out["max_weight_share"] = float(p.max())
    out["mean_weight_magnitude"] = float(magnitudes.mean())
    for category, key in (
        (Category.ESSENTIAL, "frac_essential"),
        (Category.IMPORTANT, "frac_important"),
        (Category.OPTIONAL, "frac_optional"),
        (Category.PITFALL, "frac_pitfall"),
    ):
        out[key] = sum(1 for c in items if c.category is category) / n
    return out


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class IntrinsicResults:
    """Per-question (and per-criterion) intrinsic rows plus a summary."""

    per_question_rows: list[dict[str, Any]] = field(default_factory=list)
    per_criterion_rows: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Mean and bootstrap CI of each intrinsic metric, per rubric source.

        Every metric ``m`` below is reported as ``m`` (mean over questions with
        a finite value), ``m_ci_low`` / ``m_ci_high`` (bootstrap percentile CI)
        and ``m_n`` (how many questions contributed).

        Keys under ``by_source[src]``

        ``n_questions`` / ``n_ok`` / ``n_errors``
            Rows for this source in total, without an ``error``, and with one.
            A row can carry an error from one sub-call and still contribute the
            metrics produced by the others.
        ``redundancy_tfidf``
            Fraction of criterion pairs with TF-IDF cosine above threshold.
            **Descriptive only — do not report this as a quality metric.** The
            forensics showed this naive form overstates real redundancy by an
            order of magnitude because the tokeniser drops the numbers and rare
            entities that distinguish parallel criteria (see the module note).
        ``mean_pair_cosine`` / ``max_pair_cosine``
            Descriptive lexical-overlap context.
        ``objective_fraction`` / ``subjective_fraction`` / ``underspecified_fraction``
            Verifiability mix; ``objective_fraction`` **higher is better**.
        ``self_agreement_rate``
            Mean fraction of criterion verdicts that match across repeated
            judgements of the same gold response. **Higher is better.**
        ``kappa``
            Mean Cohen's kappa of the same repeats; NaN rows are excluded.
        ``n_kappa_undefined``
            Count of questions where kappa was undefined (a repeat was constant);
            read ``self_agreement_rate`` for those.
        ``mean_checks_per_criterion``
            Mean bundled assertions per criterion. **Closer to 1.0 is better.**
        ``atomic_fraction``
            Fraction of criteria with exactly one assertion. **Higher is better.**
        ``weight_entropy`` / ``weight_entropy_norm`` / ``max_weight_share``
            Weight-distribution shape (descriptive).
        ``n_items``, ``frac_essential``, ``frac_important``, ``frac_optional``,
        ``frac_pitfall``
            Rubric size and category mix (descriptive).
        """
        by_source: dict[str, Any] = {}
        for source in self._sources():
            rows = [r for r in self.per_question_rows if r["rubric_source"] == source]
            n_ok = sum(1 for r in rows if not r.get("error"))
            entry: dict[str, Any] = {
                "n_questions": len(rows),
                "n_ok": n_ok,
                "n_errors": len(rows) - n_ok,
                "n_kappa_undefined": sum(
                    1 for r in rows if not _is_finite(r.get("kappa"))
                ),
            }
            for metric in (
                "redundancy_tfidf",
                "mean_pair_cosine",
                "max_pair_cosine",
                "objective_fraction",
                "subjective_fraction",
                "underspecified_fraction",
                "self_agreement_rate",
                "kappa",
                "mean_checks_per_criterion",
                "atomic_fraction",
                "weight_entropy",
                "weight_entropy_norm",
                "max_weight_share",
                "n_items",
                "frac_essential",
                "frac_important",
                "frac_optional",
                "frac_pitfall",
            ):
                ci = mean_ci([r.get(metric) for r in rows], iters=2000, seed=0)
                entry[metric] = ci["mean"]
                entry[f"{metric}_ci_low"] = ci["ci_low"]
                entry[f"{metric}_ci_high"] = ci["ci_high"]
                entry[f"{metric}_n"] = ci["n"]
            by_source[source] = entry
        return {
            "metric_family": "intrinsic",
            "params": dict(self.params),
            "by_source": by_source,
            "n_errors": len(self.errors),
        }

    def _sources(self) -> list[str]:
        seen: list[str] = []
        for row in self.per_question_rows:
            if row["rubric_source"] not in seen:
                seen.append(row["rubric_source"])
        return seen


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_intrinsic(
    engine: LLMEngine,
    examples: Sequence[Example],
    rubrics_by_source: Mapping[str, Mapping[str, Rubric]],
    *,
    config: EvalConfig,
    gold_responses: Mapping[str, str] | None = None,
    run_dir: RunDir | None = None,
) -> IntrinsicResults:
    """Score every rubric on redundancy, verifiability, stability and atomicity.

    Parameters
    ----------
    gold_responses:
        ``{uid: response_text}`` used for the self-agreement probe. Defaults to
        the example's ``reference_answer``, which is the natural choice: a rubric
        should at least be stable on the answer it was written from.
    """
    repeats = max(0, int(config.self_agreement_repeats))
    results = IntrinsicResults(
        params={
            "sources": list(rubrics_by_source),
            "self_agreement_repeats": repeats,
            "tfidf_threshold": TFIDF_REDUNDANCY_THRESHOLD,
            "gold_responses": "supplied" if gold_responses else "reference_answer",
        }
    )

    cells: list[tuple[Example, str, Rubric]] = []
    for ex in examples:
        for source, per_uid in rubrics_by_source.items():
            rubric = per_uid.get(ex.uid)
            if rubric is not None:
                cells.append((ex, source, rubric))

    outcomes = await asyncio.gather(
        *(
            _score_cell(
                engine,
                ex,
                source,
                rubric,
                config,
                gold_response=(gold_responses or {}).get(ex.uid) or ex.reference_answer,
                repeats=repeats,
            )
            for ex, source, rubric in cells
        ),
        return_exceptions=True,
    )
    for (ex, source, rubric), outcome in zip(cells, outcomes):
        if isinstance(outcome, BaseException):
            row = _blank_row(ex, source, rubric)
            row["error"] = f"exception: {outcome}"[:300]
            criterion_rows: list[dict[str, Any]] = []
        else:
            row, criterion_rows = outcome
        results.per_question_rows.append(row)
        results.per_criterion_rows.extend(criterion_rows)
        if row.get("error"):
            results.errors.append(
                {"uid": ex.uid, "rubric_source": source, "error": row["error"]}
            )

    _persist(results, run_dir)
    return results


def _blank_row(example: Example, source: str, rubric: Rubric) -> dict[str, Any]:
    row: dict[str, Any] = {
        "uid": example.uid,
        "domain": example.domain,
        "rubric_source": source,
        "n_criteria": len(rubric),
        "redundancy_tfidf": NAN,
        "mean_pair_cosine": NAN,
        "max_pair_cosine": NAN,
        "n_pairs": 0,
        "n_redundant_pairs": 0,
        "objective_fraction": NAN,
        "subjective_fraction": NAN,
        "underspecified_fraction": NAN,
        "n_criteria_labelled": 0,
        "self_agreement_rate": NAN,
        "kappa": NAN,
        "n_self_agreement_runs": 0,
        "n_self_agreement_criteria": 0,
        "mean_checks_per_criterion": NAN,
        "atomic_fraction": NAN,
        "max_checks_per_criterion": 0,
        "error": None,
    }
    row.update(weight_profile(rubric))
    return row


async def _score_cell(
    engine: LLMEngine,
    example: Example,
    source: str,
    rubric: Rubric,
    config: EvalConfig,
    *,
    gold_response: str,
    repeats: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row = _blank_row(example, source, rubric)
    if len(rubric) == 0:
        row["error"] = "empty rubric"
        return row, []

    row.update(tfidf_redundancy(rubric.items))
    lexical_pairs = {tuple(p) for p in row.pop("redundant_pairs", [])}

    verifiability, atomicity, repeats_out = await asyncio.gather(
        _verifiability(engine, example, rubric, config),
        _atomicity(engine, example, rubric, config),
        _self_agreement(engine, example, source, rubric, config, gold_response, repeats),
    )

    problems: list[str] = []
    for part in (verifiability, atomicity, repeats_out):
        if part.get("error"):
            problems.append(str(part["error"]))
        row.update({k: v for k, v in part.items() if k != "error"})

    llm_pairs: set[tuple[int, int]] = set()
    per_criterion = _per_criterion_rows(
        example,
        source,
        rubric,
        labels=row.pop("verifiability_labels", {}),
        checks=row.pop("atomicity_checks", {}),
        agreement=row.pop("per_criterion_agreement", {}),
        lexical_pairs=lexical_pairs,
        llm_pairs=llm_pairs,
    )
    if problems:
        row["error"] = "; ".join(problems)[:300]
    return row, per_criterion


async def _verifiability(
    engine: LLMEngine, example: Example, rubric: Rubric, config: EvalConfig
) -> dict[str, Any]:
    try:
        parsed = await engine.chat_json(
            build_verifiability_prompt(example.question, rubric.items),
            system=VERIFIABILITY_SYSTEM,
            expect="array",
            max_tokens=int(config.judge_max_tokens),
            tag="intrinsic:verifiability",
        )
    except (JSONParseError, RuntimeError) as exc:
        return {"error": f"verifiability failed: {exc}"[:250]}

    labels = _collect_labels(parsed, len(rubric), VERIFIABILITY_LABELS)
    if not labels:
        return {"error": "no usable verifiability labels"}
    total = len(labels)
    counts = {label: 0 for label in VERIFIABILITY_LABELS}
    for label in labels.values():
        counts[label] += 1
    return {
        "objective_fraction": counts["objective"] / total,
        "subjective_fraction": counts["subjective"] / total,
        "underspecified_fraction": counts["underspecified"] / total,
        "n_criteria_labelled": total,
        "verifiability_labels": labels,
    }


async def _atomicity(
    engine: LLMEngine, example: Example, rubric: Rubric, config: EvalConfig
) -> dict[str, Any]:
    try:
        parsed = await engine.chat_json(
            build_atomicity_prompt(example.question, rubric.items),
            system=ATOMICITY_SYSTEM,
            expect="array",
            max_tokens=int(config.judge_max_tokens),
            tag="intrinsic:atomicity",
        )
    except (JSONParseError, RuntimeError) as exc:
        return {"error": f"atomicity failed: {exc}"[:250]}

    n = len(rubric)
    checks: dict[int, int] = {}
    for pos, entry in enumerate(parsed if isinstance(parsed, list) else [], start=1):
        if not isinstance(entry, dict):
            continue
        cid = _as_int(entry.get("criterion_id"), pos)
        if not 1 <= cid <= n or cid in checks:
            continue
        value = _as_int(entry.get("n_checks"), -1)
        if value >= 1:
            checks[cid] = min(value, 20)
    if not checks:
        return {"error": "no usable atomicity counts"}
    values = list(checks.values())
    return {
        "mean_checks_per_criterion": sum(values) / len(values),
        "atomic_fraction": sum(1 for v in values if v == 1) / len(values),
        "max_checks_per_criterion": max(values),
        "atomicity_checks": checks,
    }


async def _self_agreement(
    engine: LLMEngine,
    example: Example,
    source: str,
    rubric: Rubric,
    config: EvalConfig,
    gold_response: str,
    repeats: int,
) -> dict[str, Any]:
    """Repeat the same judgement with different salts and measure stability."""
    if repeats < 2:
        return {
            "self_agreement_rate": NAN,
            "kappa": NAN,
            "n_self_agreement_runs": 0,
            "per_criterion_agreement": {},
        }
    results: Sequence[JudgeResult] = await asyncio.gather(
        *(
            judge_rubric(
                engine,
                uid=example.uid,
                question=example.question,
                response=gold_response,
                rubric=rubric,
                rubric_source=source,
                response_id=GOLD_RESPONSE_ID,
                shuffle=bool(config.shuffle_criteria),
                shuffle_seed=int(config.shuffle_seed),
                max_tokens=int(config.judge_max_tokens),
                repeat_salt=f"selfagree{i}",
            )
            for i in range(repeats)
        )
    )
    runs: list[dict[int, bool]] = [
        {v.index: bool(v.met) for v in r.verdicts} for r in results if r.verdicts
    ]
    if len(runs) < 2:
        errors = [r.error for r in results if r.error]
        return {
            "self_agreement_rate": NAN,
            "kappa": NAN,
            "n_self_agreement_runs": len(runs),
            "per_criterion_agreement": {},
            "error": f"self-agreement: {errors[0] if errors else 'too few runs'}"[:250],
        }

    shared = sorted(set.intersection(*(set(run) for run in runs)))
    if not shared:
        return {
            "self_agreement_rate": NAN,
            "kappa": NAN,
            "n_self_agreement_runs": len(runs),
            "per_criterion_agreement": {},
            "error": "self-agreement: no shared criteria across repeats",
        }

    agreements: list[float] = []
    kappas: list[float] = []
    per_criterion: dict[int, float] = {i: 0.0 for i in shared}
    n_comparisons = 0
    for a, b in itertools.combinations(range(len(runs)), 2):
        va = [runs[a][i] for i in shared]
        vb = [runs[b][i] for i in shared]
        agreements.append(sum(1.0 for x, y in zip(va, vb) if x == y) / len(shared))
        kappas.append(cohens_kappa(va, vb))
        for i, (x, y) in zip(shared, zip(va, vb)):
            per_criterion[i] += float(x == y)
        n_comparisons += 1
    defined = [k for k in kappas if _is_finite(k)]
    return {
        "self_agreement_rate": safe_mean(agreements),
        "kappa": safe_mean(defined) if defined else NAN,
        "n_self_agreement_runs": len(runs),
        "n_self_agreement_criteria": len(shared),
        "per_criterion_agreement": {
            i: v / n_comparisons for i, v in per_criterion.items()
        },
    }


def _per_criterion_rows(
    example: Example,
    source: str,
    rubric: Rubric,
    *,
    labels: Mapping[int, str],
    checks: Mapping[int, int],
    agreement: Mapping[int, float],
    lexical_pairs: set[tuple[int, ...]],
    llm_pairs: set[tuple[int, ...]],
) -> list[dict[str, Any]]:
    lexical_flagged = {i for pair in lexical_pairs for i in pair}
    llm_flagged = {i for pair in llm_pairs for i in pair}
    rows: list[dict[str, Any]] = []
    for i, criterion in enumerate(rubric.items):
        rows.append(
            {
                "uid": example.uid,
                "rubric_source": source,
                "criterion_index": i,
                "title": criterion.title,
                "category": criterion.category.value,
                "weight": int(criterion.weight),
                "verifiability": labels.get(i + 1),
                "n_checks": checks.get(i + 1),
                "agreement": agreement.get(i),
                "in_redundant_pair_tfidf": i in lexical_flagged,
                "in_redundant_pair_llm": i in llm_flagged,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Parsing + persistence helpers
# ---------------------------------------------------------------------------


def _collect_labels(
    parsed: Any, n: int, allowed: Sequence[str]
) -> dict[int, str]:
    labels: dict[int, str] = {}
    for pos, entry in enumerate(parsed if isinstance(parsed, list) else [], start=1):
        if not isinstance(entry, dict):
            continue
        cid = _as_int(entry.get("criterion_id"), pos)
        if not 1 <= cid <= n or cid in labels:
            continue
        label = str(entry.get("label", "")).strip().lower()
        if label in allowed:
            labels[cid] = label
    return labels


def _as_int(value: Any, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _is_finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _persist(results: IntrinsicResults, run_dir: RunDir | None) -> None:
    if run_dir is None:
        return
    try:
        for row in results.per_question_rows:
            run_dir.writer("intrinsic_per_question").write(row)
        for row in results.per_criterion_rows:
            run_dir.writer("intrinsic_per_criterion").write(row)
        run_dir.write_json("intrinsic_summary.json", results.summary())
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to persist intrinsic artefacts: %s", exc)
