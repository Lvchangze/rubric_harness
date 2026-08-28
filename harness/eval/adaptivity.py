"""Metric family 6 — F1: does the rubric's item budget respond to the question?

Zero LLM calls. The forensics' first and most consequential finding is that the
shipped budget is *a constant*: RaR-Science gives 7 items to 64.56% of questions,
its item-count coefficient of variation is 0.107, and item count correlates with
question length at only r = 0.19 — a 15-word multiple-choice question and a
100-word three-part derivation with a 730-word reference answer both get 8
criteria (``docs/01_data_forensics.md`` §3, §8.1, F1). Everything downstream
follows from that: with the budget pinned at 7 and Essential facts eating 3-5
slots, the remainder can only be filled with generic padding, and genuinely
complex questions are structurally under-covered.

So this module measures the *response* of ``n_items`` to three complexity
proxies, none of which needs a model:

* **sub-questions** — via :func:`analysis.rubric_grounding.detect_subparts`,
  which is the detector that matters here because a multiple-choice option list
  ("(a) ... (b) ... (c) ...") is indistinguishable from a sub-question list at
  the regex level; a span only counts as a sub-question when it actually asks
  for something. Science: 2.95% of questions are genuinely multi-part, a further
  9.36% are MCQs.
* **question word count** and **reference word count**.

Published shipped-corpus reference points and the targets F1 sets
----------------------------------------------------------------
========================================= ============ ============ =========
metric                                     science      medicine     target
========================================= ============ ============ =========
``items_cv``                               0.107        0.140        > 0.35
``modal_count_share``                      0.6456 (@7)  0.3618 (@10) < 0.25
``r_items_vs_question_len``                0.19         0.26         > 0.5
``r_items_vs_reference_len``               0.24         0.30         —
``items_per_subpart`` (4 sub-questions)    2.07         n/a          —
``orphan_subpart_rate``                    0.0445*      n/a          → 0
``weight_corr_with_uniform``               0.943        0.932        —
========================================= ============ ============ =========

\\* Our IDF-free overlap measure scores the same corpus at 0.0495; see
:data:`ORPHAN_OVERLAP_THRESHOLD`. Note also that only ~3% of science questions
are multi-part, so at n=80/domain a run has one or two of them and every
sub-part statistic is anecdotal — read ``n_multi_part_questions`` first.

Two of these statistics are far less stable at small n than the rest, and a
reader comparing a run against the table above should know which: the
``r_items_vs_*`` correlations (a Pearson r on 80 points has SE ~0.11, and it is
sensitive to a single long question — the pilot's shipped column reads 0.42
science / 0.51 medicine against the published 0.19 / 0.26, while the Spearman
``rho_items_vs_question_len`` sits much closer at 0.28) and
``modal_count_share`` when the count distribution is flat. ``items_cv``,
``n_items`` and ``weight_corr_with_uniform`` reproduce the published values to
three decimals at n=80 and are the ones to trust.

Count-controlled comparison
---------------------------
:func:`truncate_to_common_size` exists because a coverage or grounding win is
uninteresting if the agentic rubrics simply have *more* criteria. It cuts every
source to the same per-question item count — the per-question minimum across the
sources present — so the eval CLI can re-run selected metrics on a set where
``n_items`` is held constant by construction. It is pure: the input mapping and
every ``Rubric``/``Criterion`` in it are left untouched (criteria are deep-copied
into the new rubrics).

Rate metrics such as ``generic_criterion_rate`` are already per-criterion and so
count-insensitive; for the ones that are not, the count-normalised view is
emitted directly as ``weight_mass_per_criterion`` and
``paper_weight_mass_per_criterion``.
"""

from __future__ import annotations

import collections
import copy
import logging
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..config import EvalConfig
from ..schema import Category, Criterion, Example, Rubric
from ..tracing import RunDir
from .grounding import CATEGORY_WEIGHT_REMAP, SHIPPED_FULL_CORPUS_REFERENCE
from .stats import mean_ci, rankdata

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "analysis")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from analysis.common import content_tokens, words  # noqa: E402
from analysis.rubric_grounding import detect_subparts  # noqa: E402

logger = logging.getLogger(__name__)

__all__ = [
    "AdaptivityResults",
    "ORPHAN_OVERLAP_THRESHOLD",
    "TRUNCATION_STRATEGIES",
    "question_complexity",
    "run_adaptivity",
    "subpart_coverage",
    "truncate_to_common_size",
]

NAN = float("nan")

#: Binary content-token cosine below which a sub-question counts as an *orphan*
#: — no criterion of the rubric is about it. The forensics used a TF-IDF cosine
#: with a 0.08 cut, fitted over all multi-part questions of the corpus (§8.1:
#: 4.45% orphan rate). A corpus-fitted IDF is meaningless for a 160-question run,
#: so we use the IDF-free binary-token cosine ``|A ∩ B| / sqrt(|A| · |B|)``
#: instead and calibrated the cut on the *same* population the published number
#: comes from: over the 1,777 sub-parts of the 677 multi-part science questions,
#: 0.10 gives a 4.95% orphan rate against the forensics' 4.45%. Close enough to
#: read on the same axis, not identical in construction.
ORPHAN_OVERLAP_THRESHOLD = 0.10
#: Random binary criterion outcomes drawn per question for the Eq. 1 vs
#: unweighted-mean correlation, matching ``analysis/rubric_stats.py``.
WEIGHT_SIMULATION_DRAWS = 8
WEIGHT_SIMULATION_SEED = 0

#: Priority order used by the ``category_priority`` truncation strategy.
_CATEGORY_KEEP_RANK: dict[Category, int] = {
    Category.ESSENTIAL: 3,
    Category.IMPORTANT: 2,
    Category.OPTIONAL: 1,
    Category.PITFALL: 0,
}


# ---------------------------------------------------------------------------
# Complexity proxies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QuestionComplexity:
    """Model-free complexity proxies for one question."""

    uid: str
    domain: str
    n_subparts: int
    subpart_kind: str
    is_mcq: bool
    question_words: int
    reference_words: int
    subpart_texts: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "n_subparts": self.n_subparts,
            "subpart_kind": self.subpart_kind,
            "is_mcq": self.is_mcq,
            "question_words": self.question_words,
            "reference_words": self.reference_words,
        }


def question_complexity(example: Example) -> QuestionComplexity:
    """Sub-question count, MCQ flag and length proxies for one example.

    ``n_subparts`` counts only *genuine* sub-questions: ``detect_subparts``
    returns ``"mcq"`` for an enumerated span list where fewer than two spans
    actually ask for anything, and those contribute 0 sub-parts (they are still
    flagged via ``is_mcq``, which matters because §8.2 shows MCQs get 7.44
    criteria of which 4.3 are unrelated to the answer).
    """
    try:
        spans, kind = detect_subparts(example.question or "")
    except Exception as exc:  # noqa: BLE001 - a metric must never raise
        logger.warning("subpart detection failed for %s: %s", example.uid, exc)
        spans, kind = [], "none"
    subquestions = kind == "subquestions"
    return QuestionComplexity(
        uid=example.uid,
        domain=example.domain,
        n_subparts=len(spans) if subquestions else 0,
        subpart_kind=kind,
        is_mcq=kind == "mcq",
        question_words=len(words(example.question or "")),
        reference_words=len(words(example.reference_answer or "")),
        subpart_texts=tuple(text for _label, text in spans) if subquestions else (),
    )


def _binary_cosine(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / math.sqrt(len(a) * len(b))


def subpart_coverage(
    subpart_texts: Sequence[str],
    criteria: Sequence[Criterion],
    *,
    threshold: float = ORPHAN_OVERLAP_THRESHOLD,
) -> dict[str, Any]:
    """Is every sub-question represented by at least one criterion?

    A sub-question is *covered* when some criterion of the same rubric reaches
    ``threshold`` binary content-token cosine with it, and an *orphan* otherwise.
    Returns NaN rates when the question has no detected sub-questions, so
    single-part questions do not dilute the rate.
    """
    out: dict[str, Any] = {
        "n_subparts_scored": 0,
        "n_orphan_subparts": 0,
        "orphan_subpart_rate": NAN,
        "mean_best_subpart_overlap": NAN,
    }
    if not subpart_texts or not criteria:
        return out
    criterion_tokens = [
        set(content_tokens((c.description or "").lower())) | set(content_tokens(c.title or ""))
        for c in criteria
    ]
    best = [
        max((_binary_cosine(set(content_tokens(text.lower())), tokens) for tokens in criterion_tokens), default=0.0)
        for text in subpart_texts
    ]
    orphans = [value < threshold for value in best]
    out.update(
        {
            "n_subparts_scored": len(best),
            "n_orphan_subparts": int(sum(orphans)),
            "orphan_subpart_rate": float(np.mean(orphans)),
            "mean_best_subpart_overlap": float(np.mean(best)),
        }
    )
    return out


# ---------------------------------------------------------------------------
# Count-controlled comparison
# ---------------------------------------------------------------------------


def _keep_max_weight(rubric: Rubric, k: int) -> list[int]:
    """Indices of the ``k`` highest-magnitude criteria.

    Tie-break: **original position**, earliest first. Generators emit criteria
    in the order they consider important (Essential first in every shipped and
    baseline rubric we have seen), so position is the most informative available
    tie-break, and using it makes the selection fully deterministic — no RNG, no
    dependence on dict iteration order.
    """
    order = sorted(range(len(rubric.items)), key=lambda i: (-rubric.items[i].magnitude, i))
    return sorted(order[:k])


def _keep_first(rubric: Rubric, k: int) -> list[int]:
    """The first ``k`` criteria as emitted. The null strategy, for robustness checks."""
    return list(range(min(k, len(rubric.items))))


def _keep_category_priority(rubric: Rubric, k: int) -> list[int]:
    """Essential > Important > Optional > Pitfall, then magnitude, then position.

    Useful as a sensitivity check on ``max_weight``: the shipped weights are
    close to a deterministic function of the category (F5), so the two strategies
    mostly agree, and where they disagree it is because a rubric violates its own
    category order (F5 again).
    """
    order = sorted(
        range(len(rubric.items)),
        key=lambda i: (
            -_CATEGORY_KEEP_RANK.get(rubric.items[i].category, 0),
            -rubric.items[i].magnitude,
            i,
        ),
    )
    return sorted(order[:k])


#: Pluggable selection rules for :func:`truncate_to_common_size`. Each maps
#: ``(rubric, k)`` to the *sorted* indices of the criteria to keep.
TRUNCATION_STRATEGIES: dict[str, Callable[[Rubric, int], list[int]]] = {
    "max_weight": _keep_max_weight,
    "first_k": _keep_first,
    "category_priority": _keep_category_priority,
}


def truncate_to_common_size(
    rubrics_by_source: Mapping[str, Mapping[str, Rubric]],
    *,
    strategy: str | Callable[[Rubric, int], list[int]] = "max_weight",
) -> dict[str, dict[str, Rubric]]:
    """Cut every source to the same per-question item count.

    Parameters
    ----------
    rubrics_by_source:
        ``{source: {uid: Rubric}}``. Never mutated — neither the mapping, nor the
        ``Rubric`` objects, nor their ``Criterion`` objects (those are
        deep-copied into the result).
    strategy:
        A key of :data:`TRUNCATION_STRATEGIES` (default ``"max_weight"``: keep
        each rubric's own highest-``|weight|`` criteria, ties broken by original
        position, earliest first) or a callable ``(rubric, k) -> list[int]``
        returning the indices to keep.

    Returns
    -------
    A new ``{source: {uid: Rubric}}`` where, for each uid, every source has
    exactly ``min_s len(rubric_s[uid])`` criteria — so any remaining difference
    between sources cannot be a difference in rubric size. Criteria keep their
    original relative order. Each result rubric's ``meta`` gains
    ``truncated_from``, ``truncated_to`` and ``truncate_strategy``; uids absent
    from a source stay absent.
    """
    selector = (
        TRUNCATION_STRATEGIES.get(strategy, _keep_max_weight)
        if isinstance(strategy, str)
        else strategy
    )
    name = strategy if isinstance(strategy, str) else getattr(strategy, "__name__", "custom")
    if isinstance(strategy, str) and strategy not in TRUNCATION_STRATEGIES:
        logger.warning("unknown truncation strategy %r; falling back to max_weight", strategy)
        name = "max_weight"

    sizes: dict[str, list[int]] = collections.defaultdict(list)
    for per_uid in rubrics_by_source.values():
        for uid, rubric in per_uid.items():
            sizes[uid].append(len(rubric.items))
    target = {uid: min(values) for uid, values in sizes.items() if values}

    out: dict[str, dict[str, Rubric]] = {}
    for source, per_uid in rubrics_by_source.items():
        cut: dict[str, Rubric] = {}
        for uid, rubric in per_uid.items():
            k = target.get(uid, len(rubric.items))
            try:
                keep = sorted(int(i) for i in selector(rubric, k))
            except Exception as exc:  # noqa: BLE001 - degrade to the null strategy
                logger.warning("truncation strategy failed for %s/%s: %s", source, uid, exc)
                keep = _keep_first(rubric, k)
            keep = [i for i in keep if 0 <= i < len(rubric.items)][:k]
            cut[uid] = Rubric(
                items=[copy.deepcopy(rubric.items[i]) for i in keep],
                meta={
                    **dict(rubric.meta),
                    "truncated_from": len(rubric.items),
                    "truncated_to": len(keep),
                    "truncate_strategy": name,
                },
            )
        out[source] = cut
    return out


# ---------------------------------------------------------------------------
# Distribution-level statistics
# ---------------------------------------------------------------------------


def _safe_mean(values: Sequence[Any]) -> float:
    finite: list[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(number):
            finite.append(number)
    return float(np.mean(finite)) if finite else NAN


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return NAN
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 3:
        return NAN
    return _pearson(rankdata(a), rankdata(b))


def _weight_corr_with_uniform(rubrics: Sequence[Rubric], *, seed: int = WEIGHT_SIMULATION_SEED) -> float:
    """Correlation between the paper's Eq. 1 reward and a plain unweighted mean.

    Replicates ``analysis/rubric_stats.py``'s simulation: draw
    :data:`WEIGHT_SIMULATION_DRAWS` random binary criterion outcomes per rubric,
    score each draw with the RaR-EXPLICIT category remap and with an unweighted
    mean, and correlate. Near 1.0 means the whole weighting scheme is an identity
    transform at the reward level — shipped: 0.943 science / 0.932 medicine
    (§9.2). Reported for every source so a win cannot come from weighting alone.
    """
    rng = np.random.default_rng(seed)
    weighted: list[float] = []
    uniform: list[float] = []
    for rubric in rubrics:
        if not rubric.items:
            continue
        w = np.asarray(
            [CATEGORY_WEIGHT_REMAP.get(c.category, 0.5) for c in rubric.items], dtype=float
        )
        if w.sum() == 0:
            continue
        for _ in range(WEIGHT_SIMULATION_DRAWS):
            outcome = rng.integers(0, 2, size=w.size).astype(float)
            weighted.append(float((w * outcome).sum() / w.sum()))
            uniform.append(float(outcome.mean()))
    return _pearson(weighted, uniform)


def _count_distribution(counts: Sequence[float], rubrics: Sequence[Rubric]) -> dict[str, Any]:
    values = np.asarray([c for c in counts if math.isfinite(c)], dtype=float)
    out: dict[str, Any] = {
        "items_mean": NAN,
        "items_std": NAN,
        "items_cv": NAN,
        "items_min": NAN,
        "items_max": NAN,
        "items_modal": NAN,
        "modal_count_share": NAN,
        "top2_count_share": NAN,
        "prompt_range_used": NAN,
        "weight_corr_with_uniform": _weight_corr_with_uniform(rubrics),
    }
    if values.size == 0:
        return out
    histogram = collections.Counter(int(v) for v in values)
    ordered = histogram.most_common()
    mean = float(values.mean())
    out.update(
        {
            "items_mean": mean,
            "items_std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "items_cv": (float(values.std(ddof=1)) / mean) if values.size > 1 and mean else NAN,
            "items_min": float(values.min()),
            "items_max": float(values.max()),
            "items_modal": float(ordered[0][0]),
            "modal_count_share": ordered[0][1] / values.size,
            "top2_count_share": sum(n for _v, n in ordered[:2]) / values.size,
            # Share of the prompt's allowed 7-20 window (14 values) actually used.
            "prompt_range_used": (values.max() - values.min() + 1) / 14.0,
            "items_histogram": {str(k): int(v) for k, v in sorted(histogram.items())},
        }
    )
    return out


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

#: Per-question metrics that get a mean + bootstrap CI in the summary.
_SUMMARY_METRICS: tuple[str, ...] = (
    "n_items",
    "items_per_subpart",
    "orphan_subpart_rate",
    "mean_best_subpart_overlap",
    "weight_mass",
    "weight_mass_per_criterion",
    "paper_weight_mass",
    "paper_weight_mass_per_criterion",
    "n_subparts",
    "question_words",
    "reference_words",
    "is_mcq",
)


@dataclass
class AdaptivityResults:
    """Per-question budget rows plus per-source distribution statistics."""

    per_question_rows: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Kept outside the dataclass fields: the rubrics are needed to recompute
        # the weight simulation in summary(), but they must not travel into any
        # serialisation of this result object.
        self.rubrics: dict[str, dict[str, Rubric]] = {}

    def summary(self) -> dict[str, Any]:
        """``{rubric_source: {metric: value}}`` plus underscore-prefixed metadata.

        Per-question metrics carry ``m``, ``m_ci_low``, ``m_ci_high``, ``m_n``.
        The distribution-level statistics have no per-question counterpart and so
        appear as bare values:

        ``items_cv``
            Coefficient of variation of ``n_items``. **Higher is better** here —
            it means the budget moves with the question. Shipped science 0.107;
            F1's target is > 0.35.
        ``modal_count_share`` / ``items_modal`` / ``top2_count_share``
            How concentrated the budget is. Shipped science puts 64.6% of
            questions on exactly 7 items; target < 0.25.
        ``r_items_vs_question_len`` / ``r_items_vs_reference_len`` /
        ``r_items_vs_subparts``
            Pearson correlation of ``n_items`` with each complexity proxy
            (Spearman under ``rho_...``). Shipped science r = 0.19 vs question
            length; target > 0.5.
        ``weight_corr_with_uniform``
            Eq. 1 reward vs an unweighted mean over the same criteria; shipped
            0.943 / 0.932 (§9.2). Descriptive.

        Metadata: ``_metric_family``, ``_params``, ``_by_domain`` (the same
        statistics split by domain, which is where the published numbers live)
        and ``_shipped_full_corpus_reference``.
        """
        out: dict[str, Any] = {}
        for source in self._sources():
            rows = [r for r in self.per_question_rows if r.get("rubric_source") == source]
            out[source] = self._entry(rows, source)

        by_domain: dict[str, dict[str, Any]] = {}
        for domain in sorted({str(r.get("domain")) for r in self.per_question_rows}):
            by_domain[domain] = {}
            for source in self._sources():
                rows = [
                    r
                    for r in self.per_question_rows
                    if r.get("rubric_source") == source and str(r.get("domain")) == domain
                ]
                if rows:
                    by_domain[domain][source] = self._entry(rows, source, with_ci=False)

        out["_metric_family"] = "adaptivity"
        out["_params"] = dict(self.params)
        out["_by_domain"] = by_domain
        out["_shipped_full_corpus_reference"] = SHIPPED_FULL_CORPUS_REFERENCE
        out["_n_errors"] = len(self.errors)
        return out

    def _entry(
        self, rows: Sequence[Mapping[str, Any]], source: str, *, with_ci: bool = True
    ) -> dict[str, Any]:
        entry: dict[str, Any] = {
            "n_questions": len(rows),
            "n_errors": sum(1 for r in rows if r.get("error")),
        }
        for metric in _SUMMARY_METRICS:
            values = [r.get(metric) for r in rows]
            if with_ci:
                ci = mean_ci(values, iters=2000, seed=0)
                entry[metric] = ci["mean"]
                entry[f"{metric}_ci_low"] = ci["ci_low"]
                entry[f"{metric}_ci_high"] = ci["ci_high"]
                entry[f"{metric}_n"] = ci["n"]
            else:
                entry[metric] = _safe_mean(values)

        counts = [float(r.get("n_items", NAN) or NAN) for r in rows]
        rubrics = [
            self.rubrics.get(source, {}).get(str(r.get("uid"))) for r in rows
        ]
        entry.update(_count_distribution(counts, [r for r in rubrics if r is not None]))

        for key, column in (
            ("question_len", "question_words"),
            ("reference_len", "reference_words"),
            ("subparts", "n_subparts"),
        ):
            proxy = [float(r.get(column, NAN) or 0.0) for r in rows]
            entry[f"r_items_vs_{key}"] = _pearson(counts, proxy)
            entry[f"rho_items_vs_{key}"] = _spearman(counts, proxy)

        multi = [r for r in rows if float(r.get("n_subparts") or 0) >= 2]
        entry["n_multi_part_questions"] = len(multi)
        entry["items_per_subpart_multi_part"] = (
            float(np.mean([r["items_per_subpart"] for r in multi])) if multi else NAN
        )
        return entry

    def _sources(self) -> list[str]:
        seen: list[str] = []
        for row in self.per_question_rows:
            source = str(row.get("rubric_source"))
            if source not in seen:
                seen.append(source)
        return seen


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_adaptivity(
    engine: Any,  # unused: accepted for API uniformity across metric modules
    examples: Sequence[Example],
    rubrics_by_source: Mapping[str, Mapping[str, Rubric]],
    *,
    config: EvalConfig,
    run_dir: RunDir | None = None,
) -> AdaptivityResults:
    """Measure how each source's item budget responds to question complexity."""
    del engine, config  # zero-LLM metric; kept in the signature for uniformity

    results = AdaptivityResults(
        params={
            "sources": list(rubrics_by_source),
            "orphan_overlap_threshold": ORPHAN_OVERLAP_THRESHOLD,
            "orphan_overlap_measure": "binary content-token cosine |A n B| / sqrt(|A||B|)",
            "weight_simulation_draws": WEIGHT_SIMULATION_DRAWS,
            "subpart_detector": "analysis.rubric_grounding.detect_subparts",
        }
    )
    examples = [ex for ex in examples if ex is not None]
    if not examples:
        return results
    results.rubrics = {
        source: dict(per_uid) for source, per_uid in rubrics_by_source.items()
    }

    complexity = {ex.uid: question_complexity(ex) for ex in examples}
    results.params["n_multi_part_questions"] = sum(
        1 for c in complexity.values() if c.n_subparts >= 2
    )
    results.params["n_mcq_questions"] = sum(1 for c in complexity.values() if c.is_mcq)

    for ex in examples:
        for source, per_uid in rubrics_by_source.items():
            rubric = per_uid.get(ex.uid)
            if rubric is None:
                continue
            try:
                row = _score_rubric(ex, source, rubric, complexity[ex.uid])
            except Exception as exc:  # noqa: BLE001 - degrade per (question, source)
                logger.warning("adaptivity failed for %s/%s: %s", source, ex.uid, exc)
                row = {
                    "uid": ex.uid,
                    "domain": ex.domain,
                    "rubric_source": source,
                    "n_items": len(rubric),
                    "error": f"exception: {exc}"[:300],
                }
            results.per_question_rows.append(row)
            if row.get("error"):
                results.errors.append(
                    {"uid": ex.uid, "rubric_source": source, "error": row["error"]}
                )

    _persist(results, run_dir)
    return results


def _score_rubric(
    example: Example, source: str, rubric: Rubric, complexity: QuestionComplexity
) -> dict[str, Any]:
    n = len(rubric.items)
    raw_mass = float(sum(c.magnitude for c in rubric.items))
    paper_mass = float(
        sum(CATEGORY_WEIGHT_REMAP.get(c.category, 0.5) for c in rubric.items)
    )
    coverage = subpart_coverage(complexity.subpart_texts, rubric.items)
    return {
        "uid": example.uid,
        "domain": example.domain,
        "rubric_source": source,
        "n_items": n,
        **complexity.to_dict(),
        # A single-part question is treated as one task, matching §8.1's
        # "extra items per additional sub-part" regression.
        "items_per_subpart": n / max(1, complexity.n_subparts),
        "weight_mass": raw_mass,
        "weight_mass_per_criterion": raw_mass / n if n else NAN,
        "paper_weight_mass": paper_mass,
        "paper_weight_mass_per_criterion": paper_mass / n if n else NAN,
        **coverage,
        "error": None if n else "empty rubric",
    }


def _persist(results: AdaptivityResults, run_dir: RunDir | None) -> None:
    if run_dir is None:
        return
    try:
        for row in results.per_question_rows:
            run_dir.writer("adaptivity_per_question").write(row)
        run_dir.write_json("adaptivity_summary.json", results.summary())
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to persist adaptivity artefacts: %s", exc)
