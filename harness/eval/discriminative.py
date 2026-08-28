"""Metric family 1 — does a rubric separate good responses from bad ones?

This is the headline result. For every question we hold the candidate responses
fixed (see :mod:`harness.eval.responses`) and vary only the rubric source, then
ask how well the resulting scores reproduce the ladder's known quality order.

Four views, because they fail differently:

* **Margins.** ``gold - degraded`` per question. ``min_margin`` is the strict
  reading: a rubric that cannot separate gold from its *easiest* degradation is
  weak no matter how well it does on average.
* **Ranking / AUC.** Order agreement over pairs of *different* known quality.
  Ties get half credit — the same convention as the tie-corrected Mann-Whitney
  statistic used for AUC, so the two numbers stay comparable — and the tie rate
  is reported separately because a rubric that ties everything is useless as a
  reward even at 0.5 "accuracy".
* **Dynamic range.** Per-question spread of scores. A rubric that puts every
  response at 0.8 carries no gradient.
* **Reward-hacking probes.** Verbose-but-empty and confidently-wrong responses,
  called out on their own, plus the saturation rate of degraded responses.

Every metric is computed under all three of the judge's weightings — ``.score``
(RaR categorical weights, primary, unsuffixed), ``.score_numeric``
(``__numeric``) and ``.score_unweighted`` (``__unweighted``) — so "the agentic
rubrics just weight better" is checkable rather than arguable.

Confidence intervals are deliberately *not* computed here. This module emits
clean per-question rows keyed by ``(uid, rubric_source)``; the statistics module
owns the paired bootstrap over them.
"""

from __future__ import annotations

import asyncio
import logging
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np

from ..config import EvalConfig
from ..llm import LLMEngine
from ..schema import Example, Rubric
from ..tracing import RunDir
from .judge import JudgeResult, PolarityMode, judge_many
from .responses import GOOD_QUALITY_THRESHOLD, QUALITY_LEVELS, ResponseSet

logger = logging.getLogger(__name__)

__all__ = [
    "SATURATION_THRESHOLD",
    "SCORE_FIELDS",
    "DiscriminativeResults",
    "run_discriminative",
    "rank_auc",
    "pairwise_rank_accuracy",
    "average_ranks",
]

# "Scored as if it were a good answer": the level above which a degraded
# response is indistinguishable from gold for reward purposes.
SATURATION_THRESHOLD = 0.8

# (JudgeResult attribute, key suffix). The primary weighting is unsuffixed so
# report code can read ``rank_acc`` without knowing about the robustness pair.
SCORE_FIELDS: tuple[tuple[str, str], ...] = (
    ("score", ""),
    ("score_numeric", "__numeric"),
    ("score_unweighted", "__unweighted"),
)

_ROW_META_KEYS = frozenset(
    {"uid", "domain", "rubric_source", "response_id", "variant", "response_set_fingerprint",
     "variants_scored", "error", "validation_ok"}
)


# ---------------------------------------------------------------------------
# Ranking primitives (numpy only — no sklearn in this project)
# ---------------------------------------------------------------------------


def average_ranks(values: np.ndarray) -> np.ndarray:
    """Midranks (1-based), tied values sharing the mean of their positions."""
    order = np.argsort(values, kind="mergesort")
    ordered = values[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    ends = np.r_[starts[1:], ordered.size]
    ranks_sorted = np.empty(ordered.size, dtype=float)
    for start, end in zip(starts, ends):
        ranks_sorted[start:end] = 0.5 * (start + end - 1) + 1.0
    ranks = np.empty(values.size, dtype=float)
    ranks[order] = ranks_sorted
    return ranks


def rank_auc(positive: np.ndarray, negative: np.ndarray) -> float | None:
    """AUC via the tie-corrected Mann-Whitney U statistic.

    ``U / (n_pos * n_neg)`` where ties contribute 0.5, i.e. exactly the
    probability that a random positive outranks a random negative with ties
    broken by a coin flip. ``None`` when either class is empty.
    """
    n_pos, n_neg = positive.size, negative.size
    if n_pos == 0 or n_neg == 0:
        return None
    ranks = average_ranks(np.concatenate([positive, negative]))
    u = float(ranks[:n_pos].sum()) - n_pos * (n_pos + 1) / 2.0
    return u / (n_pos * n_neg)


def pairwise_rank_accuracy(
    scores: np.ndarray, levels: np.ndarray
) -> tuple[float | None, float, int, int]:
    """Order agreement over pairs of *different* known quality.

    Returns ``(accuracy, credit, n_pairs, n_ties)``. Only pairs whose ground
    truth levels differ are counted, so the deliberate ties in
    :data:`~harness.eval.responses.QUALITY_LEVELS` never punish a rubric. A
    scoring tie earns 0.5 credit and is also counted in ``n_ties``.
    """
    credit = 0.0
    n_pairs = n_ties = 0
    for i in range(scores.size):
        for j in range(i + 1, scores.size):
            if levels[i] == levels[j]:
                continue
            n_pairs += 1
            better, worse = (i, j) if levels[i] > levels[j] else (j, i)
            if scores[better] > scores[worse]:
                credit += 1.0
            elif scores[better] == scores[worse]:
                credit += 0.5
                n_ties += 1
    return (credit / n_pairs if n_pairs else None), credit, n_pairs, n_ties


def _finite(value: float | None) -> float | None:
    """Guard the JSONL artefacts against NaN/Inf, which are not valid JSON."""
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


# ---------------------------------------------------------------------------
# Per-question metric block
# ---------------------------------------------------------------------------


def _question_metrics(
    scores: Mapping[str, float], levels: Mapping[str, int]
) -> dict[str, Any]:
    """All scalars for one (question, rubric_source, weighting).

    ``scores`` holds one usable score per variant; ``levels`` their ground-truth
    quality. Counts are emitted alongside the ratios so the caller can pool them
    into micro-averages instead of averaging ratios of different denominators.
    """
    variants = list(scores)
    arr = np.array([float(scores[v]) for v in variants], dtype=float)
    lvl = np.array([int(levels[v]) for v in variants], dtype=int)
    gold = scores.get("gold")
    degraded = [v for v in variants if v != "gold"]

    out: dict[str, Any] = {}

    for variant in degraded:
        out[f"margin__{variant}"] = (
            _finite(gold - scores[variant]) if gold is not None else None
        )
    if gold is not None and degraded:
        margins = np.array([gold - scores[v] for v in degraded], dtype=float)
        out["mean_margin"] = _finite(margins.mean())
        out["min_margin"] = _finite(margins.min())
    else:
        out["mean_margin"] = None
        out["min_margin"] = None

    accuracy, credit, n_pairs, n_ties = pairwise_rank_accuracy(arr, lvl)
    out["rank_acc"] = _finite(accuracy)
    out["rank_credit"] = credit
    out["n_rank_pairs"] = n_pairs
    out["n_rank_ties"] = n_ties
    out["rank_tie_rate"] = _finite(n_ties / n_pairs) if n_pairs else None

    positives = {
        "good_vs_bad": lvl >= GOOD_QUALITY_THRESHOLD,
        "gold_vs_degraded": np.array([v == "gold" for v in variants], dtype=bool),
    }
    for name, mask in positives.items():
        pos, neg = arr[mask], arr[~mask]
        auc = rank_auc(pos, neg)
        out[f"auc_{name}"] = _finite(auc)
        out[f"n_auc_pairs_{name}"] = int(pos.size * neg.size)
        out[f"auc_credit_{name}"] = (
            _finite(auc * pos.size * neg.size) if auc is not None else None
        )

    out["score_mean"] = _finite(arr.mean())
    out["score_std"] = _finite(arr.std(ddof=0))
    out["score_range"] = _finite(arr.max() - arr.min())

    for probe, variant in (
        ("hack_verbose", "verbose_empty"),
        ("hack_confident_wrong", "right_method_wrong_answer"),
    ):
        probe_score = scores.get(variant)
        out[probe] = _finite(probe_score)
        out[f"{probe}_margin"] = (
            _finite(gold - probe_score)
            if probe_score is not None and gold is not None
            else None
        )

    n_saturated = sum(1 for v in degraded if scores[v] >= SATURATION_THRESHOLD)
    out["n_degraded"] = len(degraded)
    out["n_saturated"] = n_saturated
    out["saturation_rate"] = _finite(n_saturated / len(degraded)) if degraded else None
    return out


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

# Micro-averaged views: ratio-of-sums over questions, which weights a question
# by how many comparisons it actually contributed.
_MICRO: tuple[tuple[str, str, str], ...] = (
    ("rank_acc_micro", "rank_credit", "n_rank_pairs"),
    ("rank_tie_rate_micro", "n_rank_ties", "n_rank_pairs"),
    ("auc_good_vs_bad_micro", "auc_credit_good_vs_bad", "n_auc_pairs_good_vs_bad"),
    ("auc_gold_vs_degraded_micro", "auc_credit_gold_vs_degraded", "n_auc_pairs_gold_vs_degraded"),
    ("saturation_rate_micro", "n_saturated", "n_degraded"),
)


def _is_summed(key: str) -> bool:
    return key.startswith("n_") or "_credit" in key


def _aggregate_rows(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Macro-average (or sum, for counts) each numeric column, then add micros."""
    keys: list[str] = list(
        dict.fromkeys(k for row in rows for k in row if k not in _ROW_META_KEYS)
    )
    out: dict[str, Any] = {"n_questions": len(rows)}
    for key in keys:
        values = [
            float(row[key])
            for row in rows
            if isinstance(row.get(key), (int, float)) and not isinstance(row.get(key), bool)
        ]
        if not values:
            out[key] = None
        elif _is_summed(key):
            total = sum(values)
            out[key] = int(total) if key.startswith("n_") else _finite(total)
        else:
            out[key] = _finite(sum(values) / len(values))
            if len(values) != len(rows):
                # Flag partial coverage rather than hiding it behind a mean.
                out[f"{key}__n"] = len(values)

    for name, numerator, denominator in _MICRO:
        for _, suffix in SCORE_FIELDS:
            num, den = out.get(f"{numerator}{suffix}"), out.get(f"{denominator}{suffix}")
            out[f"{name}{suffix}"] = (
                _finite(num / den) if num is not None and den else None
            )
    return out


@dataclass
class DiscriminativeResults:
    """Rows plus a per-source summary. Statistics live in another module."""

    per_response_rows: list[dict[str, Any]] = field(default_factory=list)
    per_question_rows: list[dict[str, Any]] = field(default_factory=list)
    #: One row per (rubric, response, criterion). Storing only the aggregate
    #: score makes every later question about *which* criteria fired
    #: unanswerable, and re-deriving them depends on the judge cache still being
    #: on disk — which the cache is gitignored precisely not to guarantee.
    per_criterion_rows: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    quality_levels: dict[str, int] = field(default_factory=lambda: dict(QUALITY_LEVELS))

    def rows_for(self, source: str) -> list[dict[str, Any]]:
        return [r for r in self.per_question_rows if r.get("rubric_source") == source]

    def summary(self) -> dict[str, dict[str, Any]]:
        """``{rubric_source: {metric: value}}``.

        Keys match the per-question row keys and are macro-averages over
        questions (counts are sums); ``*_micro`` keys are ratio-of-sums.
        ``<key>__n`` appears only when fewer than all questions contributed.
        """
        out: dict[str, dict[str, Any]] = {}
        for source in self.sources:
            rows = self.rows_for(source)
            if rows:
                out[source] = _aggregate_rows(rows)
                out[source].update(self._weight_informativeness(source))
        return out

    def _weight_informativeness(self, source: str) -> dict[str, Any]:
        """How much the weighting actually changes the reward (forensics F5).

        The forensics estimated this by simulating random 0/1 verdicts and found
        the paper's category weights correlate with a plain mean at Pearson 0.94
        — i.e. the whole weight scheme is nearly an identity transform. Here the
        same quantity is computed from *real* judge verdicts on the response
        ladder, which is strictly more informative than the simulation.

        A generator whose weights carry information should sit well below 0.94.
        On its own that means nothing — weights can differ from uniform and still
        be wrong — so it is only meaningful read together with the
        discriminative margins: lower correlation *and* higher margin is what
        shows the calibration earned its keep.
        """
        weighted, uniform, numeric = [], [], []
        for row in self.per_response_rows:
            if row.get("rubric_source") != source or not row.get("usable"):
                continue
            w, u, n = row.get("score"), row.get("score_unweighted"), row.get("score_numeric")
            if w is None or u is None:
                continue
            weighted.append(float(w))
            uniform.append(float(u))
            numeric.append(float(n) if n is not None else float("nan"))
        if len(weighted) < 3:
            return {}
        out: dict[str, Any] = {
            "weight_corr_with_uniform": _pearson(weighted, uniform),
            "weight_mean_abs_diff_vs_uniform": float(
                np.mean(np.abs(np.asarray(weighted) - np.asarray(uniform)))
            ),
            "weight_corr__n": len(weighted),
        }
        numeric_arr = np.asarray(numeric)
        if np.isfinite(numeric_arr).all():
            out["raw_weight_corr_with_uniform"] = _pearson(numeric, uniform)
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "quality_levels": dict(self.quality_levels),
            "good_quality_threshold": GOOD_QUALITY_THRESHOLD,
            "saturation_threshold": SATURATION_THRESHOLD,
            "n_questions": len({r["uid"] for r in self.per_question_rows}),
            "n_judgements": len(self.per_response_rows),
            "summary": self.summary(),
            "errors": list(self.errors),
        }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    """Pearson r, or None when either series is constant."""
    x, y = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if x.size < 3 or x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _usable(result: JudgeResult) -> bool:
    """A judgement is usable when at least one verdict came back."""
    return bool(result.verdicts)


def _polarity_for(source: str, config: EvalConfig) -> PolarityMode:
    """Scoring convention for one source: detection, for every source alike.

    This used to route comparator sources through ``PolarityMode.FAVOURABLE`` to
    give them the benefit of the doubt on ambiguous Pitfalls. That mode turned
    out to be an exact alias of ``DETECTED`` (see :func:`judge.effective_polarity`),
    so the concession never applied and the two paths are now collapsed rather
    than left looking like they differ.

    The real best-case comparator reading is the whole-rubric sensitivity
    analysis: every source is additionally rescored under ``ALL_AVOIDANCE`` and
    ``ALL_FAILURE`` from the same verdicts, and the comparators are then given
    whichever convention favours them most. That runs post hoc at zero cost and
    is reported alongside the main result.

    A residual asymmetry remains and cannot be removed here: our own Pitfalls
    declare their polarity (``polarity_method == "enforced"``) whereas the
    comparators' are inferred by regex, so a comparator's polarity can be
    misread and ours cannot.
    """
    return PolarityMode.coerce_mode(config.polarity_mode)


def _criterion_rows(
    example: Example,
    rubric: Rubric,
    result: JudgeResult,
    polarity_mode: PolarityMode,
) -> list[dict[str, Any]]:
    """Flatten one judgement into its individual criterion verdicts.

    ``literally_true`` is the judge's raw answer and ``met`` is the
    polarity-corrected reading; keeping both is what makes it possible to ask
    later whether a polarity convention changed anything, without re-running the
    judge. ``polarity_method`` records how the polarity was arrived at, which
    matters because our own rubrics declare it while the comparators have it
    inferred from phrasing — an asymmetry that only shows up if it is stored.
    """
    rows: list[dict[str, Any]] = []
    for verdict in result.verdicts:
        criterion = (
            rubric.items[verdict.index] if 0 <= verdict.index < len(rubric.items) else None
        )
        rows.append(
            {
                "uid": result.uid,
                "domain": example.domain,
                "rubric_source": result.rubric_source,
                "response_id": result.response_id,
                "criterion_index": verdict.index,
                "title": verdict.title,
                "category": verdict.category.value,
                "weight": int(verdict.weight),
                "polarity": verdict.polarity.value,
                "polarity_method": getattr(criterion, "polarity_method", "") or "",
                "literally_true": bool(verdict.literally_true),
                "met": bool(verdict.met),
                "checkable": bool(verdict.checkable),
                "polarity_mode": polarity_mode.value,
            }
        )
    return rows


def _response_row(
    example: Example,
    response_set: ResponseSet,
    result: JudgeResult,
    fingerprint: str,
) -> dict[str, Any]:
    candidate = response_set.get(result.response_id)
    return {
        "uid": result.uid,
        "domain": example.domain,
        "rubric_source": result.rubric_source,
        "response_id": result.response_id,
        "variant": candidate.variant if candidate else result.response_id,
        "quality_level": candidate.quality_level if candidate else None,
        "score": _finite(result.score),
        "score_numeric": _finite(result.score_numeric),
        "score_unweighted": _finite(result.score_unweighted),
        # Same verdicts re-aggregated under each Pitfall polarity convention, so
        # the F6 sensitivity table costs no extra LLM calls.
        "scores_by_polarity_mode": {
            k: _finite(v) for k, v in result.scores_by_polarity_mode.items()
        },
        "n_items": result.n_items,
        "n_met": sum(1 for v in result.verdicts if v.met),
        "n_unmet": sum(1 for v in result.verdicts if not v.met),
        "n_literally_true": sum(1 for v in result.verdicts if v.literally_true),
        "n_negative_polarity": sum(1 for v in result.verdicts if v.polarity.value == "negative"),
        "usable": _usable(result),
        "response_chars": len(candidate.text) if candidate else None,
        "validation_ok": candidate.validation_ok if candidate else None,
        "response_set_fingerprint": fingerprint,
        "error": result.error,
    }


def _question_row(
    example: Example,
    source: str,
    rubric: Rubric,
    response_set: ResponseSet,
    results: Sequence[JudgeResult],
    fingerprint: str,
) -> dict[str, Any]:
    usable: list[JudgeResult] = []
    levels: dict[str, int] = {}
    for result in results:
        candidate = response_set.get(result.response_id)
        if candidate is None or not _usable(result):
            continue
        usable.append(result)
        levels[result.response_id] = candidate.quality_level

    row: dict[str, Any] = {
        "uid": example.uid,
        "domain": example.domain,
        "rubric_source": source,
        "n_items": len(rubric),
        "n_responses_scored": len(usable),
        "n_judge_errors": sum(1 for r in results if not _usable(r)),
        "variants_scored": sorted(levels),
        "response_set_fingerprint": fingerprint,
    }
    for attribute, suffix in SCORE_FIELDS:
        scores = {r.response_id: float(getattr(r, attribute)) for r in usable}
        for variant, value in scores.items():
            row[f"{attribute}__{variant}"] = _finite(value)
        for key, value in _question_metrics(scores, levels).items():
            row[f"{key}{suffix}"] = value
    return row


async def run_discriminative(
    engine: LLMEngine,
    examples: Sequence[Example],
    rubrics_by_source: Mapping[str, Mapping[str, Rubric]],
    response_sets: Mapping[str, ResponseSet],
    *,
    config: EvalConfig | None = None,
    run_dir: RunDir | None = None,
    concurrency: int = 6,
) -> DiscriminativeResults:
    """Score every rubric source on the shared ladder and reduce to metrics.

    ``response_sets`` is keyed by ``uid`` only: there is no per-source response
    axis to get wrong, and each row records
    :meth:`~harness.eval.responses.ResponseSet.fingerprint` so a reader can
    confirm all sources were scored on identical text.

    Questions and sources degrade independently — a missing ladder, an empty
    rubric or a judge failure is recorded in :attr:`DiscriminativeResults.errors`
    and skipped. ``concurrency`` bounds questions in flight; the real limiter is
    the engine's own semaphore.
    """
    config = config or EvalConfig()
    sources = list(rubrics_by_source)
    results = DiscriminativeResults(sources=sources)
    lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(max(1, concurrency))
    # Alternative polarity conventions are recomputed from the same verdicts, so
    # the whole F6 sensitivity analysis is free.
    sensitivity_modes = [
        PolarityMode.coerce_mode(m) for m in (config.polarity_sensitivity_modes or ())
    ]

    response_writer = run_dir.writer("discriminative_responses") if run_dir else None
    question_writer = run_dir.writer("discriminative_questions") if run_dir else None

    async def one_question(example: Example) -> None:
        response_set = response_sets.get(example.uid)
        if response_set is None or len(response_set) < 2:
            async with lock:
                results.errors.append(
                    {
                        "uid": example.uid,
                        "stage": "responses",
                        "error": "missing or degenerate response set",
                        "n_responses": 0 if response_set is None else len(response_set),
                    }
                )
            return

        fingerprint = response_set.fingerprint()
        active: list[tuple[str, Rubric]] = []
        local_errors: list[dict[str, Any]] = []
        for source in sources:
            rubric = rubrics_by_source[source].get(example.uid)
            if rubric is None or len(rubric) == 0:
                local_errors.append(
                    {
                        "uid": example.uid,
                        "rubric_source": source,
                        "stage": "rubric",
                        "error": "missing or empty rubric",
                    }
                )
                continue
            active.append((source, rubric))

        jobs = [
            {
                "uid": example.uid,
                "question": example.question,
                "response": candidate.text,
                "rubric": rubric,
                "rubric_source": source,
                "response_id": candidate.response_id,
                "shuffle": bool(config.shuffle_criteria),
                "shuffle_seed": int(config.shuffle_seed),
                "max_tokens": int(config.judge_max_tokens),
                "polarity_mode": _polarity_for(source, config),
                "extra_polarity_modes": sensitivity_modes,
            }
            for source, rubric in active
            for candidate in response_set.responses
        ]
        if not jobs:
            async with lock:
                results.errors.extend(local_errors)
            return

        async with semaphore:
            judged = await judge_many(engine, jobs)

        by_source: dict[str, list[JudgeResult]] = {source: [] for source, _ in active}
        for result in judged:
            by_source.setdefault(result.rubric_source, []).append(result)

        response_rows: list[dict[str, Any]] = []
        question_rows: list[dict[str, Any]] = []
        criterion_rows: list[dict[str, Any]] = []
        for source, rubric in active:
            source_results = by_source.get(source, [])
            for result in source_results:
                row = _response_row(example, response_set, result, fingerprint)
                response_rows.append(row)
                criterion_rows.extend(
                    _criterion_rows(example, rubric, result, _polarity_for(source, config))
                )
                if result.error:
                    local_errors.append(
                        {
                            "uid": example.uid,
                            "rubric_source": source,
                            "response_id": result.response_id,
                            "stage": "judge",
                            "error": result.error,
                        }
                    )
            if not any(_usable(r) for r in source_results):
                local_errors.append(
                    {
                        "uid": example.uid,
                        "rubric_source": source,
                        "stage": "judge",
                        "error": "no usable judgements for this question",
                    }
                )
                continue
            question_rows.append(
                _question_row(
                    example, source, rubric, response_set, source_results, fingerprint
                )
            )

        async with lock:
            results.per_response_rows.extend(response_rows)
            results.per_question_rows.extend(question_rows)
            results.per_criterion_rows.extend(criterion_rows)
            results.errors.extend(local_errors)
            if response_writer is not None:
                for row in response_rows:
                    response_writer.write(row)
            if question_writer is not None:
                for row in question_rows:
                    question_writer.write(row)

    outcomes = await asyncio.gather(
        *(one_question(ex) for ex in examples), return_exceptions=True
    )
    for example, outcome in zip(examples, outcomes):
        if isinstance(outcome, BaseException):
            logger.error("discriminative failed uid=%s: %s", example.uid, outcome)
            results.errors.append(
                {"uid": example.uid, "stage": "question", "error": f"{outcome}"[:300]}
            )

    # Stable order regardless of completion order, so artefacts diff cleanly.
    source_rank = {source: i for i, source in enumerate(sources)}
    uid_rank = {ex.uid: i for i, ex in enumerate(examples)}
    level_rank = {v: -level for v, level in QUALITY_LEVELS.items()}
    results.per_question_rows.sort(
        key=lambda r: (uid_rank.get(r["uid"], 1 << 30), source_rank.get(r["rubric_source"], 99))
    )
    results.per_response_rows.sort(
        key=lambda r: (
            uid_rank.get(r["uid"], 1 << 30),
            source_rank.get(r["rubric_source"], 99),
            level_rank.get(r.get("variant", ""), 0),
        )
    )
    logger.info(
        "discriminative: %d questions x %d sources -> %d judgements, %d issues",
        len(response_sets),
        len(sources),
        len(results.per_response_rows),
        len(results.errors),
    )
    if run_dir is not None:
        run_dir.write_json("discriminative_summary.json", results.to_dict())
    return results
