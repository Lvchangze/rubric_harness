"""Leakage-free rubric evaluation against real model rollouts.

This is the RL-faithful counterpart to ``discriminative.py``. There, positives
and negatives are synthesised *from* ``reference_answer`` and the reference
itself is the top of the ladder, so a generator that legitimately consults the
reference looks like it is cheating. Here the scored objects are fresh policy
rollouts and the labels come from an independent oracle, so the reference is
never a scored object and the gold-side filter has no path to leak.

Metrics, in the order they answer the question "is this rubric a usable reward":

``auc``
    Can the rubric's score rank a correct rollout above an incorrect one?
    Computed per question over that question's rollouts, then averaged. This is
    the headline.
``best_of_n_accuracy``
    Pick the highest-scoring rollout; how often is it actually correct? This is
    the most direct analogue of what a reward model does in practice, and it is
    reported against both a random-choice floor and the oracle ceiling.
``spearman``
    Rank correlation between score and correctness, as a monotonicity check that
    does not assume a threshold.
``separation``
    Mean score of correct rollouts minus mean score of incorrect ones, with a
    scale-normalised variant so a rubric cannot win by inflating everything.
``length_hack_r``
    Among rollouts the oracle marked **incorrect**, the correlation between
    rubric score and response length. Positive means the rubric pays for verbose
    wrong answers — a real reward-hacking signal measured on real failures
    rather than on a synthesised "verbose but empty" tier.

Only questions with mixed labels contribute: if every rollout is correct (or
every one wrong) there is no pair to order, and averaging in a 0.5 would dilute
real signal with noise. Those questions are counted and reported separately.
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
from .judge import PolarityMode, judge_rubric
from .rollouts import RolloutSet
from .stats import auc, mean_ci, paired_diff, rankdata

logger = logging.getLogger(__name__)

NAN = float("nan")


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    """Spearman rho via Pearson on ranks; NaN when either side is constant."""
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 3:
        return NAN
    ra, rb = rankdata(a), rankdata(b)
    sa, sb = ra.std(), rb.std()
    if sa < 1e-12 or sb < 1e-12:
        return NAN
    return float(((ra - ra.mean()) * (rb - rb.mean())).mean() / (sa * sb))


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    keep = np.isfinite(a) & np.isfinite(b)
    a, b = a[keep], b[keep]
    if a.size < 3:
        return NAN
    sa, sb = a.std(), b.std()
    if sa < 1e-12 or sb < 1e-12:
        return NAN
    return float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))


def _best_of_n(scores: Sequence[float], correct: Sequence[bool]) -> float:
    """Accuracy of picking the top-scoring rollout; ties share credit.

    Ties are resolved by expected value over the tied set rather than by taking
    the first, which would otherwise reward whatever order the rollouts happen
    to be stored in — a rubric that scores everything identically must land on
    the random-choice baseline, not above it.
    """
    if not scores:
        return NAN
    arr = np.asarray(scores, dtype=float)
    best = np.flatnonzero(arr >= arr.max() - 1e-12)
    return float(np.mean([1.0 if correct[i] else 0.0 for i in best]))


@dataclass
class RolloutEvalResults:
    per_question_rows: list[dict[str, Any]] = field(default_factory=list)
    per_rollout_rows: list[dict[str, Any]] = field(default_factory=list)
    #: One row per (rubric, rollout, criterion). Aggregated scores alone cannot
    #: support any after-the-fact question about *which* criteria fired — the
    #: polarity sensitivity analysis, for one, is uncomputable without these —
    #: and re-deriving them later needs the judge cache to still exist.
    per_criterion_rows: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        by_source: dict[str, dict[str, Any]] = {}
        sources = sorted({r["rubric_source"] for r in self.per_question_rows})
        metrics = [
            "auc", "best_of_n_accuracy", "best_of_n_lift", "spearman",
            "separation", "z_separation", "mean_correct", "mean_incorrect",
            "random_baseline", "length_hack_r", "score_std",
        ]
        for source in sources:
            rows = [r for r in self.per_question_rows if r["rubric_source"] == source]
            entry: dict[str, Any] = {"n_questions": len(rows)}
            for metric in metrics:
                stats = mean_ci([r.get(metric) for r in rows])
                entry[metric] = stats["mean"]
                entry[f"{metric}_ci_low"] = stats["ci_low"]
                entry[f"{metric}_ci_high"] = stats["ci_high"]
                entry[f"{metric}_n"] = stats["n"]
            # Pooled over all incorrect rollouts, which is the honest way to read
            # a length correlation: per-question samples are tiny.
            pooled = [
                (r["score"], r["n_chars"])
                for r in self.per_rollout_rows
                if r["rubric_source"] == source and r.get("correct") is False
            ]
            entry["length_hack_r_pooled"] = (
                _pearson([p[0] for p in pooled], [p[1] for p in pooled]) if len(pooled) > 2 else NAN
            )
            entry["n_incorrect_rollouts"] = len(pooled)
            by_source[source] = entry
        return {
            "metric_family": "rollout_eval",
            "by_source": by_source,
            "n_errors": len(self.errors),
            **self.meta,
        }


async def run_rollout_eval(
    engine: LLMEngine,
    examples: Sequence[Example],
    rubrics_by_source: Mapping[str, Mapping[str, Rubric]],
    rollout_sets: Mapping[str, RolloutSet],
    *,
    config: EvalConfig | None = None,
    run_dir: RunDir | None = None,
) -> RolloutEvalResults:
    """Score every source's rubric against the shared rollout set."""
    config = config or EvalConfig()
    results = RolloutEvalResults()
    by_uid = {ex.uid: ex for ex in examples}
    sources = list(rubrics_by_source)

    informative = {
        uid: rs for uid, rs in rollout_sets.items()
        if rs.is_informative and uid in by_uid
    }
    skipped = {
        uid: rs for uid, rs in rollout_sets.items()
        if uid in by_uid and not rs.is_informative
    }
    results.meta = {
        "n_informative_questions": len(informative),
        "n_skipped_uninformative": len(skipped),
        "n_skipped_all_correct": sum(
            1 for rs in skipped.values() if rs.usable and rs.n_correct == len(rs.usable)
        ),
        "n_skipped_all_incorrect": sum(
            1 for rs in skipped.values() if rs.usable and rs.n_correct == 0
        ),
        "aggregation": str(config.aggregation),
    }
    logger.info(
        "rollout eval: %d informative questions (%d skipped: labels not mixed)",
        len(informative), len(skipped),
    )
    if not informative:
        return results

    # The unit of work is one (question, source) pair, not one question. Looping
    # over sources inside a question would serialise seven independent judge
    # batches behind each other and leave the engine's concurrency mostly idle.
    lock = asyncio.Lock()

    async def score_pair(uid: str, rollout_set: RolloutSet, source: str) -> None:
        example = by_uid[uid]
        candidates = rollout_set.as_candidates()
        rubric = (rubrics_by_source.get(source) or {}).get(uid)
        if rubric is None or not rubric.items:
            results.errors.append({"uid": uid, "rubric_source": source,
                                   "error": "missing or empty rubric"})
            return
        verdicts = await asyncio.gather(
            *(
                judge_rubric(
                    engine, uid=uid, question=example.question, response=c.text,
                    rubric=rubric, rubric_source=source, response_id=c.response_id,
                    shuffle=config.shuffle_criteria, shuffle_seed=config.shuffle_seed,
                    max_tokens=config.judge_max_tokens,
                    polarity_mode=PolarityMode.coerce_mode(config.polarity_mode),
                )
                for c in candidates
            ),
            return_exceptions=True,
        )
        scores: list[float] = []
        correct: list[bool] = []
        lengths: list[float] = []
        for candidate, verdict in zip(candidates, verdicts):
            if isinstance(verdict, BaseException) or getattr(verdict, "error", None):
                results.errors.append({"uid": uid, "rubric_source": source,
                                       "response_id": candidate.response_id,
                                       "error": str(verdict)[:200]})
                continue
            score = float(getattr(verdict, "score", NAN))
            if not math.isfinite(score):
                continue
            is_correct = bool(candidate.meta.get("correct"))
            scores.append(score)
            correct.append(is_correct)
            lengths.append(float(candidate.meta.get("n_chars") or len(candidate.text)))
            results.per_rollout_rows.append({
                "uid": uid, "domain": rollout_set.domain, "rubric_source": source,
                "response_id": candidate.response_id,
                "setting": candidate.meta.get("setting"),
                "score": score,
                "score_numeric": getattr(verdict, "score_numeric", None),
                "score_unweighted": getattr(verdict, "score_unweighted", None),
                "scores_by_polarity_mode": getattr(verdict, "scores_by_polarity_mode", {}),
                "correct": is_correct,
                "n_chars": int(lengths[-1]), "n_items": getattr(verdict, "n_items", None),
            })
            for cv in getattr(verdict, "verdicts", []) or []:
                results.per_criterion_rows.append({
                    "uid": uid, "domain": rollout_set.domain,
                    "rubric_source": source, "response_id": candidate.response_id,
                    "correct": is_correct,
                    "criterion_index": cv.index, "title": cv.title,
                    "category": cv.category.value, "weight": int(cv.weight),
                    "polarity": cv.polarity.value,
                    "polarity_method": getattr(
                        rubric.items[cv.index] if cv.index < len(rubric.items) else None,
                        "polarity_method", "",
                    ),
                    "literally_true": bool(cv.literally_true),
                    "met": bool(cv.met), "checkable": bool(cv.checkable),
                    "aggregation": str(config.aggregation),
                    "polarity_mode": str(config.polarity_mode),
                })

        if len(scores) < 2 or len(set(correct)) < 2:
            results.errors.append({"uid": uid, "rubric_source": source,
                                   "error": "fewer than 2 usable scored rollouts, or labels no longer mixed"})
            return

        good = [s for s, c in zip(scores, correct) if c]
        bad = [s for s, c in zip(scores, correct) if not c]
        sd = float(np.std(scores, ddof=1)) if len(scores) > 1 else NAN
        separation = float(np.mean(good) - np.mean(bad))
        wrong_scores = [s for s, c in zip(scores, correct) if not c]
        wrong_lengths = [l for l, c in zip(lengths, correct) if not c]

        results.per_question_rows.append({
            "uid": uid, "domain": rollout_set.domain, "rubric_source": source,
            "n_rollouts": len(scores),
            "n_correct": len(good), "n_incorrect": len(bad),
            "auc": auc(scores, correct),
            "best_of_n_accuracy": _best_of_n(scores, correct),
            "best_of_n_lift": _best_of_n(scores, correct) - (len(good) / len(scores)),
            "random_baseline": len(good) / len(scores),
            "spearman": _spearman(scores, [1.0 if c else 0.0 for c in correct]),
            "separation": separation,
            "z_separation": separation / sd if math.isfinite(sd) and sd > 1e-9 else NAN,
            "mean_correct": float(np.mean(good)),
            "mean_incorrect": float(np.mean(bad)),
            "score_std": sd,
            "length_hack_r": _pearson(wrong_scores, wrong_lengths),
        })

    done = 0
    pairs = [(uid, rs, src) for uid, rs in sorted(informative.items()) for src in sources]

    async def guarded(uid: str, rollout_set: RolloutSet, source: str) -> None:
        nonlocal done
        try:
            await score_pair(uid, rollout_set, source)
        except Exception as exc:  # noqa: BLE001
            logger.exception("rollout eval failed for %s/%s", uid, source)
            results.errors.append({"uid": uid, "rubric_source": source,
                                   "error": f"{type(exc).__name__}: {exc}"})
        async with lock:
            done += 1
            if done % 50 == 0:
                logger.info("rollout eval %d/%d (question, source) pairs", done, len(pairs))

    await asyncio.gather(*(guarded(uid, rs, src) for uid, rs, src in pairs))

    # Row files are written by the caller (`scripts/eval_rubrics.py`), which
    # merges them per source rather than truncating, so they are not written
    # here as well.
    return results


def paired_table(
    rows: Sequence[Mapping[str, Any]], *, metric: str, reference: str = "baseline"
) -> dict[str, Any]:
    """Paired contrasts of one metric against ``reference``, over shared uids."""
    by_source: dict[str, dict[str, float]] = {}
    for row in rows:
        by_source.setdefault(row["rubric_source"], {})[row["uid"]] = row.get(metric, NAN)
    if reference not in by_source:
        return {}
    shared = set.intersection(*(set(v) for v in by_source.values())) if by_source else set()
    uids = sorted(shared)
    out: dict[str, Any] = {"metric": metric, "reference": reference, "n": len(uids)}
    for source, values in by_source.items():
        out[source] = mean_ci([values[u] for u in uids])
        if source != reference:
            out[f"delta__{source}"] = paired_diff(
                [values[u] for u in uids], [by_source[reference][u] for u in uids]
            )
    return out
