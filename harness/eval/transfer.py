"""Metric family 2 — query specificity by rubric transfer.

The cheapest sharp test of whether a rubric is *about its question*: take
question A's rubric and apply it to the gold answer of an unrelated question B
from the same domain. A criterion that genuinely encodes what A requires
("states that the total mechanical energy before equals that after") must FAIL
on B's answer. A boilerplate criterion ("is numerically accurate", "explains the
reasoning clearly") passes on B's answer just as happily as on A's — that is
precisely what makes it worthless as a reward signal.

So: low ``cross_pass_rate`` is good, and ``specificity_gap = same-question gold
pass rate - cross_pass_rate`` is the headline number (higher is better). A rubric
that everything passes and a rubric that nothing passes both look bad on the
gap; only a rubric that separates its own answer from a stranger's scores well.

The key design decision — which question the judge sees
-------------------------------------------------------
The judge prompt always contains a question, a response and a checklist. When we
pose rubric A against answer B there are two coherent ways to fill the question
slot, and they measure different things:

``question_mode="own"`` (default)
    Show **question A** — the rubric's own question — next to **answer B**. Only
    the response text changes relative to the real evaluation, so
    ``specificity_gap`` is a clean single-factor contrast: same question, same
    rubric, same judge, different answer. The judge's task is also identical to
    the one it performs everywhere else in the harness, so no new grading
    regime is introduced.

``question_mode="distractor"``
    Show **question B** next to **answer B**, so the pair is coherent and only
    the rubric is foreign. Two things change at once (question *and* response),
    which makes the gap harder to attribute, but it lets the judge reinterpret a
    vague criterion in B's context — and that reinterpretation is itself
    diagnostic.

Measured contrast (4 ``rar_science`` questions, shipped rubrics versus a
deliberately content-free "boilerplate" rubric)::

    cross_pass_rate     mode="own"    mode="distractor"
    shipped                  0.054                0.089
    boilerplate              0.275                0.675

Both modes rank the two sources correctly, and ``"distractor"`` separates them
more aggressively because a content-free criterion becomes fully applicable once
it is shown a question it can attach to. ``"own"`` is kept as the default
precisely because it is the *conservative* choice: it yields the smaller
apparent difference between sources, so an advantage measured under ``"own"`` is
the harder one to obtain. Run both before drawing a conclusion.

Because rubric A is judged against answer B under the same judge, the same
prompt and the same shuffling as everywhere else, this metric costs one judge
call per (question, source, distractor) and needs no new grader.
"""

from __future__ import annotations

import logging
import math
import random
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..config import EvalConfig
from ..schema import Example, Rubric
from ..tracing import RunDir
from .judge import JudgeResult, judge_many
from .stats import mean_ci, safe_mean

logger = logging.getLogger(__name__)

__all__ = ["TransferResults", "build_distractor_pairs", "run_transfer"]

NAN = float("nan")
QUESTION_MODES = ("own", "distractor")
GOLD_RESPONSE_ID = "gold"


# ---------------------------------------------------------------------------
# Deterministic pairing
# ---------------------------------------------------------------------------


def build_distractor_pairs(
    examples: Sequence[Example], *, n_distractors: int = 2, seed: int = 7
) -> dict[str, list[str]]:
    """Pick ``n_distractors`` same-domain partners for each example.

    Reproducible and order-independent: the candidate pool is sorted by uid and
    the RNG is seeded from the example's own uid, so the pairing for a given
    question depends only on which questions are in the sample, not on the order
    they arrive in. An example is never paired with itself. Domains with a
    single example yield an empty list.
    """
    by_domain: dict[str, list[str]] = {}
    for ex in examples:
        by_domain.setdefault(ex.domain, []).append(ex.uid)
    for uids in by_domain.values():
        uids.sort()

    pairs: dict[str, list[str]] = {}
    for ex in examples:
        pool = [uid for uid in by_domain.get(ex.domain, []) if uid != ex.uid]
        if not pool:
            pairs[ex.uid] = []
            continue
        rng = random.Random(f"{seed}:transfer:{ex.uid}")
        pairs[ex.uid] = sorted(rng.sample(pool, k=min(int(n_distractors), len(pool))))
    return pairs


def _normalise_supplied_rates(
    supplied: Mapping[Any, Any] | None,
) -> dict[tuple[str, str], float]:
    """Accept ``{(uid, source): rate}`` or ``{"uid::source": rate}``."""
    out: dict[tuple[str, str], float] = {}
    if not supplied:
        return out
    for key, value in supplied.items():
        if isinstance(key, tuple) and len(key) == 2:
            pair = (str(key[0]), str(key[1]))
        elif isinstance(key, str) and "::" in key:
            uid, _, source = key.partition("::")
            pair = (uid, source)
        else:
            continue
        try:
            out[pair] = float(value)
        except (TypeError, ValueError):
            continue
    return out


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class TransferResults:
    """Per-question and per-criterion transfer rows plus a source-level summary."""

    per_question_rows: list[dict[str, Any]] = field(default_factory=list)
    per_criterion_rows: list[dict[str, Any]] = field(default_factory=list)
    pairs: dict[str, list[str]] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Mean and bootstrap CI of each transfer metric, per rubric source.

        Every metric ``m`` below is reported as ``m`` (mean over questions with
        a finite value), ``m_ci_low`` / ``m_ci_high`` (bootstrap percentile CI)
        and ``m_n`` (how many questions contributed).

        Keys under ``by_source[src]``

        ``n_questions`` / ``n_ok`` / ``n_errors``
            Rows for this source in total, without an ``error``, and with one.
            A row can carry an error and still contribute some metrics.
        ``cross_pass_rate``
            Mean over questions of the criterion pass rate on distractor gold
            answers. **Lower is better.** Reported with ``_ci_low``/``_ci_high``.
        ``same_question_pass_rate``
            Mean criterion pass rate on the question's own gold answer, either
            supplied by the caller or measured here.
        ``specificity_gap``
            Mean of ``same_question_pass_rate - cross_pass_rate`` per question.
            **Higher is better** — this is the headline number.
        ``boilerplate_fraction``
            Mean fraction of criteria that pass on *every* distractor answer;
            these are the question-independent ones. **Lower is better.**
        ``specific_fraction``
            Mean fraction of criteria that pass on the own gold answer and fail
            on every distractor — the ideal behaviour. NaN when gold verdicts
            were not measured here. **Higher is better.**
        ``cross_score_weighted``
            Same as ``cross_pass_rate`` but using the judge's categorical
            weighting, as a robustness check.
        ``n_criteria``
            Mean rubric length, for context.
        """
        by_source: dict[str, Any] = {}
        for source in self._sources():
            rows = [r for r in self.per_question_rows if r["rubric_source"] == source]
            n_ok = sum(1 for r in rows if not r.get("error"))
            entry: dict[str, Any] = {
                "n_questions": len(rows),
                "n_ok": n_ok,
                "n_errors": len(rows) - n_ok,
            }
            for metric in (
                "cross_pass_rate",
                "same_question_pass_rate",
                "specificity_gap",
                "boilerplate_fraction",
                "specific_fraction",
                "cross_score_weighted",
                "n_criteria",
            ):
                ci = mean_ci([r.get(metric) for r in rows], iters=2000, seed=0)
                entry[metric] = ci["mean"]
                entry[f"{metric}_ci_low"] = ci["ci_low"]
                entry[f"{metric}_ci_high"] = ci["ci_high"]
                entry[f"{metric}_n"] = ci["n"]
            by_source[source] = entry
        return {
            "metric_family": "transfer",
            "params": dict(self.params),
            "n_pairs_total": sum(len(v) for v in self.pairs.values()),
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


async def run_transfer(
    engine: Any,
    examples: Sequence[Example],
    rubrics_by_source: Mapping[str, Mapping[str, Rubric]],
    *,
    config: EvalConfig,
    n_distractors: int = 2,
    question_mode: str = "own",
    same_question_pass_rates: Mapping[Any, Any] | None = None,
    run_dir: RunDir | None = None,
    pair_seed: int | None = None,
) -> TransferResults:
    """Judge every rubric against unrelated same-domain gold answers.

    Parameters
    ----------
    rubrics_by_source:
        ``{source_name: {uid: Rubric}}``. A source missing a uid is skipped for
        that question rather than failing the run.
    n_distractors:
        Same-domain partners per question (see :func:`build_distractor_pairs`).
    question_mode:
        ``"own"`` or ``"distractor"`` — see the module docstring.
    same_question_pass_rates:
        Optional ``{(uid, source): pass_rate}`` measured elsewhere (e.g. by the
        discriminative family, which already judges gold responses). When
        omitted, this function judges each rubric against its own gold answer,
        costing one extra call per (question, source) but also enabling
        ``specific_fraction``.
    """
    if question_mode not in QUESTION_MODES:
        raise ValueError(f"question_mode must be one of {QUESTION_MODES}, got {question_mode!r}")

    seed = int(pair_seed if pair_seed is not None else config.shuffle_seed)
    by_uid = {ex.uid: ex for ex in examples}
    pairs = build_distractor_pairs(examples, n_distractors=n_distractors, seed=seed)
    supplied = _normalise_supplied_rates(same_question_pass_rates)
    measure_gold = not supplied

    results = TransferResults(
        pairs=pairs,
        params={
            "n_distractors": int(n_distractors),
            "question_mode": question_mode,
            "pair_seed": seed,
            "gold_pass_rates": "supplied" if supplied else "measured",
            "sources": list(rubrics_by_source),
        },
    )

    jobs: list[dict[str, Any]] = []
    tags: list[tuple[str, str, str]] = []  # (uid, source, response_id)
    for ex in examples:
        for source, per_uid in rubrics_by_source.items():
            rubric = per_uid.get(ex.uid)
            if rubric is None or len(rubric) == 0:
                continue
            targets: list[tuple[str, str, str]] = []
            if measure_gold or (ex.uid, source) not in supplied:
                targets.append((GOLD_RESPONSE_ID, ex.question, ex.reference_answer))
            for d_uid in pairs.get(ex.uid, []):
                distractor = by_uid.get(d_uid)
                if distractor is None:
                    continue
                shown_question = (
                    ex.question if question_mode == "own" else distractor.question
                )
                targets.append(
                    (f"xfer:{d_uid}", shown_question, distractor.reference_answer)
                )
            for response_id, shown_question, response in targets:
                jobs.append(
                    {
                        "uid": ex.uid,
                        "question": shown_question,
                        "response": response,
                        "rubric": rubric,
                        "rubric_source": source,
                        "response_id": response_id,
                        "shuffle": bool(config.shuffle_criteria),
                        "shuffle_seed": int(config.shuffle_seed),
                        "max_tokens": int(config.judge_max_tokens),
                    }
                )
                tags.append((ex.uid, source, response_id))

    judged = await judge_many(engine, jobs) if jobs else []
    by_cell: dict[tuple[str, str], dict[str, JudgeResult]] = {}
    for (uid, source, response_id), result in zip(tags, judged):
        by_cell.setdefault((uid, source), {})[response_id] = result

    for ex in examples:
        for source, per_uid in rubrics_by_source.items():
            rubric = per_uid.get(ex.uid)
            if rubric is None:
                continue
            row, criterion_rows = _assemble(
                example=ex,
                source=source,
                rubric=rubric,
                judged=by_cell.get((ex.uid, source), {}),
                distractor_uids=pairs.get(ex.uid, []),
                supplied_rate=supplied.get((ex.uid, source)),
                question_mode=question_mode,
            )
            results.per_question_rows.append(row)
            results.per_criterion_rows.extend(criterion_rows)
            if row.get("error"):
                results.errors.append(
                    {"uid": ex.uid, "rubric_source": source, "error": row["error"]}
                )

    _persist(results, run_dir)
    return results


def _assemble(
    *,
    example: Example,
    source: str,
    rubric: Rubric,
    judged: Mapping[str, JudgeResult],
    distractor_uids: Sequence[str],
    supplied_rate: float | None,
    question_mode: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Turn one cell's judge results into a per-question row + per-criterion rows."""
    n_criteria = len(rubric)
    row: dict[str, Any] = {
        "uid": example.uid,
        "domain": example.domain,
        "rubric_source": source,
        "question_mode": question_mode,
        "n_criteria": n_criteria,
        "n_distractors_requested": len(distractor_uids),
        "n_distractors_used": 0,
        "distractor_uids": list(distractor_uids),
        "cross_pass_rate": NAN,
        "cross_score_weighted": NAN,
        "same_question_pass_rate": NAN,
        "same_question_measured": False,
        "specificity_gap": NAN,
        "boilerplate_fraction": NAN,
        "n_universal_criteria": 0,
        "specific_fraction": NAN,
        "gold_pass_fraction": NAN,
        "error": None,
    }
    if n_criteria == 0:
        row["error"] = "empty rubric"
        return row, []

    gold = judged.get(GOLD_RESPONSE_ID)
    gold_met: dict[int, bool] | None = None
    if supplied_rate is not None and math.isfinite(supplied_rate):
        row["same_question_pass_rate"] = float(supplied_rate)
    elif gold is not None and not gold.error and gold.verdicts:
        row["same_question_pass_rate"] = float(gold.score_unweighted)
        row["same_question_measured"] = True
        gold_met = {v.index: bool(v.met) for v in gold.verdicts}
    elif gold is not None and gold.error:
        row["error"] = f"gold judge: {gold.error}"[:300]

    cross_results: list[JudgeResult] = []
    failed: list[str] = []
    for d_uid in distractor_uids:
        result = judged.get(f"xfer:{d_uid}")
        if result is None:
            failed.append(f"{d_uid}: no result")
        elif result.error and not result.verdicts:
            failed.append(f"{d_uid}: {result.error}")
        else:
            cross_results.append(result)

    row["n_distractors_used"] = len(cross_results)
    if not cross_results:
        detail = "; ".join(failed) if failed else "no same-domain distractor available"
        row["error"] = "; ".join(filter(None, [row.get("error"), detail]))[:300] or detail
        return row, []
    if failed:
        row["error"] = "; ".join(filter(None, [row.get("error"), *failed]))[:300]

    row["cross_pass_rate"] = safe_mean([r.score_unweighted for r in cross_results])
    row["cross_score_weighted"] = safe_mean([r.score for r in cross_results])
    if math.isfinite(row["same_question_pass_rate"]):
        row["specificity_gap"] = row["same_question_pass_rate"] - row["cross_pass_rate"]

    # Per-criterion: how many distractor answers did each criterion wave through?
    pass_counts: dict[int, int] = {i: 0 for i in range(n_criteria)}
    seen_counts: dict[int, int] = {i: 0 for i in range(n_criteria)}
    for result in cross_results:
        for verdict in result.verdicts:
            if verdict.index in seen_counts:
                seen_counts[verdict.index] += 1
                pass_counts[verdict.index] += int(verdict.met)

    criterion_rows: list[dict[str, Any]] = []
    n_universal = 0
    n_specific = 0
    n_gold_pass = 0
    for i, criterion in enumerate(rubric.items):
        seen = seen_counts[i]
        n_pass = pass_counts[i]
        universal = bool(seen > 0 and n_pass == seen)
        n_universal += int(universal)
        met_gold = None if gold_met is None else bool(gold_met.get(i, False))
        specific = None
        if met_gold is not None and seen > 0:
            specific = bool(met_gold and n_pass == 0)
            n_specific += int(specific)
            n_gold_pass += int(met_gold)
        criterion_rows.append(
            {
                "uid": example.uid,
                "rubric_source": source,
                "criterion_index": i,
                "title": criterion.title,
                "category": criterion.category.value,
                "weight": int(criterion.weight),
                "n_distractors": seen,
                "n_distractor_pass": n_pass,
                "cross_pass_rate": (n_pass / seen) if seen else NAN,
                "universal": universal,
                "gold_met": met_gold,
                "specific": specific,
            }
        )

    row["boilerplate_fraction"] = n_universal / n_criteria
    row["n_universal_criteria"] = n_universal
    if gold_met is not None:
        row["specific_fraction"] = n_specific / n_criteria
        row["gold_pass_fraction"] = n_gold_pass / n_criteria
    return row, criterion_rows


def _persist(results: TransferResults, run_dir: RunDir | None) -> None:
    if run_dir is None:
        return
    try:
        for row in results.per_question_rows:
            run_dir.writer("transfer_per_question").write(row)
        for row in results.per_criterion_rows:
            run_dir.writer("transfer_per_criterion").write(row)
        run_dir.write_json("transfer_summary.json", results.summary())
        run_dir.write_json("transfer_pairs.json", results.pairs)
    except Exception as exc:  # noqa: BLE001 - artefacts must never sink a run
        logger.warning("failed to persist transfer artefacts: %s", exc)
