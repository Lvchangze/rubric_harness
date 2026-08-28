"""Metric family 5 — blind pairwise preference between rubric sources.

Supporting evidence only. "An LLM prefers rubric X" is a weaker claim than "X
separates good responses from bad ones" (family 1) or "X's criteria are actually
about this question" (family 2), because the preference judge shares the biases
of the rubric writer and cannot be validated against ground truth. It is
included because it is the comparison a reader will ask for, and because a
*disagreement* between this family and the others is itself a finding.

Two controls make the number worth reporting at all:

**Anonymisation.** The two rubrics are shown as "Rubric A" and "Rubric B" with
no provenance, and :func:`~harness.prompts.eval_prompts.render_criteria` emits
only the display fields, never the ``provenance``/``validation`` metadata that
would identify the agentic generator.

**Position swapping.** Every comparison is run twice with the slots exchanged.
An LLM asked to choose between two options has a strong, well-documented pull
toward one slot, so a raw win rate over single presentations mostly measures
that pull. Both numbers are reported:

* ``raw_win_rate`` — each of the two presentations counts as one vote. This is
  what a naive single-order experiment would have produced.
* ``corrected_win_rate`` — a source wins only if it wins in **both** orders;
  otherwise the comparison is a tie. Order-dependent preferences are exactly the
  ones the model cannot defend, so discarding them is the conservative choice.

``position_bias_rate`` is the fraction of decisive verdicts that landed on the
first slot, measured over the whole run — 0.5 means no bias, 1.0 means the model
always picked whichever rubric was shown first. ``flip_rate`` is the fraction of
comparisons whose winner changed when the slots were swapped; it is the
inconsistency the correction absorbs. Feeding two *identical* rubrics through
this module is the calibration check: the corrected win rate must come out at
0/0 with a 100% tie rate no matter how biased the raw votes are.

Cost: two calls per (question, unordered source pair).
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from ..config import EvalConfig
from ..llm import JSONParseError, LLMEngine
from ..prompts.eval_prompts import HEADTOHEAD_SYSTEM, build_headtohead_prompt
from ..schema import Example, Rubric
from ..tracing import RunDir

logger = logging.getLogger(__name__)

__all__ = ["HeadToHeadResults", "run_headtohead"]

NAN = float("nan")
TIE = "tie"


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class HeadToHeadResults:
    """Per-comparison rows plus win rates aggregated over each source pair."""

    per_pair_rows: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Win rates per unordered source pair, plus global position bias.

        Top-level keys

        ``metric_family`` / ``params``
            ``"headtohead"`` and the run parameters (sources compared, etc.).
        ``position_bias_rate``
            Fraction of decisive verdicts across the whole run that chose the
            first slot. 0.5 is unbiased; departures justify the correction. NaN
            when nothing was decisive — which is what two identical rubrics
            produce, and is why the calibration check reads the tie rate rather
            than this number.
        ``n_verdicts`` / ``n_decisive_verdicts`` / ``n_tie_verdicts`` /
        ``n_failed_verdicts``
            Individual presentations attempted, those that produced a winner,
            those the model called a tie, and those that errored.
        ``pairs``
            One entry per unordered source pair, keyed ``"<a>_vs_<b>"`` with:

            ``source_a`` / ``source_b``
                The two sources, in the order they appear in the input mapping.
            ``n_questions``
                Questions where both sources had a usable rubric.
            ``n_complete``
                Questions where *both* orders returned a verdict; only these
                enter the corrected rates.
            ``raw_win_rate_a`` / ``raw_win_rate_b`` / ``raw_tie_rate``
                Share of individual votes (2 per complete question). Naive,
                position-contaminated.
            ``corrected_win_rate_a`` / ``corrected_win_rate_b`` /
            ``corrected_tie_rate``
                Share of questions where the source won **both** orders. These
                three sum to 1 and are the numbers to quote.
            ``flip_rate``
                Share of complete questions whose winner changed on swap.
            ``n_tie_verdicts``
                Presentations the model called a tie within this pair.
            ``position_bias_rate``
                First-slot share of decisive verdicts within this pair.
        """
        pairs: dict[str, Any] = {}
        decisive = first_slot = failed = verdicts = 0

        for key, rows in self._by_pair().items():
            complete = [r for r in rows if r["n_verdicts"] == 2]
            n_complete = len(complete)
            votes_a = sum(int(r["raw_wins_a"]) for r in complete)
            votes_b = sum(int(r["raw_wins_b"]) for r in complete)
            n_votes = 2 * n_complete
            corrected_a = sum(1 for r in complete if r["corrected_winner"] == "a")
            corrected_b = sum(1 for r in complete if r["corrected_winner"] == "b")
            flips = sum(
                1
                for r in complete
                if r["verdict_order1"] != r["verdict_order2"]
            )
            pair_decisive = sum(int(r["n_decisive"]) for r in rows)
            pair_first = sum(int(r["n_first_slot_wins"]) for r in rows)

            decisive += pair_decisive
            first_slot += pair_first
            failed += sum(int(r["n_failed"]) for r in rows)
            verdicts += sum(int(r["n_verdicts"]) for r in rows)

            pairs[key] = {
                "source_a": rows[0]["source_a"],
                "source_b": rows[0]["source_b"],
                "n_questions": len(rows),
                "n_complete": n_complete,
                "raw_win_rate_a": (votes_a / n_votes) if n_votes else NAN,
                "raw_win_rate_b": (votes_b / n_votes) if n_votes else NAN,
                "raw_tie_rate": (
                    (n_votes - votes_a - votes_b) / n_votes if n_votes else NAN
                ),
                "corrected_win_rate_a": (corrected_a / n_complete) if n_complete else NAN,
                "corrected_win_rate_b": (corrected_b / n_complete) if n_complete else NAN,
                "corrected_tie_rate": (
                    (n_complete - corrected_a - corrected_b) / n_complete
                    if n_complete
                    else NAN
                ),
                "flip_rate": (flips / n_complete) if n_complete else NAN,
                "n_tie_verdicts": sum(int(r["n_verdicts"]) for r in rows) - pair_decisive,
                "position_bias_rate": (
                    (pair_first / pair_decisive) if pair_decisive else NAN
                ),
            }

        return {
            "metric_family": "headtohead",
            "params": dict(self.params),
            "position_bias_rate": (first_slot / decisive) if decisive else NAN,
            "n_verdicts": verdicts,
            "n_decisive_verdicts": decisive,
            "n_tie_verdicts": verdicts - decisive,
            "n_failed_verdicts": failed,
            "pairs": pairs,
            "n_errors": len(self.errors),
        }

    def _by_pair(self) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for row in self.per_pair_rows:
            out.setdefault(row["pair"], []).append(row)
        return out


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_headtohead(
    engine: LLMEngine,
    examples: Sequence[Example],
    rubrics_by_source: Mapping[str, Mapping[str, Rubric]],
    *,
    config: EvalConfig,
    run_dir: RunDir | None = None,
) -> HeadToHeadResults:
    """Blind pairwise preference over every unordered pair of rubric sources.

    Source pairs follow the insertion order of ``rubrics_by_source``, so the
    ``source_a``/``source_b`` labelling is stable across reruns. Questions where
    either source lacks a non-empty rubric are skipped for that pair.
    """
    sources = list(rubrics_by_source)
    source_pairs = list(itertools.combinations(sources, 2))
    results = HeadToHeadResults(
        params={"sources": sources, "n_source_pairs": len(source_pairs)}
    )

    cells: list[tuple[Example, str, str, Rubric, Rubric]] = []
    for ex in examples:
        for src_a, src_b in source_pairs:
            rub_a = rubrics_by_source[src_a].get(ex.uid)
            rub_b = rubrics_by_source[src_b].get(ex.uid)
            if rub_a is None or rub_b is None or len(rub_a) == 0 or len(rub_b) == 0:
                continue
            cells.append((ex, src_a, src_b, rub_a, rub_b))

    outcomes = await asyncio.gather(
        *(
            _compare(engine, ex, src_a, src_b, rub_a, rub_b, config)
            for ex, src_a, src_b, rub_a, rub_b in cells
        ),
        return_exceptions=True,
    )
    for (ex, src_a, src_b, rub_a, rub_b), outcome in zip(cells, outcomes):
        if isinstance(outcome, BaseException):
            row = _blank_row(ex, src_a, src_b, rub_a, rub_b)
            row["error"] = f"exception: {outcome}"[:300]
            row["n_failed"] = 2
        else:
            row = outcome
        results.per_pair_rows.append(row)
        if row.get("error"):
            results.errors.append(
                {"uid": ex.uid, "pair": row["pair"], "error": row["error"]}
            )

    _persist(results, run_dir)
    return results


def _blank_row(
    example: Example, src_a: str, src_b: str, rub_a: Rubric, rub_b: Rubric
) -> dict[str, Any]:
    return {
        "uid": example.uid,
        "domain": example.domain,
        "pair": f"{src_a}_vs_{src_b}",
        "source_a": src_a,
        "source_b": src_b,
        "n_items_a": len(rub_a),
        "n_items_b": len(rub_b),
        # Winners expressed as sources ("a"/"b"/"tie"), not slots.
        "verdict_order1": None,   # order 1: source_a shown in slot A
        "verdict_order2": None,   # order 2: source_b shown in slot A
        "raw_wins_a": 0,
        "raw_wins_b": 0,
        "corrected_winner": TIE,
        "n_verdicts": 0,
        "n_decisive": 0,
        "n_first_slot_wins": 0,
        "n_failed": 0,
        "why_order1": "",
        "why_order2": "",
        "error": None,
    }


async def _compare(
    engine: LLMEngine,
    example: Example,
    src_a: str,
    src_b: str,
    rub_a: Rubric,
    rub_b: Rubric,
    config: EvalConfig,
) -> dict[str, Any]:
    """Run one question's comparison in both slot orders and fold the verdicts."""
    row = _blank_row(example, src_a, src_b, rub_a, rub_b)
    order1, order2 = await asyncio.gather(
        _ask(engine, example, rub_a, rub_b, config),
        _ask(engine, example, rub_b, rub_a, config),
    )

    problems: list[str] = []
    # Order 1 shows source_a in slot A; order 2 shows source_b in slot A.
    for outcome, key, slot_a_source, slot_b_source in (
        (order1, "order1", "a", "b"),
        (order2, "order2", "b", "a"),
    ):
        if outcome.get("error"):
            problems.append(str(outcome["error"]))
            row["n_failed"] += 1
            continue
        row["n_verdicts"] += 1
        row[f"why_{key}"] = str(outcome.get("why", ""))[:300]
        slot = outcome["winner"]
        if slot == TIE:
            row[f"verdict_{key}"] = TIE
            continue
        winner = slot_a_source if slot == "A" else slot_b_source
        row[f"verdict_{key}"] = winner
        row["n_decisive"] += 1
        row["n_first_slot_wins"] += int(slot == "A")
        row[f"raw_wins_{winner}"] += 1

    if row["verdict_order1"] == row["verdict_order2"] and row["verdict_order1"] in {
        "a",
        "b",
    }:
        row["corrected_winner"] = row["verdict_order1"]
    else:
        row["corrected_winner"] = TIE

    if problems:
        row["error"] = "; ".join(problems)[:300]
    return row


async def _ask(
    engine: LLMEngine,
    example: Example,
    slot_a: Rubric,
    slot_b: Rubric,
    config: EvalConfig,
) -> dict[str, Any]:
    """One preference call; returns the winning *slot* ("A"/"B"/"tie")."""
    try:
        parsed = await engine.chat_json(
            build_headtohead_prompt(
                example.question, example.reference_answer, slot_a.items, slot_b.items
            ),
            system=HEADTOHEAD_SYSTEM,
            expect="object",
            max_tokens=int(config.judge_max_tokens),
            tag="headtohead",
        )
    except (JSONParseError, RuntimeError) as exc:
        return {"error": f"preference call failed: {exc}"[:250]}

    raw = str((parsed or {}).get("winner", "")).strip().lower()
    if raw in {"a", "rubric a", "rubric_a"}:
        winner = "A"
    elif raw in {"b", "rubric b", "rubric_b"}:
        winner = "B"
    elif raw in {"tie", "draw", "equal", "neither", "both"}:
        winner = TIE
    else:
        return {"error": f"unparsable winner {raw!r}"[:250]}
    return {"winner": winner, "why": (parsed or {}).get("why", "")}


def _persist(results: HeadToHeadResults, run_dir: RunDir | None) -> None:
    if run_dir is None:
        return
    try:
        for row in results.per_pair_rows:
            run_dir.writer("headtohead_per_pair").write(row)
        run_dir.write_json("headtohead_summary.json", results.summary())
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to persist head-to-head artefacts: %s", exc)
