#!/usr/bin/env python
"""Dump a side-by-side rubric comparison for one question, ready to paste into
the report.

Picks (or takes) a uid and renders every source's rubric plus, for the agentic
source, the Stage-6 validation verdict that kept or killed each criterion.

Examples::

    python scripts/case_study.py --run-name pilot --rank-by agentic_advantage --top 2
    python scripts/case_study.py --run-name pilot --uid sci-da904481bff8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.schema import Rubric  # noqa: E402
from harness.tracing import read_jsonl  # noqa: E402

logger = logging.getLogger("case_study")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default="pilot")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--uid", default=None)
    p.add_argument("--sources", nargs="+", default=["shipped", "baseline", "agentic"])
    p.add_argument("--rank-by", default="agentic_advantage",
                   choices=["agentic_advantage", "agentic_deficit", "none"],
                   help="pick the questions where agentic most helps (or most hurts)")
    p.add_argument("--metric", default="mean_margin")
    p.add_argument("--top", type=int, default=2)
    p.add_argument("--max-question-chars", type=int, default=2000)
    p.add_argument("--out", default=None, help="write markdown here instead of stdout")
    return p.parse_args()


def load_examples(run_path: Path) -> dict[str, dict[str, Any]]:
    return {rec["uid"]: rec for rec in read_jsonl(run_path / "examples.jsonl")}


def load_rubrics(run_path: Path, source: str) -> dict[str, Rubric]:
    out: dict[str, Rubric] = {}
    for rec in read_jsonl(run_path / f"rubrics_{source}.jsonl"):
        if rec.get("uid"):
            out[rec["uid"]] = Rubric.from_dict(rec.get("rubric") or {})
    return out


def pick_uids(run_path: Path, args: argparse.Namespace, candidates: set[str]) -> list[str]:
    if args.uid:
        return [args.uid]
    if args.rank_by == "none":
        return sorted(candidates)[: args.top]
    path = run_path / "metrics" / "discriminative_per_question_rows.jsonl"
    by_uid: dict[str, dict[str, float]] = {}
    for rec in read_jsonl(path):
        value = rec.get(args.metric)
        if value is None or rec.get("uid") not in candidates:
            continue
        by_uid.setdefault(rec["uid"], {})[rec["rubric_source"]] = float(value)
    deltas = {
        uid: scores["agentic"] - max(scores.get("baseline", 0.0), scores.get("shipped", 0.0))
        for uid, scores in by_uid.items()
        if "agentic" in scores and ("baseline" in scores or "shipped" in scores)
    }
    if not deltas:
        return sorted(candidates)[: args.top]
    reverse = args.rank_by == "agentic_advantage"
    return [uid for uid, _ in sorted(deltas.items(), key=lambda kv: kv[1], reverse=reverse)][: args.top]


def render_rubric(rubric: Rubric, *, show_validation: bool) -> str:
    lines: list[str] = []
    for i, c in enumerate(rubric.items, start=1):
        lines.append(f"{i}. **[{c.category.value} w={c.weight}] {c.title}** — {c.description}")
        extras: list[str] = []
        if show_validation and c.provenance:
            origin = c.provenance.get("evidence") or c.provenance.get("stage")
            if origin:
                extras.append(f"来源证据: `{origin}`")
        if show_validation and c.validation:
            v = c.validation
            extras.append(
                f"验证: gold_pass={v.get('gold_pass')}, "
                f"negatives_failed={v.get('n_negatives_failed')}/{v.get('n_negatives')}, "
                f"discrimination={v.get('discrimination')}"
            )
        if extras:
            lines.append(f"   - <sub>{' · '.join(str(e) for e in extras)}</sub>")
    if not lines:
        lines.append("_(空 rubric)_")
    return "\n".join(lines)


def render_case(uid: str, example: dict[str, Any], rubrics: dict[str, Rubric],
                trace_summary: dict[str, Any] | None, max_q: int) -> str:
    out = [f"### Case `{uid}` — {example.get('domain')} / {example.get('question_source')}", ""]
    question = example["question"]
    if len(question) > max_q:
        question = question[:max_q] + "\n…(截断)"
    out += ["**题目**", "", "```text", question, "```", ""]
    if trace_summary:
        out += ["**agentic 中间证据摘要**", "", "```json",
                json.dumps(trace_summary, ensure_ascii=False, indent=2)[:3000], "```", ""]
    for source, rubric in rubrics.items():
        out += [f"**`{source}` rubric（{len(rubric)} 条）**", "",
                render_rubric(rubric, show_validation=source.startswith("agentic")), ""]
    return "\n".join(out)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()
    run_path = Path(args.runs_dir) / args.run_name
    examples = load_examples(run_path)
    all_rubrics = {s: load_rubrics(run_path, s) for s in args.sources}
    common = set(examples)
    for source_rubrics in all_rubrics.values():
        common &= set(source_rubrics)

    chunks: list[str] = []
    for uid in pick_uids(run_path, args, common):
        trace_summary = None
        trace_path = run_path / "traces" / f"agentic_{uid}.json"
        if trace_path.exists():
            try:
                trace = json.loads(trace_path.read_text(encoding="utf-8"))
                trace_summary = {
                    k: trace.get(k)
                    for k in ("reconcile", "validation_summary", "stages_failed")
                    if trace.get(k) is not None
                }
            except Exception:  # noqa: BLE001
                logger.warning("could not read trace for %s", uid)
        chunks.append(
            render_case(
                uid, examples[uid],
                {s: all_rubrics[s][uid] for s in args.sources if uid in all_rubrics[s]},
                trace_summary, args.max_question_chars,
            )
        )

    markdown = "\n\n---\n\n".join(chunks)
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(markdown, encoding="utf-8")
        logger.info("wrote %s", path)
    else:
        print(markdown)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
