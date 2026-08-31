#!/usr/bin/env python3
"""Print the cases one source gets wrong and another gets right.

Reading these is the only way to form a hypothesis worth testing; the aggregate
tables say a gap exists but never say what it is made of. Restricted to a case
list so failure analysis stays inside dev.

Usage::

    python scripts/rubricbench_failures.py --wrong framed --right expert \
        --group chat --case-ids results/rubricbench/split.json:dev -n 6
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness.rubricbench import group_of, load_cases  # noqa: E402
from rubricbench_split import load_case_ids  # noqa: E402

RESULTS = Path("results/rubricbench")
LETTER = {"A": 0, "B": 1}


def verdicts(source: str, root: Path) -> dict[str, dict]:
    path = root / f"{source}_verdicts.jsonl"
    rows = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            rows[row["case_id"]] = row
    return rows


def hit(row: dict | None) -> int | None:
    if row is None or row.get("forward") is None:
        return None
    return int(LETTER[row["forward"]] == row["label"])


def rubrics(source: str, root: Path) -> dict[str, str]:
    path = root / f"{source}_rubrics.json"
    if not path.exists():
        return {}
    return {r["case_id"]: r["rubric"] for r in json.loads(path.read_text(encoding="utf-8"))}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--wrong", default="framed")
    p.add_argument("--right", default="expert")
    p.add_argument("--group", default=None)
    p.add_argument("--case-ids", default="results/rubricbench/split.json:dev")
    p.add_argument("-n", type=int, default=6)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--chars", type=int, default=1100, help="per-response excerpt budget")
    p.add_argument("--results-dir", default=str(RESULTS))
    p.add_argument("--ids-only", action="store_true")
    args = p.parse_args()

    root = Path(args.results_dir)
    wrong_v, right_v = verdicts(args.wrong, root), verdicts(args.right, root)
    wrong_r, right_r = rubrics(args.wrong, root), rubrics(args.right, root)
    allowed = set(load_case_ids(args.case_ids))
    cases = {c.case_id: c for c in load_cases()}

    picked = [
        cid for cid in sorted(allowed)
        if hit(wrong_v.get(cid)) == 0 and hit(right_v.get(cid)) == 1
        and (args.group is None or group_of(cases[cid].domain) == args.group)
    ]
    print(f"{len(picked)} cases where `{args.wrong}` is wrong and `{args.right}` is right"
          f"{f' in {args.group.upper()}' if args.group else ''} "
          f"(of {sum(1 for c in allowed if args.group is None or group_of(cases[c].domain) == args.group)})")
    if args.ids_only:
        print(" ".join(picked))
        return 0

    for cid in random.Random(args.seed).sample(picked, min(args.n, len(picked))):
        case = cases[cid]
        won = "A" if case.label == 0 else "B"
        lost = "B" if case.label == 0 else "A"
        print("\n" + "=" * 100)
        print(f"{cid}  domain={case.domain}  human preferred {won}")
        print("-" * 30 + " INSTRUCTION " + "-" * 30)
        print(case.instruction[:1600])
        print("-" * 26 + f" PREFERRED ({won}) " + "-" * 26)
        print((case.response_a if won == "A" else case.response_b)[:args.chars])
        print("-" * 26 + f" REJECTED ({lost}) " + "-" * 26)
        print((case.response_a if lost == "A" else case.response_b)[:args.chars])
        print("-" * 26 + f" {args.wrong} RUBRIC (judge said {wrong_v[cid]['forward']}) " + "-" * 12)
        print(wrong_r.get(cid, "(none)"))
        print("-" * 26 + f" {args.right} RUBRIC (judge said {right_v[cid]['forward']}) " + "-" * 12)
        print(right_r.get(cid, "(none)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
