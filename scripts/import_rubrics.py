#!/usr/bin/env python3
"""Carry rubrics for already-generated questions into a new run.

The screening pool is a superset of the pilot's sample, so some selected
questions already have rubrics under every source in ``runs/pilot_v2``. Those
were produced by the same generators from the same config; regenerating them
would return identical results through the LLM cache while still costing the
wall-clock time of walking the agentic pipeline again.

Copying them in lets ``generate_rubrics`` skip those uids on resume. Only uids
present in the target run's ``examples.jsonl`` are copied, so no question enters
the analysis that the screener did not select.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from-run", default="runs/pilot_v2")
    ap.add_argument("--to-run", default="runs/rollout_v2")
    ap.add_argument("--sources", nargs="+", required=True)
    args = ap.parse_args()

    src = Path(args.from_run)
    dst = Path(args.to_run)
    wanted = {
        json.loads(line)["uid"]
        for line in (dst / "examples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    print(f"target run has {len(wanted)} selected questions")

    for source in args.sources:
        source_path = src / f"rubrics_{source}.jsonl"
        if not source_path.exists():
            print(f"  {source}: nothing to import")
            continue
        target_path = dst / f"rubrics_{source}.jsonl"
        already = set()
        if target_path.exists():
            already = {
                json.loads(line)["uid"]
                for line in target_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
        kept = []
        for line in source_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("uid") in wanted and row["uid"] not in already:
                kept.append(line)
        if kept:
            with target_path.open("a", encoding="utf-8") as fh:
                fh.write("\n".join(kept) + "\n")
        print(f"  {source}: imported {len(kept)} (already had {len(already)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
