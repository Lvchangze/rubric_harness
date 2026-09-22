#!/usr/bin/env python
"""Distribute an unsharded run's finished questions into per-shard files.

Resharding a run that is already part-done creates a trap: each shard resumes
only from its own file, so questions the unsharded run finished look unfinished
to whichever shard now owns them. Redoing them is not free either — the cache
key includes the model name, so work cached under one deployment is unreachable
from the other even when both serve the same weights.

This copies each finished row into its owning shard's file, once, and never
touches the source. Rows already present in a shard file are left alone, so
running it twice is a no-op.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gen_rubrics import shard_of  # noqa: E402 - single definition of the partition


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True)
    p.add_argument("--source", required=True)
    p.add_argument("--shards", type=int, required=True)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    src = run_dir / f"rubrics_{args.source}.jsonl"
    if not src.exists():
        print(f"  no unsharded {src.name} to seed from; nothing to do")
        return 0

    rows = [json.loads(l) for l in src.open(encoding="utf-8") if l.strip()]
    targets = {i: run_dir / f"rubrics_{args.source}.s{i}-of-{args.shards}.jsonl"
               for i in range(args.shards)}
    have = {i: {json.loads(l)["uid"] for l in p.open(encoding="utf-8") if l.strip()}
            if p.exists() else set()
            for i, p in targets.items()}

    added = {i: 0 for i in targets}
    handles = {i: p.open("a", encoding="utf-8") for i, p in targets.items()}
    try:
        for row in rows:
            uid = row["uid"]
            i = shard_of(uid, args.shards)
            if uid in have[i]:
                continue
            handles[i].write(json.dumps(row, ensure_ascii=False) + "\n")
            have[i].add(uid)
            added[i] += 1
    finally:
        for fh in handles.values():
            fh.close()

    print(f"  seeded from {src.name} ({len(rows)} rows)")
    for i, p in targets.items():
        total = sum(1 for l in p.open(encoding="utf-8") if l.strip())
        print(f"    {p.name:46} +{added[i]:5d}  -> {total} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
