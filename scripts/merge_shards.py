#!/usr/bin/env python
"""Recombine the per-shard outputs of a sharded `gen_rubrics.py` run.

Sharding exists so one source can be generated across several endpoints at
once; each shard writes `rubrics_<source>.sI-of-N.jsonl` because two processes
appending rubric-sized lines to one file would interleave mid-line. This puts
them back together.

Writes to a new path and refuses to clobber, which is deliberate. The
overwrite-in-place bug (HANDOFF.md §7.1) has cost this project real data four
separate times, and a merge step reading N files and writing one is exactly
the shape that keeps reintroducing it.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="e.g. runs/train_full_glm53")
    p.add_argument("--source", required=True, help="e.g. agentic-tools")
    p.add_argument("--out", default=None, help="default: <run-dir>/rubrics_<source>.merged.jsonl")
    p.add_argument("--force", action="store_true", help="allow overwriting --out")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    shards = sorted(run_dir.glob(f"rubrics_{args.source}.s*-of-*.jsonl"))
    # The unsharded file is a legitimate input too: a run that started
    # unsharded and was resharded later leaves its early questions there.
    plain = run_dir / f"rubrics_{args.source}.jsonl"
    inputs = ([plain] if plain.exists() else []) + shards
    if not inputs:
        print(f"no inputs matching rubrics_{args.source}[.sI-of-N].jsonl under {run_dir}", file=sys.stderr)
        return 1

    out = Path(args.out) if args.out else run_dir / f"rubrics_{args.source}.merged.jsonl"
    if out.exists() and not args.force:
        print(f"refusing to overwrite {out} (pass --force)", file=sys.stderr)
        return 1
    if out.resolve() in {p.resolve() for p in inputs}:
        print(f"--out {out} is also an input; that is the truncation bug", file=sys.stderr)
        return 1

    rows: dict[str, dict] = {}
    per_file: list[tuple[str, int]] = []
    dupes = 0
    for path in inputs:
        n = 0
        for line in path.open(encoding="utf-8"):
            if not line.strip():
                continue
            row = json.loads(line)
            n += 1
            uid = row["uid"]
            if uid in rows:
                dupes += 1
                # Keep the row that actually produced a rubric. A shard can
                # legitimately redo a uid the unsharded run had already failed.
                if not row.get("rubric", {}).get("items"):
                    continue
            rows[uid] = row
        per_file.append((path.name, n))

    models = Counter(r.get("model", "<unstamped>") for r in rows.values())
    errors = sum(1 for r in rows.values() if r.get("error"))
    empty = sum(1 for r in rows.values() if not r.get("rubric", {}).get("items"))

    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for uid in sorted(rows):
            fh.write(json.dumps(rows[uid], ensure_ascii=False) + "\n")
    written = sum(1 for line in tmp.open(encoding="utf-8") if line.strip())
    assert written == len(rows), f"wrote {written}, expected {len(rows)}"
    tmp.replace(out)

    for name, n in per_file:
        print(f"  {name:48} {n:7d}")
    print(f"  {'=> ' + out.name:48} {len(rows):7d} unique  ({dupes} duplicate uids collapsed)")
    print(f"  models: {dict(models)}")
    print(f"  rows with error: {errors}, rows with empty rubric: {empty}")
    if len(models) > 1:
        print("  NOTE: more than one generator model in this file. t1/t2 are the same")
        print("        weights on different hardware and are fine to mix; a GLM-5.2 or")
        print("        Kimi row is not (HANDOFF.md §6.0).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
