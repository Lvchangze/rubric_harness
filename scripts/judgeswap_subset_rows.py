#!/usr/bin/env python
"""Project the incumbent judge's per-response rows onto the swap's question set.

The swap evaluates 114 questions; the pilot evaluated every question it could.
Comparing the two directly would mix a judge change with a sample change. This
writes a run directory holding *only* the incumbent judge's rows for the swap's
uids, so ``scripts/confound_audit.py`` can be pointed at it and every reported
difference between the two judges is measured on the same questions, the same
rubrics and the same responses.

No LLM calls: this only filters rows that already exist on disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-run", default="runs/pilot_v2")
    ap.add_argument("--dst-run", default="runs/oldjudge_sub")
    ap.add_argument("--manifest", default="runs/judgeswap/judgeswap_manifest.json")
    ap.add_argument("--prefix", default="")
    args = ap.parse_args()

    uids = set(json.loads(Path(args.manifest).read_text())["uids"])
    src = Path(args.src_run) / "metrics" / f"{args.prefix}discriminative_per_response_rows.jsonl"
    dst_dir = Path(args.dst_run) / "metrics"
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / f"{args.prefix}discriminative_per_response_rows.jsonl"

    kept = total = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            total += 1
            row = json.loads(line)
            if row.get("uid") in uids:
                fout.write(line + "\n")
                kept += 1
    print(f"{src} -> {dst}: kept {kept}/{total} rows for {len(uids)} uids")


if __name__ == "__main__":
    main()
