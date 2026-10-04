#!/usr/bin/env python3
"""Give every question still missing from an `agentic-tools-contrast2` run its delivered rubric.

The contrast2 generator writes a question whose contrastive stage failed on an
LLM call as an empty row with an error, so that a dead endpoint is caught and a
resume retries it. A few questions fail that way every time: on hard science
problems the simulation call runs past the 1200 s per-call cap on each attempt.
Run this only after a retry pass. Each uid with no usable row gets a row holding
its `agentic-tools` rubric unchanged, marked `contrast_applied: False` with the
last error as the reason and stamped with the model that wrote that rubric, so
that the three train arms stay paired question for question. Error rows are
dropped; the original file is kept as `.prefill.bak`.

    python scripts/fill_contrast_fallbacks.py --run-dir runs/train_full_glm53
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

SOURCE = "agentic-tools-contrast2"


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="runs/train_full_glm53")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    run = Path(args.run_dir)
    out_path = run / f"rubrics_{SOURCE}.jsonl"

    examples = {r["uid"]: r for r in read_rows(run / "examples.jsonl")}
    base: dict[str, dict] = {}
    for row in read_rows(run / "rubrics_agentic-tools.jsonl"):
        if (row.get("rubric") or {}).get("items"):
            base.setdefault(row["uid"], row)

    good: dict[str, dict] = {}
    last_error: dict[str, str] = {}
    for row in read_rows(out_path):
        if not row.get("error") and (row.get("rubric") or {}).get("items"):
            good.setdefault(row["uid"], row)
        elif row.get("error"):
            last_error[row["uid"]] = row["error"]

    missing = [uid for uid in examples if uid not in good]
    fills = []
    for uid in missing:
        b = base[uid]
        meta = dict(b["rubric"].get("meta") or {})
        meta.update({"source": SOURCE, "contrast_applied": False,
                     "contrast_fallback": last_error.get(uid, "no contrast2 row")})
        fills.append({
            "uid": uid, "source": SOURCE,
            "rubric": {"items": b["rubric"]["items"], "meta": meta},
            "error": None, "n_llm_calls": 0, "wall_seconds": 0.0,
            "domain": examples[uid]["domain"], "model": b.get("model", "<unstamped>"),
        })
    print(f"{len(good)} questions have a contrast2 row; {len(fills)} get their agentic-tools rubric")
    for f in fills:
        print(f"  {f['uid']}  {f['rubric']['meta']['contrast_fallback'][:110]}")
    if args.dry_run or not fills:
        return 0

    shutil.copyfile(out_path, out_path.with_suffix(".jsonl.prefill.bak"))
    tmp = out_path.with_suffix(".jsonl.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        for row in list(good.values()) + fills:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, out_path)
    print(f"wrote {len(good) + len(fills)} rows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
