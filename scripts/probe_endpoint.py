#!/usr/bin/env python3
"""Measure an endpoint's usable concurrency before committing a long run.

Concurrency settings do not transfer between deployments, and the failure mode
when they are too high is not a clean error: throughput collapses while the
job still looks alive. This issues a fixed batch of realistic-length calls at
each level and reports completion rate and latency, so the launch setting is
chosen from this endpoint's numbers rather than the previous one's.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.llm import LLMEngine  # noqa: E402

PROMPT = (
    "A 54-year-old presents with crushing substernal chest pain radiating to the "
    "left arm, diaphoresis, and ST elevation in leads II, III, and aVF. "
    "Name the most likely occluded vessel and justify it in three sentences."
)


async def _one(client: LLMEngine, idx: int) -> tuple[bool, float, str]:
    start = time.monotonic()
    try:
        # Unique salt per call: a cache hit would measure the filesystem.
        text = await client.chat(
            PROMPT, max_tokens=2048, cache_salt=f"probe-{idx}-{start}", use_cache=False
        )
        ok = bool(text and text.strip())
        return ok, time.monotonic() - start, "" if ok else "empty"
    except Exception as exc:  # noqa: BLE001 - the error type is the finding
        return False, time.monotonic() - start, f"{type(exc).__name__}: {exc}"[:120]


async def _level(model: str, level: int, n: int) -> dict:
    client = LLMEngine(model=model, concurrency=level, cache_dir=None, max_attempts=2)
    start = time.monotonic()
    results = await asyncio.gather(*(_one(client, i) for i in range(n)))
    wall = time.monotonic() - start

    lats = [d for ok, d, _ in results if ok]
    errs = [e for ok, _, e in results if not ok]
    return {
        "level": level,
        "n": n,
        "ok": len(lats),
        "wall_s": wall,
        "rps": len(lats) / wall if wall else 0.0,
        "p50": statistics.median(lats) if lats else float("nan"),
        "p95": (sorted(lats)[int(len(lats) * 0.95)] if len(lats) > 2 else float("nan")),
        "errors": errs[:3],
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--levels", type=int, nargs="+", default=[8, 32, 64, 128])
    ap.add_argument("--calls", type=int, default=None, help="calls per level (default 2x level)")
    args = ap.parse_args()

    print(f"model={args.model}\n")
    print(f"{'conc':>5} {'ok/n':>9} {'rps':>7} {'p50 s':>7} {'p95 s':>7}  notes")
    rows = []
    for level in args.levels:
        n = args.calls or max(16, level * 2)
        row = await _level(args.model, level, n)
        rows.append(row)
        note = row["errors"][0] if row["errors"] else ""
        print(
            f"{row['level']:>5} {row['ok']:>4}/{row['n']:<4} {row['rps']:>7.2f} "
            f"{row['p50']:>7.1f} {row['p95']:>7.1f}  {note}"
        )
        # A level that cannot complete its batch will not improve by going wider.
        if row["ok"] < 0.9 * row["n"]:
            print(f"\n  stopping: completion fell to {row['ok']}/{row['n']} at {level}")
            break

    good = [r for r in rows if r["ok"] >= 0.9 * r["n"]]
    if good:
        best = max(good, key=lambda r: r["rps"])
        print(f"\n  best sustained throughput: {best['rps']:.2f} rps at concurrency {best['level']}")
        if best is rows[-1] and len(rows) == len(args.levels):
            print("  (this is the top level tested — the ceiling may be higher)")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
