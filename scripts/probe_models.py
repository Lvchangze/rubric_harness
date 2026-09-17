#!/usr/bin/env python3
"""Find which candidate deployments are alive, and how fast they really are.

Two passes, because they answer different questions and the cheap one filters
the expensive one:

  liveness   one short call per model - is anything serving this name at all
  throughput a batch of calls shaped like the real generation workload

The second pass exists because a short-prompt/short-output probe badly
misleads on a token-bound endpoint: measured that way Kimi looked like 4.2
completed requests/s, while the actual rubric workload gets 0.71. Throughput
here is therefore reported in output characters/s as well as requests/s.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.llm import LLMEngine  # noqa: E402

PING = "Reply with exactly: OK"

# Shaped like a baseline rubric call: real question, long structured output.
WORKLOAD = """You are writing an evaluation rubric for the question below.

Question: A 54-year-old presents with crushing substernal chest pain radiating
to the left arm, diaphoresis, and ST elevation in leads II, III, and aVF.
Which vessel is most likely occluded, and what is the immediate management?

Write 10 specific, checkable criteria that a grader would use to score a
candidate answer. For each, give a title, a one-sentence description of what
must be present, and an integer weight from 1 to 5. Return JSON:
{"items": [{"title": ..., "description": ..., "weight": ...}]}"""


async def _ping(model: str) -> tuple[str, bool, float, str]:
    start = time.monotonic()
    try:
        eng = LLMEngine(model=model, concurrency=1, cache_dir=None, max_attempts=1)
        txt = await eng.chat(PING, max_tokens=2048, use_cache=False)
        dt = time.monotonic() - start
        return (model, bool(txt and txt.strip()), dt, (txt or "").strip()[:20])
    except Exception as exc:  # noqa: BLE001
        return (model, False, time.monotonic() - start, f"{type(exc).__name__}"[:40])


async def _throughput(model: str, level: int, n: int) -> dict:
    eng = LLMEngine(model=model, concurrency=level, cache_dir=None, max_attempts=2)

    async def one(i: int) -> tuple[bool, int]:
        try:
            txt = await eng.chat(
                WORKLOAD, max_tokens=16384, use_cache=False, cache_salt=f"w{i}"
            )
            return bool(txt and txt.strip()), len(txt or "")
        except Exception:  # noqa: BLE001
            return False, 0

    start = time.monotonic()
    res = await asyncio.gather(*(one(i) for i in range(n)))
    wall = time.monotonic() - start
    ok = [c for good, c in res if good]
    return {
        "model": model,
        "level": level,
        "ok": len(ok),
        "n": n,
        "rps": len(ok) / wall,
        "cps": sum(ok) / wall,
        "mean_chars": (sum(ok) / len(ok)) if ok else 0,
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--level", type=int, default=32)
    ap.add_argument("--calls", type=int, default=32)
    args = ap.parse_args()

    print("=== liveness ===")
    live = []
    for model, ok, dt, note in await asyncio.gather(*(_ping(m) for m in args.models)):
        print(f"  {'UP  ' if ok else 'DOWN'} {model:<44} {dt:5.1f}s  {note}")
        if ok:
            live.append(model)

    if not live:
        print("\n  nothing alive")
        return 1

    print(f"\n=== throughput, workload-shaped, concurrency {args.level} ===")
    print(f"  {'model':<44} {'ok/n':>8} {'req/s':>7} {'chars/s':>9} {'chars/req':>10}")
    rows = []
    for model in live:
        r = await _throughput(model, args.level, args.calls)
        rows.append(r)
        print(
            f"  {r['model']:<44} {r['ok']:>3}/{r['n']:<4} {r['rps']:>7.2f} "
            f"{r['cps']:>9.0f} {r['mean_chars']:>10.0f}"
        )

    usable = [r for r in rows if r["ok"] >= 0.9 * r["n"]]
    if usable:
        best = max(usable, key=lambda r: r["rps"])
        print(f"\n  fastest clean: {best['model']} at {best['rps']:.2f} req/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
