#!/usr/bin/env python
"""Smoke-test candidate judge models before spending budget on a full rerun.

Three things have to be true before a model can replace the judge:

* the endpoint answers at all (credentials / routing / budget);
* ``LLMEngine`` normalises whatever ``chat()`` returns — the incumbent hands
  back ``{'reasoning_content', 'response'}``, others return a bare ``str``, and
  :func:`harness.llm.extract_response_text` is supposed to absorb both;
* it emits a parsable JSON array of per-criterion verdicts under the *unchanged*
  judge prompt, since the prompt is what must stay fixed across the swap.

The third check runs the real :func:`harness.eval.judge.judge_rubric` against a
real rubric and a real response from the pilot, so a model that passes here is
known to work for the metric itself, not merely for a toy prompt.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.data import load_examples_jsonl  # noqa: E402
from harness.eval.judge import judge_rubric  # noqa: E402
from harness.eval.responses import load_response_sets  # noqa: E402
from harness.llm import LLMEngine  # noqa: E402
from harness.pipeline import load_rubrics  # noqa: E402
from harness.tracing import RunDir  # noqa: E402

CANDIDATES = [
    "api_azure_openai_gpt-5.1",
    "api_aws_third_anthropic.claude-opus-5",
    "api_ali_qwen3.8-max",
    "api_moonshot_kimi-k3",
    "api_azure_openai_gpt-5.6-sol",
]


async def probe_raw(model: str, cache_dir: str | None) -> dict:
    """Round 1: does the endpoint answer, and in what shape?"""
    out: dict = {"model": model}
    started = time.time()
    try:
        engine = LLMEngine(
            model=model,
            concurrency=4,
            reasoning_effort="high",
            cache_dir=cache_dir,
            max_attempts=2,
            default_max_tokens=4096,
            request_timeout_s=1200,
        )
    except Exception as exc:  # noqa: BLE001
        out["ok"] = False
        out["stage"] = "construct"
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
        return out

    # Bypass LLMEngine so the *raw* return shape is visible: this is the
    # dict-vs-str compatibility question the swap depends on.
    try:
        raw = await engine._client.chat(
            "Reply with exactly the word PONG and nothing else.",
            system="You are terse.",
            max_tokens=2048,
        )
        out["raw_type"] = type(raw).__name__
        out["raw_keys"] = sorted(raw.keys()) if isinstance(raw, dict) else None
        from harness.llm import extract_response_text

        out["normalised"] = extract_response_text(raw)[:120]
        out["normalised_ok"] = bool(out["normalised"].strip())
        out["latency_s"] = round(time.time() - started, 2)
    except Exception as exc:  # noqa: BLE001
        out["ok"] = False
        out["stage"] = "raw_chat"
        out["error"] = f"{type(exc).__name__}: {exc}"[:300]
        out["latency_s"] = round(time.time() - started, 2)
        return out

    out["ok"] = bool(out.get("normalised_ok"))
    out["stage"] = "raw_chat"
    out["_engine"] = engine
    return out


async def probe_judge(engine: LLMEngine, fixtures: list[dict], max_tokens: int) -> dict:
    """Round 2: the real judge prompt on real (rubric, response) pairs."""
    started = time.time()
    tasks = [
        judge_rubric(
            engine,
            uid=f["uid"],
            question=f["question"],
            response=f["response"],
            rubric=f["rubric"],
            rubric_source=f["source"],
            response_id=f["variant"],
            shuffle=True,
            shuffle_seed=7,
            max_tokens=max_tokens,
        )
        for f in fixtures
    ]
    results = await asyncio.gather(*tasks, return_exceptions=True)
    elapsed = time.time() - started

    detail = []
    n_ok = 0
    for f, r in zip(fixtures, results):
        if isinstance(r, BaseException):
            detail.append({"uid": f["uid"], "source": f["source"], "variant": f["variant"],
                           "error": f"{type(r).__name__}: {r}"[:200]})
            continue
        ok = bool(r.verdicts) and r.error is None
        n_ok += int(ok)
        detail.append({
            "uid": f["uid"], "source": f["source"], "variant": f["variant"],
            "n_items": r.n_items, "n_verdicts": len(r.verdicts),
            "n_true": sum(1 for v in r.verdicts if v.literally_true),
            "score": round(r.score, 4) if r.verdicts else None,
            "error": r.error,
        })
    return {
        "n_fixtures": len(fixtures),
        "n_ok": n_ok,
        "elapsed_s": round(elapsed, 2),
        "per_call_s": round(elapsed / max(1, len(fixtures)), 2),
        "detail": detail,
        "stats": engine.stats_snapshot(),
    }


def build_fixtures(run_dir: RunDir, sources: list[str], n: int) -> list[dict]:
    """Real pilot inputs: same rubrics and same responses the rerun will use."""
    examples = {ex.uid: ex for ex in load_examples_jsonl(run_dir.path / "examples.jsonl")}
    response_sets = load_response_sets(run_dir)
    fixtures: list[dict] = []
    for source in sources:
        rubrics = load_rubrics(run_dir, source)
        picked = 0
        for uid, rubric in rubrics.items():
            if picked >= n:
                break
            ex = examples.get(uid)
            rs = response_sets.get(uid)
            if ex is None or rs is None or len(rubric) == 0:
                continue
            for variant in ("gold", "off_topic"):
                cand = next((c for c in rs.responses if c.variant == variant), None)
                if cand is None:
                    continue
                fixtures.append({
                    "uid": uid, "question": ex.question, "response": cand.text,
                    "rubric": rubric, "source": source, "variant": variant,
                })
            picked += 1
    return fixtures


async def main_async() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="pilot_v2")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--models", nargs="*", default=CANDIDATES)
    ap.add_argument("--sources", nargs="*", default=["agentic", "baseline"])
    ap.add_argument("--n-per-source", type=int, default=1)
    ap.add_argument("--judge-max-tokens", type=int, default=12288)
    ap.add_argument("--cache-dir", default="runs/judgeswap_cache")
    ap.add_argument("--out", default="runs/pilot_v2/judgeswap_smoke.json")
    args = ap.parse_args()

    run_dir = RunDir(args.runs_dir, args.run_name)
    fixtures = build_fixtures(run_dir, args.sources, args.n_per_source)
    print(f"built {len(fixtures)} judge fixtures from real pilot data", flush=True)

    report = {"fixtures": [{k: v for k, v in f.items() if k != "rubric"} | {"n_items": len(f["rubric"])}
                           for f in fixtures],
              "candidates": {}}

    for model in args.models:
        print(f"\n{'=' * 70}\n{model}\n{'=' * 70}", flush=True)
        raw = await probe_raw(model, args.cache_dir)
        engine = raw.pop("_engine", None)
        print(json.dumps(raw, ensure_ascii=False, indent=2), flush=True)
        entry = {"raw": raw}
        if raw.get("ok") and engine is not None and fixtures:
            judged = await probe_judge(engine, fixtures, args.judge_max_tokens)
            print(json.dumps({k: v for k, v in judged.items() if k != "stats"},
                             ensure_ascii=False, indent=2), flush=True)
            entry["judge"] = judged
        report["candidates"][model] = entry

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str))
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
