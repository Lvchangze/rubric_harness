#!/usr/bin/env python
"""Build the shared, quality-stratified response ladder for a run.

Responses are generated **once** and reused to score every rubric source; that
shared set is what makes the comparison fair, so this is a separate CLI step
rather than something the evaluator does implicitly.

Example::

    python scripts/build_responses.py --config configs/pilot.yaml
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.config import load_config  # noqa: E402
from harness.pipeline import build_engine, resolve_examples  # noqa: E402
from harness.tracing import RunDir  # noqa: E402

logger = logging.getLogger("build_responses")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--variants", nargs="+", default=None)
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


async def main_async() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    config = load_config(
        args.config,
        **{
            "run_name": args.run_name,
            "llm.concurrency": args.concurrency,
            "eval.response_variants": args.variants,
        },
    )
    run_dir = RunDir(config.runs_dir, config.run_name)
    examples = resolve_examples(config, run_dir)
    engine = build_engine(config)

    from harness.eval.responses import build_all_response_sets  # noqa: PLC0415

    started = time.time()
    sets = await build_all_response_sets(
        engine,
        examples,
        variants=config.eval.response_variants,
        run_dir=run_dir,
        reuse=not args.no_resume,
        concurrency=config.llm.concurrency,
    )
    n_resp = sum(len(s.responses) for s in sets.values())
    logger.info(
        "built %d response sets (%d responses) in %.1fs", len(sets), n_resp, time.time() - started
    )
    run_dir.write_json("llm_stats_responses.json", engine.stats_snapshot())
    logger.info("LLM stats: %s", engine.stats_snapshot())
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
