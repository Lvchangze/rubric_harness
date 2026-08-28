#!/usr/bin/env python
"""Generate rubrics for one run, one generator source at a time.

Examples
--------
Smoke (5 questions/domain, baseline + agentic)::

    python scripts/gen_rubrics.py --config configs/smoke.yaml --sources baseline agentic

Pilot with the ablation::

    python scripts/gen_rubrics.py --config configs/pilot.yaml --sources agentic-noval
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.config import RunConfig, load_config  # noqa: E402
from harness.pipeline import build_engine, generate_rubrics, resolve_examples  # noqa: E402
from harness.tracing import RunDir  # noqa: E402

logger = logging.getLogger("gen_rubrics")

# ``agentic-noval`` is the Stage-6 ablation: same generator, critic switched off.
#: Stage-6 ablations. `goldonly` and `negonly` split the validation loop by
#: whether the evidence it filters on leaks into the evaluation: the gold half
#: judges against `reference_answer`, which the discriminative metric also uses
#: as its positive, while the negative half judges against independently
#: synthesised flawed answers the evaluation never sees. Comparing the two
#: separates "the rubric became a better reward signal" from "the rubric was
#: fitted to this particular gold text".
ABLATIONS: dict[str, dict[str, bool | int]] = {
    "agentic-noval": {"enable_critic": False},
    "agentic-goldonly": {"use_negative_signal": False},
    "agentic-negonly": {"use_gold_signal": False},
    # Same negative-only critic as `agentic-negonly`, but the counterexamples
    # are real failed rollouts instead of synthesised flawed answers. Held to
    # the same gold-signal-off setting so the only variable against
    # `agentic-negonly` is where the negatives came from.
    "agentic-realneg": {
        "use_gold_signal": False,
        "negatives_from_rollouts_only": True,
        "max_rollout_negatives": 3,
    },
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="YAML/JSON config path")
    p.add_argument("--run-name", default=None)
    p.add_argument("--sources", nargs="+", default=None,
                   help="generator names: shipped baseline agentic agentic-noval")
    p.add_argument("--domains", nargs="+", default=None)
    p.add_argument("--n-per-domain", type=int, default=None)
    p.add_argument("--split", default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--n-rollouts", type=int, default=None)
    p.add_argument("--no-resume", action="store_true", help="regenerate even if results exist")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def resolve_config(args: argparse.Namespace) -> RunConfig:
    return load_config(
        args.config,
        **{
            "run_name": args.run_name,
            "sample.domains": args.domains,
            "sample.n_per_domain": args.n_per_domain,
            "sample.split": args.split,
            "sample.seed": args.seed,
            "llm.model": args.model,
            "llm.concurrency": args.concurrency,
            "agentic.n_rollouts": args.n_rollouts,
        },
    )


def generator_kwargs(source: str, config: RunConfig) -> tuple[str, dict]:
    """Map a run-level source name onto (generator name, constructor kwargs)."""
    from dataclasses import replace

    if source in ABLATIONS:
        return "agentic", {"config": replace(config.agentic, **ABLATIONS[source])}
    if source == "agentic":
        return "agentic", {"config": config.agentic}
    return source, {}


async def main_async() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    config = resolve_config(args)
    sources = args.sources or config.generators

    run_dir = RunDir(config.runs_dir, config.run_name)
    run_dir.snapshot_config(config)
    examples = resolve_examples(config, run_dir)
    engine = build_engine(config)

    logger.info("run=%s examples=%d sources=%s model=%s concurrency=%d",
                config.run_name, len(examples), sources, config.llm.model, config.llm.concurrency)

    for source in sources:
        gen_name, kwargs = generator_kwargs(source, config)
        started = time.time()
        # Output file is keyed by the *run-level* source name so the ablation
        # lands in its own file rather than overwriting `agentic`.
        original_writer_name = source
        rubrics = await _generate_as(
            engine, examples, gen_name, original_writer_name, run_dir, config, kwargs,
            resume=not args.no_resume,
        )
        sizes = [len(r) for r in rubrics.values()]
        logger.info(
            "source=%s done in %.1fs: %d rubrics, mean_items=%.1f",
            source, time.time() - started, len(rubrics),
            sum(sizes) / len(sizes) if sizes else 0.0,
        )

    run_dir.write_json("llm_stats_gen.json", engine.stats_snapshot())
    logger.info("LLM stats: %s", engine.stats_snapshot())
    return 0


async def _generate_as(engine, examples, gen_name, writer_name, run_dir, config, kwargs, *, resume):
    """Run generator ``gen_name`` but persist under ``writer_name``."""
    if gen_name == writer_name:
        return await generate_rubrics(
            engine, examples, gen_name, run_dir=run_dir, config=config,
            generator_kwargs=kwargs, resume=resume,
        )
    from harness.generators import REGISTRY, register

    alias = type(f"Alias_{writer_name}", (REGISTRY[gen_name],), {"name": writer_name})
    register(alias)
    return await generate_rubrics(
        engine, examples, writer_name, run_dir=run_dir, config=config,
        generator_kwargs=kwargs, resume=resume,
    )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
