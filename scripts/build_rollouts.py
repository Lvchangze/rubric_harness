#!/usr/bin/env python
"""Sample fresh policy rollouts and label them with a rubric-blind oracle.

These rollouts are the scored objects for the RL-faithful evaluation protocol
(``harness/eval/rollout_metrics.py``). Unlike the synthesised response ladder,
they are not derived from ``reference_answer``, so a rubric built with access to
the reference gains no unfair advantage when scoring them.

The reference answer is still used — as an input to the oracle that labels each
rollout correct or incorrect. That is exactly its role in a real pipeline: it
defines ground truth, but it is never the thing being scored.

    python scripts/build_rollouts.py --config configs/pilot.yaml --k 6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.config import load_config  # noqa: E402
from harness.eval.rollouts import (  # noqa: E402
    assert_disjoint_from_generator_rollouts,
    build_all_rollout_sets,
    oracle_report,
)
from harness.pipeline import build_engine, resolve_examples  # noqa: E402
from harness.tracing import RunDir  # noqa: E402

logger = logging.getLogger("build_rollouts")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--config", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--k", type=int, default=6, help="rollouts per question")
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument("--max-tokens", type=int, default=8192)
    p.add_argument("--limit", type=int, default=None, help="only the first N questions (smoke)")
    p.add_argument("--agreement-probe-fraction", type=float, default=0.15,
                   help="share of questions given a second oracle pass")
    p.add_argument("--top-up", type=int, default=0,
                   help="extra draws for questions whose labels are not yet mixed; "
                        "such questions contribute to no discriminative metric")
    p.add_argument("--no-reuse", action="store_true")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


async def main_async() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    config = load_config(args.config, overrides={"run_name": args.run_name} if args.run_name else None)
    run_dir = RunDir(config.runs_dir, config.run_name)
    engine = build_engine(config, concurrency=args.concurrency)
    examples = resolve_examples(config, run_dir)
    if args.limit:
        examples = examples[: args.limit]

    logger.info("sampling k=%d rollouts for %d questions", args.k, len(examples))
    sets = await build_all_rollout_sets(
        engine, examples, k=args.k, max_tokens=args.max_tokens,
        concurrency=args.concurrency or config.llm.concurrency,
        agreement_probe_fraction=args.agreement_probe_fraction,
        top_up=args.top_up, run_dir=run_dir, reuse=not args.no_reuse,
    )

    report = oracle_report(sets)
    disjoint = assert_disjoint_from_generator_rollouts(sets, run_dir)
    report["leakage_check"] = disjoint
    run_dir.write_json("rollout_report.json", report)

    print("\n=== rollout / oracle report ===")
    for key in ("n_questions", "n_rollouts_total", "n_rollouts_usable",
                "rollout_correct_rate", "mean_per_question_correct_rate",
                "oracle_self_agreement", "n_oracle_agreement_probed",
                "low_confidence_rate", "missing_final_answer_rate",
                "n_informative_questions", "n_all_correct_questions",
                "n_all_incorrect_questions", "informative_fraction"):
        value = report.get(key)
        print(f"  {key:34s} {value if not isinstance(value, float) else round(value, 4)}")
    print(f"\n  leakage check: disjoint={disjoint['disjoint']} "
          f"({disjoint['n_eval_rollouts']} eval vs "
          f"{disjoint['n_generator_rollouts_checked']} generator rollouts, "
          f"{disjoint['n_collisions']} collisions)")
    if not disjoint["disjoint"]:
        logger.error("evaluation rollouts overlap generator rollouts; results would leak")
        return 2
    print(f"\n  llm: {json.dumps(engine.stats.snapshot(), default=str)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
