#!/usr/bin/env python3
"""Screen a large question pool down to the ones a reward signal can act on.

Why screen at all
-----------------
The first rollout evaluation drew rubrics for 200 questions and then discovered
that only 61 of them had mixed outcomes: the policy got 123 of them right on
every sample and all 15 remaining ones wrong. Those 138 questions cost a full
rubric-generation budget and contributed nothing, because a question with no
correct/incorrect pair has no AUC, no best-of-n choice, and no separation.

The fix is to reverse the order — screen first, generate second — and it is not
merely an efficiency trick. In GRPO the advantage within a group is computed
against the group's own mean, so a group whose rollouts are all correct (or all
incorrect) has zero advantage for every member and contributes no gradient. The
quality of the rubric on such a question is irrelevant to training by
construction. **Mixed-outcome questions are therefore not a convenience subset
of the population; they are the population on which a reward signal does any
work at all.** Screening to them measures the right thing and costs less.

Design
------
Screening draws k=4 rollouts per question under the balanced prefix of
``ROLLOUT_SETTINGS`` (2 careful, 2 degraded) and labels them with the same
rubric-blind oracle used in the evaluation. Questions with 1..k-1 correct
survive.

Two properties keep this from biasing the comparison:

- The screen is **rubric-blind**. It runs before any rubric exists for these
  questions and consults only the question, the reference answer, and the
  policy. No rubric source can influence which questions are selected.
- Every source is later scored on the **same** rollouts. The degraded sampling
  settings that create the outcome spread are shared, so they cannot favour one
  source's rubrics over another's.

The screened-in questions keep their four screening rollouts; the evaluation
tops them up to k=8, and the first four come back from the LLM cache.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.config import RunConfig, load_config  # noqa: E402
from harness.data import sample_examples, write_examples_jsonl  # noqa: E402
from harness.eval.rollouts import (  # noqa: E402
    RolloutSet,
    build_all_rollout_sets,
    load_rollout_sets,
    oracle_report,
    save_rollout_sets,
)
from harness.pipeline import build_engine  # noqa: E402
from harness.schema import Example  # noqa: E402
from harness.tracing import RunDir  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s | %(message)s"
)
logger = logging.getLogger("screen_questions")


def build_pool(config: RunConfig, per_domain: int) -> list[Example]:
    """Draw the screening pool, de-duplicated on question text.

    ``sample_examples`` shuffles a deterministic index list and takes a prefix,
    so drawing ``per_domain`` with the pilot's seed yields a strict superset of
    the pilot's own sample whose first 100 entries are exactly the pilot's
    questions. That is what lets already-generated rubrics be reused instead of
    paid for twice.
    """
    pool: list[Example] = []
    for domain in config.sample.domains:
        drawn = sample_examples(
            domain,
            split=config.sample.split,
            n=per_domain,
            seed=config.sample.seed,
            deduplicate_questions=config.sample.deduplicate_questions,
        )
        logger.info("pool %s: %d questions", domain, len(drawn))
        pool.extend(drawn)
    return pool


def import_existing(
    pool: Sequence[Example], sources: Sequence[Path], target: Path
) -> dict[str, RolloutSet]:
    """Seed the screen with rollout sets already labelled in earlier runs.

    These were drawn at k=8 under the same salt namespace and the same oracle,
    so they are the same measurement, only wider. Re-screening them at k=4 would
    re-derive a label already on disk.
    """
    # Merge into whatever the target already holds. Writing only the imported
    # sets would truncate a checkpoint from an earlier screening pass, which is
    # the same overwriting-write failure that once destroyed the zero-LLM metric
    # rows; the cost here is not lost calls (they are cached) but lost hours.
    seeded: dict[str, RolloutSet] = dict(load_rollout_sets(target))
    n_existing = len(seeded)
    wanted = {ex.uid for ex in pool}
    imported: set[str] = set()
    for source in sources:
        for uid, rollout_set in load_rollout_sets(source).items():
            if uid in wanted and uid not in seeded:
                rollout_set.meta["imported_from"] = str(source)
                seeded[uid] = rollout_set
                imported.add(uid)
    logger.info(
        "checkpoint had %d sets; imported %d more from earlier runs (%d total)",
        n_existing, len(imported), len(seeded),
    )
    if seeded:
        save_rollout_sets(seeded, target)
    # Questions carried over from an earlier run are the ones whose rubrics
    # already exist, whether they arrived just now or in a previous pass.
    return {uid: rs for uid, rs in seeded.items() if rs.meta.get("imported_from")}


def select(
    sets: dict[str, RolloutSet],
    pool: Sequence[Example],
    *,
    per_domain: int,
    prefer: set[str],
) -> tuple[list[Example], dict[str, Any]]:
    """Keep mixed-outcome questions, balanced across domains.

    Questions whose rubrics already exist are taken first. That is a budget
    decision, not a quality one: preference is applied *after* the mixed-outcome
    filter, which is rubric-blind, so it cannot select for questions where any
    source happens to look good.
    """
    by_domain: dict[str, list[Example]] = defaultdict(list)
    stats: Counter[str] = Counter()
    by_uid = {ex.uid: ex for ex in pool}

    for uid, rollout_set in sets.items():
        example = by_uid.get(uid)
        if example is None:
            continue
        stats[f"{example.domain}/screened"] += 1
        if not rollout_set.usable:
            stats[f"{example.domain}/unusable"] += 1
            continue
        if rollout_set.is_informative:
            stats[f"{example.domain}/mixed"] += 1
            by_domain[example.domain].append(example)
        elif rollout_set.n_correct == 0:
            stats[f"{example.domain}/all_wrong"] += 1
        else:
            stats[f"{example.domain}/all_right"] += 1

    chosen: list[Example] = []
    for domain, candidates in sorted(by_domain.items()):
        # Deterministic: existing-rubric questions first, then by uid.
        candidates.sort(key=lambda e: (e.uid not in prefer, e.uid))
        chosen.extend(candidates[:per_domain])
        logger.info(
            "select %s: %d mixed, keeping %d (%d with rubrics already generated)",
            domain, len(candidates), min(per_domain, len(candidates)),
            sum(1 for e in candidates[:per_domain] if e.uid in prefer),
        )
    # Provenance, because the two groups were discovered at different depths.
    # The *inclusion criterion* is uniform -- mixed outcomes among the k=8
    # evaluation rollouts -- but imported questions were tested at k=8 straight
    # away while screened ones passed at k=4. Mixing at k=4 implies mixing at
    # k=8 (the disagreeing pair is still there once more samples are added), so
    # the screen has no false positives; it only misses questions that would
    # have turned mixed on the later draws. Recording which is which lets a
    # reader check that the difference does not track any result.
    for example in chosen:
        stats[f"provenance/{'imported_k8' if example.uid in prefer else 'screened_k4'}"] += 1
    return chosen, dict(stats)


async def run(args: argparse.Namespace) -> None:
    config = load_config(args.config, run_name=args.run_name)
    if args.concurrency:
        config.llm.concurrency = args.concurrency
    run_dir = RunDir(config.runs_dir, args.run_name)
    engine = build_engine(config)

    pool = build_pool(config, args.pool_per_domain)
    write_examples_jsonl(pool, run_dir.path / "screening_pool.jsonl")

    target = run_dir.path / "rollouts.jsonl"
    prefer: set[str] = set()
    if args.import_from:
        imported = import_existing(pool, [Path(p) for p in args.import_from], target)
        prefer = set(imported)

    if args.select_only:
        # Select from whatever the screen has already labelled. The screen walks
        # the pool in a fixed order but *completes* in order of latency, so a
        # partial screen over-represents questions the policy answers quickly.
        # That skew is rubric-blind and every source is scored on the same
        # questions, so it cannot favour a source; it only narrows what the
        # result generalises to, and the actual screened count is recorded below
        # rather than the count that was planned.
        sets = load_rollout_sets(target)
        logger.info("select-only: %d rollout sets already on disk", len(sets))
    else:
        sets = await build_all_rollout_sets(
            engine,
            pool,
            k=args.k,
            max_tokens=args.max_tokens,
            concurrency=args.concurrency or config.llm.concurrency,
            agreement_probe_fraction=args.agreement_probe_fraction,
            run_dir=run_dir,
            reuse=True,
        )

    report = oracle_report(sets)
    chosen, stats = select(
        sets, pool, per_domain=args.select_per_domain, prefer=prefer
    )
    write_examples_jsonl(chosen, run_dir.path / "examples.jsonl")

    summary = {
        "pool_per_domain": args.pool_per_domain,
        "screen_k": args.k,
        "n_pool": len(pool),
        "n_pool_actually_screened": len(sets),
        "n_screened": len(sets),
        "n_selected": len(chosen),
        "selected_by_domain": dict(Counter(e.domain for e in chosen)),
        "n_selected_with_existing_rubrics": sum(1 for e in chosen if e.uid in prefer),
        "screen_stats": stats,
        "oracle": report,
    }
    (run_dir.path / "screening_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (run_dir.path / "llm_stats_screen.json").write_text(
        json.dumps(engine.stats_snapshot(), indent=2, default=str), encoding="utf-8"
    )

    logger.info("screened %d, selected %d -> %s",
                len(sets), len(chosen), run_dir.path / "examples.jsonl")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="configs/pilot.yaml")
    ap.add_argument("--run-name", default="rollout_v2")
    ap.add_argument("--pool-per-domain", type=int, default=350)
    ap.add_argument("--select-per-domain", type=int, default=110)
    ap.add_argument("--k", type=int, default=4, help="screening rollouts per question")
    ap.add_argument("--max-tokens", type=int, default=8192)
    ap.add_argument("--concurrency", type=int, default=0)
    ap.add_argument("--select-only", action="store_true",
                    help="select from the existing checkpoint without sampling more")
    ap.add_argument("--agreement-probe-fraction", type=float, default=0.05)
    ap.add_argument(
        "--import-from", nargs="*", default=["runs/pilot_v2/rollouts.jsonl"],
        help="rollout sets from earlier runs to reuse instead of re-screening",
    )
    return ap.parse_args()


if __name__ == "__main__":
    asyncio.run(run(parse_args()))
