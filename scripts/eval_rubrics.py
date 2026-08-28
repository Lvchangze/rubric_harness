#!/usr/bin/env python
"""Score every rubric source in a run with the shared judge and metric suite.

All metric families read the same examples, the same responses and the same
judge configuration; ``--sources`` is the only free variable.

Example::

    python scripts/eval_rubrics.py --config configs/pilot.yaml \
        --sources shipped baseline agentic --metrics discriminative transfer coverage
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.config import RunConfig, load_config  # noqa: E402
from harness.pipeline import build_engine, load_all_rubrics, resolve_examples  # noqa: E402
from harness.tracing import RunDir, jsonable  # noqa: E402

logger = logging.getLogger("eval_rubrics")

#: Ordered so the zero-LLM structural metrics run first: they cost nothing and
#: their `shipped` column can be checked against the forensics' n=45k baselines,
#: which catches a bad sample before any judge budget is spent.
METRIC_MODULES: dict[str, tuple[str, str]] = {
    "grounding": ("harness.eval.grounding", "run_grounding"),
    "lint": ("harness.eval.lint", "run_lint"),
    "adaptivity": ("harness.eval.adaptivity", "run_adaptivity"),
    "discriminative": ("harness.eval.discriminative", "run_discriminative"),
    "rollout": ("harness.eval.rollout_metrics", "run_rollout_eval"),
    "transfer": ("harness.eval.transfer", "run_transfer"),
    "coverage": ("harness.eval.coverage", "run_coverage"),
    "intrinsic": ("harness.eval.intrinsic", "run_intrinsic"),
    "headtohead": ("harness.eval.headtohead", "run_headtohead"),
}

#: Metrics that make no LLM calls, so they can run before the response ladder
#: exists and need no judge engine.
ZERO_LLM_METRICS = frozenset({"grounding", "lint", "adaptivity"})


def _write_rows_preserving_other_sources(
    path: Path, rows: list[dict], sources: Sequence[str]
) -> None:
    """Replace only the rows belonging to ``sources``; keep every other source.

    Metric row files are keyed by ``rubric_source``, and a run scoped with
    ``--sources`` must not destroy the sources it was not asked to evaluate. A
    plain truncating write does exactly that, silently: evaluating one added
    ablation would leave the file holding only that ablation, and the loss is
    invisible until an aggregate turns up empty.

    Not every metric family is source-keyed. Head-to-head rows describe a *pair*
    of sources and carry no ``rubric_source`` at all, so matching on membership
    keeps them all and appends a second copy on every re-run — the same silent
    corruption in the opposite direction. Such rows cannot be updated
    selectively, so whenever the incoming batch contains them the existing ones
    are dropped wholesale, which is correct because those metrics are always
    recomputed over every pair at once.

    Losing rows is the failure this function exists to prevent, so it also
    verifies after writing that no source ended up with fewer rows than it
    started with.
    """
    import json  # noqa: PLC0415

    def count_by_source(records: Sequence[dict]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in records:
            label = str(record.get("rubric_source") or "<unattributed>")
            counts[label] = counts.get(label, 0) + 1
        return counts

    existing: list[dict] = []
    if path.exists():
        try:
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if not line:
                    continue
                existing.append(json.loads(line))
        except (OSError, ValueError):
            logger.warning("could not merge existing %s; rewriting from scratch", path.name)
            existing = []

    replaced = set(sources)
    incoming_unattributed = any(not row.get("rubric_source") for row in rows)
    keep = [
        row for row in existing
        if (row.get("rubric_source") not in replaced)
        and not (incoming_unattributed and not row.get("rubric_source"))
    ]

    with path.open("w", encoding="utf-8") as fh:
        for row in keep + rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    before, after = count_by_source(existing), count_by_source(keep + rows)
    shrunk = {s: (n, after.get(s, 0)) for s, n in before.items() if after.get(s, 0) < n}
    if shrunk:
        logger.error(
            "%s: row count DROPPED for %s — this file previously held data that "
            "is no longer present; check the --sources scope before trusting any "
            "aggregate built from it",
            path.name,
            ", ".join(f"{s} {old}->{new}" for s, (old, new) in sorted(shrunk.items())),
        )
    if keep:
        logger.info(
            "%s: kept %d pre-existing rows from other sources, wrote %d new",
            path.name, len(keep), len(rows),
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--run-name", default=None)
    p.add_argument("--sources", nargs="+", default=None)
    p.add_argument("--metrics", nargs="+", default=None, choices=sorted(METRIC_MODULES) + ["all"])
    p.add_argument("--judge-model", default=None)
    p.add_argument("--concurrency", type=int, default=None)
    p.add_argument(
        "--count-controlled",
        action="store_true",
        help="truncate every source to the same per-question item count before "
             "evaluating, to check that wins are not just from having more criteria",
    )
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _load(name: str) -> Callable[..., Any]:
    module_path, fn_name = METRIC_MODULES[name]
    module = __import__(module_path, fromlist=[fn_name])
    return getattr(module, fn_name)


def _call_kwargs(fn: Callable[..., Any], available: dict[str, Any]) -> dict[str, Any]:
    """Pass only the kwargs a metric actually declares, so the suite tolerates
    metric modules that legitimately need different inputs."""
    params = inspect.signature(fn).parameters
    return {k: v for k, v in available.items() if k in params}


async def main_async() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    config: RunConfig = load_config(
        args.config,
        **{
            "run_name": args.run_name,
            "eval.judge_model": args.judge_model,
            "llm.concurrency": args.concurrency,
            "generators": args.sources,
        },
    )
    sources = args.sources or config.generators
    metrics = config.eval.metrics if not args.metrics else (
        list(METRIC_MODULES) if "all" in args.metrics else args.metrics
    )

    run_dir = RunDir(config.runs_dir, config.run_name)
    examples = resolve_examples(config, run_dir)
    rubrics_by_source = load_all_rubrics(run_dir, sources, examples)
    kept = {ex.uid for ex in examples} & set(next(iter(rubrics_by_source.values())).keys())
    examples = [ex for ex in examples if ex.uid in kept]
    logger.info(
        "evaluating %d paired questions x %d sources: %s",
        len(examples), len(sources),
        {s: len(v) for s, v in rubrics_by_source.items()},
    )
    if not examples:
        logger.error("no paired questions — did generation finish for every source?")
        return 1

    summaries_name = "metric_summaries.json"
    metric_prefix = ""
    if args.count_controlled:
        # A coverage win that comes purely from writing more criteria is not a
        # win. Cutting every source to the same per-question item count removes
        # that confound; results land beside the full-size ones so the report can
        # show both (docs/01_data_forensics.md F1).
        from harness.eval.adaptivity import truncate_to_common_size  # noqa: PLC0415

        before = {s: sum(len(r) for r in v.values()) for s, v in rubrics_by_source.items()}
        rubrics_by_source = truncate_to_common_size(rubrics_by_source)
        after = {s: sum(len(r) for r in v.values()) for s, v in rubrics_by_source.items()}
        logger.info("count-controlled: criteria per source %s -> %s", before, after)
        summaries_name = "metric_summaries_count_controlled.json"
        metric_prefix = "cc_"

    engine = build_engine(config, model=config.eval.judge_model)

    response_sets = None
    if "discriminative" in metrics:
        from harness.eval.responses import load_response_sets  # noqa: PLC0415

        response_sets = load_response_sets(run_dir)
        missing = [ex.uid for ex in examples if ex.uid not in response_sets]
        if missing:
            logger.error(
                "%d/%d questions have no response set — run scripts/build_responses.py first",
                len(missing), len(examples),
            )
            return 1

    rollout_sets = None
    if "rollout" in metrics:
        from harness.eval.rollouts import load_rollout_sets  # noqa: PLC0415

        rollout_sets = load_rollout_sets(run_dir)
        if not rollout_sets:
            logger.error("no rollouts on disk — run scripts/build_rollouts.py first")
            return 1

    summaries: dict[str, Any] = {}
    for metric in metrics:
        fn = _load(metric)
        available = {
            "engine": engine,
            "examples": examples,
            "rubrics_by_source": rubrics_by_source,
            "response_sets": response_sets,
            "rollout_sets": rollout_sets,
            "config": config.eval,
            "run_dir": run_dir,
        }
        started = time.time()
        logger.info("--- metric %s ---", metric)
        try:
            result = await fn(**_call_kwargs(fn, available))
        except Exception:  # noqa: BLE001 - one bad metric must not lose the others
            logger.exception("metric %s failed", metric)
            summaries[metric] = {"error": "failed; see log"}
            continue
        summaries[metric] = jsonable(result.summary())
        for attr in ("per_question_rows", "per_response_rows", "per_criterion_rows",
                     "per_pair_rows", "per_rollout_rows"):
            rows = getattr(result, attr, None)
            if rows:
                path = run_dir.file(f"metrics/{metric_prefix}{metric}_{attr}.jsonl")
                _write_rows_preserving_other_sources(
                    path, [jsonable(row) for row in rows], sources
                )
        logger.info("metric %s done in %.1fs", metric, time.time() - started)

    # Merge rather than overwrite so partial runs (e.g. the cheap structural
    # pass) accumulate instead of clobbering each other.
    summaries_path: Path = run_dir.path / summaries_name
    if summaries_path.exists():
        import json  # noqa: PLC0415

        try:
            previous = json.loads(summaries_path.read_text(encoding="utf-8"))
            summaries = {**previous, **summaries} if isinstance(previous, dict) else summaries
        except (OSError, ValueError):
            logger.warning("could not merge existing %s; overwriting", summaries_name)
    run_dir.write_json(summaries_name, summaries)
    run_dir.write_json("llm_stats_eval.json", engine.stats_snapshot())
    logger.info("LLM stats: %s", engine.stats_snapshot())
    logger.info("summaries -> %s", summaries_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
