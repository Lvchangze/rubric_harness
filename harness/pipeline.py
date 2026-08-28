"""Orchestration shared by the CLI entry points in ``scripts/``.

Keeps three invariants that the experiment's validity depends on:

1. **One sample, many stages.** ``resolve_examples`` writes ``examples.jsonl``
   into the run directory on first use and reloads it afterwards, so generation
   and evaluation provably operate on the same questions in the same order.
2. **Resumability.** Generation and response building skip uids already present
   in their output jsonl, so an interrupted long run is restarted, not redone.
3. **Isolation of the free variable.** Rubrics are keyed ``source -> uid ->
   Rubric``; evaluation never sees anything else about how a rubric was made.
"""

from __future__ import annotations

import asyncio
import logging
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

from .config import RunConfig
from .data import load_examples_jsonl, sample_examples, write_examples_jsonl
from .llm import LLMEngine
from .schema import Example, Rubric
from .tracing import RunDir, read_jsonl

logger = logging.getLogger(__name__)

__all__ = [
    "build_engine",
    "resolve_examples",
    "generate_rubrics",
    "load_rubrics",
    "load_all_rubrics",
]


def build_engine(config: RunConfig, *, model: str | None = None, concurrency: int | None = None) -> LLMEngine:
    llm = config.llm
    return LLMEngine(
        model=model or llm.model,
        concurrency=concurrency or llm.concurrency,
        reasoning_effort=llm.reasoning_effort,
        cache_dir=llm.cache_dir,
        max_attempts=llm.max_attempts,
        default_max_tokens=llm.max_tokens,
        request_timeout_s=llm.request_timeout_s,
    )


def resolve_examples(config: RunConfig, run_dir: RunDir) -> list[Example]:
    """Sample (or reload) the run's examples. Idempotent across invocations."""
    path = run_dir.path / "examples.jsonl"
    if path.exists():
        examples = load_examples_jsonl(path)
        logger.info("reusing %d examples from %s", len(examples), path)
        return examples

    examples: list[Example] = []
    for domain in config.sample.domains:
        examples.extend(
            sample_examples(
                domain,
                split=config.sample.split,
                n=config.sample.n_per_domain,
                seed=config.sample.seed,
                deduplicate_questions=config.sample.deduplicate_questions,
            )
        )
    write_examples_jsonl(examples, path)
    logger.info("sampled %d examples -> %s", len(examples), path)
    return examples


async def generate_rubrics(
    engine: LLMEngine,
    examples: Sequence[Example],
    source: str,
    *,
    run_dir: RunDir,
    config: RunConfig,
    generator_kwargs: dict[str, Any] | None = None,
    resume: bool = True,
    progress_every: int = 10,
) -> dict[str, Rubric]:
    """Run one generator over ``examples``, streaming results to disk."""
    from .generators import build_generator  # local import keeps import graph shallow

    generator = build_generator(source, engine=engine, **(generator_kwargs or {}))
    writer = run_dir.writer(f"rubrics_{source}")
    done = run_dir.existing_uids(f"rubrics_{source}") if resume else set()
    todo = [ex for ex in examples if ex.uid not in done]
    logger.info("generator=%s: %d to do, %d already done", source, len(todo), len(done))

    rubrics: dict[str, Rubric] = {}
    for record in read_jsonl(run_dir.path / f"rubrics_{source}.jsonl"):
        if record.get("uid") in done:
            rubrics[record["uid"]] = Rubric.from_dict(record["rubric"])

    counter = {"n": 0}
    started = time.time()

    async def one(example: Example) -> None:
        t0 = time.time()
        try:
            result = await generator.generate(example)
        except Exception as exc:  # noqa: BLE001 - a generator bug must not kill the run
            logger.exception("generator %s crashed on %s", source, example.uid)
            record = {"uid": example.uid, "source": source, "rubric": {"items": [], "meta": {"source": source}},
                      "error": f"crash: {exc}"[:400]}
            writer.write(record)
            return
        result.wall_seconds = result.wall_seconds or (time.time() - t0)
        payload = result.to_dict()
        payload["domain"] = example.domain
        writer.write(payload)
        if result.trace:
            run_dir.write_trace(f"{source}_{example.uid}", result.trace)
        rubrics[example.uid] = result.rubric
        counter["n"] += 1
        if counter["n"] % progress_every == 0:
            rate = counter["n"] / max(1e-6, time.time() - started)
            logger.info(
                "generator=%s progress %d/%d (%.2f samples/s) llm_calls=%d cache_hits=%d",
                source, counter["n"], len(todo), rate, engine.stats.calls, engine.stats.cache_hits,
            )

    await asyncio.gather(*(one(ex) for ex in todo))
    return rubrics


def load_rubrics(run_dir: RunDir, source: str) -> dict[str, Rubric]:
    out: dict[str, Rubric] = {}
    for record in read_jsonl(run_dir.path / f"rubrics_{source}.jsonl"):
        uid = record.get("uid")
        if not uid:
            continue
        rubric = Rubric.from_dict(record.get("rubric") or {})
        rubric.meta.setdefault("source", source)
        if len(rubric) == 0:
            logger.warning("empty rubric uid=%s source=%s error=%s", uid, source, record.get("error"))
        out[uid] = rubric
    return out


def load_all_rubrics(
    run_dir: RunDir, sources: Iterable[str], examples: Sequence[Example] | None = None
) -> dict[str, dict[str, Rubric]]:
    """Load every source and, if ``examples`` is given, restrict to uids present
    in *all* sources so every metric is computed on an identical, paired set."""
    rubrics = {src: load_rubrics(run_dir, src) for src in sources}
    if examples is None:
        return rubrics
    wanted = {ex.uid for ex in examples}
    common = set.intersection(*[set(v) for v in rubrics.values()]) if rubrics else set()
    common &= wanted
    dropped = {src: sorted(wanted & set(v.keys()) - common) for src, v in rubrics.items()}
    for src, missing in dropped.items():
        if missing:
            logger.warning("source=%s: %d uids dropped (absent from some other source)", src, len(missing))
    also_empty = {
        uid: sorted(src for src in rubrics if len(rubrics[src].get(uid, Rubric())) == 0)
        for uid in common
    }
    also_empty = {uid: srcs for uid, srcs in also_empty.items() if srcs}
    if also_empty:
        # Dropping these keeps every metric paired, but it is not a neutral
        # exclusion: producing no rubric at all *is* a result, and discarding
        # the question hides it. The exclusion favours whichever source failed,
        # so it works against this study's own method — but it still has to be
        # stated, and `intent_to_treat_failures` is written out so the report
        # can quantify it rather than assert it.
        by_source: dict[str, int] = {}
        for srcs in also_empty.values():
            for src in srcs:
                by_source[src] = by_source.get(src, 0) + 1
        logger.warning(
            "dropping %d uids where at least one source produced an empty rubric; "
            "per-source failure counts %s — these are excluded from every paired "
            "metric, which flatters the failing source",
            len(also_empty), by_source,
        )
        try:
            run_dir.write_json(
                "intent_to_treat_failures.json",
                {
                    "n_dropped": len(also_empty),
                    "failures_by_source": by_source,
                    "uids": {uid: srcs for uid, srcs in sorted(also_empty.items())},
                    "note": (
                        "Questions where some source emitted an empty rubric. They are "
                        "excluded from the paired analysis; an intent-to-treat reading "
                        "scores each such failure as 0 for the failing source."
                    ),
                },
            )
        except Exception:  # noqa: BLE001 - bookkeeping must not break a run
            logger.debug("could not persist intent-to-treat record", exc_info=True)
        common -= set(also_empty)
    return {src: {uid: r for uid, r in v.items() if uid in common} for src, v in rubrics.items()}
