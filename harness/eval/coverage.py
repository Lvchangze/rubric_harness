"""Metric family 3 — coverage of, and grounding in, the reference answer.

Two complementary questions about a rubric, both answered against the dataset's
own reference answer:

* **Recall** — of the things the gold answer actually says, how many does the
  rubric check? A rubric can be beautifully written and still miss the step that
  matters.
* **Precision / groundedness** — of the criteria the rubric contains, how many
  check real content rather than reusable style boilerplate, and how many demand
  something the reference does not support at all (those are not merely useless,
  they are *wrong*, and they punish correct responses).

Fairness note — the single most important detail in this module
---------------------------------------------------------------
The gold claim list is extracted **once per question** and reused verbatim for
every rubric source. If each source were scored against its own freshly
extracted claim list, a source could look better simply because the extractor
happened to produce a shorter or differently-sliced list on that pass; recall
would stop being comparable across sources. Extraction therefore never sees a
rubric, is cached in memory for the run, and is persisted to ``claims.jsonl`` in
the run directory so that a rerun (or a later ablation) reuses byte-identical
claims. Use ``load_claims`` / the ``claims`` argument to share them further.

Cost: one extraction call per question (shared), then two calls per (question,
source) — one coverage mapping, one groundedness labelling.
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import EvalConfig
from ..llm import JSONParseError, LLMEngine
from ..prompts.eval_prompts import (
    CLAIM_EXTRACTION_SYSTEM,
    COVERAGE_MAP_SYSTEM,
    GROUNDEDNESS_SYSTEM,
    build_claim_extraction_prompt,
    build_coverage_map_prompt,
    build_groundedness_prompt,
)
from ..schema import Example, Rubric
from ..tracing import RunDir, read_jsonl
from .stats import mean_ci

logger = logging.getLogger(__name__)

__all__ = [
    "ClaimSet",
    "CoverageResults",
    "extract_claims",
    "extract_claims_many",
    "load_claims",
    "run_coverage",
]

NAN = float("nan")
GROUNDEDNESS_LABELS = ("grounded", "generic", "unsupported")
CLAIMS_FILENAME = "claims.jsonl"


# ---------------------------------------------------------------------------
# Shared claim extraction
# ---------------------------------------------------------------------------


@dataclass
class ClaimSet:
    """Atomic claims and sub-questions extracted from one reference answer."""

    uid: str
    claims: list[dict[str, Any]] = field(default_factory=list)
    subquestions: list[dict[str, Any]] = field(default_factory=list)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.claims) and self.error is None

    def core_ids(self) -> set[int]:
        return {
            int(c["id"])
            for c in self.claims
            if str(c.get("importance", "")).lower() == "core"
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "claims": self.claims,
            "subquestions": self.subquestions,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ClaimSet":
        return cls(
            uid=str(raw.get("uid", "")),
            claims=list(raw.get("claims") or []),
            subquestions=list(raw.get("subquestions") or []),
            error=raw.get("error"),
        )


def load_claims(path: str | Path) -> dict[str, ClaimSet]:
    """Read a previously written ``claims.jsonl`` keyed by uid."""
    out: dict[str, ClaimSet] = {}
    for record in read_jsonl(path):
        claim_set = ClaimSet.from_dict(record)
        if claim_set.uid and claim_set.ok:
            out[claim_set.uid] = claim_set
    return out


async def extract_claims(
    engine: LLMEngine, example: Example, *, max_tokens: int = 8192
) -> ClaimSet:
    """Decompose one reference answer into atomic claims; never raises."""
    try:
        parsed = await engine.chat_json(
            build_claim_extraction_prompt(example.question, example.reference_answer),
            system=CLAIM_EXTRACTION_SYSTEM,
            expect="object",
            max_tokens=max_tokens,
            tag="coverage:claims",
        )
    except (JSONParseError, RuntimeError) as exc:
        logger.warning("claim extraction failed uid=%s: %s", example.uid, exc)
        return ClaimSet(uid=example.uid, error=f"claim extraction failed: {exc}"[:300])

    claims: list[dict[str, Any]] = []
    for i, raw in enumerate(_as_list(parsed, "claims"), start=1):
        if not isinstance(raw, dict):
            continue
        text = str(raw.get("text", "")).strip()
        if not text:
            continue
        claims.append(
            {
                "id": _as_int(raw.get("id"), default=i),
                "text": text,
                "kind": str(raw.get("kind", "fact")).strip().lower() or "fact",
                "importance": (
                    "core"
                    if str(raw.get("importance", "")).strip().lower() == "core"
                    else "supporting"
                ),
            }
        )
    # Renumber defensively: the mapping prompt relies on ids being unique.
    seen: set[int] = set()
    for i, claim in enumerate(claims, start=1):
        if claim["id"] in seen:
            claim["id"] = i
        seen.add(claim["id"])

    subquestions: list[dict[str, Any]] = []
    for i, raw in enumerate(_as_list(parsed, "subquestions"), start=1):
        if isinstance(raw, dict):
            text = str(raw.get("text", "")).strip()
            sid = str(raw.get("id", i)).strip() or str(i)
        elif isinstance(raw, str):
            text, sid = raw.strip(), str(i)
        else:
            continue
        if text:
            subquestions.append({"id": sid, "text": text})

    claim_set = ClaimSet(uid=example.uid, claims=claims, subquestions=subquestions)
    if not claims:
        claim_set.error = "no claims extracted"
    return claim_set


async def extract_claims_many(
    engine: LLMEngine,
    examples: Sequence[Example],
    *,
    cache: Mapping[str, ClaimSet] | None = None,
    max_tokens: int = 8192,
) -> dict[str, ClaimSet]:
    """Extract claims for every example, skipping any already in ``cache``."""
    have = dict(cache or {})
    todo = [ex for ex in examples if ex.uid not in have or not have[ex.uid].ok]
    if todo:
        fresh = await asyncio.gather(
            *(extract_claims(engine, ex, max_tokens=max_tokens) for ex in todo),
            return_exceptions=True,
        )
        for ex, result in zip(todo, fresh):
            if isinstance(result, BaseException):
                have[ex.uid] = ClaimSet(uid=ex.uid, error=f"exception: {result}"[:300])
            else:
                have[ex.uid] = result
    return {ex.uid: have[ex.uid] for ex in examples if ex.uid in have}


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class CoverageResults:
    """Per-question coverage rows plus a source-level summary."""

    per_question_rows: list[dict[str, Any]] = field(default_factory=list)
    claims: dict[str, ClaimSet] = field(default_factory=dict)
    errors: list[dict[str, Any]] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """Mean and bootstrap CI of each coverage metric, per rubric source.

        Every metric ``m`` below is reported as ``m`` (mean over questions with
        a finite value), ``m_ci_low`` / ``m_ci_high`` (bootstrap percentile CI)
        and ``m_n`` (how many questions contributed).

        Keys under ``by_source[src]``

        ``n_questions`` / ``n_ok`` / ``n_errors``
            Rows for this source in total, without an ``error``, and with one.
            A row can carry an error from one sub-call and still contribute the
            metrics produced by the other.
        ``claim_recall``
            Mean fraction of gold claims covered by at least one criterion.
            **Higher is better.** Reported with ``_ci_low``/``_ci_high``.
        ``claim_recall_core``
            Same, restricted to claims the extractor marked ``core``.
        ``grounded_fraction``
            Mean fraction of criteria checking specific reference content.
            **Higher is better.**
        ``generic_fraction``
            Mean fraction that are reusable style/quality boilerplate.
            **Lower is better.**
        ``unsupported_fraction``
            Mean fraction demanding something the reference does not support —
            candidate *wrong* criteria. **Lower is better.**
        ``subquestion_coverage``
            Mean fraction of sub-questions with at least one criterion; NaN on
            single-part questions, which are excluded from the mean.
        ``n_claims`` / ``n_criteria``
            Mean gold claim count (identical across sources by construction) and
            mean rubric length.
        ``mean_criteria_per_claim``
            Mean number of criteria mapped to a covered claim; a proxy for how
            much the rubric piles onto the same content.
        """
        by_source: dict[str, Any] = {}
        for source in self._sources():
            rows = [r for r in self.per_question_rows if r["rubric_source"] == source]
            n_ok = sum(1 for r in rows if not r.get("error"))
            entry: dict[str, Any] = {
                "n_questions": len(rows),
                "n_ok": n_ok,
                "n_errors": len(rows) - n_ok,
                "n_multipart_questions": sum(
                    1 for r in rows if int(r.get("n_subquestions", 0)) > 0
                ),
            }
            for metric in (
                "claim_recall",
                "claim_recall_core",
                "grounded_fraction",
                "generic_fraction",
                "unsupported_fraction",
                "subquestion_coverage",
                "n_claims",
                "n_criteria",
                "mean_criteria_per_claim",
            ):
                ci = mean_ci([r.get(metric) for r in rows], iters=2000, seed=0)
                entry[metric] = ci["mean"]
                entry[f"{metric}_ci_low"] = ci["ci_low"]
                entry[f"{metric}_ci_high"] = ci["ci_high"]
                entry[f"{metric}_n"] = ci["n"]
            by_source[source] = entry
        return {
            "metric_family": "coverage",
            "params": dict(self.params),
            "n_questions_with_claims": sum(1 for c in self.claims.values() if c.ok),
            "n_claim_extraction_failures": sum(1 for c in self.claims.values() if not c.ok),
            "by_source": by_source,
            "n_errors": len(self.errors),
        }

    def _sources(self) -> list[str]:
        seen: list[str] = []
        for row in self.per_question_rows:
            if row["rubric_source"] not in seen:
                seen.append(row["rubric_source"])
        return seen


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_coverage(
    engine: LLMEngine,
    examples: Sequence[Example],
    rubrics_by_source: Mapping[str, Mapping[str, Rubric]],
    *,
    config: EvalConfig,
    run_dir: RunDir | None = None,
    claims: Mapping[str, ClaimSet] | None = None,
    reuse_claims: bool = True,
) -> CoverageResults:
    """Score every rubric for recall of, and grounding in, the gold answer.

    Parameters
    ----------
    claims:
        Pre-extracted claim sets to reuse. Anything missing is extracted here.
    reuse_claims:
        When True and ``run_dir`` is given, seed the cache from the run
        directory's ``claims.jsonl`` before extracting anything.
    """
    cache: dict[str, ClaimSet] = dict(claims or {})
    on_disk: set[str] = set()
    if reuse_claims and run_dir is not None:
        for uid, claim_set in load_claims(run_dir.path / CLAIMS_FILENAME).items():
            on_disk.add(uid)
            cache.setdefault(uid, claim_set)
    known_before = set(cache)

    claim_sets = await extract_claims_many(engine, examples, cache=cache)
    results = CoverageResults(
        claims=claim_sets,
        params={
            "sources": list(rubrics_by_source),
            "n_claims_reused": len(known_before & set(claim_sets)),
            "n_claims_extracted": len(set(claim_sets) - known_before),
        },
    )
    _persist_claims(claim_sets, run_dir, already_on_disk=on_disk)

    cells: list[tuple[Example, str, Rubric, ClaimSet]] = []
    for ex in examples:
        claim_set = claim_sets.get(ex.uid) or ClaimSet(uid=ex.uid, error="no claim set")
        for source, per_uid in rubrics_by_source.items():
            rubric = per_uid.get(ex.uid)
            if rubric is None:
                continue
            cells.append((ex, source, rubric, claim_set))

    rows = await asyncio.gather(
        *(
            _score_cell(engine, ex, source, rubric, claim_set, config)
            for ex, source, rubric, claim_set in cells
        ),
        return_exceptions=True,
    )
    for (ex, source, rubric, _), outcome in zip(cells, rows):
        if isinstance(outcome, BaseException):
            row = _blank_row(ex, source, rubric)
            row["error"] = f"exception: {outcome}"[:300]
        else:
            row = outcome
        results.per_question_rows.append(row)
        if row.get("error"):
            results.errors.append(
                {"uid": ex.uid, "rubric_source": source, "error": row["error"]}
            )

    _persist(results, run_dir)
    return results


def _blank_row(example: Example, source: str, rubric: Rubric) -> dict[str, Any]:
    return {
        "uid": example.uid,
        "domain": example.domain,
        "rubric_source": source,
        "n_criteria": len(rubric),
        "n_claims": 0,
        "n_core_claims": 0,
        "n_subquestions": 0,
        "claim_recall": NAN,
        "claim_recall_core": NAN,
        "grounded_fraction": NAN,
        "generic_fraction": NAN,
        "unsupported_fraction": NAN,
        "subquestion_coverage": NAN,
        "mean_criteria_per_claim": NAN,
        "n_uncovered_claims": 0,
        "n_criteria_labelled": 0,
        "uncovered_claim_ids": [],
        "unsupported_criteria": [],
        "error": None,
    }


async def _score_cell(
    engine: LLMEngine,
    example: Example,
    source: str,
    rubric: Rubric,
    claim_set: ClaimSet,
    config: EvalConfig,
) -> dict[str, Any]:
    """Coverage mapping + groundedness labelling for one (question, source)."""
    row = _blank_row(example, source, rubric)
    row["n_claims"] = len(claim_set.claims)
    row["n_core_claims"] = len(claim_set.core_ids())
    row["n_subquestions"] = len(claim_set.subquestions)
    if len(rubric) == 0:
        row["error"] = "empty rubric"
        return row

    problems: list[str] = []
    mapping_task = (
        _coverage_map(engine, example, rubric, claim_set, config)
        if claim_set.ok
        else None
    )
    grounded_task = _groundedness(engine, example, rubric, config)
    if mapping_task is None:
        problems.append(claim_set.error or "no claims")
        grounded = await grounded_task
        mapping = None
    else:
        mapping, grounded = await asyncio.gather(mapping_task, grounded_task)

    if mapping is not None:
        if mapping.get("error"):
            problems.append(str(mapping["error"]))
        else:
            row.update(_recall_metrics(mapping, claim_set, len(rubric)))
    if grounded.get("error"):
        problems.append(str(grounded["error"]))
    else:
        row.update(_grounding_metrics(grounded, rubric))

    if problems:
        row["error"] = "; ".join(problems)[:300]
    return row


async def _coverage_map(
    engine: LLMEngine,
    example: Example,
    rubric: Rubric,
    claim_set: ClaimSet,
    config: EvalConfig,
) -> dict[str, Any]:
    try:
        parsed = await engine.chat_json(
            build_coverage_map_prompt(
                example.question, claim_set.claims, claim_set.subquestions, rubric.items
            ),
            system=COVERAGE_MAP_SYSTEM,
            expect="object",
            max_tokens=int(config.judge_max_tokens),
            tag="coverage:map",
        )
    except (JSONParseError, RuntimeError) as exc:
        return {"error": f"coverage map failed: {exc}"[:250]}
    return {
        "claim_coverage": _as_list(parsed, "claim_coverage"),
        "subquestion_coverage": _as_list(parsed, "subquestion_coverage"),
    }


async def _groundedness(
    engine: LLMEngine, example: Example, rubric: Rubric, config: EvalConfig
) -> dict[str, Any]:
    try:
        parsed = await engine.chat_json(
            build_groundedness_prompt(
                example.question, example.reference_answer, rubric.items
            ),
            system=GROUNDEDNESS_SYSTEM,
            expect="array",
            max_tokens=int(config.judge_max_tokens),
            tag="coverage:grounding",
        )
    except (JSONParseError, RuntimeError) as exc:
        return {"error": f"groundedness failed: {exc}"[:250]}
    return {"labels": parsed if isinstance(parsed, list) else []}


def _recall_metrics(
    mapping: Mapping[str, Any], claim_set: ClaimSet, n_criteria: int
) -> dict[str, Any]:
    valid_criteria = set(range(1, n_criteria + 1))
    covered: dict[int, list[int]] = {}
    for entry in mapping.get("claim_coverage") or []:
        if not isinstance(entry, dict):
            continue
        claim_id = _as_int(entry.get("claim_id"), default=-1)
        if claim_id < 0:
            continue
        ids = [
            cid
            for cid in (
                _as_int(v, default=-1) for v in _as_id_list(entry.get("criterion_ids"))
            )
            if cid in valid_criteria
        ]
        covered.setdefault(claim_id, []).extend(ids)

    claim_ids = [int(c["id"]) for c in claim_set.claims]
    hits = [1.0 if covered.get(cid) else 0.0 for cid in claim_ids]
    core_ids = claim_set.core_ids()
    core_hits = [1.0 if covered.get(cid) else 0.0 for cid in claim_ids if cid in core_ids]
    per_covered = [len(set(covered.get(cid, []))) for cid in claim_ids if covered.get(cid)]

    out: dict[str, Any] = {
        "claim_recall": (sum(hits) / len(hits)) if hits else NAN,
        "claim_recall_core": (sum(core_hits) / len(core_hits)) if core_hits else NAN,
        "mean_criteria_per_claim": (
            sum(per_covered) / len(per_covered) if per_covered else NAN
        ),
        "n_uncovered_claims": int(len(hits) - sum(hits)),
        "uncovered_claim_ids": [cid for cid in claim_ids if not covered.get(cid)],
    }

    sub_ids = [str(s["id"]) for s in claim_set.subquestions]
    if sub_ids:
        sub_covered: dict[str, list[int]] = {}
        for entry in mapping.get("subquestion_coverage") or []:
            if not isinstance(entry, dict):
                continue
            sid = str(entry.get("subquestion_id", "")).strip()
            ids = [
                cid
                for cid in (
                    _as_int(v, default=-1)
                    for v in _as_id_list(entry.get("criterion_ids"))
                )
                if cid in valid_criteria
            ]
            sub_covered.setdefault(sid, []).extend(ids)
        out["subquestion_coverage"] = sum(
            1.0 for sid in sub_ids if sub_covered.get(sid)
        ) / len(sub_ids)
    return out


def _grounding_metrics(grounded: Mapping[str, Any], rubric: Rubric) -> dict[str, Any]:
    n = len(rubric)
    counts = {label: 0 for label in GROUNDEDNESS_LABELS}
    labelled: dict[int, str] = {}
    for pos, entry in enumerate(grounded.get("labels") or [], start=1):
        if not isinstance(entry, dict):
            continue
        cid = _as_int(entry.get("criterion_id"), default=pos)
        if not 1 <= cid <= n or cid in labelled:
            continue
        label = str(entry.get("label", "")).strip().lower()
        if label not in counts:
            continue
        labelled[cid] = label
        counts[label] += 1

    total = len(labelled)
    if total == 0:
        return {"error": "no usable groundedness labels"}
    unsupported = [
        {"criterion_index": cid - 1, "title": rubric.items[cid - 1].title}
        for cid, label in sorted(labelled.items())
        if label == "unsupported"
    ]
    return {
        "grounded_fraction": counts["grounded"] / total,
        "generic_fraction": counts["generic"] / total,
        "unsupported_fraction": counts["unsupported"] / total,
        "n_criteria_labelled": total,
        "unsupported_criteria": unsupported,
    }


# ---------------------------------------------------------------------------
# Parsing + persistence helpers
# ---------------------------------------------------------------------------


def _as_list(parsed: Any, key: str) -> list[Any]:
    if isinstance(parsed, dict):
        value = parsed.get(key)
        return value if isinstance(value, list) else []
    if isinstance(parsed, list) and key in {"claims", "labels"}:
        return parsed
    return []


def _as_id_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or value == "":
        return []
    return [value]


def _as_int(value: Any, *, default: int) -> int:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return default


def _persist_claims(
    claim_sets: Mapping[str, ClaimSet],
    run_dir: RunDir | None,
    *,
    already_on_disk: set[str],
) -> None:
    if run_dir is None:
        return
    try:
        path = run_dir.path / CLAIMS_FILENAME
        fresh = [cs for uid, cs in claim_sets.items() if uid not in already_on_disk]
        if not fresh:
            return
        with path.open("a", encoding="utf-8") as fh:
            for claim_set in fresh:
                fh.write(json.dumps(claim_set.to_dict(), ensure_ascii=False) + "\n")
    except Exception as exc:  # noqa: BLE001 - artefacts must never sink a run
        logger.warning("failed to persist claims: %s", exc)


def _persist(results: CoverageResults, run_dir: RunDir | None) -> None:
    if run_dir is None:
        return
    try:
        for row in results.per_question_rows:
            run_dir.writer("coverage_per_question").write(row)
        run_dir.write_json("coverage_summary.json", results.summary())
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to persist coverage artefacts: %s", exc)
