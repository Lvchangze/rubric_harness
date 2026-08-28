"""The controlled response ladder: one quality-stratified answer set per question.

The whole comparison between rubric sources rests on a single control: all three
sources are scored on the **same** candidate responses. So the ladder is built
once per *question* and keyed by ``uid`` alone — :class:`ResponseSet` has no
rubric-source axis, which is what makes "score the agentic rubric on nicer
responses" unrepresentable rather than merely discouraged. Every emitted row
carries :meth:`ResponseSet.fingerprint` so a downstream reader can prove the
sources saw identical text.

Quality is known by construction rather than measured: ``gold`` is the dataset's
reference answer verbatim, and each degradation is generated from it with exactly
one named defect (see :mod:`harness.prompts.responses`). The resulting ordinal
levels in :data:`QUALITY_LEVELS` are the ground truth that
:mod:`harness.eval.discriminative` ranks against.

Construction is best-effort by design: a variant that fails to generate, comes
back empty, or comes back indistinguishable from gold is *dropped and recorded*
in :attr:`ResponseSet.failures`, never allowed to abort the question. A cheap LLM
QC pass then checks that each degradation really carries its intended defect;
its verdict is recorded in ``CandidateResponse.meta['validation']`` and never
used to silently discard anything.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..llm import JSONParseError, LLMEngine
from ..prompts.responses import (
    VALIDATION_SYSTEM,
    VARIANT_CHECKS,
    VARIANT_INSTRUCTIONS,
    build_degradation_prompt,
    build_validation_prompt,
    system_for_variant,
)
from ..schema import Example
from ..tracing import JsonlWriter, RunDir, read_jsonl

logger = logging.getLogger(__name__)

__all__ = [
    "QUALITY_LEVELS",
    "GOOD_QUALITY_THRESHOLD",
    "DEFAULT_VARIANTS",
    "DEGRADED_VARIANTS",
    "CandidateResponse",
    "ResponseSet",
    "assign_distractors",
    "build_response_set",
    "build_all_response_sets",
    "save_response_sets",
    "load_response_sets",
    "validation_report",
]


# Ordinal ground-truth quality, higher = better. Ties are intentional: we can
# defend "gold beats a wrong answer" but not "a propagated arithmetic slip beats
# a botched evaluation of the right formula", so those share a level and ranking
# metrics simply skip the pair.
#
# 5 gold                      reference answer: correct result, correct work
# 4 terse_correct             correct result, no work shown
# 3 missing_step              correct result, one load-bearing step absent
# 2 right_method_wrong_answer correct method, wrong result
# 2 numeric_error             wrong result from a propagated early slip
# 1 verbose_empty             fluent, confident, no derivation and no result
# 0 off_topic                 competent answer to a different question
#
# The one debatable adjacency is 4 vs 3: both state the correct result, and they
# are separated on the view that an unsupported *claim* is worth more than an
# argument that does not establish its own conclusion. A reader who disagrees can
# level them — ranking metrics skip equal-level pairs, so nothing else changes.
QUALITY_LEVELS: dict[str, int] = {
    "gold": 5,
    "terse_correct": 4,
    "missing_step": 3,
    "right_method_wrong_answer": 2,
    "numeric_error": 2,
    "verbose_empty": 1,
    "off_topic": 0,
}

# "Good" side of the binary split used for the headline AUC: the response is
# usable as an answer — its stated result is correct — even if the work is thin.
GOOD_QUALITY_THRESHOLD: int = 3

DEFAULT_VARIANTS: tuple[str, ...] = (
    "gold",
    "missing_step",
    "numeric_error",
    "right_method_wrong_answer",
    "verbose_empty",
    "terse_correct",
    "off_topic",
)

DEGRADED_VARIANTS: tuple[str, ...] = tuple(v for v in DEFAULT_VARIANTS if v != "gold")

DEFAULT_MAX_TOKENS = 8192
DEFAULT_VALIDATION_MAX_TOKENS = 2048

_WS_RE = re.compile(r"\s+")
_WHOLE_FENCE_RE = re.compile(r"\A\s*```[A-Za-z0-9_+-]*\s*\n(?P<body>.*?)\n\s*```\s*\Z", re.DOTALL)
_PREAMBLE_RE = re.compile(
    r"\A\s*(?:sure|certainly|of course|okay|ok|here(?:'s| is)|below is|"
    r"candidate\s+(?:answer|response))\b[^\n]{0,90}:\s*\n+",
    re.IGNORECASE,
)
# Markers that the generator broke the no-self-disclosure rule. Recorded, not
# edited away: a rewritten artefact is worse than a flagged one.
_DISCLOSURE_RE = re.compile(
    r"intentional|deliberat|on purpose|\[?\s*step\s+(?:omitted|removed)\s*\]?|"
    r"\bomitted here\b|this is (?:in)?correct on purpose|as an exercise|"
    r"note[:,] the (?:above|following) (?:contains|is)",
    re.IGNORECASE,
)


def _norm(text: str) -> str:
    return _WS_RE.sub(" ", (text or "").strip()).casefold()


def _clean_text(raw: str) -> str:
    """Strip the formatting artefacts a chat model wraps around plain text."""
    text = (raw or "").strip()
    fenced = _WHOLE_FENCE_RE.match(text)
    if fenced:
        text = fenced.group("body").strip()
    text = _PREAMBLE_RE.sub("", text, count=1)
    return text.strip()


@dataclass
class CandidateResponse:
    """One candidate answer with its construction-time quality level.

    ``response_id`` is the variant name: exactly one response per variant per
    question, which keeps every downstream row addressable as
    ``(uid, rubric_source, response_id)``.
    """

    uid: str
    response_id: str
    text: str
    quality_level: int
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def variant(self) -> str:
        return str(self.meta.get("variant") or self.response_id)

    @property
    def is_gold(self) -> bool:
        return self.variant == "gold"

    @property
    def validation_ok(self) -> bool | None:
        """QC verdict: ``True``/``False``, or ``None`` if never checked."""
        validation = self.meta.get("validation")
        if not isinstance(validation, dict):
            return None
        ok = validation.get("ok")
        return bool(ok) if isinstance(ok, bool) else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "response_id": self.response_id,
            "text": self.text,
            "quality_level": int(self.quality_level),
            "meta": dict(self.meta),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "CandidateResponse":
        return cls(
            uid=str(raw["uid"]),
            response_id=str(raw["response_id"]),
            text=str(raw.get("text", "")),
            quality_level=int(raw.get("quality_level", 0)),
            meta=dict(raw.get("meta") or {}),
        )


@dataclass
class ResponseSet:
    """The ladder for one question: shared across all rubric sources."""

    uid: str
    responses: list[CandidateResponse] = field(default_factory=list)
    failures: dict[str, str] = field(default_factory=dict)
    meta: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.responses)

    def __iter__(self):
        return iter(self.responses)

    def by_variant(self) -> dict[str, CandidateResponse]:
        return {r.variant: r for r in self.responses}

    def get(self, variant: str) -> CandidateResponse | None:
        return self.by_variant().get(variant)

    def levels(self) -> dict[str, int]:
        return {r.response_id: r.quality_level for r in self.responses}

    def distinct_levels(self) -> int:
        return len({r.quality_level for r in self.responses})

    def fingerprint(self) -> str:
        """Content hash of the ladder, so shared-response use is auditable."""
        digest = hashlib.sha1()
        for r in sorted(self.responses, key=lambda r: r.response_id):
            digest.update(r.response_id.encode("utf-8"))
            digest.update(b"\x00")
            digest.update(r.text.encode("utf-8"))
            digest.update(b"\x00")
        return digest.hexdigest()[:16]

    def validation_summary(self) -> dict[str, Any]:
        """Counts of the QC pass over the degraded variants of this question."""
        checked = confirmed = unchecked = 0
        by_variant: dict[str, bool | None] = {}
        for r in self.responses:
            if r.is_gold:
                continue
            ok = r.validation_ok
            by_variant[r.variant] = ok
            if ok is None:
                unchecked += 1
            else:
                checked += 1
                confirmed += int(ok)
        return {
            "n_checked": checked,
            "n_confirmed": confirmed,
            "n_unchecked": unchecked,
            "rate": (confirmed / checked) if checked else None,
            "by_variant": by_variant,
        }

    def covers(self, variants: Iterable[str]) -> bool:
        """True if every requested variant was attempted (built or recorded)."""
        attempted = {r.variant for r in self.responses} | set(self.failures)
        return all(v in attempted for v in variants)

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "responses": [r.to_dict() for r in self.responses],
            "failures": dict(self.failures),
            "meta": dict(self.meta),
            "fingerprint": self.fingerprint(),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "ResponseSet":
        return cls(
            uid=str(raw["uid"]),
            responses=[CandidateResponse.from_dict(r) for r in raw.get("responses") or []],
            failures=dict(raw.get("failures") or {}),
            meta=dict(raw.get("meta") or {}),
        )


# ---------------------------------------------------------------------------
# Distractors for off_topic
# ---------------------------------------------------------------------------


def assign_distractors(examples: Sequence[Example]) -> dict[str, Example | None]:
    """Pair every example with another example from the *same domain*.

    Same domain keeps ``off_topic`` a relevance probe rather than a domain-shift
    probe: the answer must be recognisably wrong for *this* question, not merely
    from another field. The pairing is a deterministic cyclic shift over
    uid-sorted groups, so a rerun reproduces it exactly.
    """
    by_domain: dict[str, list[Example]] = {}
    for ex in examples:
        by_domain.setdefault(ex.domain, []).append(ex)

    out: dict[str, Example | None] = {}
    for group in by_domain.values():
        ordered = sorted(group, key=lambda e: e.uid)
        n = len(ordered)
        for i, ex in enumerate(ordered):
            pick: Example | None = None
            for offset in range(1, n):
                cand = ordered[(i + offset) % n]
                if (
                    cand.uid != ex.uid
                    and cand.reference_answer.strip()
                    and _norm(cand.question) != _norm(ex.question)
                ):
                    pick = cand
                    break
            out[ex.uid] = pick
    return out


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


async def _generate_variant(
    engine: LLMEngine,
    example: Example,
    variant: str,
    *,
    max_tokens: int,
    cache_salt: str | None,
) -> str:
    prompt = build_degradation_prompt(
        variant, question=example.question, reference_answer=example.reference_answer
    )
    text = await engine.chat(
        prompt,
        system=system_for_variant(variant),
        max_tokens=max_tokens,
        tag=f"response:{variant}",
        cache_salt=cache_salt,
    )
    return _clean_text(text if isinstance(text, str) else str(text))


async def _validate_variant(
    engine: LLMEngine,
    example: Example,
    response: CandidateResponse,
    *,
    max_tokens: int,
) -> dict[str, Any]:
    """Cheap QC verdict for one degraded candidate; never raises."""
    variant = response.variant
    if variant not in VARIANT_CHECKS:
        return {"ok": None, "error": f"no check defined for {variant!r}"}
    prompt = build_validation_prompt(
        variant,
        question=example.question,
        reference_answer=example.reference_answer,
        response=response.text,
    )
    try:
        parsed = await engine.chat_json(
            prompt,
            system=VALIDATION_SYSTEM,
            expect="object",
            max_tokens=max_tokens,
            tag=f"response_check:{variant}",
            reasoning_effort="low",
        )
    except (JSONParseError, RuntimeError) as exc:
        return {"ok": None, "error": f"check failed: {exc}"[:300]}

    if not isinstance(parsed, dict):
        return {"ok": None, "error": "check returned non-object"}

    has_defect = _as_bool(parsed.get("has_intended_defect"))
    fluent = _as_bool(parsed.get("fluent"), default=True)
    disclosing = _as_bool(parsed.get("self_disclosing"))
    return {
        "ok": bool(has_defect and fluent and not disclosing),
        "has_intended_defect": has_defect,
        "fluent": fluent,
        "self_disclosing": disclosing,
        "notes": str(parsed.get("notes", ""))[:200],
    }


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "y", "1"}:
            return True
        if text in {"false", "no", "n", "0"}:
            return False
    return default


async def build_response_set(
    engine: LLMEngine,
    example: Example,
    *,
    variants: Sequence[str] = DEFAULT_VARIANTS,
    distractor_example: Example | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    validate: bool = True,
    validation_max_tokens: int = DEFAULT_VALIDATION_MAX_TOKENS,
    cache_salt: str | None = None,
) -> ResponseSet:
    """Build the quality-stratified ladder for one question.

    ``gold`` costs no LLM call (it is ``example.reference_answer`` verbatim), and
    ``off_topic`` costs none either when ``distractor_example`` is supplied — its
    reference answer is reused, which is both cheaper and safer than asking a
    model for something "unrelated but plausible".

    Never raises: variants that fail are dropped into
    :attr:`ResponseSet.failures` with a reason.
    """
    started = time.time()
    requested = list(dict.fromkeys(variants))
    order = {v: i for i, v in enumerate(requested)}
    failures: dict[str, str] = {}
    built: dict[str, CandidateResponse] = {}
    n_calls = 0

    def add(variant: str, text: str, **meta: Any) -> None:
        built[variant] = CandidateResponse(
            uid=example.uid,
            response_id=variant,
            text=text,
            quality_level=QUALITY_LEVELS[variant],
            meta={"variant": variant, "n_chars": len(text), **meta},
        )

    gold_text = (example.reference_answer or "").strip()
    gold_norm = _norm(gold_text)

    for variant in requested:
        if variant not in QUALITY_LEVELS:
            failures[variant] = "unknown variant"

    if "gold" in order:
        if gold_text:
            add("gold", gold_text, method="reference_verbatim")
        else:
            failures["gold"] = "empty reference_answer"

    # off_topic from a sibling example needs no generation call.
    llm_variants = [
        v for v in requested if v in VARIANT_INSTRUCTIONS and v not in failures
    ]
    if "off_topic" in llm_variants and distractor_example is not None:
        distractor_text = (distractor_example.reference_answer or "").strip()
        if distractor_text and _norm(distractor_text) != gold_norm:
            add(
                "off_topic",
                distractor_text,
                method="distractor_reference",
                distractor_uid=distractor_example.uid,
            )
            llm_variants.remove("off_topic")

    if not gold_text and llm_variants:
        # Without a reference there is no anchor for the ladder, so the whole
        # question is unusable rather than partially built.
        for variant in llm_variants:
            failures[variant] = "no reference_answer to build a ladder from"
        llm_variants = []

    results = await asyncio.gather(
        *(
            _generate_variant(
                engine, example, v, max_tokens=max_tokens, cache_salt=cache_salt
            )
            for v in llm_variants
        ),
        return_exceptions=True,
    )
    n_calls += len(llm_variants)

    for variant, result in zip(llm_variants, results):
        if isinstance(result, BaseException):
            failures[variant] = f"generation failed: {result}"[:300]
            logger.warning("variant %s failed uid=%s: %s", variant, example.uid, result)
            continue
        text = str(result)
        if not text:
            failures[variant] = "empty generation"
            continue
        if _norm(text) == gold_norm:
            # Identical text cannot discriminate; keeping it would manufacture a
            # tie at two different quality levels.
            failures[variant] = "identical to gold"
            continue
        disclosure = _DISCLOSURE_RE.search(text)
        add(
            variant,
            text,
            method="llm_degradation",
            disclosure_marker=disclosure.group(0) if disclosure else None,
        )

    if validate and built:
        to_check = [r for v, r in built.items() if v != "gold"]
        verdicts = await asyncio.gather(
            *(
                _validate_variant(engine, example, r, max_tokens=validation_max_tokens)
                for r in to_check
            ),
            return_exceptions=True,
        )
        n_calls += len(to_check)
        for response, verdict in zip(to_check, verdicts):
            if isinstance(verdict, BaseException):
                verdict = {"ok": None, "error": f"check exception: {verdict}"[:200]}
            response.meta["validation"] = verdict

    responses = sorted(
        built.values(), key=lambda r: (-r.quality_level, order.get(r.variant, 99))
    )
    response_set = ResponseSet(
        uid=example.uid,
        responses=responses,
        failures=failures,
        meta={
            "domain": example.domain,
            "variants_requested": requested,
            "distractor_uid": getattr(distractor_example, "uid", None),
            "n_llm_calls": n_calls,
            "validated": bool(validate),
            "wall_seconds": round(time.time() - started, 2),
        },
    )
    response_set.meta["validation"] = response_set.validation_summary()
    if response_set.distinct_levels() < 2:
        logger.warning(
            "uid=%s ladder has <2 distinct quality levels (%d responses) — "
            "ranking metrics will skip it",
            example.uid,
            len(responses),
        )
    return response_set


async def build_all_response_sets(
    engine: LLMEngine,
    examples: Sequence[Example],
    *,
    variants: Sequence[str] = DEFAULT_VARIANTS,
    distractors: Mapping[str, Example | None] | None = None,
    cache_path: str | Path | None = None,
    run_dir: RunDir | None = None,
    reuse: bool = True,
    concurrency: int = 8,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    validate: bool = True,
    validation_max_tokens: int = DEFAULT_VALIDATION_MAX_TOKENS,
) -> dict[str, ResponseSet]:
    """Build ladders for many questions, reusing a previous ``responses.jsonl``.

    Resolution order for the artefact path is ``cache_path`` then
    ``run_dir.file("responses.jsonl")``. Cached sets are reused only when they
    attempted every requested variant, so widening ``variants`` rebuilds rather
    than silently returning a short ladder. New sets are appended as they finish,
    which makes an interrupted run resumable.
    """
    path = Path(cache_path) if cache_path else (run_dir.file("responses.jsonl") if run_dir else None)
    existing: dict[str, ResponseSet] = {}
    if path is not None and reuse:
        existing = {
            uid: rs
            for uid, rs in load_response_sets(path).items()
            if rs.covers(variants) and rs.responses
        }
        if existing:
            logger.info("reusing %d cached response sets from %s", len(existing), path)

    if distractors is None:
        distractors = assign_distractors(examples)

    pending = [ex for ex in examples if ex.uid not in existing]
    writer = JsonlWriter(path) if path is not None else None
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def one(example: Example) -> ResponseSet:
        async with semaphore:
            response_set = await build_response_set(
                engine,
                example,
                variants=variants,
                distractor_example=distractors.get(example.uid) if distractors else None,
                max_tokens=max_tokens,
                validate=validate,
                validation_max_tokens=validation_max_tokens,
            )
        if writer is not None:
            writer.write(response_set)
        return response_set

    fresh = await asyncio.gather(*(one(ex) for ex in pending), return_exceptions=True)
    for example, result in zip(pending, fresh):
        if isinstance(result, BaseException):
            logger.error("response set failed uid=%s: %s", example.uid, result)
            existing[example.uid] = ResponseSet(
                uid=example.uid,
                failures={"*": f"build failed: {result}"[:300]},
                meta={"domain": example.domain},
            )
        else:
            existing[example.uid] = result

    return {ex.uid: existing[ex.uid] for ex in examples if ex.uid in existing}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_response_sets(
    response_sets: Mapping[str, ResponseSet], path: str | Path
) -> Path:
    """Write one JSON object per ladder (overwrites)."""
    out = Path(path)
    if out.exists():
        out.unlink()
    writer = JsonlWriter(out)
    for uid in sorted(response_sets):
        writer.write(response_sets[uid])
    return out


def load_response_sets(path: "str | Path | RunDir") -> dict[str, ResponseSet]:
    """Read ``responses.jsonl``; later records win, so appends act as updates.

    Accepts either the artefact path or the :class:`RunDir` that contains it, so
    callers that already hold a run directory need not know the file name.
    """
    if isinstance(path, RunDir):
        path = path.file("responses.jsonl")
    out: dict[str, ResponseSet] = {}
    for record in read_jsonl(path):
        if not isinstance(record, dict) or "uid" not in record:
            continue
        try:
            response_set = ResponseSet.from_dict(record)
        except Exception as exc:  # noqa: BLE001 - a bad line must not kill a run
            logger.warning("skipping malformed response record: %s", exc)
            continue
        out[response_set.uid] = response_set
    return out


# ---------------------------------------------------------------------------
# Quality control reporting
# ---------------------------------------------------------------------------


def validation_report(response_sets: Mapping[str, ResponseSet]) -> dict[str, Any]:
    """Aggregate QC pass rate over degraded variants, overall and per variant.

    Reported, not enforced: a low rate for one variant is a finding about the
    degradation prompt, and the affected responses stay in the ladder so the
    number can be checked against them.
    """
    per_variant: dict[str, dict[str, int]] = {}
    disclosure_hits = 0
    n_responses = 0
    failures: dict[str, int] = {}

    for response_set in response_sets.values():
        for variant, reason in response_set.failures.items():
            key = f"{variant}:{reason.split(':')[0]}"
            failures[key] = failures.get(key, 0) + 1
        for response in response_set.responses:
            n_responses += 1
            if response.is_gold:
                continue
            bucket = per_variant.setdefault(
                response.variant,
                {"n": 0, "checked": 0, "confirmed": 0, "unchecked": 0,
                 "defect_absent": 0, "not_fluent": 0, "self_disclosing": 0},
            )
            bucket["n"] += 1
            if response.meta.get("disclosure_marker"):
                disclosure_hits += 1
            validation = response.meta.get("validation")
            if not isinstance(validation, dict) or not isinstance(validation.get("ok"), bool):
                bucket["unchecked"] += 1
                continue
            bucket["checked"] += 1
            bucket["confirmed"] += int(bool(validation["ok"]))
            bucket["defect_absent"] += int(not validation.get("has_intended_defect", True))
            bucket["not_fluent"] += int(not validation.get("fluent", True))
            bucket["self_disclosing"] += int(bool(validation.get("self_disclosing")))

    checked = sum(b["checked"] for b in per_variant.values())
    confirmed = sum(b["confirmed"] for b in per_variant.values())
    for bucket in per_variant.values():
        bucket["rate"] = round(bucket["confirmed"] / bucket["checked"], 4) if bucket["checked"] else None  # type: ignore[assignment]

    return {
        "n_questions": len(response_sets),
        "n_responses": n_responses,
        "n_degraded_checked": checked,
        "n_degraded_confirmed": confirmed,
        "pass_rate": round(confirmed / checked, 4) if checked else None,
        "n_unchecked": sum(b["unchecked"] for b in per_variant.values()),
        "n_disclosure_markers": disclosure_hits,
        "by_variant": {k: per_variant[k] for k in sorted(per_variant)},
        "build_failures": dict(sorted(failures.items())),
    }
