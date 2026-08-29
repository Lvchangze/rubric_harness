"""Tools that put a criterion in front of real text and see what happens.

The distinguishing claim of this pipeline is that criteria are *executed* before
they are kept, rather than written and trusted. These two tools are that
execution, exposed so the drafting agent can run it on its own initiative
instead of waiting for a fixed critic stage:

* :class:`ExtractReferenceClaimsTool` — what does a complete answer have to say?
* :class:`ExecuteCriterionTool` — does this criterion actually fire on a correct
  answer, and does it actually *fail* on a wrong one?

The second is the important one. A criterion that a wrong answer also passes
carries no signal, and that is not visible from reading the criterion; it is
only visible by running it.

Both reuse the evaluation-side prompts verbatim (``CLAIM_EXTRACTION_SYSTEM``,
``JUDGE_SYSTEM``) so a criterion that looks good here is being measured by the
same instrument that will grade it later.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from .base import Tool, ToolContext, ToolResult, as_str_list

logger = logging.getLogger(__name__)

__all__ = ["ExtractReferenceClaimsTool", "ExecuteCriterionTool", "resolve_targets"]

#: Response text embedded in one execution call.
TARGET_EXCERPT_CHARS = 12_000
#: Criteria per execution call. Beyond this the judge starts losing ids.
MAX_CRITERIA_PER_CALL = 20


def _clip(text: str, limit: int = TARGET_EXCERPT_CHARS) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return text[:head] + f"\n…[{len(text) - limit} chars omitted]…\n" + text[-(limit - head):]


def resolve_targets(ctx: ToolContext) -> dict[str, str]:
    """Named texts a criterion can be executed against.

    ``reference`` is always present. The generator publishes its rollouts and
    any counterexamples into ``ctx.extras['texts']``, so a criterion can be
    tested against the model's own failed attempts without the agent having to
    paste them back in.
    """
    texts: dict[str, str] = {}
    reference = ctx.reference_answer
    if reference:
        texts["reference"] = reference
    extra = ctx.extras.get("texts")
    if isinstance(extra, dict):
        for key, value in extra.items():
            if isinstance(value, str) and value.strip():
                texts[str(key)] = value
    return texts


class ExtractReferenceClaimsTool(Tool):
    name = "extract_reference_claims"
    description = """
    Break the reference answer into the atomic claims a complete response would
    have to make — setup facts, each reasoning step, each value and unit, the
    conclusion, and any caveat — each tagged 'core' or 'supporting'. Call this
    first when you need to know what the rubric has to cover, and check your
    draft against the returned list so nothing essential is missing and nothing
    is invented.
    """
    uses_llm = True
    parameters = {
        "type": "object",
        "properties": {
            "focus": {
                "type": "string",
                "description": "Optional: restrict to one sub-question, e.g. 'part b'.",
            }
        },
    }

    def __init__(self, *, max_tokens: int = 12288) -> None:
        self.max_tokens = max_tokens

    async def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        if ctx.engine is None:
            return ToolResult.failure("no LLM engine is available in this run")
        if not ctx.reference_answer:
            return ToolResult.failure("this example has no reference answer")

        from ..prompts.eval_prompts import (  # noqa: PLC0415 - shared with evaluation
            CLAIM_EXTRACTION_SYSTEM,
            build_claim_extraction_prompt,
        )

        prompt = build_claim_extraction_prompt(ctx.question, ctx.reference_answer)
        focus = str(kwargs.get("focus") or "").strip()
        if focus:
            prompt += f"\n\nRestrict the claims to this part of the question: {focus}"

        parsed = await ctx.engine.chat_json(
            prompt,
            system=CLAIM_EXTRACTION_SYSTEM,
            expect="object",
            max_tokens=self.max_tokens,
            tag="tool:extract_claims",
            cache_salt=f"claims:{focus}" if focus else None,
        )
        claims = parsed.get("claims") if isinstance(parsed, dict) else None
        if not isinstance(claims, list) or not claims:
            return ToolResult.failure("claim extraction returned nothing usable")

        cleaned: list[dict[str, Any]] = []
        for i, raw in enumerate(claims, start=1):
            if not isinstance(raw, dict):
                continue
            cleaned.append(
                {
                    "id": raw.get("id", i),
                    "claim": str(raw.get("claim") or raw.get("text") or "")[:400],
                    "kind": str(raw.get("kind") or "fact"),
                    "importance": str(raw.get("importance") or "supporting"),
                }
            )
        subquestions = parsed.get("subquestions") if isinstance(parsed, dict) else None
        n_core = sum(1 for c in cleaned if c["importance"] == "core")
        return ToolResult(
            ok=True,
            data={
                "n_claims": len(cleaned),
                "n_core": n_core,
                "claims": cleaned,
                "subquestions": subquestions if isinstance(subquestions, list) else [],
                "hint": "every 'core' claim should be checked by at least one criterion",
            },
            meta={"uid": ctx.uid},
        )


class ExecuteCriterionTool(Tool):
    name = "execute_criterion"
    description = """
    Actually run one or more draft criteria against real texts and report, for
    each pair, whether the criterion is literally true of that text. Targets are
    named: 'reference' is the reference answer, and any rollouts or
    counterexamples produced so far are available under their own names (the
    list is in the result of a failed lookup, and in your task brief). You may
    also pass raw text.

    Use it to check two things a criterion must satisfy: the reference answer
    PASSES it (otherwise the criterion is wrong or unverifiable), and at least
    one wrong answer FAILS it (otherwise it does not discriminate and is
    decoration). Criteria that fail either test should be rewritten or dropped.
    """
    uses_llm = True
    parameters = {
        "type": "object",
        "properties": {
            "criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Criterion descriptions to execute, as full sentences.",
            },
            "targets": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Names of texts to run against, e.g. ['reference', 'rollout_2'].",
            },
            "response_text": {
                "type": "string",
                "description": "Optional raw text to run against, in addition to named targets.",
            },
        },
        "required": ["criteria"],
    }

    def __init__(self, *, max_tokens: int = 12288) -> None:
        self.max_tokens = max_tokens

    async def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        if ctx.engine is None:
            return ToolResult.failure("no LLM engine is available in this run")

        criteria = as_str_list(kwargs.get("criteria"))
        if not criteria:
            return ToolResult.failure("'criteria' must be a non-empty array of strings")
        if len(criteria) > MAX_CRITERIA_PER_CALL:
            return ToolResult.failure(
                f"at most {MAX_CRITERIA_PER_CALL} criteria per call; you sent {len(criteria)}"
            )

        available = resolve_targets(ctx)
        names = as_str_list(kwargs.get("targets"))
        if not names and not kwargs.get("response_text"):
            names = ["reference"]

        selected: list[tuple[str, str]] = []
        unknown: list[str] = []
        for name in names:
            if name in available:
                selected.append((name, available[name]))
            else:
                unknown.append(name)
        if inline := str(kwargs.get("response_text") or "").strip():
            selected.append(("inline_text", inline))

        if not selected:
            return ToolResult.failure(
                f"no usable target. Unknown: {unknown or names}. "
                f"Available: {', '.join(sorted(available)) or 'none'}"
            )

        from ..schema import Criterion  # noqa: PLC0415

        # Titles are dropped: the judge only needs the statement, and a title
        # would leak the drafter's intent into the verdict.
        objects = [Criterion(title="", description=c, weight=3) for c in criteria]

        per_target: dict[str, list[dict[str, Any]]] = {}
        for target_name, text in selected:
            verdicts = await self._judge_one(ctx, objects, text, target_name)
            per_target[target_name] = verdicts

        return ToolResult(
            ok=True,
            data={
                "targets_run": [n for n, _ in selected],
                "unknown_targets": unknown,
                "available_targets": sorted(available),
                "results": self._summarise(criteria, per_target),
            },
            meta={"uid": ctx.uid, "n_criteria": len(criteria)},
        )

    async def _judge_one(
        self,
        ctx: ToolContext,
        objects: Sequence[Any],
        text: str,
        target_name: str,
    ) -> list[dict[str, Any]]:
        from ..eval.judge import JUDGE_SYSTEM, build_judge_prompt  # noqa: PLC0415

        prompt = build_judge_prompt(_clip(ctx.question, 8000), _clip(text), objects)
        try:
            parsed = await ctx.engine.chat_json(
                prompt,
                system=JUDGE_SYSTEM,
                expect="array",
                max_tokens=self.max_tokens,
                tag="tool:execute_criterion",
                cache_salt=f"exec:{target_name}",
            )
        except Exception as exc:  # noqa: BLE001 - reported per target, not fatal
            logger.warning("execute_criterion failed on %s: %s", target_name, str(exc)[:200])
            return [{"id": i + 1, "error": str(exc)[:120]} for i in range(len(objects))]

        by_id: dict[int, dict[str, Any]] = {}
        if isinstance(parsed, list):
            for pos, entry in enumerate(parsed, start=1):
                if isinstance(entry, dict):
                    try:
                        by_id.setdefault(int(entry.get("id", pos)), entry)
                    except (TypeError, ValueError):
                        by_id.setdefault(pos, entry)
        out: list[dict[str, Any]] = []
        for i in range(1, len(objects) + 1):
            entry = by_id.get(i) or {}
            out.append(
                {
                    "id": i,
                    "true": bool(entry.get("true")),
                    "checkable": entry.get("checkable", True) is not False,
                    "why": str(entry.get("why") or "")[:160],
                }
            )
        return out

    @staticmethod
    def _summarise(
        criteria: Sequence[str], per_target: dict[str, list[dict[str, Any]]]
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for i, text in enumerate(criteria):
            outcomes = {
                target: verdicts[i] for target, verdicts in per_target.items() if i < len(verdicts)
            }
            passes = {t: bool(v.get("true")) for t, v in outcomes.items()}
            unchecked = [t for t, v in outcomes.items() if v.get("checkable") is False]
            reference_pass = passes.get("reference")
            others = {t: p for t, p in passes.items() if t != "reference"}

            diagnosis: list[str] = []
            if reference_pass is False:
                diagnosis.append(
                    "the reference answer FAILS this criterion — it is wrong, or it demands "
                    "something a correct answer need not say"
                )
            if others and all(others.values()) and reference_pass is not False:
                diagnosis.append(
                    "every wrong answer also passes — this criterion does not discriminate"
                )
            if unchecked:
                diagnosis.append(f"not objectively checkable against: {', '.join(unchecked)}")

            rows.append(
                {
                    "index": i,
                    "criterion": text[:200],
                    "passes": passes,
                    "why": {t: v.get("why", "") for t, v in outcomes.items()},
                    "diagnosis": diagnosis,
                    "verdict": "keep" if not diagnosis else "revise",
                }
            )
        return rows
