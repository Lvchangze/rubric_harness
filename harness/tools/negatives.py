"""Counterexample construction, with verification that the counterexample is wrong.

A criterion earns its place by failing something. That requires wrong answers,
and wrong answers are scarce: on this corpus the policy solves roughly 83% of
questions, so most examples yield no natural failure at all.

The tool generates one on demand and then *checks* it — an "incorrect" answer
the model actually got right is worse than no counterexample, because it turns
the discrimination test into noise pointing the wrong way. The verification uses
a conclusion-only comparison against the reference, with no rubric in sight, so
it stays independent of whatever criterion is being tested.

Successful counterexamples are published into ``ctx.extras['texts']`` so
``execute_criterion`` can target them by name in a later turn.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

__all__ = ["MakeCounterexampleTool", "FLAW_TYPES"]

#: Flaw catalogue. Each entry is a way a plausible answer can be wrong; the
#: descriptions are what the model is asked to produce, phrased so the result
#: still looks like a serious attempt rather than obvious nonsense.
FLAW_TYPES: dict[str, str] = {
    "missing_step": (
        "Omit one load-bearing reasoning step or condition, but keep the surrounding "
        "work and state a final answer confidently."
    ),
    "numeric_error": (
        "Follow the correct method but make one arithmetic or algebraic slip partway "
        "through, and carry it faithfully to a wrong final answer."
    ),
    "wrong_principle": (
        "Apply a plausible but incorrect principle, law, or formula for this problem, "
        "and reason correctly from it."
    ),
    "unit_error": (
        "Do the physics or chemistry correctly but mishandle a unit or a conversion, "
        "so the final value is off by a factor."
    ),
    "right_method_wrong_answer": (
        "Set the problem up correctly and then report a final answer that does not "
        "follow from the setup."
    ),
    "overconfident_incomplete": (
        "Answer only part of what was asked, while presenting it as a complete answer."
    ),
    "plausible_but_unsupported": (
        "Give a fluent, well-organised answer whose key assertion is simply not "
        "supported — the kind of answer that reads well and is wrong."
    ),
}

_GENERATE_SYSTEM = """You write realistic INCORRECT answers, for use as test cases \
when validating a grading rubric.

The answer you produce must look like a serious attempt: same register, same \
structure, same level of detail as a real solution. It must be genuinely wrong in \
the specific way requested, and wrong in a way a competent respondent could \
plausibly be wrong.

Do NOT signal the error. No hedging, no "note: this is incorrect", no comments \
pointing at the flaw. A grader must have to catch it by checking the work.

Output ONLY the answer text. No preamble, no explanation of what you did."""

_VERIFY_SYSTEM = """You are checking whether a candidate answer reached the same \
conclusion as a reference answer.

Judge ONLY whether the candidate's final conclusion agrees with the reference's. \
Ignore style, length, and whether the candidate showed its work. Accept equivalent \
notation, units, and ordinary rounding. If the question has several parts, the \
candidate is correct only if it gets every part the reference states.

Return ONLY this JSON object:

{"correct": true or false, "why": "<one sentence>"}"""


class MakeCounterexampleTool(Tool):
    name = "make_counterexample"
    description = """
    Build a realistic WRONG answer to this question, of a flaw type you choose,
    and verify that it is actually wrong before returning it. Use it when you
    need something for a criterion to fail: a criterion no wrong answer fails
    cannot separate good responses from bad ones, and this is how you get a
    wrong response to test against.

    The result is registered under a name you can pass to execute_criterion's
    'targets'. If verification finds the answer was accidentally correct, that
    is reported and the counterexample is not registered — ask for a different
    flaw type.
    """
    uses_llm = True
    parameters = {
        "type": "object",
        "properties": {
            "flaw_type": {
                "type": "string",
                "enum": sorted(FLAW_TYPES),
                "description": "Which way the answer should be wrong.",
            },
            "focus": {
                "type": "string",
                "description": (
                    "Optional: what specifically to get wrong, e.g. 'the angular momentum "
                    "step in part a'. Steers the flaw at the criterion you are testing."
                ),
            },
            "verify": {
                "type": "boolean",
                "description": "Check that the result really is wrong. Default true; leave it on.",
            },
        },
        "required": ["flaw_type"],
    }

    def __init__(self, *, max_tokens: int = 8192, excerpt_chars: int = 1200) -> None:
        self.max_tokens = max_tokens
        self.excerpt_chars = excerpt_chars

    async def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        if ctx.engine is None:
            return ToolResult.failure("no LLM engine is available in this run")

        flaw = str(kwargs.get("flaw_type") or "").strip()
        if flaw not in FLAW_TYPES:
            return ToolResult.failure(
                f"unknown flaw_type {flaw!r}; choose one of: {', '.join(sorted(FLAW_TYPES))}"
            )
        focus = str(kwargs.get("focus") or "").strip()
        verify = kwargs.get("verify", True) is not False

        instruction = FLAW_TYPES[flaw]
        if focus:
            instruction += f"\n\nDirect the flaw at this specifically: {focus}"
        prompt = (
            f"<question>\n{ctx.question}\n</question>\n\n"
            f"<required_flaw>\n{instruction}\n</required_flaw>\n\n"
            "Write the incorrect answer now."
        )
        salt = f"neg:{flaw}:{focus[:60]}"
        try:
            text = await ctx.engine.chat(
                prompt,
                system=_GENERATE_SYSTEM,
                max_tokens=self.max_tokens,
                temperature=0.9,
                tag="tool:make_counterexample",
                cache_salt=salt,
            )
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"could not generate a counterexample: {str(exc)[:200]}")

        text = (text or "").strip()
        if not text:
            return ToolResult.failure("the generated counterexample was empty")

        verdict: dict[str, Any] = {"checked": False}
        if verify and ctx.reference_answer:
            verdict = await self._verify(ctx, text, salt)

        # Accidentally-correct counterexamples are withheld deliberately: naming
        # one would let a later execute_criterion call treat a correct answer as
        # the negative, which inverts the discrimination test.
        accidentally_correct = verdict.get("checked") and verdict.get("correct") is True
        name = f"counterexample_{flaw}"
        if not accidentally_correct:
            ctx.extras.setdefault("texts", {})[name] = text

        data: dict[str, Any] = {
            "registered_as": None if accidentally_correct else name,
            "flaw_type": flaw,
            "verified_incorrect": (not accidentally_correct) if verdict.get("checked") else None,
            "verification": verdict,
            "excerpt": text[: self.excerpt_chars]
            + ("…" if len(text) > self.excerpt_chars else ""),
            "length_chars": len(text),
        }
        if accidentally_correct:
            data["warning"] = (
                "this attempt came out CORRECT, so it is useless as a negative and was not "
                "registered — try a different flaw_type or a more specific focus"
            )
        return ToolResult(ok=True, data=data, meta={"uid": ctx.uid, "salt": salt})

    async def _verify(self, ctx: ToolContext, text: str, salt: str) -> dict[str, Any]:
        prompt = (
            f"<question>\n{ctx.question}\n</question>\n\n"
            f"<reference_answer>\n{ctx.reference_answer[:12000]}\n</reference_answer>\n\n"
            f"<candidate_answer>\n{text[:12000]}\n</candidate_answer>\n\n"
            "Does the candidate reach the same conclusion as the reference? Output the JSON now."
        )
        try:
            parsed = await ctx.engine.chat_json(
                prompt,
                system=_VERIFY_SYSTEM,
                expect="object",
                max_tokens=4096,
                tag="tool:counterexample_verify",
                cache_salt=f"verify:{salt}",
            )
        except Exception as exc:  # noqa: BLE001 - an unverified negative is still usable
            logger.debug("counterexample verification failed: %s", str(exc)[:200])
            return {"checked": False, "error": str(exc)[:160]}
        return {
            "checked": True,
            "correct": bool(parsed.get("correct")) if isinstance(parsed, dict) else None,
            "why": str((parsed or {}).get("why", ""))[:200] if isinstance(parsed, dict) else "",
        }
