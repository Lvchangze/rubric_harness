"""Prompts for the tool-using stages.

These differ from the fixed-chain prompts in one way that matters: they do not
tell the model what evidence to produce, they tell it what it will be held to
and give it instruments. The drafting stage downstream refuses criteria that
carry no verified backing, so the incentive is to actually call the tools rather
than to assert conclusions in their shape.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

__all__ = [
    "INVESTIGATE_SYSTEM",
    "CRITIC_TOOL_SYSTEM",
    "INVESTIGATE_CLOSING",
    "CRITIC_TOOL_CLOSING",
    "build_investigate_user",
    "build_critic_tool_user",
]

#: Sent when the tool budget runs out. Without it a model that is still
#: investigating returns nothing and the whole stage is discarded, which is how
#: a budget overrun turns into a total loss instead of a truncated one.
INVESTIGATE_CLOSING = (
    "You have used your tool budget — no further tool calls are possible. "
    "Write up what you established, using only findings your tool calls actually "
    "support. Anything you were still checking goes in notes_for_drafter as an "
    "open question rather than a verified fact. Output ONLY the JSON object now."
)

CRITIC_TOOL_CLOSING = (
    "You have used your tool budget — no further tool calls are possible. "
    "Decide on every remaining candidate using what you have already tested. "
    "Where you never tested one, say so in 'reason' and keep it. "
    "Output ONLY the JSON object now, with one decision per candidate id."
)

MAX_REFERENCE_CHARS = 12_000
MAX_QUESTION_CHARS = 8_000


def _clip(text: str | None, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return text[:head] + f"\n…[{len(text) - limit} chars omitted]…\n" + text[-(limit - head):]


def _json_block(value: Any, limit: int = 4000) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, indent=2, default=str)
    except (TypeError, ValueError):
        text = str(value)
    return text[:limit]


# ---------------------------------------------------------------------------
# Investigation
# ---------------------------------------------------------------------------

INVESTIGATE_SYSTEM = """You are preparing the evidence base for a grading rubric.

Someone else will write the rubric. Your job is to establish, by CHECKING rather \
than by asserting, what actually separates a correct answer to this question from \
a plausible wrong one.

You have tools. Use them — a finding you did not verify is worth nothing here, and \
the drafter is instructed to discard any claim you cannot back. In particular:

- Recompute the reference answer's numbers rather than trusting them. Reference \
answers in this corpus do contain arithmetic slips, and a rubric that enshrines one \
will mark correct responses wrong.
- Establish what a complete answer must contain before deciding what to check.
- Where attempts at this question disagreed with each other, find out which side is \
right. Disagreement is the most informative thing you have: content every attempt \
already gets right cannot discriminate between responses, so a criterion about it is \
dead weight no matter how central it looks.
- Build at least one realistic wrong answer and find out which checks it survives. A \
check that a wrong answer also passes does not discriminate.
- Test whether your intended checks are specific to THIS question. A check that fits \
unrelated questions carries no information about this one.

Work efficiently: batch related questions into one tool call, do not re-verify what \
you have already established, and stop when further checking would not change what \
the rubric should say.

When you are done, output ONLY this JSON object and nothing else:

{
  "verified_facts": [
    {"fact": "<what is true, stated concretely with the actual quantities>",
     "evidence": "<which tool call established it, and what it returned>",
     "discriminative": true|false,
     "why_discriminative": "<what a wrong answer would do differently here>"}
  ],
  "reference_issues": [
    {"issue": "<anything in the reference answer your checks contradict>",
     "evidence": "<what you computed>"}
  ],
  "failure_modes": [
    {"mode": "<a specific way a plausible answer goes wrong on THIS question>",
     "tested": true|false,
     "evidence": "<what happened when you tested it>"}
  ],
  "checks_that_do_not_discriminate": [
    {"check": "<a check you considered and rejected>",
     "reason": "<what it failed: everything passes it / it fits other questions / it is unverifiable>"}
  ],
  "notes_for_drafter": "<two or three sentences: where the rubric's weight belongs and why>"
}

Every entry in verified_facts must correspond to something a tool told you. If a tool \
failed or you ran out of turns, say so in notes_for_drafter rather than inventing \
coverage."""


def build_investigate_user(
    question: str,
    reference_answer: str,
    *,
    spec: Mapping[str, Any] | None = None,
    reconciliation: Mapping[str, Any] | None = None,
    rollouts: Sequence[Mapping[str, Any]] = (),
    available_targets: Sequence[str] = (),
    domain: str = "",
) -> str:
    parts = [
        f"<question>\n{_clip(question, MAX_QUESTION_CHARS)}\n</question>",
        f"<reference_answer>\n{_clip(reference_answer, MAX_REFERENCE_CHARS)}\n</reference_answer>",
    ]
    if domain:
        parts.append(f"<domain>{domain}</domain>")
    if spec:
        parts.append(f"<question_analysis>\n{_json_block(spec)}\n</question_analysis>")
    if reconciliation:
        parts.append(
            "<what_independent_attempts_did>\n"
            f"{_json_block(reconciliation)}\n"
            "</what_independent_attempts_did>"
        )
    if rollouts:
        summaries = [
            {"label": r.get("label"), "summary": r.get("summary")} for r in rollouts
        ]
        parts.append(f"<attempt_summaries>\n{_json_block(summaries, 2500)}\n</attempt_summaries>")
    if available_targets:
        parts.append(
            "<texts_you_can_run_criteria_against>\n"
            + ", ".join(available_targets)
            + "\nPass these names to execute_criterion's 'targets'. New counterexamples "
            "you create are added to this list.\n"
            "</texts_you_can_run_criteria_against>"
        )
    parts.append(
        "Investigate now. Use the tools, then output the JSON object. "
        "Do not output the JSON until you have finished checking."
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Tool-assisted critic
# ---------------------------------------------------------------------------

CRITIC_TOOL_SYSTEM = """You are deciding which candidate rubric criteria survive.

A criterion earns its place only if BOTH hold:

1. A correct answer satisfies it. If the reference answer fails a criterion, the \
criterion is wrong, or it demands something a correct answer need not say. Either \
way it will punish good responses.
2. Some wrong answer fails it. A criterion everything passes adds a constant to \
every score and separates nothing.

Do not judge this by reading. Run the criteria with execute_criterion, against the \
reference and against wrong answers — use the counterexamples already available, and \
build more with make_counterexample when you need a specific kind of failure. Verify \
any numeric or algebraic claim a criterion makes before keeping it; check_equivalence \
and python_eval exist so a criterion does not hard-code one spelling of a value or \
enshrine a miscalculation.

Then output ONLY this JSON object:

{
  "decisions": [
    {"id": <candidate id>,
     "decision": "keep" | "revise" | "drop",
     "revised_description": "<required when decision is 'revise'>",
     "reference_passes": true|false|null,
     "discriminates": true|false|null,
     "evidence": "<which tool call showed this>",
     "reason": "<one sentence>"}
  ],
  "summary": "<one sentence on what the surviving set now measures>"
}

Include every candidate id exactly once. Base each decision on a tool result; where \
you could not obtain one, say so in 'reason' and prefer 'keep' over guessing."""


def build_critic_tool_user(
    question: str,
    reference_answer: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    available_targets: Sequence[str] = (),
    investigation: Mapping[str, Any] | None = None,
) -> str:
    lines = [
        f"<question>\n{_clip(question, MAX_QUESTION_CHARS)}\n</question>",
        f"<reference_answer>\n{_clip(reference_answer, MAX_REFERENCE_CHARS)}\n</reference_answer>",
    ]
    if investigation:
        lines.append(f"<verified_evidence>\n{_json_block(investigation, 3000)}\n</verified_evidence>")
    lines.append(f"<candidates>\n{_json_block(list(candidates), 8000)}\n</candidates>")
    if available_targets:
        lines.append(
            "<targets_for_execute_criterion>\n"
            + ", ".join(available_targets)
            + "\n</targets_for_execute_criterion>"
        )
    lines.append(
        f"Test the {len(candidates)} candidates and output the JSON object with one "
        "decision per candidate id."
    )
    return "\n\n".join(lines)
