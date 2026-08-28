"""Prompts for metric families 2-5 (transfer, coverage, intrinsic, head-to-head).

All prompt text lives here so that a reviewer can audit every instruction the
judge model ever sees in one file, and so the metric modules contain only
logic. Two conventions hold throughout:

* **Source blinding.** Nothing rendered here names the rubric's provenance.
  :func:`render_criteria` deliberately emits only ``title``/``category``/
  ``description``/``weight`` — never ``provenance`` or ``validation``, which
  would identify the agentic generator. The head-to-head prompt goes further
  and labels the two rubrics only "A" and "B".
* **Stable 1-based ids.** Every batched prompt numbers its items from 1 and
  asks for those ids back, so a reply can be re-aligned even when the model
  reorders, drops or duplicates entries.

Metric family 2 (transfer) reuses :mod:`harness.eval.judge`'s prompt unchanged
— that is the point of the design — so it contributes no prompt here.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

__all__ = [
    "ATOMICITY_SYSTEM",
    "CLAIM_EXTRACTION_SYSTEM",
    "COVERAGE_MAP_SYSTEM",
    "GROUNDEDNESS_SYSTEM",
    "HEADTOHEAD_SYSTEM",
    "REDUNDANCY_SYSTEM",
    "VERIFIABILITY_SYSTEM",
    "build_atomicity_prompt",
    "build_claim_extraction_prompt",
    "build_coverage_map_prompt",
    "build_groundedness_prompt",
    "build_headtohead_prompt",
    "build_redundancy_prompt",
    "build_verifiability_prompt",
    "render_claims",
    "render_criteria",
    "render_subquestions",
]

# Reference answers are pasted into several prompts alongside a rubric; this cap
# keeps the total inside the judge's context on the longest RaR rows.
MAX_REFERENCE_CHARS = 12_000
MAX_QUESTION_CHARS = 8_000


def _clip(text: str | None, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[:limit] + "\n[...truncated...]"


# ---------------------------------------------------------------------------
# Shared renderers
# ---------------------------------------------------------------------------


def render_criteria(
    criteria: Sequence[Any],
    *,
    with_category: bool = True,
    with_weight: bool = False,
) -> str:
    """Number a criterion list 1..N for a batched prompt.

    Accepts :class:`~harness.schema.Criterion` objects. Only the four public
    display fields are rendered; provenance metadata is never shown.
    """
    lines: list[str] = []
    for i, c in enumerate(criteria, start=1):
        parts: list[str] = [f"{i}."]
        if with_category:
            parts.append(f"({getattr(getattr(c, 'category', None), 'value', 'Important')})")
        title = (getattr(c, "title", "") or "").strip()
        if title:
            parts.append(f"[{title}]")
        parts.append((getattr(c, "description", "") or "").strip())
        if with_weight:
            parts.append(f"(weight {int(getattr(c, 'weight', 0))})")
        lines.append(" ".join(parts))
    return "\n".join(lines)


def render_claims(claims: Sequence[Mapping[str, Any]]) -> str:
    """Number an extracted claim list; ids come from the claim records."""
    lines = []
    for claim in claims:
        cid = claim.get("id")
        importance = claim.get("importance") or "supporting"
        lines.append(f"{cid}. ({importance}) {str(claim.get('text', '')).strip()}")
    return "\n".join(lines)


def render_subquestions(subquestions: Sequence[Mapping[str, Any]]) -> str:
    lines = []
    for sub in subquestions:
        lines.append(f"{sub.get('id')}. {str(sub.get('text', '')).strip()}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Metric family 3 — coverage / grounding
# ---------------------------------------------------------------------------

CLAIM_EXTRACTION_SYSTEM = (
    "You decompose a reference answer into the atomic, checkable claims a good response "
    "would need to contain.\n\n"
    "Rules:\n"
    "- One claim = ONE assertion, step, value, definition or caveat. Never join two "
    "assertions with 'and'.\n"
    "- Cover the whole reference answer: setup facts, each reasoning step, each numeric "
    "value or unit, the final conclusion, and any explicit caveat or assumption.\n"
    "- Use the reference answer's own substance. Do NOT invent claims it does not make, "
    "and do NOT emit style requirements ('is clear', 'is well organised') — those are not "
    "claims about content.\n"
    "- Write each claim so it can be checked against a candidate response without reading "
    "the reference answer.\n"
    "- Emit 5-30 claims depending on the answer's length and density.\n"
    "- kind: 'fact' (a stated fact or definition), 'step' (a reasoning or calculation step), "
    "'value' (a specific number, unit, name or answer choice), 'caveat' (an assumption, "
    "limitation or exception), 'conclusion' (the final answer).\n"
    "- importance: 'core' if a response omitting it would be substantively wrong or "
    "incomplete; 'supporting' otherwise.\n"
    "- subquestions: if the question explicitly asks several separately-answerable parts "
    "(labelled (a)/(b)/(c), numbered, or joined by 'and also compute...'), list them. "
    "If the question asks exactly one thing, return an empty list.\n\n"
    "Output ONLY this JSON object:\n"
    '{"claims": [{"id": 1, "text": "...", "kind": "fact", "importance": "core"}], '
    '"subquestions": [{"id": "a", "text": "..."}]}\n'
    "No prose outside the JSON object."
)


def build_claim_extraction_prompt(question: str, reference_answer: str) -> str:
    return (
        f"<question>\n{_clip(question, MAX_QUESTION_CHARS)}\n</question>\n\n"
        f"<reference_answer>\n{_clip(reference_answer, MAX_REFERENCE_CHARS)}\n"
        f"</reference_answer>\n\n"
        "Decompose the reference answer into atomic claims and list the question's "
        "sub-questions. Output the JSON object now."
    )


COVERAGE_MAP_SYSTEM = (
    "You audit whether a grading rubric covers the content of a reference answer.\n\n"
    "You are given: numbered CLAIMS extracted from the reference answer, optional numbered "
    "SUB-QUESTIONS, and a numbered CHECKLIST of rubric criteria.\n\n"
    "For each claim, list the criterion ids that would cause a grader to check that claim. "
    "Rules:\n"
    "- A criterion covers a claim only if verifying the criterion requires the response to "
    "contain (or contradict) that claim's substance. Topical adjacency is not coverage.\n"
    "- A generic criterion ('is accurate', 'is clearly written') covers NOTHING. Return an "
    "empty list for claims whose only matches are generic.\n"
    "- A single criterion may cover several claims, and a claim may be covered by several "
    "criteria. Return an empty list for claims no criterion covers.\n"
    "- For each sub-question, list the criterion ids that specifically address that part.\n"
    "- Use every claim id exactly once in claim_coverage, and every sub-question id exactly "
    "once in subquestion_coverage.\n\n"
    "Output ONLY this JSON object:\n"
    '{"claim_coverage": [{"claim_id": 1, "criterion_ids": [3, 7]}], '
    '"subquestion_coverage": [{"subquestion_id": "a", "criterion_ids": [2]}]}\n'
    "No prose outside the JSON object."
)


def build_coverage_map_prompt(
    question: str,
    claims: Sequence[Mapping[str, Any]],
    subquestions: Sequence[Mapping[str, Any]],
    criteria: Sequence[Any],
) -> str:
    sub_block = (
        f"<subquestions>\n{render_subquestions(subquestions)}\n</subquestions>\n\n"
        if subquestions
        else "<subquestions>\n(none)\n</subquestions>\n\n"
    )
    return (
        f"<question>\n{_clip(question, MAX_QUESTION_CHARS)}\n</question>\n\n"
        f"<claims>\n{render_claims(claims)}\n</claims>\n\n"
        f"{sub_block}"
        f"<checklist>\n{render_criteria(criteria)}\n</checklist>\n\n"
        f"Map all {len(claims)} claims and {len(subquestions)} sub-questions. "
        f"Criterion ids are 1..{len(criteria)}. Output the JSON object now."
    )


GROUNDEDNESS_SYSTEM = (
    "You label how each criterion of a grading rubric relates to the reference answer.\n\n"
    "Label each criterion with exactly one of:\n"
    "- 'grounded': it checks specific substantive content that the reference answer states "
    "or clearly implies (a fact, a step, a value, a named entity, a required caveat).\n"
    "- 'generic': it is a style, format, tone, clarity, completeness or general-quality "
    "requirement that could be pasted onto any question in this subject without change. "
    "Judge by CONTENT, not by wording: 'accurately performs the calculation' is generic "
    "unless it names what is being calculated.\n"
    "- 'unsupported': it asserts, requires or presupposes something that is NOT in the "
    "reference answer and NOT implied by it — including a value that contradicts the "
    "reference. These are potentially wrong criteria.\n\n"
    "When torn between 'grounded' and 'generic', ask: would this criterion still make sense "
    "verbatim for a different question in this subject? If yes, it is 'generic'.\n"
    "A criterion naming a specific fact that is merely absent from the reference (rather "
    "than contradicted) is still 'unsupported'.\n\n"
    "Output ONLY a JSON array, one object per criterion, in the order given:\n"
    '[{"criterion_id": 1, "label": "grounded", "why": "<=15 words"}]\n'
    "No prose outside the JSON array."
)


def build_groundedness_prompt(
    question: str, reference_answer: str, criteria: Sequence[Any]
) -> str:
    return (
        f"<question>\n{_clip(question, MAX_QUESTION_CHARS)}\n</question>\n\n"
        f"<reference_answer>\n{_clip(reference_answer, MAX_REFERENCE_CHARS)}\n"
        f"</reference_answer>\n\n"
        f"<checklist>\n{render_criteria(criteria)}\n</checklist>\n\n"
        f"Return exactly {len(criteria)} objects with criterion_id 1..{len(criteria)}."
    )


# ---------------------------------------------------------------------------
# Metric family 4 — intrinsic quality
# ---------------------------------------------------------------------------

REDUNDANCY_SYSTEM = (
    "You find redundant pairs inside a single grading rubric.\n\n"
    "Two criteria are REDUNDANT when a grader checking one would almost always reach the "
    "same verdict on the other, because they test the same underlying content — even if "
    "worded differently, or if one is a strict special case of the other.\n\n"
    "They are NOT redundant when they test different facts, different steps, or different "
    "aspects (e.g. 'states the value' vs 'derives the value'), or when one is positive and "
    "the other names a distinct pitfall.\n\n"
    "Report each redundant pair once, with a < b. Report no pair more than once. "
    "If there are none, return an empty list.\n\n"
    "Output ONLY this JSON object:\n"
    '{"redundant_pairs": [{"a": 2, "b": 5, "why": "<=15 words"}]}\n'
    "No prose outside the JSON object."
)


def build_redundancy_prompt(question: str, criteria: Sequence[Any]) -> str:
    return (
        f"<question>\n{_clip(question, MAX_QUESTION_CHARS)}\n</question>\n\n"
        f"<checklist>\n{render_criteria(criteria)}\n</checklist>\n\n"
        f"Criterion ids are 1..{len(criteria)}. Output the JSON object now."
    )


VERIFIABILITY_SYSTEM = (
    "You label how objectively each grading criterion can be applied.\n\n"
    "Label each criterion with exactly one of:\n"
    "- 'objective': two competent graders reading only the candidate response would reach "
    "the same yes/no verdict, because the criterion names a concrete, checkable thing "
    "(a stated fact, a value, a required step, a specific omission).\n"
    "- 'subjective': the verdict depends on taste or degree — 'clear', 'well organised', "
    "'appropriately detailed', 'demonstrates strong understanding'.\n"
    "- 'underspecified': the criterion is concrete in intent but does not say enough to be "
    "checked (it refers to 'the correct value' or 'the relevant factors' without naming "
    "them, so the grader must already know the answer).\n\n"
    "Pitfall criteria are objective when the mistake they name is concrete.\n"
    "Judge only the criterion text; you are not grading any response.\n\n"
    "Output ONLY a JSON array, one object per criterion, in the order given:\n"
    '[{"criterion_id": 1, "label": "objective"}]\n'
    "No prose outside the JSON array."
)


def build_verifiability_prompt(question: str, criteria: Sequence[Any]) -> str:
    return (
        f"<question>\n{_clip(question, MAX_QUESTION_CHARS)}\n</question>\n\n"
        f"<checklist>\n{render_criteria(criteria)}\n</checklist>\n\n"
        f"Return exactly {len(criteria)} objects with criterion_id 1..{len(criteria)}."
    )


ATOMICITY_SYSTEM = (
    "You count how many distinct assertions each grading criterion bundles together.\n\n"
    "n_checks = the number of independent things a grader must verify before deciding the "
    "criterion is satisfied. A criterion a response could satisfy halfway is not atomic.\n\n"
    "- 'States that energy is conserved' -> 1\n"
    "- 'States that energy is conserved and computes the final velocity' -> 2\n"
    "- 'Identifies (B), explains why, and notes the exception' -> 3\n"
    "- A list of alternatives the response may satisfy in any ONE way ('mentions X or Y') "
    "-> 1\n"
    "- A single claim restated for emphasis -> 1\n\n"
    "Output ONLY a JSON array, one object per criterion, in the order given:\n"
    '[{"criterion_id": 1, "n_checks": 1}]\n'
    "No prose outside the JSON array."
)


def build_atomicity_prompt(question: str, criteria: Sequence[Any]) -> str:
    return (
        f"<question>\n{_clip(question, MAX_QUESTION_CHARS)}\n</question>\n\n"
        f"<checklist>\n{render_criteria(criteria)}\n</checklist>\n\n"
        f"Return exactly {len(criteria)} objects with criterion_id 1..{len(criteria)}."
    )


# ---------------------------------------------------------------------------
# Metric family 5 — blind pairwise preference
# ---------------------------------------------------------------------------

HEADTOHEAD_SYSTEM = (
    "You compare two candidate grading rubrics for the SAME question and pick the better "
    "one. Both were written by anonymous authors; you know nothing about their origin and "
    "must not speculate about it.\n\n"
    "A better rubric is one that would let a grader separate a genuinely good response from "
    "a plausible-looking bad one. Weigh, in this order:\n"
    "1. Query specificity — its criteria are about THIS question's content, not statements "
    "that would apply to any question in the subject.\n"
    "2. Coverage — the substance a correct answer needs is actually checked, including the "
    "final answer and any sub-parts.\n"
    "3. Checkability — a grader can decide each criterion yes/no from the response alone.\n"
    "4. Correctness — no criterion demands something false or absent from the reference.\n"
    "5. Economy — criteria are atomic and non-redundant, and the weights match how much "
    "each item actually matters.\n\n"
    "Do NOT reward a rubric for merely being longer, more verbose, or more elaborately "
    "formatted. Length is only a virtue where it buys coverage.\n"
    "Choose 'tie' only when the two are genuinely indistinguishable in quality.\n\n"
    "Output ONLY this JSON object:\n"
    '{"winner": "A", "why": "<=40 words"}\n'
    'winner must be exactly "A", "B" or "tie". No prose outside the JSON object.'
)


def build_headtohead_prompt(
    question: str,
    reference_answer: str,
    rubric_a: Sequence[Any],
    rubric_b: Sequence[Any],
) -> str:
    return (
        f"<question>\n{_clip(question, MAX_QUESTION_CHARS)}\n</question>\n\n"
        f"<reference_answer>\n{_clip(reference_answer, MAX_REFERENCE_CHARS)}\n"
        f"</reference_answer>\n\n"
        f"<rubric_A>\n{render_criteria(rubric_a, with_weight=True)}\n</rubric_A>\n\n"
        f"<rubric_B>\n{render_criteria(rubric_b, with_weight=True)}\n</rubric_B>\n\n"
        "Which rubric would grade responses to this question better? "
        "Output the JSON object now."
    )
