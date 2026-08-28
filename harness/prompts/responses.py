"""Prompts for the controlled response ladder and its quality-control pass.

The ladder needs candidate answers whose *quality is known by construction*: a
gold answer, and degradations that each carry exactly one named defect. Two
properties of these prompts do the real work:

* **No self-disclosure.** A degraded answer that says "note: this step is
  intentionally wrong" measures the judge's ability to read disclaimers, not to
  detect defects. Every template forbids annotation, hedging and preamble.
* **One flaw at a time.** Everything except the named defect must stay faithful
  to the reference, so a score drop is attributable to that defect alone.
  Length is held near the reference except where the variant is *defined* by its
  length (``verbose_empty``, ``terse_correct``).

``build_degradation_prompt`` renders the user turn and ``system_for_variant``
picks the system turn: the flaw-injection framing is *not* used for
``terse_correct`` or ``off_topic``, because a system prompt that demands a
deliberate flaw makes the model corrupt a value it was asked to copy — observed,
then fixed by splitting the framing. ``build_validation_prompt`` drives the QC
pass that checks a degradation actually landed.
"""

from __future__ import annotations

__all__ = [
    "DEGRADATION_SYSTEM",
    "TRANSCRIPTION_SYSTEM",
    "OFF_TOPIC_SYSTEM",
    "VARIANT_SYSTEMS",
    "VARIANT_INSTRUCTIONS",
    "VARIANTS_NEEDING_REFERENCE",
    "VALIDATION_SYSTEM",
    "VARIANT_CHECKS",
    "system_for_variant",
    "build_degradation_prompt",
    "build_validation_prompt",
]


_OUTPUT_RULES = """Output rules:
- Output ONLY the candidate answer text. No preamble, no sign-off, no heading such as "Candidate answer:", no code fence around the whole answer, and no explanation of what you did.
- Never comment on how the answer was constructed: no annotations, no disclaimers, no notes to the reader, nothing like "note that this is incorrect", "intentionally", "(wrong)", "[step omitted]", "as an exercise".
- Keep the writing fluent, confident and well organised, in the same language, notation and register as the reference.
- The answer must read exactly like a sincere attempt by a competent student, so that a grader who has never seen the reference finds nothing suspicious about the prose itself."""


DEGRADATION_SYSTEM = f"""You are a data engineer building a controlled test set for research on automatic graders. You take a correct reference solution and produce ONE candidate answer that carries one specific, deliberately introduced flaw.

- Introduce exactly the one flaw you are asked for, and no others. Everything else stays faithful to the reference: same approach, same notation, same units, same domain conventions.
- The grader must detect the flaw unaided, so nothing in the text may advertise it.

{_OUTPUT_RULES}"""


TRANSCRIPTION_SYSTEM = f"""You are a data engineer building a controlled test set for research on automatic graders. You take a correct reference solution and re-present its content under a strict format constraint.

This is NOT a solving task and NOT an error-injection task. Whatever content you keep must remain exactly as correct as it was in the reference: copy values, fractions, units, signs, options and the wording of conclusions instead of recomputing, converting or rephrasing them. Introducing an error, or any claim the reference does not make, destroys the item - the format constraint is the only thing that may change.

{_OUTPUT_RULES}"""


OFF_TOPIC_SYSTEM = f"""You are a data engineer building a controlled test set for research on automatic graders. You write a competent answer to a question OTHER than the one shown, to serve as a relevance distractor.

The answer must be genuinely good on its own terms, and must not answer, partially answer or set up the question shown. Never signal that you are answering something else.

{_OUTPUT_RULES}"""


# Task blocks. Rendered under the shared <question>/<reference_solution> frame,
# so they only describe the defect, never the inputs.
VARIANT_INSTRUCTIONS: dict[str, str] = {
    "missing_step": """Reproduce the reference solution with exactly ONE load-bearing reasoning step silently removed.

- The removed step must be genuinely load-bearing: a reader could not reproduce the result without it. Good candidates are the key substitution, the governing relation being applied, a case distinction, the combinatorial factor, the argument that fixes a sign or direction. Do NOT remove a restatement, a unit conversion, a numerical double-check or a decorative aside.
- The cut must be silent. No ellipsis, no "it can be shown that", no "skipping ahead", no gap marker, no sentence that draws attention to the missing link. Bridge the surrounding sentences so the prose reads as a continuous, complete argument.
- Everything else survives unchanged: the same approach, the same remaining intermediate quantities, and the SAME final answer as the reference, stated explicitly and correctly at the end.
- Remove the step; do not corrupt what remains. Every equation, value and claim still on the page must be exactly as correct as it was in the reference. In particular, do not leave behind an expression that no longer evaluates to the answer you state - if a factor or term belongs to the removed step, drop the whole displayed line rather than silently deleting the factor from it.
- Length: the reference minus the removed step. Do not compensate by padding elsewhere.""",
    "numeric_error": """Reproduce the reference solution with exactly ONE wrong value introduced early and then propagated consistently to the end.

- Corrupt one specific quantity, coefficient, exponent or sign in an early step. It must look like an ordinary slip: a misread datum, a dropped factor of 2, a flipped sign, an off-by-one exponent, a swapped pair of values. Not an absurd magnitude.
- Propagate it. Every later step must be arithmetically and algebraically CONSISTENT with the corrupted value, so a reader who only checks self-consistency finds nothing wrong. Never mention, correct or revert the original value.
- Consequently the final answer is wrong. State it clearly and confidently as the answer, with no hedging and without mentioning the reference's correct value anywhere.
- Method, formulas, structure and length stay as in the reference.
- If the reference contains no numbers at all, corrupt one specific qualitative determination instead - reverse a direction, a sign convention, an inequality or an ordering - and propagate that reversal just as consistently, so the stated conclusion is wrong.""",
    "right_method_wrong_answer": """Write an answer whose setup is correct but whose result is wrong.

- Keep the correct framing, in full: the right governing principle named, the right formulas written out correctly, the right quantities identified, the right plan of attack described. Someone grading only the method should be satisfied.
- Put the failure in the execution or in the final conclusion, not in the choice of method: mishandle the algebra between the correct formula and the number, evaluate a correct expression into a wrong result, or draw the wrong conclusion from correct machinery.
- The final answer(s) must be wrong by a clear margin - a different value, option or conclusion than the reference. Not a rounding difference, not the reference answer in other units, not a sign-only variant that could be read as a convention.
- Never state the reference's correct final answer anywhere in the text, not even as a possibility. Close by asserting the wrong result confidently, with no alternatives offered.
- Comparable length to the reference.""",
    "verbose_empty": """Write a long, polished, authoritative answer that contains NO actual solution.

- 350-600 words. Confident expert register, clean structure (a few short paragraphs or headed sections), vocabulary drawn from the question's own field.
- Fill it with framing: restate and reframe the question, explain why it matters, survey which general principles and considerations are relevant, describe in the abstract what a careful treatment "would involve", list caveats, assumptions and edge cases, comment on how practitioners usually approach such problems.
- It must contain NO derivation, NO computation, NO substitution of values, and NO final answer of any kind - not the correct one, not a guess, not in passing, not as a hedged summary, not in a closing "in short" line. If you catch yourself about to name a result, replace it with more framing.
- Include no formula or statement that would let a reader finish the job or that uniquely pins down the answer.
- Never admit anything is missing. Do not say "I cannot", "beyond the scope", "consult an expert", "more information is needed", and do not ask a clarifying question. To a skimming reader it must look like a complete, satisfied, even generous answer.""",
    "terse_correct": """Extract the reference solution's final result(s) and present them as the whole answer, with no supporting work.

- This is a transcription task, not a solving task. Copy each result exactly as the reference states it - same values, fractions, units, signs, options and labels. Never recompute, re-derive, simplify, convert or re-word the substance of a result: if the reference concludes 3/8, write 3/8; if it concludes "perpendicular to both", write perpendicular to both.
- Include every final result the reference commits to, covering every part of the question, and nothing else. Add no claim, qualifier or detail that is not already in the reference.
- No derivation, no formula, no justification, no restatement of the question, no caveats, no explanation of how the result is obtained. Do not name the principle used.
- One to three sentences, at most about 45 words.""",
    "off_topic": """Write a competent, fluent answer to a DIFFERENT question from the same field as the question below.

- Choose a standard, clearly unrelated topic from the same discipline, at a similar level of technicality, and answer it well. On its own terms the answer must be correct, specific and confident, with the derivation or explanation such a topic deserves.
- It must NOT answer the question below and must contain nothing that would satisfy it: do not mention the specific objects, quantities, conditions or the result the question asks about, and do not drift back towards its subject matter.
- Do not reference the question, do not compare topics, and never signal that you are answering something else. Simply answer your chosen topic directly, as if it had been the question asked.
- 120-250 words.""",
}

# ``off_topic`` deliberately never sees the reference solution: showing it is the
# main way an "unrelated" answer leaks a correct statement of the real answer.
VARIANTS_NEEDING_REFERENCE: frozenset[str] = frozenset(
    v for v in VARIANT_INSTRUCTIONS if v != "off_topic"
)

VARIANT_SYSTEMS: dict[str, str] = {
    "terse_correct": TRANSCRIPTION_SYSTEM,
    "off_topic": OFF_TOPIC_SYSTEM,
}


def system_for_variant(variant: str) -> str:
    """System turn matching the *kind* of task the variant is."""
    return VARIANT_SYSTEMS.get(variant, DEGRADATION_SYSTEM)


def build_degradation_prompt(
    variant: str,
    *,
    question: str,
    reference_answer: str,
) -> str:
    """User turn asking for one degraded candidate answer.

    Raises
    ------
    KeyError
        If ``variant`` has no template (``gold`` is built without an LLM call).
    """
    instruction = VARIANT_INSTRUCTIONS[variant]
    blocks = [f"<question>\n{question.strip()}\n</question>"]
    if variant in VARIANTS_NEEDING_REFERENCE:
        blocks.append(
            f"<reference_solution>\n{reference_answer.strip()}\n</reference_solution>"
        )
    blocks.append(f"<task>\n{instruction}\n</task>")
    blocks.append("Output the candidate answer text now, and nothing else.")
    return "\n\n".join(blocks)


VALIDATION_SYSTEM = """You are a quality-control checker for an evaluation dataset. You are given a question, the correct reference solution, and a candidate answer that was deliberately constructed to carry ONE specific defect. Decide whether the candidate really carries that defect and whether it still reads as a sincere attempt.

Rules:
- Judge only what the candidate text actually contains. Do not give credit for what it seems to be aiming at.
- Be strict. If the intended defect is absent, partially present, or replaced by a different defect, the check fails.
- "self_disclosing" is true if the candidate advertises its own flaw or its own construction in any way: disclaimers, annotations, "intentionally", gap markers, notes to the reader, meta-commentary, or a preamble such as "Here is the answer".
- "fluent" is true if the writing is coherent, well formed and free of obvious artefacts, judged as prose only - a wrong answer can still be perfectly fluent.

Output ONLY a JSON object, no prose:
{"has_intended_defect": true, "fluent": true, "self_disclosing": false, "notes": "<=25 words"}"""


# What the checker must confirm, per variant. Phrased as the property that makes
# the variant usable, so a false verdict is always actionable.
VARIANT_CHECKS: dict[str, str] = {
    "missing_step": (
        "Exactly one load-bearing reasoning step from the reference is absent, so the "
        "argument as written no longer establishes its own conclusion; the omission is "
        "silent (no gap marker or 'it can be shown'); the final answer is still stated "
        "and still agrees with the reference."
    ),
    "numeric_error": (
        "The candidate's final answer is WRONG compared with the reference (a wrong "
        "value, or a reversed direction/sign/ordering if the problem is qualitative); "
        "the error traces back to a single early corrupted value or determination and is "
        "carried through the later steps consistently rather than flagged or corrected; "
        "the reference's correct final value is never stated."
    ),
    "right_method_wrong_answer": (
        "The method, governing principle and formulas are stated correctly, yet the final "
        "answer is clearly wrong compared with the reference and is asserted confidently; "
        "the reference's correct final answer appears nowhere."
    ),
    "verbose_empty": (
        "The candidate is long and confident but contains no derivation, no computation "
        "and no final answer at all - in particular it never states the reference's "
        "answer, nor any other specific answer, even in passing or as a hedged summary; "
        "it also never admits that the answer is missing."
    ),
    "terse_correct": (
        "The candidate states the reference's final result(s) correctly and completely, in no "
        "more than about three sentences, with no derivation, formula or justification; it "
        "must not alter a value or make any claim that contradicts or goes beyond the "
        "reference, since the point of this variant is a correct answer with the work stripped."
    ),
    # Judged on irrelevance only. Whether the topic sits in the same subfield is
    # a generation-side preference, not a defect, and asking about it here
    # produced false negatives.
    "off_topic": (
        "The candidate is a competent, fluent answer to some DIFFERENT question, and it does "
        "not answer the question asked: it never states the reference's answer and contains "
        "nothing that would satisfy the question."
    ),
}


def build_validation_prompt(
    variant: str,
    *,
    question: str,
    reference_answer: str,
    response: str,
) -> str:
    """User turn for the QC pass on one degraded candidate."""
    check = VARIANT_CHECKS[variant]
    return (
        f"<question>\n{question.strip()}\n</question>\n\n"
        f"<reference_solution>\n{reference_answer.strip()}\n</reference_solution>\n\n"
        f"<candidate_answer>\n{response.strip()}\n</candidate_answer>\n\n"
        f"<intended_defect name=\"{variant}\">\n{check}\n</intended_defect>\n\n"
        "Set has_intended_defect=true only if every clause of the intended defect holds. "
        "Return the JSON object now."
    )
