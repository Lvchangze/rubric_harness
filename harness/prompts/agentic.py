"""Prompt templates for the agentic (search-then-validate) rubric pipeline.

One constant per stage system prompt, one ``build_*`` function per stage user
turn. The pipeline in :mod:`harness.generators.agentic` owns all control flow;
this module owns all wording, so a prompt can be revised without touching the
orchestration (and diffed cleanly for the paper appendix).

Every stage that must be machine-read declares its output shape as a literal
JSON skeleton appended to the user turn. The skeletons are plain strings rather
than f-strings so the braces stay literal.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

__all__ = [
    "EVIDENCE_KINDS",
    "BANNED_CRITERION_PHRASES",
    "POLARITY_RULE",
    "domain_guidance",
    "render_candidates",
    "render_anchors",
    "DECOMPOSE_SYSTEM",
    "build_decompose_user",
    "ROLLOUT_SYSTEM",
    "build_rollout_user",
    "RECONCILE_SYSTEM",
    "build_reconcile_user",
    "PITFALL_SYSTEM",
    "build_pitfall_user",
    "DRAFT_SYSTEM",
    "build_draft_user",
    "build_draft_retry_user",
    "LINT_SYSTEM",
    "build_lint_user",
    "NEGATIVE_SYSTEM",
    "build_negative_user",
    "CRITIC_GOLD_SYSTEM",
    "CRITIC_NEGATIVE_SYSTEM",
    "build_critic_user",
    "CALIBRATE_SYSTEM",
    "build_calibrate_user",
]


#: Evidence classes a drafted criterion may cite. Ordered by discriminative
#: value: a criterion that separates rollouts from the reference is worth more
#: than one restating something every rollout already got right.
EVIDENCE_KINDS: tuple[str, ...] = (
    "collective_error",
    "divergent",
    "pitfall",
    "reference_only",
    "sub_question",
)

#: Generic filler that passes for a rubric item but discriminates nothing.
#: Injected verbatim into the drafting and calibration prompts as a blocklist.
BANNED_CRITERION_PHRASES: tuple[str, ...] = (
    "is clear and well-organized",
    "is numerically accurate",
    "shows the work",
    "uses correct units",
    "is factually correct",
    "demonstrates understanding",
    "provides a complete answer",
    "is logically structured",
    "avoids errors",
    "explains the reasoning",
)

_BANNED_BLOCK = "\n".join(f'- "{p}"' for p in BANNED_CRITERION_PHRASES)

#: The single-polarity contract (forensics F6). Injected into every stage that
#: writes or edits criterion text. The two shipped corpora use opposite Pitfall
#: conventions — RaR-Science writes "Avoids X" (true = good) while RaR-Medicine
#: writes "Does not mention X" (true = bad) — and both carry weight -1/-2, so any
#: single sign convention scores one whole domain backwards. We remove the
#: ambiguity at the source instead of trying to detect it later.
POLARITY_RULE = """POLARITY — one direction, no exceptions.
Every criterion must be written so that "this sentence is literally true of the response" means "the response is GOOD". A grader reading the sentence must never have to work out which way round it counts.
- Never write a criterion whose truth means the response is bad. Forbidden openers: "Does not mention", "Does not state", "Does not include", "Fails to", "Omits", "Neglects to", "Overlooks", "Ignores", "Misses".
- A criterion about a MISTAKE is written as the avoidance of that mistake, not as the mistake, and opens with "Avoids". State the specific wrong move inside the avoidance so the sentence stays checkable. Do not reach for a bare negation instead ("does not add a spurious term"); a reader should be able to see the direction from the first word rather than from whether the thing being negated is good or bad.
- Never assert a bad action as the criterion body ("Recommends <the unsafe drug>", "Claims <the wrong value>"); those read as true exactly when the response is wrong.
- Requiring content the response must contain is already the right direction; leave those alone."""

_SCIENCE_GUIDANCE = """Domain notes (physics / chemistry / biology):
- Anchor criteria to named laws, formulas, mechanisms and to numeric results with their units and (where the question implies one) their sign and order of magnitude.
- Where a derivation has load-bearing intermediate quantities, a criterion may require the specific intermediate value, not merely "shows work".
- Stated modelling assumptions (frictionless, ideal gas, steady state, dilute solution, negligible edge effects) are legitimate criteria when the answer changes without them."""

_MEDICAL_GUIDANCE = """Domain notes (clinical / biomedical):
- Anchor criteria to the specific diagnosis, drug, dose, route, timing, threshold value, imaging modality or lab finding named by this case.
- Safety-relevant content (contraindications, red flags, monitoring, escalation, dose ceilings) outranks stylistic content.
- Where the question is multiple-choice, one criterion must name the selected option label explicitly."""


def domain_guidance(domain: str) -> str:
    """Short domain-conditioned paragraph injected into the authoring stages."""
    return _MEDICAL_GUIDANCE if "medicine" in (domain or "").lower() else _SCIENCE_GUIDANCE


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _dump(payload: Any, *, limit: int = 6000) -> str:
    """Pretty JSON for embedding in a prompt, length-bounded."""
    if payload is None:
        return "(unavailable)"
    if isinstance(payload, str):
        text = payload
    else:
        try:
            text = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
        except (TypeError, ValueError):
            text = str(payload)
    text = text.strip()
    if not text:
        return "(unavailable)"
    if len(text) > limit:
        text = text[:limit] + f"\n... [truncated at {limit} chars]"
    return text


def _section(tag: str, body: str) -> str:
    return f"<{tag}>\n{body.strip()}\n</{tag}>"


def render_candidates(
    candidates: Sequence[Mapping[str, Any]], *, with_validation: bool = False
) -> str:
    """Numbered checklist of candidate criteria, keyed by stable ``id``.

    Callers supply ``prefixed_description`` already rendered so this module
    stays independent of :mod:`harness.schema`.
    """
    lines: list[str] = []
    for cand in candidates:
        cid = cand.get("id")
        title = str(cand.get("title") or "").strip()
        head = f"{cid}. [{title}]" if title else f"{cid}."
        body = str(cand.get("prefixed_description") or cand.get("description") or "").strip()
        line = f"{head} {body}"
        annotations: list[str] = []
        evidence = cand.get("evidence")
        if evidence:
            annotations.append(f"evidence={evidence}")
        if with_validation:
            weight = cand.get("weight")
            if weight is not None:
                annotations.append(f"draft_weight={weight}")
            val = cand.get("validation") or {}
            if val:
                gold = "pass" if val.get("gold_pass") else "FAIL"
                n_neg = int(val.get("n_negatives") or 0)
                n_failed = int(val.get("n_negatives_failed") or 0)
                disc = val.get("discrimination")
                disc_txt = f"{float(disc):.2f}" if isinstance(disc, (int, float)) else "n/a"
                annotations.append(
                    f"gold={gold}; caught {n_failed}/{n_neg} negatives; discrimination={disc_txt}"
                )
        if annotations:
            line += f"\n     ({'; '.join(annotations)})"
        lines.append(line)
    return "\n".join(lines) if lines else "(none)"


def render_anchors(anchors: Sequence[str], *, limit: int = 80) -> str:
    """Anchor tokens as a flat list, numbers and long tokens first.

    Ordering matters more than completeness: the list is truncated, and a number
    or a multi-syllable entity pins a criterion to its question far harder than a
    short common word does.
    """
    ordered = sorted(anchors, key=lambda t: (not any(ch.isdigit() for ch in t), -len(t), t))
    shown = ordered[:limit]
    if not shown:
        return "(none extracted — fall back to naming quantities and entities from the question text verbatim)"
    tail = f"\n... and {len(ordered) - len(shown)} more" if len(ordered) > len(shown) else ""
    return ", ".join(shown) + tail


def _render_banned_words(words: Sequence[str]) -> str:
    return ", ".join(sorted({w.strip() for w in words if w and w.strip()})) or "(none)"


def _render_rollouts(rollouts: Sequence[Mapping[str, Any]], *, excerpt_chars: int) -> str:
    blocks: list[str] = []
    for roll in rollouts:
        label = roll.get("label") or f"Rollout {roll.get('index', '?')}"
        summary = _dump(roll.get("summary"), limit=2000)
        text = str(roll.get("text") or "").strip()
        if len(text) > excerpt_chars:
            text = text[:excerpt_chars] + "\n... [truncated]"
        blocks.append(
            f"===== {label} =====\n"
            f"--- declared final answers ---\n{summary}\n"
            f"--- worked solution ---\n{text}"
        )
    return "\n\n".join(blocks) if blocks else "(no rollouts available)"


# ---------------------------------------------------------------------------
# Stage 1 - decompose
# ---------------------------------------------------------------------------

DECOMPOSE_SYSTEM = """You are a meticulous problem analyst. You are given one question that a domain expert will be asked to answer. You do NOT answer it. You take it apart.

Your output is a structured specification that a downstream grader will use to decide what a correct answer must contain. Be exhaustive about what the question actually asks and completely silent about what you think the answer is.

Rules:
- Enumerate every distinct thing the question asks for as a separate sub-question, in the order asked. If the question labels parts (a), (b), (c) or (1), (2), use those labels; otherwise label them a, b, c, ...
- Record every quantity the question supplies, with its numeric value and unit exactly as given.
- Record every quantity the question asks for, with the unit it should be expressed in.
- Record constraints and modelling assumptions the question implies but does not state (conservation laws in force, idealisations, boundary or initial conditions, patient context, excluded mechanisms).
- Record the answer format the question demands (a single number, a multiple-choice letter, a proof, a comparison with justification, a differential diagnosis, ...).
- Never invent information that is not in the question."""

_DECOMPOSE_SCHEMA = """{
  "topic": "<short topic label, e.g. 'rotational dynamics' or 'acid-base management'>",
  "domain": "<physics|chemistry|biology|medicine|other>",
  "sub_questions": [
    {"label": "a", "asks": "<what this part demands>", "target": "<the quantity, claim or decision it wants>"}
  ],
  "given": [
    {"symbol": "<symbol or short name>", "value": "<value exactly as given>", "unit": "<unit or ''>", "note": "<what it is>"}
  ],
  "targets": [
    {"symbol": "<symbol or short name>", "unit": "<expected unit or ''>", "note": "<what must be produced>"}
  ],
  "implicit_constraints": ["<assumption or constraint a competent answer must respect>"],
  "answer_format": "<what a complete response must look like>",
  "difficulty_notes": "<one sentence on where this question is easy to get wrong>"
}"""


def build_decompose_user(question: str) -> str:
    return (
        _section("question", question)
        + "\n\nReturn ONLY a JSON object with exactly this shape:\n"
        + _DECOMPOSE_SCHEMA
    )


# ---------------------------------------------------------------------------
# Stage 2 - independent rollouts
# ---------------------------------------------------------------------------

ROLLOUT_SYSTEM = """You are a domain expert solving a question under exam conditions. You have NO access to a reference answer or a marking scheme; you must reason it out yourself.

Solve the question completely and honestly:
- Work through the reasoning explicitly, stating the laws, formulas, mechanisms or clinical rules you invoke.
- Carry units through the arithmetic and state the value of every load-bearing intermediate quantity.
- State any assumption you had to make in order to proceed.
- Answer every part of the question that was asked.
- Do not hedge into vagueness. If you are unsure, commit to your best answer and say so in the confidence field.

After the worked solution, and ONLY at the very end, emit a fenced ```json block with your final answers so they can be compared mechanically. Nothing may follow that block."""

_ROLLOUT_SCHEMA = """```json
{
  "final_answers": {"<sub-question label>": "<final answer, with unit if applicable>"},
  "key_quantities": [
    {"name": "<intermediate quantity>", "value": "<value>", "unit": "<unit or ''>"}
  ],
  "methods_used": ["<law, formula, mechanism or clinical rule invoked>"],
  "assumptions": ["<assumption made>"],
  "confidence": "<high|medium|low>"
}
```"""


def build_rollout_user(question: str, spec: Mapping[str, Any] | None = None) -> str:
    parts = [_section("question", question)]
    if spec:
        parts.append(
            _section("question_structure", _dump(spec, limit=4000))
            + "\n(The structure above is a parse of the question only. It contains no answer information.)"
        )
    parts.append(
        "Solve it now. End your response with the final JSON block in exactly this shape:\n"
        + _ROLLOUT_SCHEMA
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Stage 3 - reconcile rollouts against the reference
# ---------------------------------------------------------------------------

RECONCILE_SYSTEM = """You are analysing where a strong model's independent attempts at one question AGREE and where they DIVERGE, and how each attempt compares against a trusted reference answer.

Why this matters: a grading criterion is only worth including if it can actually separate a good response from a bad one. Content that every independent attempt already produces correctly discriminates nothing — a rubric item about it is dead weight. Content the attempts disagree about, or get uniformly wrong, is exactly where grading signal lives. Your classification is the evidence base for the rubric that follows, so be precise and be specific about values.

Treat the reference answer as ground truth for the final results, even when it is terse. If the reference is silent on something, say so rather than guessing.

Classify every substantive claim into exactly one bucket:
- consensus_correct: all attempts agree AND the reference supports them. LOW discriminative value.
- divergent: the attempts disagree with each other. HIGH discriminative value — say what each camp claims.
- collective_error: all attempts agree but the reference contradicts them. HIGHEST discriminative value — this is a trap the whole model class falls into.
- reference_only: present in the reference, absent from every attempt.

Also judge each attempt as a whole against the reference, because the wrong attempts will be reused as negative test cases.

Quote specific numbers, signs, units, entities and formulas. Never write "the value differs" without saying which values."""

_RECONCILE_SCHEMA = """{
  "rollout_verdicts": [
    {"rollout": 1, "verdict": "<correct|partially_correct|incorrect>",
     "errors": ["<specific error, naming the wrong value or wrong step>"]}
  ],
  "consensus_correct": [
    {"claim": "<what all attempts got right>", "value": "<the agreed value or statement>"}
  ],
  "divergent": [
    {"claim": "<the disputed point>",
     "positions": [{"rollouts": [1], "value": "<what these attempts claim>"}],
     "reference_says": "<what the reference says, or 'silent'>",
     "who_is_right": "<which position matches the reference, or 'unclear'>",
     "why_hard": "<one sentence>"}
  ],
  "collective_error": [
    {"claim": "<what every attempt claimed>",
     "all_rollouts_say": "<the shared wrong value or wrong reasoning>",
     "reference_says": "<the correct value or statement>",
     "why_tempting": "<one sentence on why the wrong answer looks right>"}
  ],
  "reference_only": [
    {"claim": "<content only the reference has>", "why_missed": "<one sentence>"}
  ],
  "reference_scope": "<one sentence: how complete the reference answer is - full derivation, bare final answer, partial>"
}"""


def build_reconcile_user(
    question: str,
    reference_answer: str,
    rollouts: Sequence[Mapping[str, Any]],
    *,
    excerpt_chars: int = 6000,
) -> str:
    return "\n\n".join(
        [
            _section("question", question),
            _section("reference_answer", reference_answer or "(empty)"),
            _section(
                "independent_attempts",
                _render_rollouts(rollouts, excerpt_chars=excerpt_chars),
            ),
            "Return ONLY a JSON object with exactly this shape "
            f"(one verdict per attempt, numbered 1..{max(len(rollouts), 1)}):\n"
            + _RECONCILE_SCHEMA,
        ]
    )


# ---------------------------------------------------------------------------
# Stage 4 - pitfall mining
# ---------------------------------------------------------------------------

PITFALL_SYSTEM = """You mine the specific ways a competent-looking answer to THIS question goes wrong.

You are not listing generic advice. "Forgetting units" is worthless. "Reports 780 mEq as the dose to give in the first 4 hours because the full base-deficit correction was applied at once instead of a partial correction" is what we want: a named mistake, the specific wrong artefact it produces, and why it is tempting.

Look hard for failure modes of these kinds, keeping only the ones that genuinely apply here:
- sign, direction or inequality reversed
- unit or prefix mismatch, or an unconverted quantity
- a conservation law, boundary condition or initial condition dropped
- the right formula applied outside its domain of validity
- an intermediate quantity substituted for the requested one (stopping one step early)
- an off-by-a-factor error from a definition (radius/diameter, per-mole/per-gram, half-angle, RMS/peak)
- a contraindication, exclusion criterion, safety ceiling or red flag missed
- a plausible but wrong mechanism or wrong causal direction asserted
- one sub-question silently left unanswered

Prefer mistakes with evidence behind them: errors an attempt actually made, and errors every attempt made. Then add the ones you can foresee. Order by how likely and how damaging they are."""

_PITFALL_SCHEMA = """[
  {
    "title": "<2-4 words>",
    "mistake": "<the specific wrong move, naming the quantity, step, formula or entity involved>",
    "wrong_result": "<the specific wrong value, sign, unit or claim this produces for THIS question>",
    "correct_result": "<what the correct value or claim is instead>",
    "why_tempting": "<one sentence>",
    "observed": "<rollout|reference_contrast|foreseen>",
    "severity": "<high|medium|low>"
  }
]"""


def build_pitfall_user(
    question: str,
    reference_answer: str,
    *,
    spec: Mapping[str, Any] | None = None,
    reconciliation: Mapping[str, Any] | None = None,
    domain: str = "",
    max_items: int = 8,
) -> str:
    parts = [
        _section("question", question),
        _section("reference_answer", reference_answer or "(empty)"),
    ]
    if spec:
        parts.append(_section("question_structure", _dump(spec, limit=3000)))
    if reconciliation:
        parts.append(
            _section("observed_disagreement", _dump(reconciliation, limit=6000))
            + "\n(Errors recorded above were actually committed by independent attempts at this question.)"
        )
    parts.append(domain_guidance(domain))
    parts.append(
        f"List at most {max_items} pitfalls, highest value first. "
        "Return ONLY a JSON array with exactly this shape:\n" + _PITFALL_SCHEMA
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Stage 5 - draft candidate criteria
# ---------------------------------------------------------------------------

DRAFT_SYSTEM = f"""You write grading checklists that discriminate. You are given a question, a reference answer, a structural parse of the question, an analysis of where independent expert attempts agreed and disagreed, a list of mined pitfalls, and the question's anchor vocabulary. You turn that evidence into candidate criteria.

The governing test for every criterion: if I handed you two responses — one correct, one that looks correct but commits a real mistake — would this criterion come out differently for them? If not, the criterion is worthless no matter how sensible it sounds.

Hard requirements. Every criterion must be:
1. ATOMIC. One checkable fact, value, step, or claim. If your sentence contains "and" joining two independently checkable things, split it into two criteria.
2. VERIFIABLE FROM THE RESPONSE ALONE. A grader who has only the question and the response must be able to answer yes or no without consulting a textbook. State the required value or claim inside the criterion.
3. ANCHORED TO THIS QUESTION. Every criterion must contain at least one token from the supplied anchor list, used verbatim: a numeric value, a named law or formula, a specific compound, organ, drug, organism, device, option label or labelled sub-question. Prefer numbers and named entities over common words. A criterion whose text would fit an unrelated question will be deleted, not repaired.
4. POSITIVELY POLARISED. See the polarity rule below; it admits no exceptions.
5. OBJECTIVE. The judgement must not turn on taste. Banned words appear in the request; a quality adverb attached to a checkable core is not a criterion, it is that core plus noise, so write the core alone. Never write a criterion about presentation — length, structure, tone, formatting, readability, empathy — those are graded elsewhere, not here.
6. EVIDENCE-BACKED. Cite which analysis produced it.

{POLARITY_RULE}

Automatically rejected phrasings (and anything with their flavour):
{_BANNED_BLOCK}

Evidence classes, in descending order of value:
- collective_error: every attempt got this wrong and the reference contradicts them. The strongest possible criterion; always include these.
- divergent: the attempts disagreed here. Write the criterion so that it selects the position the reference supports.
- pitfall: a mined failure mode. Usually becomes a Pitfall-category criterion.
- reference_only: content only the reference supplies.
- sub_question: a required deliverable of the question — the final answer to a named part, or the demanded output format. Use this for the anchor criteria that pin down the correct final results even when every attempt got them right.

Categories, and the weight magnitude each carries:
- Essential (5): the response is invalid without it. Correct final answers and load-bearing correct claims.
- Important (3-4): key reasoning, a required intermediate value, a required justification.
- Optional (1-2): genuine extra depth in the subject matter. Use sparingly; never for filler, never for presentation.
- Pitfall (1-5): this question's specific trap, written as the avoidance of it. Name the wrong value or wrong move inside the sentence so a grader can tell the difference between a response that dodged the trap and one that never went near the topic. Report a positive magnitude; the storage layer keeps Pitfall weights on the RaR negative scale and grades by magnitude.

Balance: a criterion that a response satisfies by staying silent rewards silence. An empty response commits no mistakes, so it clears every avoidance item automatically. Criteria demanding content the response must actually contain therefore have to carry most of the checklist.

Coverage: every labelled sub-question must be represented by at least one criterion. Every high-severity pitfall must be represented. Do not write two criteria that a grader would resolve identically, and never write an avoidance criterion that is a restatement of a positive one you already wrote — that scores one fact twice."""

_DRAFT_SCHEMA = """[
  {
    "id": 1,
    "title": "<2-4 words>",
    "category": "<Essential|Important|Optional|Pitfall>",
    "description": "<one self-contained sentence, WITHOUT any category prefix>",
    "weight": 5,
    "evidence": "<collective_error|divergent|pitfall|reference_only|sub_question>",
    "evidence_detail": "<the specific finding this came from>",
    "discriminates": "<the wrong response this criterion catches>"
  }
]"""


def build_draft_user(
    question: str,
    reference_answer: str,
    *,
    spec: Mapping[str, Any] | None = None,
    reconciliation: Mapping[str, Any] | None = None,
    pitfalls: Sequence[Mapping[str, Any]] | None = None,
    domain: str = "",
    n_candidates: int = 20,
    anchors: Sequence[str] = (),
    banned_words: Sequence[str] = (),
    target_items: int | None = None,
    max_pitfall_fraction: float = 0.25,
    investigation: Mapping[str, Any] | None = None,
) -> str:
    parts = [
        _section("question", question),
        _section("reference_answer", reference_answer or "(empty)"),
    ]
    if spec:
        parts.append(_section("question_structure", _dump(spec, limit=4000)))
    if reconciliation:
        parts.append(_section("agreement_analysis", _dump(reconciliation, limit=8000)))
    if pitfalls:
        parts.append(_section("mined_pitfalls", _dump(list(pitfalls), limit=6000)))
    if investigation:
        # Ranked above the other evidence on purpose: these findings were
        # produced by running something, not by reading. Where they disagree
        # with the reference answer they are usually right, and a criterion that
        # enshrines a reference slip will mark correct responses wrong.
        parts.append(
            _section("verified_findings", _dump(investigation, limit=7000))
            + "\n(These were established with tools — recomputation, executing draft criteria "
            "against real texts, and portability checks. Prefer them over the sections above "
            "where they conflict, including over the reference answer itself, and use "
            "'checks_that_do_not_discriminate' as a list of criteria NOT to write.)"
        )
    parts.append(
        _section("question_anchors", render_anchors(anchors))
        + "\n(Every criterion must contain at least one of these verbatim. Criteria that contain "
        "none of them are deleted automatically after this stage.)"
    )
    parts.append(
        _section("banned_words", _render_banned_words(banned_words))
        + "\n(These make a criterion a taste judgement. Drop the word and keep the checkable core.)"
    )
    parts.append(domain_guidance(domain))
    sizing = ""
    if target_items:
        sizing = (
            f" The final checklist is expected to keep about {target_items} items — a figure derived "
            "from this question's own structure, not a fixed house size — and at most "
            f"{max(1, int(target_items * max_pitfall_fraction))} of those may be Pitfall items. "
            "Cover the question's real content; do not invent filler to reach a count."
        )
    parts.append(
        f"Draft about {n_candidates} candidate criteria — deliberately more than the final rubric "
        "will keep, because each one is about to be tested empirically and the weak ones will be "
        f"pruned.{sizing} Number the ids 1..N contiguously.\n"
        "Return ONLY a JSON array with exactly this shape:\n" + _DRAFT_SCHEMA
    )
    return "\n\n".join(parts)


def build_draft_retry_user(
    question: str,
    reference_answer: str,
    *,
    domain: str = "",
    n_candidates: int = 16,
    anchors: Sequence[str] = (),
) -> str:
    """Compacted retry: same task, evidence sections dropped.

    Used only when the evidence-rich draft call fails to return usable JSON, on
    the assumption that prompt size or context confusion caused it.
    """
    return "\n\n".join(
        [
            _section("question", question),
            _section("reference_answer", reference_answer or "(empty)"),
            _section("question_anchors", render_anchors(anchors)),
            domain_guidance(domain),
            f"Draft {n_candidates} atomic candidate criteria for grading responses to this "
            "question. Each must contain at least one anchor token verbatim, and each must be "
            'phrased so that being literally true means the response is good — no "Does not '
            'mention ...", no "Fails to ...". Number the ids 1..N contiguously.\n'
            "Return ONLY a JSON array with exactly this shape:\n" + _DRAFT_SCHEMA,
        ]
    )


# ---------------------------------------------------------------------------
# Stage 5b - lint: repair polarity, grounding and subjectivity in one pass
# ---------------------------------------------------------------------------

LINT_SYSTEM = f"""You repair individual grading criteria. Each one you are given has failed at least one mechanical check. You rewrite it so that it passes, changing as little as possible.

You are editing, not authoring. Keep the criterion's subject matter, its specific values and its category. Do not make it broader, do not merge it with anything, do not invent facts that were not already in it.

The three checks, and what a repair looks like:

{POLARITY_RULE}

GROUNDING — the criterion contains no token from this question's anchor list. Rewrite it so it names the actual quantity, entity, value or sub-question it is really about, using anchor tokens verbatim. If the criterion has no question-specific content to recover — if it is a generic statement that would fit any question — do not invent one: return it with "drop": true and it will be deleted rather than padded back in.

SUBJECTIVITY — the criterion turns on a taste word. Delete the word and keep the checkable core; the core is what a grader can actually verify. If deleting every taste word leaves nothing checkable, the criterion was pure presentation: return "drop": true.

Preserve the criterion id you were given. Titles are 2-4 words and should describe what is being checked."""

_LINT_SCHEMA = """[
  {
    "id": 1,
    "title": "<2-4 words>",
    "description": "<the repaired criterion, one self-contained sentence, WITHOUT any category prefix>",
    "drop": false,
    "fixed": ["<polarity|grounding|subjectivity>"]
  }
]"""


def build_lint_user(
    question: str,
    offenders: Sequence[Mapping[str, Any]],
    *,
    anchors: Sequence[str] = (),
    banned_words: Sequence[str] = (),
) -> str:
    lines: list[str] = []
    for item in offenders:
        problems = ", ".join(str(p) for p in (item.get("problems") or [])) or "unspecified"
        lines.append(
            f"{item.get('id')}. [{item.get('title') or ''}] ({item.get('category')})\n"
            f"     text: {item.get('description')}\n"
            f"     failed: {problems}"
        )
    ids = ", ".join(str(item.get("id")) for item in offenders)
    return "\n\n".join(
        [
            _section("question", question),
            _section("question_anchors", render_anchors(anchors)),
            _section("banned_words", _render_banned_words(banned_words)),
            _section("criteria_to_repair", "\n".join(lines) if lines else "(none)"),
            f"Return a JSON array with exactly {len(offenders)} objects covering ids [{ids}], "
            "in that order, with this shape:\n" + _LINT_SCHEMA,
        ]
    )


# ---------------------------------------------------------------------------
# Stage 6a - constructed negative responses
# ---------------------------------------------------------------------------

NEGATIVE_SYSTEM = """You write realistic WRONG answers, for use as test cases that measure whether a grading checklist can actually detect a mistake.

You will be given a question, the reference answer, and one specific mistake. Write a response that a strong but flawed model would plausibly produce: it commits exactly that mistake and carries the consequences through to the final answer.

Requirements:
- Confident, competent tone. Normal structure and length for this kind of question: reasoning, then a clear final answer.
- Everything unrelated to the planted mistake should be correct and well-argued. The response must be genuinely hard to dismiss at a glance.
- The planted mistake must actually change the final result. Propagate it: if a sign flips early, every downstream value reflects the flipped sign.
- Never signal that anything is wrong. No hedging, no disclaimers, no "note that this may be incorrect", no meta-commentary, no mention of the reference answer or of this instruction.

Output the response text only. No JSON, no headings about the task, no explanation of what you did."""


def build_negative_user(
    question: str,
    reference_answer: str,
    *,
    pitfall: Mapping[str, Any] | None = None,
    spec: Mapping[str, Any] | None = None,
) -> str:
    parts = [
        _section("question", question),
        _section("reference_answer", reference_answer or "(empty)"),
    ]
    if spec:
        parts.append(_section("question_structure", _dump(spec, limit=2500)))
    if pitfall:
        parts.append(_section("mistake_to_commit", _dump(pitfall, limit=1500)))
        parts.append(
            "Write the flawed response now. It must commit precisely the mistake above and "
            "reach the wrong result that mistake implies."
        )
    else:
        parts.append(
            "Write a flawed response now. Choose ONE specific, plausible technical mistake that a "
            "strong model would realistically make on this question — a reversed sign, a dropped "
            "conversion, a misapplied formula, a skipped constraint, an unanswered sub-question — "
            "commit it, and carry it through to a wrong final answer."
        )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Stage 6b - the discrimination test
# ---------------------------------------------------------------------------

_CRITIC_SHARED_RULES = """Rules:
- Judge ONLY what the response actually contains. Do not credit what it was evidently trying to do, and do not import knowledge the response does not state.
- Every criterion is phrased so that being literally true of the response means the response is GOOD, including those under "Pitfall Criteria", which are written as avoidances ("Avoids ..."). met=true therefore always means the response satisfies the sentence as written: it contains the required content, or it genuinely avoids the named mistake.
- Judge each criterion independently. A single flaw may legitimately fail several criteria.
- Ignore length, formatting and eloquence unless a criterion explicitly asks about them."""

CRITIC_GOLD_SYSTEM = f"""You are testing a draft grading checklist against a trusted reference answer. The reference answer is, by construction, the correct response.

For each criterion, report two independent things:
- met: does the reference answer actually satisfy this criterion, as written?
- contradicted: does the reference answer state something INCOMPATIBLE with this criterion? Set this true only when the criterion asserts a fact, value or claim that the reference answer refutes — i.e. the criterion is simply wrong. A criterion the reference is merely silent about is NOT contradicted; it is met=false, contradicted=false.

That distinction decides whether a criterion gets deleted as erroneous or merely flagged as unstated, so do not conflate them.

{_CRITIC_SHARED_RULES}

Output ONLY a JSON array, one object per criterion, using the criterion ids given:
[{{"id": 1, "met": true, "contradicted": false, "why": "<=15 words"}}, ...]
No prose outside the JSON array."""

CRITIC_NEGATIVE_SYSTEM = f"""You are testing a draft grading checklist against a response that is known to be FLAWED. You are not grading the response; you are measuring which criteria detect the flaw.

Apply every criterion literally and strictly. If the response gets a value wrong, criteria demanding the correct value are met=false. Do not soften a verdict because the response is otherwise well written.

{_CRITIC_SHARED_RULES}

Output ONLY a JSON array, one object per criterion, using the criterion ids given:
[{{"id": 1, "met": false}}, ...]
No prose outside the JSON array."""


def build_critic_user(
    question: str,
    response: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    role: str = "gold",
    flaw_note: str | None = None,
) -> str:
    label = "reference_answer" if role == "gold" else "candidate_response"
    parts = [_section("question", question), _section(label, response or "(empty)")]
    if role != "gold" and flaw_note:
        parts.append(
            _section("known_flaw", flaw_note)
            + "\n(Stated for context only. Judge each criterion against the response text itself.)"
        )
    parts.append(_section("criteria", render_candidates(candidates)))
    ids = ", ".join(str(c.get("id")) for c in candidates)
    shape = (
        '{"id": <id>, "met": <true|false>, "contradicted": <true|false>, "why": "<=15 words"}'
        if role == "gold"
        else '{"id": <id>, "met": <true|false>}'
    )
    parts.append(
        f"Return a JSON array with exactly {len(candidates)} objects, one per criterion, "
        f"covering ids [{ids}], each of the form {shape}."
    )
    return "\n\n".join(parts)


# ---------------------------------------------------------------------------
# Stage 7 - calibration
# ---------------------------------------------------------------------------

CALIBRATE_SYSTEM = f"""You are finalising a grading checklist whose items have already been tested empirically against a correct response and against deliberately flawed responses. Each item arrives annotated with those results.

Read the annotations as evidence, not as decoration:
- "caught k/n negatives" is the measured discriminative power of the item. An item that caught every flawed response is proven signal. An item that caught none is decoration, however sensible it reads.
- "gold=FAIL" means the correct response did not visibly satisfy the item. Such items survived only because the checklist would otherwise be too small; keep them only if they are genuinely necessary to the answer's correctness, and never weight them highly.

Your job:
1. MERGE duplicates. Two items that a grader would always resolve identically are one item. Merge them into the sharper wording and list every source id you absorbed. This includes an avoidance item that only restates a positive item in the other direction — that scores one fact twice, so keep the positive one and absorb the avoidance into it.
2. DROP the remainder that is generic, unverifiable from the response alone, or that no flawed response failed while a near-duplicate item already covers the same ground.
3. ASSIGN the final category and weight.
4. PRESERVE wording specificity. Never generalise a criterion to make it tidier — the specific value, formula or entity is the whole point. You may sharpen wording, but you may not remove the concrete anchor.

{POLARITY_RULE}

Weighting principle: weight is proportional to (measured discriminative power) x (importance to the correctness of the answer). Weight is NOT proportional to how much text a response would need in order to satisfy the item. A one-number criterion that catches every flawed response outranks a paragraph-long criterion that catches none.
- Essential, weight 5: correctness collapses without it. Prefer items with high measured discrimination.
- Important, weight 3-4: key reasoning or a required intermediate result.
- Optional, weight 1-2: real added depth in the subject matter. At most a small minority of the checklist, and never about presentation.
- Pitfall, positive magnitude 1-5: this question's specific trap, written as the avoidance of it, naming the wrong value or wrong move. Use the high end for mistakes that were actually observed.

Two balance constraints, both of which override the raw discrimination numbers:
- Avoidance items are satisfied by silence. An empty response commits no mistakes and so clears every one of them; a checklist made mostly of them would award that empty response nearly full marks. Keep Pitfall items to the stated share and let criteria demanding real content carry the rest. Note that the flawed test responses were built from the mined pitfalls, so Pitfall items had a structural advantage in the measured discrimination — discount it accordingly.
- A checklist that is entirely Essential grades nothing, and one that is entirely Optional grades nothing either. Spread the weights.

Sizing: the item count you are given was derived from this question's own structure. Hitting it by deleting weak items is correct; hitting it by inventing generic items is not. A short, fully-grounded checklist beats a padded one — if the evidence only supports fewer items, return fewer.

Preserve coverage of every sub-question."""

_CALIBRATE_SCHEMA = """[
  {
    "id": 1,
    "source_ids": [3, 7],
    "title": "<2-4 words>",
    "category": "<Essential|Important|Optional|Pitfall>",
    "description": "<one self-contained sentence, WITHOUT any category prefix>",
    "weight": 5,
    "rationale": "<<=20 words: why this weight, referencing its measured discrimination>"
  }
]"""


def build_calibrate_user(
    question: str,
    reference_answer: str,
    candidates: Sequence[Mapping[str, Any]],
    *,
    min_items: int,
    max_items: int,
    domain: str = "",
    validation_note: str | None = None,
    anchors: Sequence[str] = (),
    banned_words: Sequence[str] = (),
    target_items: int | None = None,
    complexity_note: str | None = None,
    max_pitfall_fraction: float = 0.25,
) -> str:
    parts = [
        _section("question", question),
        _section("reference_answer", reference_answer or "(empty)"),
        _section(
            "validated_candidates",
            render_candidates(candidates, with_validation=True),
        ),
    ]
    if validation_note:
        parts.append(_section("validation_context", validation_note))
    parts.append(
        _section("question_anchors", render_anchors(anchors))
        + "\n(Every final item must contain at least one of these verbatim; items that contain "
        "none are deleted after this stage rather than replaced.)"
    )
    parts.append(_section("banned_words", _render_banned_words(banned_words)))
    parts.append(domain_guidance(domain))
    target = target_items or max_items
    sizing = (
        f"Produce about {target} final criteria — between {min_items} and {max_items}. "
        f"At most {max(1, int(target * max_pitfall_fraction))} of them may be Pitfall items."
    )
    if complexity_note:
        sizing += f" That figure comes from this question's structure: {complexity_note}"
    parts.append(
        sizing + " Every final item must list the source_ids it came from, so its measured "
        "validation results can be carried forward. Number the final ids 1..N contiguously.\n"
        "Return ONLY a JSON array with exactly this shape:\n" + _CALIBRATE_SCHEMA
    )
    return "\n\n".join(parts)
