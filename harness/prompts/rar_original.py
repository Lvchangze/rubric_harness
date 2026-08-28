"""Verbatim RaR rubric-generation prompts, transcribed from the paper appendix.

Source: "Rubrics as Rewards", Appendix A.9 (``rubric_as_reward.pdf`` pages
18-19, extracted text at ``analysis/out/paper_text.txt`` lines 1018-1101);
cross-checked character-for-character against the independent transcription in
``docs/00_paper_notes.md`` §2.1 / §2.2.

The two prompts are genuinely different and must be dispatched by domain (see
``docs/00_paper_notes.md`` §2.3). The medical one additionally supplies numeric
weight anchors (Essential=5, Important=3-4, Optional=1-2) and mandates that
Pitfall descriptions open with "Does not mention" or "Recommends"; the science
one supplies neither. That single difference explains most of the divergence the
forensics measured between the two shipped corpora, so using one prompt for both
domains would invalidate the baseline column entirely.

The only deviation from the appendix is typographic: the PDF's curly quotes,
en-dash minus signs, en-dash ranges and em-dashes are rendered as ASCII, and its
bullet glyphs as "-". These are LaTeX rendering artefacts rather than prompt
content, and ASCII avoids a tokeniser reading "-1" as anything other than
negative one. Wording, ordering, spacing and examples are otherwise unchanged --
including the appendix's own inconsistent ellipses, which are reproduced as
written.

``build_rar_prompt`` assembles the system prompt plus the user turn used by the
single-pass baseline generator.
"""

from __future__ import annotations

__all__ = [
    "MEDICAL_RUBRIC_SYSTEM",
    "SCIENCE_RUBRIC_SYSTEM",
    "system_for_domain",
    "build_rar_user_prompt",
    "IMPLICIT_JUDGE_SYSTEM",
    "build_implicit_judge_user",
    "PREDEFINED_RUBRIC",
]


MEDICAL_RUBRIC_SYSTEM = """You are an expert rubric writer. Your job is to generate a self-contained set of evaluation criteria ("rubrics") for judging how good a response is to a given question. Rubrics can cover aspects of a response such as, but not limited to, factual correctness, ideal-response characteristics, style, completeness, helpfulness, harmlessness, patient-centeredness, depth of reasoning, contextual relevance, and empathy. Each item must be self-contained - non expert readers should not need to infer anything or consult external information. Begin each description with its category: "Essential Criteria: . . . ", "Important Criteria: . . . ", "Optional Criteria: ...", or "Pitfall Criteria: Does not mention . . . ".

Inputs:
- question: The full question text.
- reference_answer: The ideal answer, including any specific facts, explanations, or advice.

Total items:
- Choose 7-20 rubric items based on the complexity of the question.

Each rubric item:
- title (2-4 words).
- description: One sentence starting with its category prefix that explicitly states exactly what to look for. For example:
  - Essential Criteria: Identifies non-contrast helical CT scan as the most sensitive modality for ureteric stones.
  - Pitfall Criteria: Does not mention identifying (B) as the correct answer.
  - Important Criteria: Explains that non-contrast helical CT detects stones of varying sizes and compositions.
  - Optional Criteria: States "The final answer is (B)" or similar answer choice formatting.
- weight: For Essential/Important/Optional, use 1-5 (5 = most important); for Pitfall, use -1 or -2.

Category guidance:
- Essential: Critical facts or safety checks; if missing, the response is invalid (weight 5).
- Important: Key reasoning, completeness, or clarity; strongly affects quality (weight 3-4).
- Optional: Helpful style or extra depth; nice to have but not deal-breaking (weight 1-2).
- Pitfall: Common mistakes or omissions specific to this prompt-identify things a respondent often forgets or misstates. Each Pitfall description must begin with "Pitfall Criteria: Does not mention . . . " or "Pitfall Criteria: Recommends . . . " and use weight -1 or -2.

To ensure self-contained guidance:
- When referring to answer choices, explicitly say "Identifies (A)", "Identifies (B)", etc., rather than vague phrasing.
- If the format requires a conclusion like "The final answer is (B)", include a rubric item such as:
  - Essential Criteria: Includes a clear statement "The final answer is (B)".
- If reasoning should precede the answer, include a rubric like:
  - Important Criteria: Presents the explanation before stating the final answer.
- If brevity is valued, include a rubric like:
  - Optional Criteria: Remains concise and avoids unnecessary detail.
- If the question context demands mention of specific findings, include that explicitly (e.g., "Essential Criteria: Mentions that CT does not require contrast").

Output: Provide a JSON array of rubric objects. Each object must contain exactly three keys-title, description, and weight. Do not copy large blocks of the question or reference_answer into the text. Each description must begin with its category prefix, and no extra keys are allowed.

Now, given the question and reference_answer, generate the rubric as described. The reference answer is an ideal response but not necessarily exhaustive; use it only as guidance."""


SCIENCE_RUBRIC_SYSTEM = """You are an expert rubric writer for science questions in the domains of Biology, Physics, and Chemistry. Your job is to generate a self-contained set of evaluation criteria ("rubrics") for judging how good a response is to a given question in one of these domains. Rubrics can cover aspects such as factual correctness, depth of reasoning, clarity, completeness, style, helpfulness, and common pitfalls. Each rubric item must be fully self-contained so that non-expert readers need not consult any external information.

Inputs:
- question: The full question text.
- reference_answer: The ideal answer, including any key facts or explanations.

Total items:
- Choose 7-20 rubric items based on question complexity.

Each rubric item must include exactly three keys:
1. title (2-4 words)
2. description: One sentence beginning with its category prefix, explicitly stating what to look for. For example:
   - Essential Criteria: States that in the described closed system, the total mechanical energy (kinetic plus potential) before the event equals the total mechanical energy after the event.
   - Important Criteria: Breaks down numerical energy values for each stage, demonstrating that initial kinetic energy plus initial potential energy equals final kinetic energy plus final potential energy.
   - Optional Criteria: Provides a concrete example, such as a pendulum converting between kinetic and potential energy, to illustrate how energy shifts within the system.
   - Pitfall Criteria: Does not mention that frictional or air-resistance losses are assumed negligible when applying conservation of mechanical energy.
3. weight: For Essential/Important/Optional, use 1-5 (5 = most important); for Pitfall, use -1 or -2.

Category guidance:
- Essential: Critical facts or safety checks; omission invalidates the response.
- Important: Key reasoning or completeness; strongly affects quality.
- Optional: Nice-to-have style or extra depth.
- Pitfall: Common mistakes or omissions; highlight things often missed.

Format notes:
- When referring to answer choices, explicitly say "Identifies (A)", "Identifies (B)", etc.
- If a clear conclusion is required (e.g. "The final answer is (B)"), include an Essential Criteria for it.
- If reasoning should precede the final answer, include an Important Criteria to that effect.
- If brevity is valued, include an Optional Criteria about conciseness.

Output: Provide a JSON array of rubric objects. Each object must contain exactly three keys-title, description, and weight. Do not copy large blocks of the question or reference_answer into the text. Each description must begin with its category prefix, and no extra keys are allowed.

Now, given the question and reference_answer, generate the rubric as described. The reference answer is an ideal response but not necessarily exhaustive; use it only as guidance."""


def system_for_domain(domain: str) -> str:
    """Pick the paper's domain-matched rubric-writer system prompt."""
    return MEDICAL_RUBRIC_SYSTEM if "medicine" in domain.lower() else SCIENCE_RUBRIC_SYSTEM


def build_rar_user_prompt(question: str, reference_answer: str) -> str:
    """User turn carrying the two inputs the paper's prompt declares."""
    return (
        f"question:\n{question}\n\n"
        f"reference_answer:\n{reference_answer}\n\n"
        "Output the JSON array now."
    )


# --------------------------------------------------------------------------
# Judge prompts (Appendix A.6). The paper gives no per-criterion prompt for
# RaR-EXPLICIT; only the IMPLICIT variant below is reproducible verbatim.
# --------------------------------------------------------------------------

IMPLICIT_JUDGE_SYSTEM = """You are an expert evaluator. Given a user prompt, a generated response, and a list of quality rubrics, please rate the overall quality of the response on a scale of 1 to 10 based on how well it satisfies the rubrics.
Consider all rubrics holistically when determining your score. A response that violates multiple rubrics should receive a lower score, while a response that satisfies all rubrics should receive a higher score.
Start your response with a valid JSON object that starts with "```json" and ends with "```". The JSON object should contain a single key "rating" and the value should be an integer between 1 and 10.
Example response:
```json
{
"rating": 7
}```"""


def build_implicit_judge_user(prompt: str, response: str, rubric_list_string: str) -> str:
    """RaR-IMPLICIT user template, Appendix A.6, verbatim."""
    return (
        "Given the following prompt, response, and rubrics, please rate the overall quality "
        "of the response on a scale of 1 to 10 based on how well it satisfies the rubrics.\n"
        f"<prompt>\n{prompt}\n</prompt>\n"
        f"<response>\n{response}\n</response>\n"
        f"<rubrics>\n{rubric_list_string}\n</rubrics>\n"
        "Your JSON Evaluation:"
    )


#: RaR-PREDEFINED's four fixed generic criteria (Appendix A.5, verbatim).
#: The paper's own control showing what a fully generic rubric does: it drops
#: HealthBench from 29.7 to 12.5. Useful as a floor anchor in our metrics.
PREDEFINED_RUBRIC: tuple[str, ...] = (
    "The response contains correct information without factual errors, inaccuracies, "
    "or hallucinations that could mislead the user.",
    "The response fully answers all essential parts of the question and provides "
    "sufficient detail where needed.",
    "The response is concise and to the point, avoiding unnecessary verbosity or repetition.",
    "The response effectively meets the user's practical needs, provides actionable "
    "information, and is genuinely helpful for their situation.",
)
