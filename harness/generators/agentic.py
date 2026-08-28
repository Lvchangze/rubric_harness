"""Agentic rubric generation: criteria are searched for, then empirically validated.

The single-pass baseline asks a model to *write* a rubric. This pipeline instead
treats rubric construction as search plus measurement:

1. ``decompose``  — parse the question into an explicit specification.
2. ``rollouts``   — solve the question k times *without* the reference answer.
3. ``reconcile``  — locate where the rollouts disagree, or agree and are wrong.
   Disagreement is the signal: content every attempt already produces correctly
   cannot discriminate between responses, so a criterion about it is dead weight.
4. ``pitfalls``   — mine question-specific "looks right but is wrong" failures.
5. ``scope``      — derive this question's anchor vocabulary and, from the
   decomposition, how many criteria it actually warrants (forensics F1/F3). No
   LLM call; both outputs constrain the stages below.
6. ``draft``      — synthesise candidate criteria from that evidence.
7. ``lint``       — verify-and-revise: polarity, grounding and subjectivity are
   checked mechanically and the offenders repaired in one batched call, with
   deterministic rewrites as the fallback (forensics F3/F4/F6). The forensics is
   explicit that a longer drafting prompt alone does not fix these, which is why
   they are re-checked here rather than merely requested above.
8. ``critic``     — run every candidate against the reference answer and against
   deliberately wrong responses. Criteria the reference fails are unverifiable;
   criteria no wrong answer fails are decoration. This is the stage the
   ``agentic-noval`` ablation removes.
9. ``calibrate``  — merge, categorise and weight by *measured* discriminative
   power.
10. ``dedup``     — whole-rubric pass: merge polarity-mirrored pairs across
   categories, re-enforce the lints, and clamp the item count to the derived
   target (forensics F7).

Every stage is individually switchable through :class:`~harness.config.AgenticConfig`
and every stage degrades to a no-op on failure, keeping whatever the previous
stage produced. A sample never aborts; at worst it returns a smaller rubric with
the failure recorded in the trace.

Polarity contract: every criterion this module emits carries
``Polarity.POSITIVE`` explicitly, with ``polarity_method="enforced"``. Nothing is
left to inference. The two shipped corpora disagree about what a negatively
weighted criterion means (``docs/01_data_forensics.md`` F6), so producing a
single, declared direction is the point rather than a detail.
"""

from __future__ import annotations

import asyncio
import logging
import re
import sys
import time
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass, field, fields, replace
from functools import lru_cache
from pathlib import Path
from typing import Any, AsyncIterator, Callable, Iterator, Mapping, Sequence

from ..config import AgenticConfig
from ..llm import JSONParseError, LLMEngine, extract_json
from ..prompts import agentic as P
from ..schema import (
    CATEGORY_PREFIXES,
    Category,
    Criterion,
    Example,
    Polarity,
    Rubric,
    detect_polarity,
    parse_category_prefix,
    strip_category_prefix,
)
from .base import GenerationResult, RubricGenerator, register

logger = logging.getLogger(__name__)

__all__ = ["AgenticGenerator", "STAGE_ORDER"]

STAGE_ORDER: tuple[str, ...] = (
    "decompose",
    "rollouts",
    "reconcile",
    "pitfalls",
    "scope",
    "draft",
    "lint",
    "critic",
    "calibrate",
    "dedup",
)

#: Per-stage output budgets. Reasoning models spend most of the budget before
#: the visible answer starts, so these are deliberately generous.
_MAX_TOKENS: dict[str, int] = {
    "decompose": 8192,
    "reconcile": 14336,
    "pitfalls": 10240,
    "draft": 16384,
    "lint": 12288,
    "negative": 8192,
    "critic": 14336,
    "calibrate": 16384,
}

#: High enough that k rollouts explore genuinely different solution paths;
#: without spread there is no disagreement signal for stage 3 to read.
ROLLOUT_TEMPERATURE = 0.9

#: Settings for mining real failures when the careful Stage 2 rollouts all
#: succeed. Each suppresses one of the behaviours that makes a careful attempt
#: correct — showing work, or checking it — so the resulting mistakes are ones
#: the model genuinely makes rather than ones a prompt asked it to plant.
_HARD_NEGATIVE_STYLES: list[dict[str, Any]] = [
    {
        "name": "rushed",
        "temperature": 1.0,
        "system": (
            "Answer the question under severe time pressure. Give your "
            "first-instinct answer. Do not double-check it, do not verify your "
            "arithmetic, and do not reconsider your approach."
        ),
    },
    {
        "name": "terse",
        "temperature": 1.0,
        "system": (
            "Answer the question directly. Do NOT show your reasoning, "
            "derivation, or working — state only the conclusion."
        ),
    },
    {
        "name": "shortcut",
        "temperature": 1.0,
        "system": (
            "Answer the question using the fastest shortcut or approximation you "
            "can find. Prefer speed over rigour; do not check edge cases, "
            "boundary conditions, or unit consistency."
        ),
    },
]

_NEGMINE_CHECK_SYSTEM = """You are checking whether a candidate answer reached
the same conclusion as a reference answer.

Judge ONLY whether the candidate's final conclusion agrees with the reference's.
Ignore style, length, and whether the candidate showed its work. Accept
equivalent notation, units, and ordinary rounding. If the question has several
parts, the candidate is correct only if it gets every part the reference states.

Return ONLY this JSON object:

{"correct": true or false, "why": "<one sentence>"}"""

#: Rollout text embedded into the reconcile prompt (full text is kept in the
#: trace and reused verbatim when a rollout serves as a negative).
RECONCILE_EXCERPT_CHARS = 6000

#: Response text embedded into a critic judge call.
JUDGE_EXCERPT_CHARS = 14000

#: Candidates asked for at stage 5, over and above the final target, so the
#: critic has slack to prune.
DRAFT_OVERSHOOT = 6

_EVIDENCE_RANK: dict[str, int] = {
    "collective_error": 4,
    "divergent": 3,
    "pitfall": 2,
    "reference_only": 1,
    "sub_question": 1,
    "other": 0,
}

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


# ---------------------------------------------------------------------------
# Forensics interop
#
# The detectors that define "generic" and "subjective" live in ``analysis/``,
# where they were calibrated against all 45k shipped rubrics. Importing them
# rather than restating them is what keeps the generation-time lint and the
# evaluation-time metric talking about the same thing.
# ---------------------------------------------------------------------------

_REPO_ROOT = Path(__file__).resolve().parents[2]

_FALLBACK_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-'\.]*")
_FALLBACK_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
_FALLBACK_STOPWORDS = frozenset(
    """a an the and or but if then than that this these those of to in on at by for with from as
    is are was were be been it its they them their there here which who what when where why how
    all any both each few more most other some such no nor not only own same so too very can will
    just should now must may might could would does do did have has had i you he she we us our
    your his her also into about over under between within without response answer criteria
    criterion""".split()
)
#: Minimal stand-in used only when ``analysis/`` cannot be imported; the real
#: list is 52 terms long and lives in ``analysis/rubric_stats.py``.
_FALLBACK_SUBJECTIVE = ("clearly", "clear ", "concise", "thorough", "appropriate", "properly")
_FALLBACK_STYLE = ("format", "structure", "organiz", "concise", "brevity", "verbos", "tone")


@dataclass(frozen=True)
class _Forensics:
    """The forensics detectors, or degraded local equivalents."""

    source: str
    content_tokens: Callable[[str], list[str]]
    instance_anchors: Callable[[str, str, set[str]], set[str]]
    number_re: re.Pattern[str]
    subjective_terms: tuple[str, ...]
    style_terms: tuple[str, ...]
    polarity_stripped: Callable[[str], str]


def _fallback_content_tokens(text: str) -> list[str]:
    return [
        t
        for t in _FALLBACK_TOKEN_RE.findall((text or "").lower())
        if t not in _FALLBACK_STOPWORDS and len(t) > 2
    ]


def _fallback_instance_anchors(question: str, reference: str, rare: set[str]) -> set[str]:
    anchors: set[str] = set()
    for text in (question, reference):
        lowered = (text or "").lower()
        anchors.update(t for t in _fallback_content_tokens(lowered) if t in rare)
        anchors.update(
            n for n in _FALLBACK_NUMBER_RE.findall(lowered) if len(n) >= 2 or float(n) > 2
        )
    return anchors


_LEADING_OPENER_RE = re.compile(
    r"^\s*(?:the\s+(?:response|answer|explanation)\s+)?"
    r"(?:must|should|shall|will|does|do|is|are)?\s*"
    r"(?:not\s+)?(?:avoid(?:s|ing)?|state[sd]?|identif(?:y|ies|ied)|mention[sd]?|includ(?:e|es|ed)"
    r"|note[sd]?|provide[sd]?|explain[sd]?|report[sd]?|specif(?:y|ies|ied)|recommend[sd]?)\s*"
    r"(?:that\s+)?",
    re.IGNORECASE,
)


def _fallback_polarity_stripped(body: str) -> str:
    return _LEADING_OPENER_RE.sub("", body or "").strip()


_FALLBACK_FORENSICS = _Forensics(
    source="fallback",
    content_tokens=_fallback_content_tokens,
    instance_anchors=_fallback_instance_anchors,
    number_re=_FALLBACK_NUMBER_RE,
    subjective_terms=_FALLBACK_SUBJECTIVE,
    style_terms=_FALLBACK_STYLE,
    polarity_stripped=_fallback_polarity_stripped,
)


@lru_cache(maxsize=1)
def _forensics() -> _Forensics:
    """Import the ``analysis/`` detectors, degrading to local copies on failure.

    ``analysis`` is a plain script directory with intra-package imports, so both
    the repo root and ``analysis/`` itself have to be importable.
    """
    try:
        for path in (str(_REPO_ROOT), str(_REPO_ROOT / "analysis")):
            if path not in sys.path:
                sys.path.insert(0, path)
        from analysis.common import NUMBER_RE, content_tokens
        from analysis.rubric_grounding import instance_anchors
        from analysis.rubric_similarity import polarity_stripped
        from analysis.rubric_stats import STYLE_TERMS, SUBJECTIVE_TERMS
    except Exception as exc:  # noqa: BLE001 - generation must not depend on analysis/
        logger.warning("analysis/ detectors unavailable (%s); using local fallbacks", exc)
        return _FALLBACK_FORENSICS
    return _Forensics(
        source="analysis",
        content_tokens=content_tokens,
        instance_anchors=instance_anchors,
        number_re=NUMBER_RE,
        subjective_terms=tuple(SUBJECTIVE_TERMS),
        style_terms=tuple(STYLE_TERMS),
        polarity_stripped=polarity_stripped,
    )


# ---------------------------------------------------------------------------
# F3 — anchoring
# ---------------------------------------------------------------------------

#: Excluded from the anchor proxy on top of the stoplist ``content_tokens``
#: already applies. These are the task-framing words a question contributes that
#: cannot pin a criterion to it: every physics question asks you to "calculate"
#: a "value", so matching on them would make the grounding lint vacuous.
_ANCHOR_STOPWORDS = frozenset(
    """calculate calculates calculated compute computes computed determine determines determined
    find finds found evaluate evaluates evaluated estimate estimates estimated derive derives
    derived show shows shown solve solves solved discuss discusses discussed describe describes
    described consider considers considered assume assumes assumed given gives given following
    follows value values result results results. answer answers question questions problem
    problems solution solutions step steps final total number numbers amount much many first
    second third next part parts case cases using use used uses need needs needed based want
    wants would could should patient year years old male female
    """.split()
)


def question_anchors(question: str, reference: str) -> frozenset[str]:
    """Generation-time proxy for the forensics' instance anchor set.

    ``analysis.rubric_grounding.instance_anchors`` keeps only content tokens that
    are *rare across the corpus* (document frequency <= 1%), which needs a
    vocabulary built from all 45k questions — far too expensive to compute inside
    a generator that runs per sample. The proxy passed here treats every content
    token of this question and its reference as a candidate anchor instead, minus
    a task-framing stoplist. That is a strict superset of the real anchor set, so
    the lint built on it is *conservative*: it flags only criteria that share no
    vocabulary at all with their own instance, and never claims a criterion is
    grounded that the corpus-level test would reject.

    The honest corpus-level measurement stays on the evaluation side
    (``harness/eval/grounding.py``), which is where any reported
    ``generic_criterion_rate`` must come from. Do not quote this proxy as that
    metric.
    """
    engine = _forensics()
    vocab = {
        token
        for text in (question or "", reference or "")
        for token in engine.content_tokens(text)
    }
    vocab -= _ANCHOR_STOPWORDS
    anchors = engine.instance_anchors(question or "", reference or "", vocab)
    return frozenset(a for a in anchors if a not in _ANCHOR_STOPWORDS and len(a) > 1)


def _anchor_hits(text: str, anchors: frozenset[str]) -> set[str]:
    """Anchors a criterion actually mentions, matched as the forensics matches."""
    if not anchors:
        return set()
    engine = _forensics()
    lowered = (text or "").lower()
    tokens = set(engine.content_tokens(lowered)) | set(engine.number_re.findall(lowered))
    return tokens & set(anchors)


# ---------------------------------------------------------------------------
# F6 — polarity enforcement
# ---------------------------------------------------------------------------

_SUBJECT = r"(?:the\s+(?:response|answer|explanation)\s+)?"

#: The unambiguous "true means the response is bad" openers. Kept narrower than
#: ``schema._FAILURE_PHRASING_RE`` on purpose: "Recommends X" is a failure
#: phrasing *inside a Pitfall* (X is the bad thing) but a perfectly positive
#: requirement under any other category, so it is not checked outside one.
_OMISSION_OPENER_RE = re.compile(
    rf"^\s*{_SUBJECT}(?:does\s+not\s+(?:mention|state|note|say|specify|include|list|provide"
    r"|discuss|explain|address|identify|describe|recommend|suggest)"
    r"|fails?\s+to|omits?|neglects?\s+to|overlooks?|ignores?|misses)\b",
    re.IGNORECASE,
)

#: Applied in order; first match wins. Between them these cover every opener
#: ``schema._FAILURE_PHRASING_RE`` recognises, so the catch-all below is a
#: backstop rather than a routine path.
_POSITIVISERS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(rf"^\s*{_SUBJECT}does\s+not\s+(?:mention|state|note|say|specify)\s+(?:that\s+)?", re.I), "States "),
    (re.compile(rf"^\s*{_SUBJECT}does\s+not\s+(?:include|list|provide)\s+", re.I), "Includes "),
    (re.compile(rf"^\s*{_SUBJECT}does\s+not\s+(?:identify|recognise|recognize)\s+", re.I), "Identifies "),
    (re.compile(rf"^\s*{_SUBJECT}does\s+not\s+(?:discuss|explain|describe|address)\s+", re.I), "Explains "),
    (re.compile(rf"^\s*{_SUBJECT}does\s+not\s+(?:recommend|suggest)\s+", re.I), "Recommends "),
    (re.compile(rf"^\s*{_SUBJECT}fails?\s+to\s+", re.I), "Does "),
    (re.compile(rf"^\s*{_SUBJECT}neglects?\s+to\s+", re.I), "Does "),
    (re.compile(rf"^\s*{_SUBJECT}(?:omits?|overlooks?|misses)\s+", re.I), "States "),
    (re.compile(rf"^\s*{_SUBJECT}ignores?\s+", re.I), "Accounts for "),
)

#: Only meaningful for Pitfall criteria, where the sentence body names the
#: mistake: "Recommends <the unsafe drug>" is true exactly when the response is
#: wrong, so the repair is to require its avoidance.
_PITFALL_POSITIVISERS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(rf"^\s*{_SUBJECT}(?:incorrectly|wrongly|falsely|erroneously|mistakenly)\s+", re.I), "Avoids incorrectly "),
    (re.compile(rf"^\s*{_SUBJECT}(?:recommends?|suggests?)\s+", re.I), "Avoids recommending "),
    (re.compile(rf"^\s*{_SUBJECT}(?:claims?|states?|asserts?|reports?)\s+", re.I), "Avoids claiming "),
    (re.compile(rf"^\s*{_SUBJECT}(?:misidentifies|confuses|mistakes)\s+", re.I), "Avoids misidentifying "),
)


def polarity_offence(description: str, category: Category) -> str | None:
    """Why this criterion does not read "literally true => the response is good".

    Returns the detection method that condemned it, or ``None`` if it is clean.
    """
    text = strip_category_prefix(description or "")
    if not text:
        return None
    polarity, method = detect_polarity(text, category)
    if polarity is Polarity.NEGATIVE:
        return method
    # ``detect_polarity`` short-circuits non-Pitfall categories to POSITIVE on
    # the assumption that requirements are positive by construction. A drafted
    # "Does not mention X" mislabelled Essential would slip through that, and it
    # is exactly the shape that scores RaR-Medicine backwards.
    if category is not Category.PITFALL and _OMISSION_OPENER_RE.match(text):
        return "failure"
    return None


def positivise(description: str, category: Category) -> str:
    """Deterministic rewrite of a failure-phrased criterion into positive form.

    The fallback for the batched LLM repair, and the last line of defence in the
    final pass where no LLM call is available.
    """
    text = strip_category_prefix(description or "").strip()
    if not text:
        return text
    tables = _POSITIVISERS + (_PITFALL_POSITIVISERS if category is Category.PITFALL else ())
    for pattern, replacement in tables:
        match = pattern.match(text)
        if match:
            rewritten = replacement + text[match.end():].lstrip()
            return _tidy(rewritten)
    return _tidy(f"Avoids {text[0].lower()}{text[1:]}")


# ---------------------------------------------------------------------------
# F4 — subjective / style filler
# ---------------------------------------------------------------------------

#: Single words that are pure quality judgements wherever they appear, so they
#: can be deleted without an LLM. Multi-word entries and words with a legitimate
#: technical sense ("clear" solution, "smooth" function, "robust" estimator) are
#: deliberately left to the batched repair call, which can read the context.
_AMBIGUOUS_SUBJECTIVE = frozenset(
    {"clear", "smooth", "robust", "accessible", "effective", "elegant", "engaging"}
)

#: Trailing clauses that exist only to praise the response.
_STYLE_CLAUSE_RE = re.compile(
    r"\s*(?:,|;)?\s*\b(?:while|whilst|ensuring|ensure|making|so\s+that|thereby|thus|without|avoiding"
    r"|which\s+(?:makes|ensures|helps))\b[^.;]*",
    re.IGNORECASE,
)


@lru_cache(maxsize=1)
def _subjective_words() -> tuple[str, ...]:
    terms = {t.strip().lower() for t in _forensics().subjective_terms}
    return tuple(sorted(t for t in terms if t and " " not in t and t not in _AMBIGUOUS_SUBJECTIVE))


@lru_cache(maxsize=1)
def _subjective_removal_re() -> re.Pattern[str]:
    alternation = "|".join(re.escape(w) for w in _subjective_words())
    return re.compile(rf"\b(?:{alternation})\b", re.IGNORECASE)


@lru_cache(maxsize=1)
def _detector_terms() -> tuple[tuple[str, ...], tuple[str, ...]]:
    engine = _forensics()
    return (
        tuple(t.lower() for t in engine.subjective_terms),
        tuple(t.lower() for t in engine.style_terms),
    )


def subjective_hits(text: str) -> list[str]:
    """Subjective/style terms present, matched exactly as ``rubric_stats`` does."""
    lowered = (text or "").lower()
    subjective, style = _detector_terms()
    return sorted({term.strip() for term in subjective + style if term in lowered})


def is_style_criterion(text: str) -> bool:
    """Whether the criterion is *about* presentation rather than substance."""
    lowered = (text or "").lower()
    _, style = _detector_terms()
    return any(term in lowered for term in style)


#: Applied in order after a word-level deletion, to repair the grammar the
#: deletion breaks: "a comprehensive and well-organised derivation" loses both
#: adjectives and would otherwise read "a and derivation".
_TIDY_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"([,;])\s*(?=[,;])"), ""),
    (re.compile(r"\b(a|an|the)\s+(?:and|or)\s+", re.I), r"\1 "),
    (re.compile(r"\b(?:is|are|was|were|remains?|stays?|reads?|appears?)\s+and\s+", re.I), ""),
    (re.compile(r"\s+(?:and|or)\s+(?:and|or)\s+", re.I), " and "),
    (re.compile(r"^\s*(?:and|or)\s+", re.I), ""),
    (re.compile(r"\s+(?:and|or)\s*(?=[,.;:]|$)", re.I), ""),
    (re.compile(r"\b(a|an|the)\s+(?=[,.;:]|$)", re.I), ""),
    (re.compile(r"\s+([,.;:])"), r"\1"),
    (re.compile(r"\s{2,}"), " "),
)


def _tidy(text: str) -> str:
    out = text
    for pattern, replacement in _TIDY_RULES:
        out = pattern.sub(replacement, out)
    out = out.strip().strip(",;").strip()
    if out:
        out = out[0].upper() + out[1:]
    return out


def destyle(description: str) -> str:
    """Strip quality adverbs and praise clauses, keeping the checkable core.

    "clearly states X" becomes "states X" — the forensics' recommended repair
    (F4(d)): the subjective modifier is what makes an otherwise binary criterion
    a taste call, and deleting it loses nothing a grader could have checked.
    """
    text = strip_category_prefix(description or "")
    if not text:
        return text
    stripped = _STYLE_CLAUSE_RE.sub(
        lambda m: "" if subjective_hits(m.group(0)) else m.group(0), text
    )
    stripped = _subjective_removal_re().sub("", stripped)
    tidied = _tidy(stripped)
    # A rewrite that deletes the sentence is not a rewrite; leave the original
    # for the drop test to handle.
    return tidied if len(tidied.split()) >= 4 else text


# ---------------------------------------------------------------------------
# Defensive parsing helpers
# ---------------------------------------------------------------------------


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        for candidate in value.values():
            if isinstance(candidate, list):
                return candidate
    return []


def _as_dict(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _as_text(value: Any, limit: int = 4000) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else str(value)
    return text.strip()[:limit]


def _as_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    try:
        return int(round(float(str(value).strip())))
    except (TypeError, ValueError):
        return None


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "y", "1", "met", "satisfied", "pass", "passed"}:
            return True
        if text in {"false", "no", "n", "0", "unmet", "unsatisfied", "fail", "failed"}:
            return False
    return default


def _clip_middle(text: str, limit: int) -> str:
    """Keep head and tail: a solution's setup and its final answer both matter."""
    if len(text) <= limit:
        return text
    head = int(limit * 0.6)
    tail = limit - head
    elided = len(text) - limit
    return f"{text[:head]}\n\n... [{elided} chars elided] ...\n\n{text[-tail:]}"


def _parse_trailing_json(text: str) -> dict[str, Any] | None:
    """Recover the summary block a rollout is asked to end with.

    Worked solutions are full of LaTeX braces, so a forward scan for the first
    ``{`` finds the wrong thing; fenced blocks are tried last-first instead.
    """
    if not text:
        return None
    blocks = _FENCE_RE.findall(text)
    for block in reversed(blocks):
        try:
            value = extract_json(block, expect="object")
        except JSONParseError:
            continue
        if isinstance(value, dict):
            return value
    tail = text[-4000:]
    for source in (tail, text):
        try:
            value = extract_json(source, expect="object")
        except JSONParseError:
            continue
        if isinstance(value, dict) and value:
            return value
    return None


def _derive_title(description: str) -> str:
    words = [w for w in description.replace("\n", " ").split(" ") if w][:4]
    return " ".join(words).strip(".,;:")[:60] or "Criterion"


def _step_weight(weight: int, category: Category, *, up: bool) -> int:
    """Move one step along the importance scale, respecting category ranges."""
    if category is Category.PITFALL:
        stepped = weight - 1 if up else weight + 1
        return max(-2, min(-1, stepped))
    stepped = weight + 1 if up else weight - 1
    return max(1, min(5, stepped))


# ---------------------------------------------------------------------------
# Per-sample state
# ---------------------------------------------------------------------------


@dataclass
class _Rollout:
    index: int              # 1-based, matching the labels used in prompts
    text: str = ""
    summary: dict[str, Any] | None = None
    error: str | None = None

    @property
    def ok(self) -> bool:
        return bool(self.text)

    @property
    def label(self) -> str:
        return f"Rollout {self.index}"

    def to_prompt_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "label": self.label,
            "summary": self.summary,
            "text": self.text,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "summary": self.summary,
            "text": self.text,
            "error": self.error,
        }


@dataclass
class _Negative:
    nid: str                # stable handle, e.g. "rollout2" / "constructed1"
    kind: str               # "rollout" | "constructed"
    text: str
    note: str = ""          # what is known to be wrong with it

    def to_dict(self) -> dict[str, Any]:
        return {"id": self.nid, "kind": self.kind, "note": self.note, "text": self.text}


@dataclass
class _Candidate:
    cid: int
    title: str
    description: str
    category: Category
    weight: int
    provenance: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] | None = None

    @property
    def evidence(self) -> str:
        return str(self.provenance.get("evidence") or "other")

    def prefixed_description(self) -> str:
        return f"{CATEGORY_PREFIXES[self.category]} {self.description}".strip()

    def to_prompt_dict(self, *, with_validation: bool = False) -> dict[str, Any]:
        out: dict[str, Any] = {
            "id": self.cid,
            "title": self.title,
            "prefixed_description": self.prefixed_description(),
            "evidence": self.evidence,
        }
        if with_validation:
            out["weight"] = self.weight
            out["validation"] = self.validation
        return out

    def to_criterion(self) -> Criterion:
        """Emit the schema object, declaring polarity rather than leaving it inferred.

        ``Criterion.__post_init__`` would otherwise call ``detect_polarity`` and
        record a *guess*. Every criterion leaving this pipeline has been linted
        into "literally true => the response is good", so the direction is a fact
        we know, and ``polarity_method="enforced"`` is what tells the judge's
        ``FAVOURABLE`` mode that it does not need to give us the benefit of the
        doubt (see ``harness/eval/judge.py::effective_polarity``).

        Pitfall weights are stored on the RaR negative scale because
        ``Criterion.__post_init__`` coerces them there and ``harness/schema.py``
        is not ours to change. That is safe rather than a contradiction: every
        consumer grades by magnitude — ``judge.aggregate`` uses ``abs(weight)``
        or the categorical remap, ``eval/intrinsic`` uses ``Criterion.magnitude``
        — and the good/bad direction is taken from ``polarity`` alone. The
        intended magnitude is preserved in provenance so the coercion is
        auditable.
        """
        provenance = dict(self.provenance)
        provenance["weight_magnitude"] = abs(int(self.weight))
        return Criterion(
            title=self.title,
            description=self.description,
            weight=self.weight,
            category=self.category,
            provenance=provenance,
            validation=dict(self.validation) if self.validation is not None else None,
            polarity=Polarity.POSITIVE,
            polarity_method="enforced",
        )

    def clone(self) -> "_Candidate":
        return _Candidate(
            cid=self.cid,
            title=self.title,
            description=self.description,
            category=self.category,
            weight=self.weight,
            provenance=dict(self.provenance),
            validation=dict(self.validation) if self.validation is not None else None,
        )


@dataclass
class _Run:
    """Everything one ``generate`` call accumulates.

    Held locally rather than on the generator so many examples can share one
    generator instance concurrently.
    """

    example: Example
    stages: list[dict[str, Any]] = field(default_factory=list)
    stages_run: list[str] = field(default_factory=list)
    stages_failed: list[str] = field(default_factory=list)
    n_calls: int = 0

    spec: dict[str, Any] | None = None
    rollouts: list[_Rollout] = field(default_factory=list)
    reconciliation: dict[str, Any] | None = None
    pitfalls: list[dict[str, Any]] = field(default_factory=list)
    candidates: list[_Candidate] = field(default_factory=list)
    negatives: list[_Negative] = field(default_factory=list)
    survivors: list[_Candidate] = field(default_factory=list)
    final: list[_Candidate] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    anchors: frozenset[str] = frozenset()
    scope: dict[str, Any] = field(default_factory=dict)
    #: Lint tallies, accumulated across the batched stage and the final pass.
    lint: dict[str, int] = field(default_factory=dict)

    @property
    def target_items(self) -> int | None:
        value = self.scope.get("target_items")
        return int(value) if isinstance(value, int) else None

    def bump(self, key: str, amount: int = 1) -> None:
        self.lint[key] = self.lint.get(key, 0) + amount


@asynccontextmanager
async def _stage(run: _Run, name: str) -> AsyncIterator[dict[str, Any]]:
    """Record one stage's input/output and absorb its failures.

    An exception inside the block marks the stage failed and is swallowed, so
    the pipeline continues with whatever earlier stages produced.
    """
    record: dict[str, Any] = {"stage": name, "ok": True, "error": None, "input": {}, "output": None}
    started = time.perf_counter()
    try:
        yield record
    except Exception as exc:  # noqa: BLE001 - stage isolation is the point
        record["ok"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"[:600]
        run.stages_failed.append(name)
        logger.warning("agentic stage %s failed uid=%s: %s", name, run.example.uid, str(exc)[:300])
    else:
        run.stages_run.append(name)
    finally:
        record["seconds"] = round(time.perf_counter() - started, 2)
        run.stages.append(record)


@contextmanager
def _sync_stage(run: _Run, name: str) -> Iterator[dict[str, Any]]:
    """:func:`_stage` for the deterministic stages that run outside the event loop."""
    record: dict[str, Any] = {"stage": name, "ok": True, "error": None, "input": {}, "output": None}
    started = time.perf_counter()
    try:
        yield record
    except Exception as exc:  # noqa: BLE001 - stage isolation is the point
        record["ok"] = False
        record["error"] = f"{type(exc).__name__}: {exc}"[:600]
        run.stages_failed.append(name)
        logger.warning("agentic stage %s failed uid=%s: %s", name, run.example.uid, str(exc)[:300])
    else:
        run.stages_run.append(name)
    finally:
        record["seconds"] = round(time.perf_counter() - started, 2)
        run.stages.append(record)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


@register
class AgenticGenerator(RubricGenerator):
    """Multi-stage, validation-in-the-loop rubric synthesis."""

    name = "agentic"

    def __init__(
        self,
        engine: LLMEngine | None = None,
        *,
        config: AgenticConfig | Mapping[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        known = {f.name for f in fields(AgenticConfig)}
        # ``build_generator`` forwards arbitrary kwargs; treat the ones that name
        # AgenticConfig fields as overrides on top of the supplied config.
        overrides = {k: kwargs.pop(k) for k in list(kwargs) if k in known}
        if isinstance(config, AgenticConfig):
            base = config
        elif isinstance(config, Mapping):
            base = AgenticConfig(**{k: v for k, v in config.items() if k in known})
        else:
            base = AgenticConfig()
        super().__init__(engine=engine, **kwargs)
        self.config: AgenticConfig = replace(base, **overrides) if overrides else base
        # These belong with the stage switches in AgenticConfig, but that file is
        # not ours to extend, so they arrive as constructor options instead. All
        # three default to on: they are the F1/F3/F4/F6 fixes, not experiments.
        self.enable_lint: bool = bool(self.options.get("enable_lint", True))
        self.adaptive_items: bool = bool(self.options.get("adaptive_items", True))
        self.drop_ungrounded: bool = bool(self.options.get("drop_ungrounded", True))

    # -- LLM plumbing -----------------------------------------------------

    async def _chat(self, run: _Run, prompt: str, **kwargs: Any) -> str:
        if self.engine is None:
            raise RuntimeError("agentic generator requires an LLMEngine")
        run.n_calls += 1
        out = await self.engine.chat(prompt, **kwargs)
        return out if isinstance(out, str) else str(out)

    async def _chat_json(self, run: _Run, prompt: str, **kwargs: Any) -> Any:
        if self.engine is None:
            raise RuntimeError("agentic generator requires an LLMEngine")
        run.n_calls += 1
        return await self.engine.chat_json(prompt, **kwargs)

    @staticmethod
    def _skip(run: _Run, name: str, reason: str) -> None:
        run.stages.append({"stage": name, "ok": True, "skipped": reason, "output": None})

    # -- entry point ------------------------------------------------------

    async def generate(self, example: Example) -> GenerationResult:
        started = time.perf_counter()
        run = _Run(example=example)
        error: str | None = None
        try:
            await self._stage_decompose(run)
            await self._stage_rollouts(run)
            await self._stage_reconcile(run)
            await self._stage_pitfalls(run)
            self._stage_scope(run)
            await self._stage_draft(run)
            if run.candidates:
                await self._stage_lint(run)
                await self._stage_critic(run)
                await self._stage_calibrate(run)
            else:
                error = "no candidate criteria were drafted"
        except Exception as exc:  # noqa: BLE001 - one bad row must not kill a run
            error = f"{type(exc).__name__}: {exc}"[:600]
            logger.exception("agentic pipeline aborted uid=%s", example.uid)

        rubric = self._finalise(run)
        if error:
            rubric.meta["error"] = error
        elif not len(rubric):
            error = "agentic pipeline produced an empty rubric"
            rubric.meta["error"] = error

        return GenerationResult(
            uid=example.uid,
            source=self.name,
            rubric=rubric,
            trace={
                "stages": run.stages,
                "stages_run": list(run.stages_run),
                "stages_failed": list(run.stages_failed),
                "config": {f.name: getattr(self.config, f.name) for f in fields(AgenticConfig)},
                "n_llm_calls": run.n_calls,
            },
            error=error,
            n_llm_calls=run.n_calls,
            wall_seconds=time.perf_counter() - started,
        )

    # -- stage 1: decompose ----------------------------------------------

    async def _stage_decompose(self, run: _Run) -> None:
        if not self.config.enable_decompose:
            self._skip(run, "decompose", "disabled by config")
            return
        async with _stage(run, "decompose") as rec:
            user = P.build_decompose_user(run.example.question)
            rec["input"] = {"system": P.DECOMPOSE_SYSTEM, "user": user}
            raw = await self._chat_json(
                run,
                user,
                system=P.DECOMPOSE_SYSTEM,
                expect="object",
                max_tokens=_MAX_TOKENS["decompose"],
                tag="gen:agentic:decompose",
            )
            rec["output"] = raw
            spec = _as_dict(raw)
            if not spec:
                raise ValueError("decompose returned no object")
            run.spec = spec

    # -- stage 2: independent rollouts ------------------------------------

    async def _stage_rollouts(self, run: _Run) -> None:
        k = int(self.config.n_rollouts)
        if not self.config.enable_rollouts or k <= 0:
            self._skip(run, "rollouts", "disabled by config")
            return
        async with _stage(run, "rollouts") as rec:
            user = P.build_rollout_user(run.example.question, run.spec)
            rec["input"] = {
                "system": P.ROLLOUT_SYSTEM,
                "user": user,
                "k": k,
                "temperature": ROLLOUT_TEMPERATURE,
                "max_tokens": self.config.rollout_max_tokens,
            }
            outcomes = await asyncio.gather(
                *(self._one_rollout(run, user, i) for i in range(1, k + 1)),
                return_exceptions=True,
            )
            rollouts: list[_Rollout] = []
            for i, outcome in enumerate(outcomes, start=1):
                if isinstance(outcome, BaseException):
                    rollouts.append(_Rollout(index=i, error=f"{type(outcome).__name__}: {outcome}"[:300]))
                else:
                    rollouts.append(outcome)
            run.rollouts = [r for r in rollouts if r.ok]
            rec["output"] = {
                "n_ok": len(run.rollouts),
                "rollouts": [r.to_dict() for r in rollouts],
            }
            if not run.rollouts:
                raise RuntimeError(f"all {k} rollouts failed")

    async def _one_rollout(self, run: _Run, user: str, index: int) -> _Rollout:
        # A distinct salt per rollout is mandatory: the response cache is
        # content-addressed, so without it k samples collapse to one answer.
        text = await self._chat(
            run,
            user,
            system=P.ROLLOUT_SYSTEM,
            max_tokens=self.config.rollout_max_tokens,
            temperature=ROLLOUT_TEMPERATURE,
            tag="gen:agentic:rollout",
            cache_salt=f"rollout{index}",
        )
        return _Rollout(index=index, text=text, summary=_parse_trailing_json(text))

    # -- stage 3: reconcile ------------------------------------------------

    async def _stage_reconcile(self, run: _Run) -> None:
        if not self.config.enable_reconcile:
            self._skip(run, "reconcile", "disabled by config")
            return
        if not run.rollouts:
            self._skip(run, "reconcile", "no successful rollouts to reconcile")
            return
        async with _stage(run, "reconcile") as rec:
            user = P.build_reconcile_user(
                run.example.question,
                run.example.reference_answer,
                [r.to_prompt_dict() for r in run.rollouts],
                excerpt_chars=RECONCILE_EXCERPT_CHARS,
            )
            rec["input"] = {"system": P.RECONCILE_SYSTEM, "user": user}
            raw = await self._chat_json(
                run,
                user,
                system=P.RECONCILE_SYSTEM,
                expect="object",
                max_tokens=_MAX_TOKENS["reconcile"],
                tag="gen:agentic:reconcile",
            )
            rec["output"] = raw
            reconciliation = _as_dict(raw)
            if not reconciliation:
                raise ValueError("reconcile returned no object")
            run.reconciliation = reconciliation
            run.summary["disagreement"] = {
                bucket: len(_as_list(reconciliation.get(bucket)))
                for bucket in ("consensus_correct", "divergent", "collective_error", "reference_only")
            }

    # -- stage 4: pitfalls --------------------------------------------------

    async def _stage_pitfalls(self, run: _Run) -> None:
        if not self.config.enable_pitfalls:
            self._skip(run, "pitfalls", "disabled by config")
            return
        async with _stage(run, "pitfalls") as rec:
            user = P.build_pitfall_user(
                run.example.question,
                run.example.reference_answer,
                spec=run.spec,
                reconciliation=run.reconciliation,
                domain=run.example.domain,
            )
            rec["input"] = {"system": P.PITFALL_SYSTEM, "user": user}
            raw = await self._chat_json(
                run,
                user,
                system=P.PITFALL_SYSTEM,
                expect="array",
                max_tokens=_MAX_TOKENS["pitfalls"],
                tag="gen:agentic:pitfalls",
            )
            rec["output"] = raw
            pitfalls = [_as_dict(p) for p in _as_list(raw)]
            run.pitfalls = [p for p in pitfalls if p.get("mistake") or p.get("title")]
            if not run.pitfalls:
                raise ValueError("pitfall mining returned no usable entries")

    # -- stage 5: scope (deterministic) --------------------------------------

    def _stage_scope(self, run: _Run) -> None:
        """Derive the anchor set and the item-count target. No LLM call.

        Both outputs exist because the forensics found prompting alone does not
        fix either problem (§11.2): the shipped generator was *told* to choose
        7-20 items "based on the complexity of the question" and produced a
        constant 7 (CV 0.107, r=0.19 against question length), and it was told to
        write query-specific criteria while leaving 24.6% of reward mass on
        criteria with no question anchor at all. So the count is computed here
        from the decomposition rather than requested, and the anchor list becomes
        a checkable constraint rather than an instruction.
        """
        with _sync_stage(run, "scope") as rec:
            run.anchors = question_anchors(
                run.example.question, run.example.reference_answer
            )
            signals = _complexity_signals(run)
            derived = _derive_target(signals, float(self.config.max_pitfall_fraction))
            low = max(1, int(self.config.target_min_items))
            high = max(low, int(self.config.target_max_items))
            if self.adaptive_items and derived is not None:
                target = max(low, min(high, derived))
            else:
                target = (low + high) // 2
            run.scope = {
                "target_items": target,
                "derived_target": derived,
                "clamped_to": [low, high],
                "adaptive": self.adaptive_items and derived is not None,
                "signals": signals,
                "note": _complexity_note(signals),
                "n_anchors": len(run.anchors),
                "anchor_source": _forensics().source,
                "anchors": sorted(run.anchors)[:120],
            }
            rec["output"] = dict(run.scope)
            run.summary["target_items"] = target
            run.summary["derived_target_items"] = derived
            run.summary["complexity_signals"] = signals
            run.summary["n_anchors"] = len(run.anchors)

    # -- stage 6: draft -----------------------------------------------------

    async def _stage_draft(self, run: _Run) -> None:
        target = run.target_items or int(self.config.target_max_items)
        n_candidates = target + DRAFT_OVERSHOOT
        async with _stage(run, "draft") as rec:
            user = P.build_draft_user(
                run.example.question,
                run.example.reference_answer,
                spec=run.spec,
                reconciliation=run.reconciliation,
                pitfalls=run.pitfalls,
                domain=run.example.domain,
                n_candidates=n_candidates,
                anchors=sorted(run.anchors),
                banned_words=_subjective_words(),
                target_items=target,
                max_pitfall_fraction=float(self.config.max_pitfall_fraction),
            )
            rec["input"] = {"system": P.DRAFT_SYSTEM, "user": user}
            candidates: list[_Candidate] = []
            raw: Any = None
            try:
                raw = await self._chat_json(
                    run,
                    user,
                    system=P.DRAFT_SYSTEM,
                    expect="array",
                    max_tokens=_MAX_TOKENS["draft"],
                    tag="gen:agentic:draft",
                )
                candidates = _parse_candidates(raw)
            except (JSONParseError, RuntimeError) as exc:
                rec["warning"] = f"primary draft failed: {exc}"[:400]

            if not candidates:
                # The load-bearing stage: retry once without the evidence
                # sections, in case prompt size caused the failure.
                retry_user = P.build_draft_retry_user(
                    run.example.question,
                    run.example.reference_answer,
                    domain=run.example.domain,
                    n_candidates=target,
                    anchors=sorted(run.anchors),
                )
                rec["input"]["retry_user"] = retry_user
                retry_raw = await self._chat_json(
                    run,
                    retry_user,
                    system=P.DRAFT_SYSTEM,
                    expect="array",
                    max_tokens=_MAX_TOKENS["draft"],
                    tag="gen:agentic:draft_retry",
                    cache_salt="draft_retry",
                )
                rec["output"] = {"primary": raw, "retry": retry_raw}
                candidates = _parse_candidates(retry_raw)
                for cand in candidates:
                    cand.provenance["degraded"] = "evidence-free retry"
            else:
                rec["output"] = raw

            if not candidates:
                raise ValueError("draft returned no usable criteria")
            run.candidates = candidates
            run.survivors = [c.clone() for c in candidates]
            run.summary["n_drafted"] = len(candidates)

    # -- stage 7: lint ------------------------------------------------------

    async def _stage_lint(self, run: _Run) -> None:
        """Verify and revise: polarity (F6), grounding (F3), subjectivity (F4).

        One batched call repairs every offender in the rubric at once, so the
        stage costs a single request regardless of how many criteria failed.
        Whatever the call does not fix, the deterministic rewrites below do; what
        neither can ground is deleted rather than kept as a placeholder, because
        the forensics is explicit that a short grounded rubric beats a padded one.
        """
        if not self.enable_lint:
            self._skip(run, "lint", "disabled by config")
            return
        async with _stage(run, "lint") as rec:
            flags = {c.cid: self._lint_flags(c, run.anchors) for c in run.candidates}
            offenders = [c for c in run.candidates if flags[c.cid]]
            rec["input"] = {
                "n_candidates": len(run.candidates),
                "n_offenders": len(offenders),
                "flags": {cid: sorted(problems) for cid, problems in flags.items() if problems},
                "anchor_source": _forensics().source,
            }
            repairs: dict[int, dict[str, Any]] = {}
            if offenders:
                payload = [
                    {
                        "id": c.cid,
                        "title": c.title,
                        "category": c.category.value,
                        "description": c.description,
                        "problems": sorted(flags[c.cid]),
                    }
                    for c in offenders
                ]
                user = P.build_lint_user(
                    run.example.question,
                    payload,
                    anchors=sorted(run.anchors),
                    banned_words=_subjective_words(),
                )
                rec["input"]["system"] = P.LINT_SYSTEM
                rec["input"]["user"] = user
                try:
                    raw = await self._chat_json(
                        run,
                        user,
                        system=P.LINT_SYSTEM,
                        expect="array",
                        max_tokens=_MAX_TOKENS["lint"],
                        tag="gen:agentic:lint",
                    )
                    rec["output"] = raw
                    repairs = _parse_repairs(raw, [c.cid for c in offenders])
                except (JSONParseError, RuntimeError) as exc:
                    # Not fatal: the deterministic rewrites below still run, they
                    # just have less context than the model would have had.
                    rec["warning"] = f"lint call failed, falling back to rewrites: {exc}"[:400]

            kept, decisions = self._apply_lint(run, repairs, flags)
            rec["output"] = {"llm": rec.get("output"), "decisions": decisions}
            if kept:
                run.candidates = kept
                run.survivors = [c.clone() for c in kept]
            else:
                rec["warning"] = "every candidate failed the lint; keeping the draft unchanged"
                run.bump("n_lint_abandoned")

    def _lint_flags(self, cand: _Candidate, anchors: frozenset[str]) -> set[str]:
        problems: set[str] = set()
        if polarity_offence(cand.description, cand.category):
            problems.add("polarity")
        if anchors and not _anchor_hits(cand.description, anchors):
            problems.add("grounding")
        if subjective_hits(cand.description):
            problems.add("subjectivity")
        return problems

    def _apply_lint(
        self,
        run: _Run,
        repairs: Mapping[int, dict[str, Any]],
        flags: Mapping[int, set[str]],
    ) -> tuple[list[_Candidate], list[dict[str, Any]]]:
        kept: list[_Candidate] = []
        decisions: list[dict[str, Any]] = []
        for cand in run.candidates:
            before = cand.description
            entry = repairs.get(cand.cid) or {}
            repaired = _as_text(entry.get("description"), limit=2000)
            dropped_by_model = _as_bool(entry.get("drop"))
            if repaired:
                cand.description = strip_category_prefix(repaired)
                cand.title = _as_text(entry.get("title"), limit=120) or cand.title
            decision = _enforce_criterion(
                cand, run.anchors, drop_ungrounded=self.drop_ungrounded
            )
            if dropped_by_model and decision["action"] != "drop":
                decision["action"] = "drop"
                decision["reason"] = "no recoverable question-specific content"
            decision.update(
                {
                    "id": cand.cid,
                    "problems": sorted(flags.get(cand.cid) or ()),
                    "llm_repaired": bool(repaired),
                    "before": before,
                    "after": cand.description,
                }
            )
            decisions.append(decision)
            _tally_lint(run, decision, flags.get(cand.cid) or set())
            if decision["action"] != "drop":
                kept.append(cand)
        return kept, decisions

    # -- stage 8: critic ----------------------------------------------------

    async def _stage_critic(self, run: _Run) -> None:
        run.survivors = [c.clone() for c in run.candidates]
        run.summary["critic_ran"] = False
        if not self.config.enable_critic:
            self._skip(run, "critic", "disabled by config (agentic-noval ablation)")
            return
        async with _stage(run, "critic") as rec:
            negatives, construction_log = await self._build_negatives(run)
            run.negatives = negatives
            rec["input"] = {
                "n_candidates": len(run.candidates),
                "negative_construction": construction_log,
            }
            if not negatives:
                raise RuntimeError("no negative test cases could be assembled")

            prompt_candidates = [c.to_prompt_dict() for c in run.candidates]
            cids = [c.cid for c in run.candidates]

            gold_user = P.build_critic_user(
                run.example.question,
                _clip_middle(run.example.reference_answer, JUDGE_EXCERPT_CHARS),
                prompt_candidates,
                role="gold",
            )
            negative_users = [
                P.build_critic_user(
                    run.example.question,
                    _clip_middle(neg.text, JUDGE_EXCERPT_CHARS),
                    prompt_candidates,
                    role="negative",
                    flaw_note=neg.note or None,
                )
                for neg in negatives
            ]
            rec["input"]["judge_calls"] = [
                {"role": "gold", "system": P.CRITIC_GOLD_SYSTEM, "user": gold_user},
                *(
                    {"role": neg.nid, "system": P.CRITIC_NEGATIVE_SYSTEM, "user": user}
                    for neg, user in zip(negatives, negative_users)
                ),
            ]

            # One batched call per (candidate-set, response) pair.
            outcomes = await asyncio.gather(
                self._judge_batch(run, gold_user, P.CRITIC_GOLD_SYSTEM, "gold"),
                *(
                    self._judge_batch(run, user, P.CRITIC_NEGATIVE_SYSTEM, neg.nid)
                    for neg, user in zip(negatives, negative_users)
                ),
                return_exceptions=True,
            )

            raw_verdicts: dict[str, Any] = {}
            gold_map: dict[int, dict[str, Any]] | None = None
            negative_maps: list[tuple[_Negative, dict[int, dict[str, Any]]]] = []
            for position, outcome in enumerate(outcomes):
                label = "gold" if position == 0 else negatives[position - 1].nid
                if isinstance(outcome, BaseException):
                    raw_verdicts[label] = {"error": f"{type(outcome).__name__}: {outcome}"[:300]}
                    logger.warning(
                        "critic judge %s failed uid=%s: %s", label, run.example.uid, str(outcome)[:200]
                    )
                    continue
                raw_verdicts[label] = outcome
                parsed = _parse_verdicts(outcome, cids)
                if position == 0:
                    gold_map = parsed
                else:
                    negative_maps.append((negatives[position - 1], parsed))

            rec["output"] = {
                "negatives": [n.to_dict() for n in negatives],
                "verdicts": raw_verdicts,
            }
            decisions = self._apply_critic_decisions(run, gold_map, negative_maps)
            rec["output"]["decisions"] = decisions
            run.summary["critic_ran"] = True

    async def _mine_hard_negatives(
        self, run: _Run, *, want: int
    ) -> tuple[list[_Negative], list[dict[str, Any]]]:
        """Resample the question under failure-prone settings until it breaks.

        Stage 2 deliberately samples carefully, which is right for finding
        disagreement but wrong for collecting counterexamples: a policy that
        solves the question three times out of three supplies none. Mining draws
        further attempts under instructions that suppress the behaviours which
        make the careful samples correct — no working shown, no verification —
        and keeps only those an independent check rules incorrect.

        The salt namespace is distinct from both Stage 2 (``rollout{i}``) and the
        evaluation rollouts (``eval-rollout:*``), so a mined negative can never
        be a text the evaluation later scores. Correctness is decided against
        ``reference_answer``, which the generator is entitled to see offline.
        """
        found: list[_Negative] = []
        log: list[dict[str, Any]] = []
        for attempt, style in enumerate(_HARD_NEGATIVE_STYLES, start=1):
            if len(found) >= want:
                break
            try:
                text = await self._chat(
                    run,
                    P.build_rollout_user(run.example.question, run.spec),
                    system=style["system"],
                    max_tokens=self.config.rollout_max_tokens,
                    temperature=float(style["temperature"]),
                    tag="gen:agentic:negmine",
                    cache_salt=f"negmine{attempt}",
                )
            except Exception as exc:  # noqa: BLE001
                log.append({"attempt": attempt, "style": style["name"],
                            "error": f"{type(exc).__name__}: {exc}"[:200]})
                continue
            text = _as_text(text, limit=60000)
            if not text:
                log.append({"attempt": attempt, "style": style["name"], "error": "empty"})
                continue
            verdict = await self._judge_rollout_correctness(run, text)
            log.append({"attempt": attempt, "style": style["name"],
                        "correct": verdict.get("correct"),
                        "why": str(verdict.get("why") or "")[:200]})
            if verdict.get("correct") is False:
                found.append(
                    _Negative(
                        nid=f"mined{attempt}",
                        kind="rollout",
                        text=text,
                        note=str(verdict.get("why") or "a real failed model attempt")[:600],
                    )
                )
        return found, log

    async def _judge_rollout_correctness(self, run: _Run, text: str) -> dict[str, Any]:
        """Is this attempt's conclusion the same as the reference's? No rubric involved."""
        user = (
            f"QUESTION:\n{run.example.question}\n\n"
            f"REFERENCE ANSWER:\n{run.example.reference_answer}\n\n"
            f"CANDIDATE ANSWER:\n{text[:20000]}"
        )
        try:
            parsed = await self._chat_json(
                run, user, system=_NEGMINE_CHECK_SYSTEM, max_tokens=4096,
                expect="object", tag="gen:agentic:negmine:check",
            )
        except Exception:  # noqa: BLE001
            return {}
        return _as_dict(parsed)

    async def _build_negatives(self, run: _Run) -> tuple[list[_Negative], list[dict[str, Any]]]:
        """Wrong rollouts plus at least one purpose-built flawed response."""
        negatives: list[_Negative] = []
        cap = max(1, int(self.config.max_rollout_negatives))
        wrong = _wrong_rollouts(run)
        for rollout, note in wrong[:cap]:
            negatives.append(
                _Negative(
                    nid=f"rollout{rollout.index}",
                    kind="rollout",
                    text=rollout.text,
                    note=note,
                )
            )

        if self.config.negatives_from_rollouts_only:
            # Real-failure-only mode. Stage 2 samples at high effort and mostly
            # succeeds, so on this corpus two thirds of questions yield no wrong
            # rollout at all — which would make this ablation a no-op on most of
            # the set and unable to answer the question it exists to answer.
            # Rather than substituting a synthetic negative (that being the very
            # variable under test), mine for a *real* failure by resampling
            # under settings the policy is known to fail at.
            mined: list[dict[str, Any]] = []
            if len(negatives) < cap and self.config.mine_hard_negatives:
                extra, mined = await self._mine_hard_negatives(
                    run, want=cap - len(negatives)
                )
                negatives.extend(extra)
            run.summary["realneg_no_failure"] = not negatives
            run.summary["realneg_n_negatives"] = len(negatives)
            run.summary["realneg_n_mined"] = len(mined)
            return negatives, [
                {"mode": "rollouts_only", "n_wrong_stage2_rollouts": len(wrong),
                 "mining": mined}
            ]

        # Always construct at least one negative so discrimination is measured
        # against a deliberate, known mistake rather than only against whatever
        # the rollouts happened to get wrong.
        n_constructed = 1 if negatives else 2
        chosen_pitfalls: list[dict[str, Any] | None] = list(run.pitfalls[:n_constructed])
        while len(chosen_pitfalls) < n_constructed:
            chosen_pitfalls.append(None)

        construction_log: list[dict[str, Any]] = []
        prompts: list[str] = []
        for pitfall in chosen_pitfalls:
            user = P.build_negative_user(
                run.example.question,
                run.example.reference_answer,
                pitfall=pitfall,
                spec=run.spec,
            )
            prompts.append(user)
            construction_log.append(
                {"system": P.NEGATIVE_SYSTEM, "user": user, "pitfall": pitfall}
            )

        outcomes = await asyncio.gather(
            *(
                self._chat(
                    run,
                    user,
                    system=P.NEGATIVE_SYSTEM,
                    max_tokens=_MAX_TOKENS["negative"],
                    temperature=0.7,
                    tag="gen:agentic:negative",
                    cache_salt=f"negative{i}",
                )
                for i, user in enumerate(prompts, start=1)
            ),
            return_exceptions=True,
        )
        for i, outcome in enumerate(outcomes):
            pitfall = chosen_pitfalls[i] or {}
            if isinstance(outcome, BaseException):
                construction_log[i]["error"] = f"{type(outcome).__name__}: {outcome}"[:300]
                continue
            text = _as_text(outcome, limit=60000)
            if not text:
                construction_log[i]["error"] = "empty response"
                continue
            note = _as_text(pitfall.get("mistake") or pitfall.get("title"), limit=400)
            negatives.append(
                _Negative(
                    nid=f"constructed{i + 1}",
                    kind="constructed",
                    text=text,
                    note=note or "an unspecified planted technical error",
                )
            )
        return negatives, construction_log

    async def _judge_batch(self, run: _Run, user: str, system: str, label: str) -> Any:
        return await self._chat_json(
            run,
            user,
            system=system,
            expect="array",
            max_tokens=_MAX_TOKENS["critic"],
            tag=f"gen:agentic:critic:{'gold' if label == 'gold' else 'negative'}",
            cache_salt=f"critic:{label}",
        )

    def _apply_critic_decisions(
        self,
        run: _Run,
        gold_map: dict[int, dict[str, Any]] | None,
        negative_maps: Sequence[tuple[_Negative, dict[int, dict[str, Any]]]],
    ) -> list[dict[str, Any]]:
        """Turn raw verdicts into per-criterion validation records and edits."""
        candidates = [c.clone() for c in run.candidates]
        gold_available = gold_map is not None

        for cand in candidates:
            gold_entry = (gold_map or {}).get(cand.cid)
            gold_seen = gold_entry is not None
            gold_pass = _as_bool(gold_entry.get("met"), default=True) if gold_seen else True
            contradicted = _as_bool(gold_entry.get("contradicted")) if gold_seen else False
            per_negative: list[dict[str, Any]] = []
            for negative, verdicts in negative_maps:
                entry = verdicts.get(cand.cid)
                if entry is None:
                    continue
                met = _as_bool(entry.get("met"), default=True)
                per_negative.append({"id": negative.nid, "kind": negative.kind, "met": met})
            n_negatives = len(per_negative)
            n_failed = sum(1 for v in per_negative if not v["met"])
            cand.validation = {
                "gold_pass": bool(gold_pass) and not contradicted,
                "gold_contradicted": bool(contradicted),
                "gold_evidence": "judged" if gold_seen else ("missing" if gold_available else "unavailable"),
                "n_negatives": n_negatives,
                "n_negatives_failed": n_failed,
                "discrimination": (n_failed / n_negatives) if n_negatives else 0.0,
                "negatives": per_negative,
                "gold_why": _as_text((gold_entry or {}).get("why"), limit=200),
                "decision": "keep",
            }

        by_cid = {c.cid: c for c in candidates}
        dropped: set[int] = set()
        gold_fail_pool: list[_Candidate] = []
        # When the gold signal is switched off, the verdicts above are still
        # recorded (they are the audit trail for the leakage analysis) but no
        # edit may be derived from them.
        use_gold = self.config.use_gold_signal
        use_negative = self.config.use_negative_signal
        for cand in candidates:
            val = cand.validation or {}
            if not use_gold:
                val["gold_signal_used"] = False
                continue
            if val.get("gold_contradicted"):
                # The reference refutes it: the criterion is simply wrong.
                val["decision"] = "drop_contradicted"
                dropped.add(cand.cid)
            elif not val.get("gold_pass") and val.get("gold_evidence") == "judged":
                gold_fail_pool.append(cand)

        # Terse reference answers legitimately fail criteria they never had room
        # to state. Drop the least valuable of those, but never below the target
        # floor — the survivors are demoted and flagged instead.
        floor = min(int(self.config.target_min_items), len(candidates) - len(dropped))
        remaining = len(candidates) - len(dropped)
        if use_gold and self.config.drop_gold_failures:
            for cand in sorted(gold_fail_pool, key=_ascending_value):
                if remaining <= floor:
                    break
                (cand.validation or {})["decision"] = "drop_gold_fail"
                dropped.add(cand.cid)
                remaining -= 1

        survivors: list[_Candidate] = []
        n_demoted = n_boosted = n_retained = 0
        for cand in candidates:
            if cand.cid in dropped:
                continue
            val = cand.validation or {}
            if use_gold and not val.get("gold_pass") and val.get("gold_evidence") == "judged":
                # Kept against the rule, so it must not also carry a high weight.
                val["decision"] = "retained_gold_fail"
                val["retained_reason"] = (
                    "target_min_items floor" if self.config.drop_gold_failures else "drop disabled"
                )
                val["gold_fail_retained"] = True
                cand.weight = _step_weight(cand.weight, cand.category, up=False)
                n_retained += 1
            elif val.get("n_negatives") and not val.get("n_negatives_failed"):
                if use_negative and self.config.demote_indiscriminate:
                    val["decision"] = "demote_indiscriminate"
                    val["indiscriminate"] = True
                    cand.weight = _step_weight(cand.weight, cand.category, up=False)
                    n_demoted += 1
                else:
                    val["decision"] = "keep_indiscriminate"
                    val["indiscriminate"] = True
            elif val.get("n_negatives_failed"):
                # Proven signal: it separates the reference from every planted flaw.
                if (
                    use_negative
                    and val.get("discrimination", 0.0) >= 1.0
                    and val.get("n_negatives", 0) >= 2
                ):
                    val["decision"] = "keep_boosted"
                    cand.weight = _step_weight(cand.weight, cand.category, up=True)
                    n_boosted += 1
                else:
                    val["decision"] = "keep"
            else:
                val["decision"] = "keep_untested"
            survivors.append(cand)

        run.survivors = survivors
        run.summary.update(
            {
                "n_negatives": len(run.negatives),
                "n_dropped_contradicted": sum(
                    1 for c in candidates if (c.validation or {}).get("decision") == "drop_contradicted"
                ),
                "n_dropped_gold_fail": sum(
                    1 for c in candidates if (c.validation or {}).get("decision") == "drop_gold_fail"
                ),
                "n_retained_gold_fail": n_retained,
                "n_demoted_indiscriminate": n_demoted,
                "n_boosted": n_boosted,
                "n_survivors": len(survivors),
                "gold_judge_ok": gold_available,
            }
        )
        return [
            {
                "id": cand.cid,
                "title": cand.title,
                "decision": (cand.validation or {}).get("decision"),
                "validation": cand.validation,
            }
            for cand in by_cid.values()
        ]

    # -- stage 9: calibrate -------------------------------------------------

    async def _stage_calibrate(self, run: _Run) -> None:
        if not self.config.enable_calibration:
            self._skip(run, "calibrate", "disabled by config")
            return
        if not run.survivors:
            self._skip(run, "calibrate", "no surviving candidates")
            return
        async with _stage(run, "calibrate") as rec:
            note = _calibration_note(run)
            low, high = self._calibration_window(run)
            user = P.build_calibrate_user(
                run.example.question,
                run.example.reference_answer,
                [c.to_prompt_dict(with_validation=True) for c in run.survivors],
                min_items=low,
                max_items=high,
                domain=run.example.domain,
                validation_note=note,
                anchors=sorted(run.anchors),
                banned_words=_subjective_words(),
                target_items=run.target_items,
                complexity_note=str(run.scope.get("note") or "") or None,
                max_pitfall_fraction=float(self.config.max_pitfall_fraction),
            )
            rec["input"] = {"system": P.CALIBRATE_SYSTEM, "user": user}
            raw = await self._chat_json(
                run,
                user,
                system=P.CALIBRATE_SYSTEM,
                expect="array",
                max_tokens=_MAX_TOKENS["calibrate"],
                tag="gen:agentic:calibrate",
            )
            rec["output"] = raw
            final = _merge_calibrated(_as_list(raw), run.survivors)
            if not final:
                raise ValueError("calibration returned no usable criteria")
            run.final = final

    def _calibration_window(self, run: _Run) -> tuple[int, int]:
        """The count range handed to the calibrator, centred on the derived target.

        Config's ``target_min_items`` / ``target_max_items`` are outer bounds, not
        the target: asking for "6 to 16" every time is how the shipped generator
        ended up at a constant 7 (F1).
        """
        low = max(1, int(self.config.target_min_items))
        high = max(low, int(self.config.target_max_items))
        target = run.target_items
        if target is None:
            return low, high
        return max(low, target - 1), min(high, max(target + 1, low))

    # -- assembly -----------------------------------------------------------

    def _finalise(self, run: _Run) -> Rubric:
        chosen = run.final or run.survivors or run.candidates
        chosen = self._stage_dedup(run, chosen)
        items = [c.to_criterion() for c in chosen]

        summary = dict(run.summary)
        summary.setdefault("n_drafted", len(run.candidates))
        summary.setdefault("critic_ran", False)
        summary.setdefault("n_negatives", len(run.negatives))
        summary.setdefault("n_dropped_gold_fail", 0)
        summary.setdefault("n_dropped_contradicted", 0)
        summary.setdefault("n_demoted_indiscriminate", 0)
        summary.setdefault("n_boosted", 0)
        summary.setdefault("n_survivors", len(run.survivors))
        summary["n_kept"] = len(items)
        summary["mean_discrimination"] = _mean_discrimination(items)
        summary.update(
            {key: run.lint.get(key, 0) for key in _LINT_SUMMARY_KEYS}
        )
        summary["lint_ran"] = self.enable_lint
        summary["anchor_source"] = _forensics().source
        summary["mean_anchors_per_criterion"] = _mean_anchor_hits(items, run.anchors)
        # Emitted so the F6 claim is checkable from the artefact alone rather
        # than only from a rerun of the generator.
        summary["polarity"] = Rubric(items=items).polarity_stats()
        summary["pitfall_fraction"] = (
            round(sum(1 for c in items if c.category is Category.PITFALL) / len(items), 4)
            if items
            else 0.0
        )
        summary["pitfall_weight_convention"] = (
            "stored on the RaR -1/-2 scale by schema.Criterion; graded by magnitude "
            "with polarity=positive"
        )

        return Rubric(
            items=items,
            meta={
                "source": self.name,
                "n_items": len(items),
                "stages_run": list(run.stages_run),
                "stages_failed": list(run.stages_failed),
                "validation_summary": summary,
                "n_llm_calls": run.n_calls,
                "config": {f.name: getattr(self.config, f.name) for f in fields(AgenticConfig)},
                "options": {
                    "enable_lint": self.enable_lint,
                    "adaptive_items": self.adaptive_items,
                    "drop_ungrounded": self.drop_ungrounded,
                },
            },
        )

    def _stage_dedup(self, run: _Run, chosen: Sequence[_Candidate]) -> list[_Candidate]:
        """Whole-rubric pass: mirror merge, re-lint, then size.

        This sees the finished checklist, which is what the shipped single-pass
        generator never does — 16.2% of RaR-Medicine questions carry a Pitfall
        that is only a polarity-flipped restatement of an Essential item, scoring
        one fact twice (F7). Deterministic throughout, so it costs no call and
        still runs when calibration failed.
        """
        items = [c.clone() for c in chosen]
        with _sync_stage(run, "dedup") as rec:
            rec["input"] = {"n_in": len(items)}
            merged, merges = _merge_mirrors(items)
            for merge in merges:
                run.bump(
                    "n_mirror_merges" if merge["cross_category"] else "n_duplicate_merges"
                )
            relinted: list[_Candidate] = []
            enforcement: list[dict[str, Any]] = []
            for cand in merged:
                before = cand.description
                decision = _enforce_criterion(
                    cand, run.anchors, drop_ungrounded=self.drop_ungrounded
                )
                decision.update({"id": cand.cid, "before": before, "after": cand.description})
                flags: set[str] = set()
                if decision["positivised"]:
                    flags.add("polarity")
                if decision["destyled"]:
                    flags.add("subjectivity")
                if not decision["grounded_after"]:
                    flags.add("grounding")
                _tally_lint(run, decision, flags)
                if decision["action"] != "drop":
                    relinted.append(cand)
                else:
                    enforcement.append(decision)
            items = self._enforce_bounds(
                relinted or merged,
                run.survivors or run.candidates,
                target=run.target_items,
                anchors=run.anchors,
            )
            rec["output"] = {
                "n_out": len(items),
                "merges": merges,
                "dropped": enforcement,
                "target_items": run.target_items,
            }
        return items

    def _enforce_bounds(
        self,
        chosen: Sequence[_Candidate],
        pool: Sequence[_Candidate],
        *,
        target: int | None = None,
        anchors: frozenset[str] = frozenset(),
    ) -> list[_Candidate]:
        """Size the rubric to the derived target, inside the configured bounds.

        Trimming is free — the least defensible items go. Filling is not: the
        forensics found a four-item grounded rubric beats a seven-item padded one
        (F3), so the only items ever added back are drafted candidates that pass
        the same grounding and polarity lint as the ones already in, and the
        floor is abandoned rather than met with filler.
        """
        low = max(1, int(self.config.target_min_items))
        high = max(low, int(self.config.target_max_items))
        if target is not None:
            high = max(low, min(high, int(target)))
            low = min(low, high)
        items = list(chosen)

        if len(items) > high:
            keep = {id(c) for c in sorted(items, key=_ascending_value, reverse=True)[:high]}
            items = [c for c in items if id(c) in keep]

        if len(items) < low:
            seen = {(c.title.lower(), c.description.lower()) for c in items}
            extras = [
                c
                for c in pool
                if (c.title.lower(), c.description.lower()) not in seen
                and not polarity_offence(c.description, c.category)
                and (
                    not (self.drop_ungrounded and anchors)
                    or _anchor_hits(c.description, anchors)
                )
            ]
            for cand in sorted(extras, key=_ascending_value, reverse=True):
                if len(items) >= low:
                    break
                items.append(cand.clone())
        return _rebalance_pitfalls(items, pool, float(self.config.max_pitfall_fraction))


# ---------------------------------------------------------------------------
# Parsing / post-processing
# ---------------------------------------------------------------------------


def _parse_candidates(raw: Any) -> list[_Candidate]:
    """Read stage-5 output into renumbered, category-normalised candidates."""
    out: list[_Candidate] = []
    for entry in _as_list(raw):
        data = _as_dict(entry)
        description = _as_text(data.get("description") or data.get("criterion"), limit=2000)
        if not description:
            continue
        inline = parse_category_prefix(description)
        if inline is not None:
            description = strip_category_prefix(description)
        category = inline or Category.coerce(data.get("category"))
        title = _as_text(data.get("title"), limit=120) or _derive_title(description)
        weight = _as_int(data.get("weight"))
        if weight is None:
            weight = -1 if category is Category.PITFALL else 3
        evidence = _as_text(data.get("evidence") or data.get("provenance"), limit=60).lower()
        if evidence not in P.EVIDENCE_KINDS:
            evidence = "other"
        provenance = {
            "stage": "draft",
            "evidence": evidence,
            "evidence_detail": _as_text(data.get("evidence_detail"), limit=600),
            "discriminates": _as_text(data.get("discriminates"), limit=400),
        }
        cid = len(out) + 1
        provenance["draft_id"] = _as_int(data.get("id")) or cid
        out.append(
            _Candidate(
                cid=cid,
                title=title,
                description=description,
                category=category,
                weight=_clamp_weight(weight, category),
                provenance=provenance,
            )
        )
    return out


_LINT_SUMMARY_KEYS: tuple[str, ...] = (
    "n_polarity_rewrites",
    "n_regrounded",
    "n_dropped_ungrounded",
    "n_dropped_style",
    "n_destyled",
    "n_mirror_merges",
    "n_duplicate_merges",
    "n_lint_abandoned",
)


def _parse_repairs(raw: Any, cids: Sequence[int]) -> dict[int, dict[str, Any]]:
    """Map the batched lint reply onto criterion ids, tolerating id drift."""
    entries = [e for e in (_as_dict(entry) for entry in _as_list(raw)) if e]
    if not entries:
        return {}
    valid = set(cids)
    out: dict[int, dict[str, Any]] = {}
    for position, entry in enumerate(entries):
        reported = _as_int(entry.get("id"))
        target = reported if reported in valid else (cids[position] if position < len(cids) else None)
        if target is None or target in out:
            continue
        out[target] = entry
    return out


def _enforce_criterion(
    cand: _Candidate, anchors: frozenset[str], *, drop_ungrounded: bool
) -> dict[str, Any]:
    """Deterministically bring one criterion into compliance, or condemn it.

    Runs after any LLM repair and again on the finished rubric, so a criterion
    that the calibrator reworded back into a failure phrasing cannot escape.
    """
    text = strip_category_prefix(cand.description or "").strip()
    destyled = False
    if subjective_hits(text):
        rewritten = destyle(text)
        if rewritten and rewritten != text:
            text, destyled = rewritten, True

    positivised = False
    if polarity_offence(text, cand.category):
        text = positivise(text, cand.category)
        positivised = True
        if polarity_offence(text, cand.category):
            # ``_AVOIDANCE_PHRASING_RE`` is checked before the failure patterns,
            # so this prefix always terminates the loop.
            text = _tidy(f"Avoids {text[0].lower()}{text[1:]}")

    cand.description = text
    cand.title = cand.title or _derive_title(text)
    hits = _anchor_hits(text, anchors)
    grounded = bool(hits) or not anchors
    style = is_style_criterion(text) and not hits

    decision: dict[str, Any] = {
        "action": "keep",
        "reason": "",
        "destyled": destyled,
        "positivised": positivised,
        "polarity_offence_after": polarity_offence(text, cand.category),
        "grounded_after": grounded,
        "style": style,
        "anchors": sorted(hits)[:8],
    }
    if not grounded and drop_ungrounded:
        decision["action"] = "drop"
        decision["reason"] = "pure style, no question anchor" if style else "no question anchor"

    lint_record = dict(cand.provenance.get("lint") or {})
    lint_record.update(
        {
            "destyled": lint_record.get("destyled", False) or destyled,
            "positivised": lint_record.get("positivised", False) or positivised,
            "anchors": decision["anchors"],
        }
    )
    cand.provenance["lint"] = lint_record
    cand.provenance["polarity"] = Polarity.POSITIVE.value
    return decision


def _tally_lint(run: _Run, decision: Mapping[str, Any], flags: set[str]) -> None:
    changed = decision.get("before") != decision.get("after")
    if "polarity" in flags and not decision.get("polarity_offence_after"):
        run.bump("n_polarity_rewrites")
    if "subjectivity" in flags and changed:
        run.bump("n_destyled")
    if decision.get("action") == "drop":
        # Every drop here is a grounding drop: nothing else deletes a criterion
        # at this point. ``n_dropped_style`` is the subset that was pure
        # presentation, reported separately because F4 and F3 are different
        # claims about the same item.
        run.bump("n_dropped_ungrounded")
        if decision.get("style"):
            run.bump("n_dropped_style")
    elif "grounding" in flags and decision.get("grounded_after"):
        run.bump("n_regrounded")


# ---------------------------------------------------------------------------
# F1 — deriving the item count from the question
# ---------------------------------------------------------------------------


def _complexity_signals(run: _Run) -> dict[str, int]:
    """Countable structure in the question, as the earlier stages found it."""
    spec = run.spec or {}
    reconciliation = run.reconciliation or {}
    quantities = sorted(
        len(_as_list((r.summary or {}).get("key_quantities"))) for r in run.rollouts
    )
    # Median rather than the union: rollouts name the same intermediate three
    # different ways, so a union counts naming variation as complexity.
    median_quantities = quantities[len(quantities) // 2] if quantities else 0
    return {
        "n_sub_questions": len(_as_list(spec.get("sub_questions"))),
        "n_targets": len(_as_list(spec.get("targets"))),
        "n_given": len(_as_list(spec.get("given"))),
        "n_implicit_constraints": len(_as_list(spec.get("implicit_constraints"))),
        "n_intermediate_quantities": int(median_quantities),
        "n_divergent": len(_as_list(reconciliation.get("divergent"))),
        "n_collective_error": len(_as_list(reconciliation.get("collective_error"))),
        "n_reference_only": len(_as_list(reconciliation.get("reference_only"))),
        "n_high_severity_pitfalls": sum(
            1
            for p in run.pitfalls
            if _as_text(p.get("severity"), limit=20).lower() == "high"
        ),
    }


def _derive_target(signals: Mapping[str, int], max_pitfall_fraction: float) -> int | None:
    """Item count implied by the question's own structure.

    One criterion per deliverable the question names, one per intermediate
    quantity a solution has to produce, one per point the independent attempts
    actually disagreed about, plus a bounded allowance for stated constraints and
    for this question's traps. Every term is capped so a single verbose stage
    cannot run away with the budget.

    Returns ``None`` when no stage produced any structure to count, in which case
    the caller falls back to the configured window.
    """
    if not any(signals.values()):
        return None
    deliverables = max(1, signals.get("n_sub_questions", 0)) + signals.get("n_targets", 0)
    intermediates = min(signals.get("n_intermediate_quantities", 0), 5)
    evidence = (
        min(signals.get("n_divergent", 0), 3)
        + min(signals.get("n_collective_error", 0), 2)
        + min(signals.get("n_reference_only", 0), 2)
    )
    # Capped hardest of the four: the decomposer lists idealisations for almost
    # every physics question, so an uncapped term here is a constant, and a
    # constant is exactly what F1 is about removing.
    constraints = min(signals.get("n_implicit_constraints", 0), 2)
    core = deliverables + intermediates + evidence + constraints
    # p <= f*(core + p)  =>  p <= core*f/(1-f): the largest pitfall allowance that
    # still leaves the finished rubric inside config.max_pitfall_fraction.
    headroom = int(core * max_pitfall_fraction / max(1e-9, 1.0 - max_pitfall_fraction))
    return core + min(signals.get("n_high_severity_pitfalls", 0), max(0, headroom))


def _complexity_note(signals: Mapping[str, int]) -> str:
    parts = [
        f"{signals.get('n_sub_questions', 0)} sub-question(s)",
        f"{signals.get('n_targets', 0)} requested quantity/quantities",
        f"{signals.get('n_intermediate_quantities', 0)} intermediate quantity/quantities",
        f"{signals.get('n_divergent', 0)} disputed point(s)",
        f"{signals.get('n_collective_error', 0)} shared error(s)",
        f"{signals.get('n_implicit_constraints', 0)} implicit constraint(s)",
    ]
    return ", ".join(parts) + "."


# ---------------------------------------------------------------------------
# F7 — polarity-mirror dedup
# ---------------------------------------------------------------------------

#: Jaccard over polarity-stripped content tokens. Chosen to sit at the shoulder
#: of the forensics' cosine>=0.6 mirror threshold while staying above the noise
#: floor of two genuinely different criteria that happen to share a topic.
_MIRROR_JACCARD = 0.6


def _mirror_key(cand: _Candidate) -> tuple[frozenset[str], frozenset[str]]:
    """Comparable content of a criterion: topic tokens and the marks that separate near-twins."""
    engine = _forensics()
    body = engine.polarity_stripped(strip_category_prefix(cand.description or "").lower())
    tokens = frozenset(engine.content_tokens(body))
    marks = frozenset(engine.number_re.findall(body)) | frozenset(
        re.findall(r"\(([a-e])\)", body)
    )
    return tokens, marks


def _merge_mirrors(items: list[_Candidate]) -> tuple[list[_Candidate], list[dict[str, Any]]]:
    """Collapse criteria that check the same fact, across categories.

    Cross-category is the point: a Pitfall phrased as the avoidance of the very
    thing an Essential item demands is one measurement wearing two hats, and
    scoring it twice biases the rubric toward whichever fact happened to get
    mirrored. Comparison is on polarity-stripped text so the two look alike, and
    a pair is only merged when its numbers and option labels also agree — the
    forensics' own correction (§1.3), without which "every 3 months" and "every
    6 months" would be merged into one.
    """
    keys = [_mirror_key(c) for c in items]
    absorbed: set[int] = set()
    merges: list[dict[str, Any]] = []
    for i in range(len(items)):
        if i in absorbed:
            continue
        for j in range(i + 1, len(items)):
            if j in absorbed:
                continue
            # Re-read both slots each time: a merge earlier in this inner loop
            # may have moved the survivor into position i.
            left, right = items[i], items[j]
            tokens_l, marks_l = keys[i]
            tokens_r, marks_r = keys[j]
            if not tokens_l or not tokens_r:
                continue
            if (marks_l or marks_r) and marks_l != marks_r:
                continue
            union = tokens_l | tokens_r
            similarity = len(tokens_l & tokens_r) / len(union) if union else 0.0
            if similarity < _MIRROR_JACCARD:
                continue
            keeper, loser = _pick_mirror_survivor(left, right)
            merges.append(
                {
                    "kept": keeper.cid,
                    "absorbed": loser.cid,
                    "similarity": round(similarity, 3),
                    "cross_category": left.category is not right.category,
                    "kept_text": keeper.description,
                    "absorbed_text": loser.description,
                }
            )
            _absorb(keeper, loser)
            if keeper is right:
                items[i], items[j] = right, left
                keys[i], keys[j] = keys[j], keys[i]
            absorbed.add(j)
    return [c for k, c in enumerate(items) if k not in absorbed], merges


def _pick_mirror_survivor(left: _Candidate, right: _Candidate) -> tuple[_Candidate, _Candidate]:
    """Keep the item that demands content over the one that forbids a mistake."""
    left_is_pitfall = left.category is Category.PITFALL
    right_is_pitfall = right.category is Category.PITFALL
    if left_is_pitfall != right_is_pitfall:
        return (right, left) if left_is_pitfall else (left, right)
    return (left, right) if _ascending_value(left) >= _ascending_value(right) else (right, left)


def _absorb(keeper: _Candidate, other: _Candidate) -> None:
    merged = list(keeper.provenance.get("merged_mirror_ids") or [])
    merged.append(other.cid)
    keeper.provenance["merged_mirror_ids"] = merged
    keeper.provenance["merged_mirror_text"] = other.description
    if keeper.category is not Category.PITFALL:
        keeper.weight = _clamp_weight(
            max(abs(keeper.weight), abs(other.weight)), keeper.category
        )


def _rebalance_pitfalls(
    items: list[_Candidate], pool: Sequence[_Candidate], max_fraction: float = 0.25
) -> list[_Candidate]:
    """Cap the Pitfall share of a finished rubric.

    Pitfall criteria are satisfied by silence, so a checklist dominated by them
    awards an empty response nearly full marks. The bias is structural rather
    than accidental: the negatives are built from the mined pitfalls, so pitfall
    criteria measure as the most discriminative. Excess pitfalls are swapped for
    the best unused positive candidates, never simply deleted, so the rubric
    keeps its size.
    """
    max_pitfalls = max(1, int(len(items) * max_fraction))
    pitfalls = [c for c in items if c.category is Category.PITFALL]
    if len(pitfalls) <= max_pitfalls:
        return items

    present = {(c.title.lower(), c.description.lower()) for c in items}
    spare = sorted(
        (
            c
            for c in pool
            if c.category is not Category.PITFALL
            and (c.title.lower(), c.description.lower()) not in present
            and not polarity_offence(c.description, c.category)
        ),
        key=_ascending_value,
        reverse=True,
    )
    n_swap = min(len(pitfalls) - max_pitfalls, len(spare))
    if n_swap <= 0:
        return items

    dropped = {id(c) for c in sorted(pitfalls, key=_ascending_value)[:n_swap]}
    return [c for c in items if id(c) not in dropped] + spare[:n_swap]


def _clamp_weight(weight: int, category: Category) -> int:
    if category is Category.PITFALL:
        return max(-2, min(-1, -abs(weight) if weight else -1))
    return max(1, min(5, abs(weight) or 3))


def _parse_verdicts(raw: Any, cids: Sequence[int]) -> dict[int, dict[str, Any]]:
    """Map a batched judge reply onto criterion ids, tolerating id drift."""
    entries = [_as_dict(e) for e in _as_list(raw)]
    entries = [e for e in entries if e]
    if not entries:
        return {}

    valid = set(cids)
    reported: list[int | None] = [_as_int(e.get("id")) for e in entries]
    # Some models number from zero even when told otherwise; detect and shift.
    seen_ids = {r for r in reported if r is not None}
    if seen_ids and seen_ids.isdisjoint(valid) and {r + 1 for r in seen_ids} <= valid:
        reported = [None if r is None else r + 1 for r in reported]

    out: dict[int, dict[str, Any]] = {}
    for position, (entry, cid) in enumerate(zip(entries, reported)):
        target = cid if cid in valid else (cids[position] if position < len(cids) else None)
        if target is None or target in out:
            continue
        out[target] = entry
    return out


def _wrong_rollouts(run: _Run) -> list[tuple[_Rollout, str]]:
    """Rollouts the reconcile stage judged wrong, worst first."""
    if not run.reconciliation:
        return []
    by_index = {r.index: r for r in run.rollouts}
    scored: list[tuple[int, _Rollout, str]] = []
    for entry in _as_list(run.reconciliation.get("rollout_verdicts")):
        data = _as_dict(entry)
        index = _as_int(data.get("rollout") if "rollout" in data else data.get("index"))
        if index is None:
            continue
        rollout = by_index.get(index) or by_index.get(index + 1)
        if rollout is None:
            continue
        verdict = _as_text(data.get("verdict"), limit=60).lower()
        if "incorrect" in verdict or verdict == "wrong":
            rank = 0
        elif "partial" in verdict:
            rank = 1
        else:
            continue
        errors = [_as_text(e, limit=300) for e in _as_list(data.get("errors"))]
        note = "; ".join(e for e in errors if e)[:600] or f"judged {verdict} against the reference"
        scored.append((rank, rollout, note))
    scored.sort(key=lambda item: (item[0], item[1].index))
    return [(rollout, note) for _, rollout, note in scored]


def _ascending_value(cand: _Candidate) -> tuple[float, int, int]:
    """Sort key for pruning: least defensible criteria first."""
    val = cand.validation or {}
    discrimination = float(val.get("discrimination") or 0.0)
    return (
        discrimination,
        _EVIDENCE_RANK.get(cand.evidence, 0),
        abs(int(cand.weight)),
    )


def _calibration_note(run: _Run) -> str:
    if not run.summary.get("critic_ran"):
        return (
            "The validation stage was disabled for this run, so no discrimination "
            "measurements are attached. Judge each candidate on its own merits."
        )
    negatives = ", ".join(f"{n.nid} ({n.kind})" for n in run.negatives) or "none"
    return (
        f"Each candidate was tested against the reference answer and against "
        f"{len(run.negatives)} deliberately flawed responses [{negatives}]. "
        f"{run.summary.get('n_dropped_gold_fail', 0)} candidates were already deleted for failing "
        f"the reference answer and {run.summary.get('n_dropped_contradicted', 0)} for being "
        "contradicted by it."
    )


def _merge_calibrated(entries: Sequence[Any], survivors: Sequence[_Candidate]) -> list[_Candidate]:
    """Rebuild final candidates, carrying provenance/validation across the merge."""
    by_cid = {c.cid: c for c in survivors}
    by_title = {c.title.strip().lower(): c for c in survivors if c.title}
    out: list[_Candidate] = []

    for entry in entries:
        data = _as_dict(entry)
        description = _as_text(data.get("description") or data.get("criterion"), limit=2000)
        if not description:
            continue
        inline = parse_category_prefix(description)
        if inline is not None:
            description = strip_category_prefix(description)
        category = inline or Category.coerce(data.get("category"))
        title = _as_text(data.get("title"), limit=120) or _derive_title(description)

        source_ids = [i for i in (_as_int(v) for v in _as_list(data.get("source_ids"))) if i is not None]
        sources = [by_cid[i] for i in source_ids if i in by_cid]
        if not sources:
            match = by_title.get(title.strip().lower())
            if match is not None:
                sources = [match]

        weight = _as_int(data.get("weight"))
        if weight is None:
            weight = sources[0].weight if sources else (-1 if category is Category.PITFALL else 3)

        provenance, validation = _merge_provenance(sources)
        provenance["stage"] = "calibrate"
        rationale = _as_text(data.get("rationale"), limit=400)
        if rationale:
            provenance["calibration_rationale"] = rationale
        if not sources:
            provenance.setdefault("evidence", "other")
            provenance["unmatched_source"] = True

        out.append(
            _Candidate(
                cid=len(out) + 1,
                title=title,
                description=description,
                category=category,
                weight=_clamp_weight(weight, category),
                provenance=provenance,
                validation=validation,
            )
        )
    return out


def _merge_provenance(
    sources: Sequence[_Candidate],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    if not sources:
        return {}, None
    primary = max(sources, key=_ascending_value)
    provenance = dict(primary.provenance)
    provenance["source_ids"] = [c.cid for c in sources]
    evidences = sorted({c.evidence for c in sources if c.evidence != "other"})
    if evidences:
        provenance["evidence"] = max(evidences, key=lambda e: _EVIDENCE_RANK.get(e, 0))
        if len(evidences) > 1:
            provenance["evidence_merged"] = evidences
    if primary.validation is None:
        return provenance, None

    validation = dict(primary.validation)
    if len(sources) > 1:
        validation["merged_from"] = [
            {
                "id": c.cid,
                "discrimination": (c.validation or {}).get("discrimination"),
                "decision": (c.validation or {}).get("decision"),
            }
            for c in sources
        ]
    return provenance, validation


def _mean_anchor_hits(items: Sequence[Criterion], anchors: frozenset[str]) -> float | None:
    """Mean anchors per criterion under the generation-time proxy.

    Reported for monitoring only; the corpus-level figure the forensics quotes
    (1.78 science / 2.07 medicine) comes from the evaluation path, which uses the
    real rare-token vocabulary. The two are not comparable.
    """
    if not items or not anchors:
        return None
    hits = [len(_anchor_hits(c.description, anchors)) for c in items]
    return round(sum(hits) / len(hits), 3)


def _mean_discrimination(items: Sequence[Criterion]) -> float | None:
    scores = [
        float(c.validation["discrimination"])
        for c in items
        if c.validation and isinstance(c.validation.get("discrimination"), (int, float))
    ]
    return round(sum(scores) / len(scores), 4) if scores else None
