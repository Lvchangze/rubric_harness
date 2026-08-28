"""The shared LLM judge — the single point of contact between rubrics and scores.

Fairness is the whole point of this module. Every rubric source is scored by the
*same* judge model, with the *same* prompt, on the *same* responses. The only
thing that varies is the criteria themselves. Three controls guard against
artefacts:

* **Order randomisation.** Criteria are shuffled with a seed derived from
  ``(uid, response_id)`` — *not* from the rubric source — so every source sees an
  equally arbitrary ordering and no source benefits from position bias.
* **Source blinding.** The judge prompt never names the rubric's provenance.
* **Polarity separation.** The judge is asked only whether each criterion is
  *literally true* of the response. Whether "true" means good or bad is applied
  afterwards from :class:`~harness.schema.Polarity`. This matters because the two
  RaR corpora use opposite Pitfall conventions (``docs/01_data_forensics.md``
  F6): RaR-Science writes "Avoids X" (true = good) and RaR-Medicine writes "Does
  not mention X" (true = bad), yet both carry weight -1/-2. Any single blanket
  convention silently scores one whole domain backwards, which would hand the
  agentic method a win it did not earn. Asking the judge a polarity-neutral
  question removes the ambiguity from the model entirely.

The paper gives no per-criterion judge prompt for RaR-EXPLICIT (Appendix A.6
covers only IMPLICIT / DIRECT-LIKERT / REFERENCE-LIKERT, see
``docs/00_paper_notes.md`` §4.3), so :data:`JUDGE_SYSTEM` is ours; the RaR
IMPLICIT prompt is reproduced verbatim in :mod:`harness.prompts.rar_original`
and available through :func:`judge_rubric_implicit`.

Aggregation follows RaR Eq. (1), ``r = Σ w_j c_j / Σ w_j``, under three
selectable weightings (see :class:`AggregationMode`).
"""

from __future__ import annotations

import asyncio
import logging
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Sequence

from ..llm import JSONParseError, LLMEngine
from ..schema import Category, Criterion, Polarity, Rubric

logger = logging.getLogger(__name__)

__all__ = [
    "AggregationMode",
    "PolarityMode",
    "CriterionVerdict",
    "JudgeResult",
    "judge_rubric",
    "judge_many",
    "aggregate",
    "CATEGORY_WEIGHTS",
]


class AggregationMode(str, Enum):
    """How Eq. (1) weights each criterion.

    ``PAPER_EXPLICIT`` is the primary result because it is what RaR-EXPLICIT
    actually trains with: the numeric weights in the data are discarded and the
    category label is remapped (``docs/00_paper_notes.md`` §4.1). The other two
    are robustness checks — the forensics found the shipped numeric weights carry
    almost no information beyond the category (conditional entropy 0.72/0.78 bit,
    Pearson 0.94 against a plain mean), so a result that only holds under one
    weighting is not a result.
    """

    PAPER_EXPLICIT = "paper_explicit"
    RAW_WEIGHT = "raw_weight"
    UNIFORM = "uniform"


class PolarityMode(str, Enum):
    """Which polarity convention to score under — the F6 sensitivity knob.

    ``DETECTED`` uses each criterion's own detected polarity and is correct.
    The other two force a single blanket convention across every criterion,
    reproducing the two ways a naive implementation can get F6 wrong; running all
    three is how the report shows the conclusion does not depend on the choice.
    ``FAVOURABLE`` gives a rubric the benefit of the doubt on every ambiguous
    Pitfall, which is what we grant the shipped rubrics so that any remaining
    agentic win is earned rather than an artefact of mis-scoring the comparator.
    """

    DETECTED = "detected"
    FAVOURABLE = "favourable"
    ALL_AVOIDANCE = "all_avoidance"   # every Pitfall read as "true = good"
    ALL_FAILURE = "all_failure"       # every Pitfall read as "true = bad"

    @classmethod
    def coerce_mode(cls, value: Any) -> "PolarityMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError:
            logger.warning("unknown polarity mode %r; falling back to 'detected'", value)
            return cls.DETECTED


# Paper-faithful categorical weights (RaR §4.4 / paper notes §4.1).
CATEGORY_WEIGHTS: dict[Category, float] = {
    Category.ESSENTIAL: 1.0,
    Category.IMPORTANT: 0.7,
    Category.OPTIONAL: 0.3,
    Category.PITFALL: 0.9,
}

JUDGE_SYSTEM = (
    "You are a meticulous, impartial grader. You are given a question, a candidate response, "
    "and a numbered list of statements about that response. For EACH statement, decide "
    "independently whether the statement is LITERALLY TRUE of the candidate response.\n\n"
    "Critical: you are NOT judging whether the response is good. You are only judging whether "
    "each statement accurately describes the response. Some statements describe desirable "
    "things and some describe mistakes; treat them identically and report only truth.\n\n"
    "Rules:\n"
    "- Judge ONLY the candidate response as written. Do not credit what it seems to intend.\n"
    "- A statement asserting the response contains something is true only if it actually does.\n"
    "- A statement asserting the response omits or lacks something is true only if it really is absent.\n"
    "- A statement asserting the response makes a specific error is true only if that error is present.\n"
    "- Judge each statement on its own terms; do not let one flaw cascade across statements.\n"
    "- Ignore formatting, length and eloquence unless a statement explicitly concerns them.\n"
    "- If a statement is too vague to check from the response alone, still give your best "
    'binary judgement and set "checkable": false.\n\n'
    "Output ONLY a JSON array, one object per statement, in the order given:\n"
    '[{"id": 1, "true": true, "checkable": true, "why": "<=20 words"}, ...]\n'
    "No prose outside the JSON array."
)


@dataclass
class CriterionVerdict:
    index: int                 # index into the original (unshuffled) rubric
    title: str
    category: Category
    weight: int
    polarity: Polarity
    literally_true: bool       # the judge's raw answer
    met: bool                  # polarity-corrected: True means good for the response
    checkable: bool = True
    why: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "title": self.title,
            "category": self.category.value,
            "weight": int(self.weight),
            "polarity": self.polarity.value,
            "literally_true": bool(self.literally_true),
            "met": bool(self.met),
            "checkable": bool(self.checkable),
            "why": self.why,
        }


@dataclass
class JudgeResult:
    uid: str
    rubric_source: str
    response_id: str
    verdicts: list[CriterionVerdict] = field(default_factory=list)
    score: float = 0.0             # primary: AggregationMode.PAPER_EXPLICIT
    score_numeric: float = 0.0     # AggregationMode.RAW_WEIGHT
    score_unweighted: float = 0.0  # AggregationMode.UNIFORM
    scores_by_polarity_mode: dict[str, float] = field(default_factory=dict)
    n_items: int = 0
    error: str | None = None

    def pass_rate(self) -> float:
        return self.score_unweighted

    def score_for(self, mode: AggregationMode) -> float:
        return {
            AggregationMode.PAPER_EXPLICIT: self.score,
            AggregationMode.RAW_WEIGHT: self.score_numeric,
            AggregationMode.UNIFORM: self.score_unweighted,
        }[mode]

    def to_dict(self, *, include_verdicts: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "uid": self.uid,
            "rubric_source": self.rubric_source,
            "response_id": self.response_id,
            "score": round(self.score, 6),
            "score_numeric": round(self.score_numeric, 6),
            "score_unweighted": round(self.score_unweighted, 6),
            "scores_by_polarity_mode": {
                k: round(v, 6) for k, v in self.scores_by_polarity_mode.items()
            },
            "n_items": self.n_items,
            "error": self.error,
        }
        if include_verdicts:
            out["verdicts"] = [v.to_dict() for v in self.verdicts]
        return out


def effective_polarity(criterion: Criterion, mode: PolarityMode) -> Polarity:
    """Polarity to score a criterion under, for the chosen convention."""
    if criterion.category is not Category.PITFALL:
        return Polarity.POSITIVE
    if mode is PolarityMode.ALL_AVOIDANCE:
        return Polarity.POSITIVE
    if mode is PolarityMode.ALL_FAILURE:
        return Polarity.NEGATIVE
    if mode is PolarityMode.FAVOURABLE:
        # Retained only so old configs still load. This mode was intended to give
        # comparator rubrics the benefit of the doubt on ambiguous Pitfalls, but
        # it cannot: `detect_polarity` already returns POSITIVE for
        # "unclassified", and "category_default" is only ever assigned to
        # non-Pitfall criteria, which return above. Every reachable branch
        # therefore agreed with DETECTED, and measurement confirmed it —
        # 0 of 12,484 criteria and a maximum absolute score difference of 0.0
        # across 18,849 judgements.
        #
        # A per-criterion "most favourable" reading is not merely unimplemented
        # but degenerate: choosing the polarity that makes each criterion count
        # as satisfied scores every response 1.0. Benefit of the doubt is only
        # meaningful one convention at a time over a whole rubric, which is what
        # ALL_AVOIDANCE / ALL_FAILURE provide and what the sensitivity analysis
        # selects between. Use those.
        logger.warning(
            "PolarityMode.FAVOURABLE is an alias for DETECTED; for a genuine "
            "best-case comparator reading use the ALL_AVOIDANCE / ALL_FAILURE "
            "sensitivity analysis instead"
        )
    return criterion.polarity or Polarity.POSITIVE


def aggregate(
    verdicts: Sequence[CriterionVerdict],
    mode: AggregationMode = AggregationMode.PAPER_EXPLICIT,
) -> float:
    """RaR Eq. (1) under one weighting. Weights are magnitudes: the good/bad
    direction lives in :attr:`CriterionVerdict.met`, never in the weight's sign,
    so the denominator cannot shrink the way the shipped negative weights make
    it shrink (forensics §9.4)."""
    if not verdicts:
        return 0.0
    numerator = denominator = 0.0
    for v in verdicts:
        if mode is AggregationMode.PAPER_EXPLICIT:
            w = CATEGORY_WEIGHTS.get(v.category, 0.7)
        elif mode is AggregationMode.RAW_WEIGHT:
            w = abs(float(v.weight)) or 1.0
        else:
            w = 1.0
        denominator += w
        if v.met:
            numerator += w
    return numerator / denominator if denominator else 0.0


def _render_checklist(criteria: Sequence[Criterion]) -> str:
    """Render criteria as neutral statements.

    The category prefix is dropped on purpose: "Pitfall Criteria:" is a hint
    about direction, and the judge is being asked only about literal truth. It
    is also the label whose meaning is inconsistent across the two corpora.
    """
    lines = []
    for i, c in enumerate(criteria, start=1):
        title = f" [{c.title}]" if c.title else ""
        lines.append(f"{i}.{title} {c.description}")
    return "\n".join(lines)


def build_judge_prompt(question: str, response: str, criteria: Sequence[Criterion]) -> str:
    return (
        f"<question>\n{question}\n</question>\n\n"
        f"<candidate_response>\n{response}\n</candidate_response>\n\n"
        f"<statements>\n{_render_checklist(criteria)}\n</statements>\n\n"
        f"Return the JSON array with exactly {len(criteria)} objects, ids 1..{len(criteria)}."
    )


async def judge_rubric(
    engine: LLMEngine,
    *,
    uid: str,
    question: str,
    response: str,
    rubric: Rubric,
    rubric_source: str,
    response_id: str,
    shuffle: bool = True,
    shuffle_seed: int = 7,
    max_tokens: int = 12288,
    repeat_salt: str | None = None,
    polarity_mode: PolarityMode = PolarityMode.DETECTED,
    extra_polarity_modes: Sequence[PolarityMode] = (),
) -> JudgeResult:
    """Score one (rubric, response) pair; never raises.

    Because the judge reports literal truth, alternative polarity conventions are
    recomputed from the *same* verdicts at zero extra cost — the F6 sensitivity
    analysis needs no additional LLM calls.

    ``repeat_salt`` forces a fresh sample from the judge for self-agreement
    measurement (it also changes the criterion ordering, which is what makes the
    repeat an honest stability probe rather than a cache lookup).
    """
    criteria = list(rubric.items)
    result = JudgeResult(
        uid=uid, rubric_source=rubric_source, response_id=response_id, n_items=len(criteria)
    )
    if not criteria:
        result.error = "empty rubric"
        return result

    order = list(range(len(criteria)))
    if shuffle:
        # Seed excludes rubric_source on purpose: ordering must not correlate
        # with which method produced the rubric.
        rng = random.Random(f"{shuffle_seed}:{uid}:{response_id}:{repeat_salt or ''}")
        rng.shuffle(order)
    shown = [criteria[i] for i in order]

    prompt = build_judge_prompt(question, response, shown)
    try:
        parsed = await engine.chat_json(
            prompt,
            system=JUDGE_SYSTEM,
            expect="array",
            max_tokens=max_tokens,
            tag="judge",
            cache_salt=repeat_salt,
        )
    except (JSONParseError, RuntimeError) as exc:
        result.error = f"judge failed: {exc}"[:400]
        logger.warning("judge failed uid=%s src=%s resp=%s: %s", uid, rubric_source, response_id, exc)
        return result

    by_id: dict[int, dict[str, Any]] = {}
    if isinstance(parsed, list):
        for pos, entry in enumerate(parsed, start=1):
            if not isinstance(entry, dict):
                continue
            try:
                idx = int(entry.get("id", pos))
            except (TypeError, ValueError):
                idx = pos
            by_id.setdefault(idx, entry)

    verdicts: list[CriterionVerdict] = []
    missing = 0
    for shown_pos, orig_idx in enumerate(order, start=1):
        c = criteria[orig_idx]
        entry = by_id.get(shown_pos)
        if entry is None:
            missing += 1
            literal, checkable, why = False, True, "no verdict returned"
        else:
            literal = _as_bool(entry.get("true", entry.get("met")))
            checkable = _as_bool(entry.get("checkable"), default=True)
            why = str(entry.get("why", ""))[:300]
        polarity = effective_polarity(c, polarity_mode)
        verdicts.append(
            CriterionVerdict(
                index=orig_idx,
                title=c.title,
                category=c.category,
                weight=c.weight,
                polarity=polarity,
                literally_true=literal,
                met=literal if polarity is Polarity.POSITIVE else not literal,
                checkable=checkable,
                why=why,
            )
        )
    verdicts.sort(key=lambda v: v.index)

    if missing == len(criteria):
        result.error = "judge returned no usable verdicts"
        return result
    if missing:
        result.error = f"{missing}/{len(criteria)} verdicts missing"

    result.verdicts = verdicts
    result.score = aggregate(verdicts, AggregationMode.PAPER_EXPLICIT)
    result.score_numeric = aggregate(verdicts, AggregationMode.RAW_WEIGHT)
    result.score_unweighted = aggregate(verdicts, AggregationMode.UNIFORM)

    for alt in extra_polarity_modes:
        alt_verdicts = _rescore_polarity(criteria, verdicts, alt)
        result.scores_by_polarity_mode[alt.value] = aggregate(
            alt_verdicts, AggregationMode.PAPER_EXPLICIT
        )
    return result


def _rescore_polarity(
    criteria: Sequence[Criterion],
    verdicts: Sequence[CriterionVerdict],
    mode: PolarityMode,
) -> list[CriterionVerdict]:
    """Re-derive ``met`` under a different convention, reusing the same verdicts."""
    out: list[CriterionVerdict] = []
    for v in verdicts:
        c = criteria[v.index]
        polarity = effective_polarity(c, mode)
        out.append(
            CriterionVerdict(
                index=v.index,
                title=v.title,
                category=v.category,
                weight=v.weight,
                polarity=polarity,
                literally_true=v.literally_true,
                met=v.literally_true if polarity is Polarity.POSITIVE else not v.literally_true,
                checkable=v.checkable,
                why=v.why,
            )
        )
    return out


def _as_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"true", "yes", "y", "1", "met", "satisfied", "pass"}:
            return True
        if text in {"false", "no", "n", "0", "unmet", "unsatisfied", "fail"}:
            return False
    return default


async def judge_many(
    engine: LLMEngine, jobs: Sequence[dict[str, Any]]
) -> list[JudgeResult]:
    """Fan out :func:`judge_rubric` over a list of kwargs dicts."""
    results = await asyncio.gather(
        *(judge_rubric(engine, **job) for job in jobs), return_exceptions=True
    )
    out: list[JudgeResult] = []
    for job, res in zip(jobs, results):
        if isinstance(res, BaseException):
            out.append(
                JudgeResult(
                    uid=job.get("uid", "?"),
                    rubric_source=job.get("rubric_source", "?"),
                    response_id=job.get("response_id", "?"),
                    error=f"exception: {res}"[:300],
                )
            )
        else:
            out.append(res)
    return out


async def judge_rubric_implicit(
    engine: LLMEngine,
    *,
    uid: str,
    question: str,
    response: str,
    rubric: Rubric,
    rubric_source: str,
    response_id: str,
    max_tokens: int = 4096,
) -> JudgeResult:
    """RaR-IMPLICIT scoring, using the paper's verbatim judge prompt.

    The paper's strongest variant: hand the judge every criterion at once and
    let it emit one 1-10 Likert rating, normalised to [0, 1]. Reported as a
    robustness check alongside the explicit per-criterion scores. Polarity is
    irrelevant here because the judge never reports per-criterion verdicts.
    """
    from ..prompts.rar_original import (  # noqa: PLC0415 - keeps the prompt module optional
        IMPLICIT_JUDGE_SYSTEM,
        build_implicit_judge_user,
    )

    result = JudgeResult(
        uid=uid, rubric_source=rubric_source, response_id=response_id, n_items=len(rubric)
    )
    if not len(rubric):
        result.error = "empty rubric"
        return result
    rubric_list = "\n".join(f"- {c.prefixed_description()}" for c in rubric.items)
    try:
        parsed = await engine.chat_json(
            build_implicit_judge_user(question, response, rubric_list),
            system=IMPLICIT_JUDGE_SYSTEM,
            expect="object",
            max_tokens=max_tokens,
            tag="judge:implicit",
        )
        rating = float(parsed.get("rating"))
    except (JSONParseError, RuntimeError, TypeError, ValueError) as exc:
        result.error = f"implicit judge failed: {exc}"[:300]
        return result
    rating = max(1.0, min(10.0, rating))
    result.score = result.score_numeric = result.score_unweighted = (rating - 1.0) / 9.0
    return result
