"""Criterion inspection tools: specificity, redundancy, corpus portability. No LLM calls.

The forensics found that roughly a quarter of the reward mass in the shipped
rubrics sits on criteria that would apply just as well to a different question
(``docs/01_data_forensics.md`` F3), and that asking a model in the prompt to be
specific does not fix it. These tools let the drafting agent measure the
property instead of promising it, using the *same detectors the evaluation uses*
so a criterion that passes here is not merely passing a friendlier test.

The corpus-portability check is the generation-time analogue of the permutation
control: a criterion that also fires on unrelated questions is recyclable, and
recyclable criteria are the ones that carry no signal.
"""

from __future__ import annotations

import functools
import logging
import random
import re
from typing import Any, Sequence

from .base import Tool, ToolContext, ToolResult, as_str_list

logger = logging.getLogger(__name__)

__all__ = ["CheckSpecificityTool", "FindSimilarQuestionsTool", "clear_corpus_cache"]

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-'\.]*")


# ---------------------------------------------------------------------------
# Detector access
#
# The detectors live in ``harness.generators.agentic``, which imports this
# package — so they are imported lazily, inside the call, to keep the module
# graph acyclic.
# ---------------------------------------------------------------------------


def _detectors() -> dict[str, Any]:
    from ..generators import agentic as A  # noqa: PLC0415 - deliberate late import
    from ..schema import Category  # noqa: PLC0415

    return {
        "question_anchors": A.question_anchors,
        "anchor_hits": A._anchor_hits,
        "subjective_hits": A.subjective_hits,
        "is_style_criterion": A.is_style_criterion,
        # Every criterion this pipeline emits is a positive requirement, so the
        # phrasing check is run against that contract.
        "polarity_offence": lambda body: A.polarity_offence(body, Category.IMPORTANT),
        "strip_prefix": A.strip_category_prefix,
    }


# ---------------------------------------------------------------------------
# Question corpus (for portability and retrieval)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=4)
def _corpus(domain: str, limit: int) -> tuple[tuple[str, str], ...]:
    """``(uid, question)`` pairs drawn from the domain's training split.

    Cached per (domain, limit). Returns empty on any failure — the tools degrade
    to their non-corpus checks rather than breaking generation.
    """
    try:
        from ..data import load_split  # noqa: PLC0415 - optional at generation time

        frame = load_split(domain, "train")
        questions = (
            frame["question"].astype(str).str.strip().drop_duplicates().tolist()
        )
        rng = random.Random(f"corpus:{domain}")
        if len(questions) > limit:
            questions = rng.sample(questions, limit)
        return tuple((f"{domain}#{i}", q) for i, q in enumerate(questions))
    except Exception as exc:  # noqa: BLE001 - corpus is an enhancement, not a dependency
        logger.warning("tools: question corpus unavailable for %s (%s)", domain, exc)
        return ()


def clear_corpus_cache() -> None:
    _corpus.cache_clear()


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD_RE.findall((text or "").lower()) if len(t) > 2}


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# check_specificity
# ---------------------------------------------------------------------------


class CheckSpecificityTool(Tool):
    name = "check_specificity"
    description = """
    Audit draft criteria before you commit to them. For each one it reports:
    which question-specific terms it actually mentions, whether it leans on
    subjective words ('clearly', 'thorough') that a judge cannot check
    objectively, whether it is really a style rule, whether its wording is
    negatively phrased, and — the important one — how many UNRELATED questions
    from the corpus it would apply to just as well. A criterion that fits other
    questions carries no information about this one. It also flags near-duplicate
    pairs within the batch. Call this on your full draft list and rewrite
    whatever it flags.
    """
    parameters = {
        "type": "object",
        "properties": {
            "criteria": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Criterion descriptions to audit, in full sentences.",
            },
            "n_probe_questions": {
                "type": "integer",
                "description": "How many unrelated questions to test portability against. Default 40.",
            },
        },
        "required": ["criteria"],
    }

    def __init__(self, *, corpus_limit: int = 4000, default_probe: int = 40) -> None:
        self.corpus_limit = corpus_limit
        self.default_probe = default_probe

    async def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        criteria = as_str_list(kwargs.get("criteria"))
        if not criteria:
            return ToolResult.failure("'criteria' must be a non-empty array of strings")
        try:
            n_probe = max(0, int(kwargs.get("n_probe_questions", self.default_probe)))
        except (TypeError, ValueError):
            n_probe = self.default_probe

        det = _detectors()
        anchors = det["question_anchors"](ctx.question, ctx.reference_answer)
        probes = self._probe_questions(ctx, n_probe)

        reports: list[dict[str, Any]] = []
        token_sets: list[set[str]] = []
        for index, text in enumerate(criteria):
            body = det["strip_prefix"](text)
            hits = sorted(det["anchor_hits"](body, anchors))
            subjective = det["subjective_hits"](body)
            offence = det["polarity_offence"](body)
            portable = self._portability(body, probes, anchors)
            token_sets.append(_tokens(body))

            problems: list[str] = []
            if len(hits) < 2:
                problems.append(
                    "mentions fewer than two question-specific terms — it is probably generic"
                )
            if subjective:
                problems.append(f"subjective wording: {', '.join(sorted(set(subjective))[:5])}")
            if det["is_style_criterion"](body):
                problems.append("this is a presentation/style rule, not a substantive check")
            if portable["portable_fraction"] >= 0.25 and probes:
                problems.append(
                    f"would apply to {portable['portable_count']}/{len(probes)} unrelated "
                    "questions — rewrite it around this question's own quantities"
                )
            reports.append(
                {
                    "index": index,
                    "anchors_mentioned": hits[:12],
                    "n_anchors": len(hits),
                    "subjective_words": sorted(set(subjective))[:8],
                    "is_style_rule": bool(det["is_style_criterion"](body)),
                    "portable_to_other_questions": portable["portable_count"],
                    "n_probe_questions": len(probes),
                    "problems": problems,
                    "verdict": "revise" if problems else "ok",
                }
            )
            if offence:
                reports[-1]["polarity_problem"] = offence

        duplicates = _near_duplicates(criteria, token_sets)
        n_revise = sum(1 for r in reports if r["verdict"] == "revise")
        return ToolResult(
            ok=True,
            data={
                "n_criteria": len(criteria),
                "n_needing_revision": n_revise,
                "question_anchor_vocabulary": sorted(anchors)[:30],
                "criteria": reports,
                "near_duplicate_pairs": duplicates,
                "summary": (
                    "all criteria look question-specific and objectively checkable"
                    if not n_revise and not duplicates
                    else f"{n_revise} of {len(criteria)} need revision"
                    + (f"; {len(duplicates)} near-duplicate pair(s)" if duplicates else "")
                ),
            },
            meta={"n_probe": len(probes), "uid": ctx.uid},
        )

    def _probe_questions(self, ctx: ToolContext, n_probe: int) -> list[str]:
        if n_probe <= 0:
            return []
        pool = _corpus(ctx.domain or "rar_science", self.corpus_limit)
        if not pool:
            return []
        own = (ctx.question or "").strip()
        rng = random.Random(f"probe:{ctx.uid}")
        picks = rng.sample(pool, min(n_probe, len(pool)))
        return [q for _, q in picks if q.strip() != own]

    @staticmethod
    def _portability(body: str, probes: Sequence[str], anchors: frozenset[str]) -> dict[str, Any]:
        """How many unrelated questions this criterion could be attached to.

        A criterion is counted portable against a probe question when none of
        the *instance-specific* terms it uses are absent from that question —
        i.e. everything it names is generic enough to be present anyway. With no
        anchors at all it is portable by definition.
        """
        if not probes:
            return {"portable_count": 0, "portable_fraction": 0.0}
        body_tokens = _tokens(body)
        specific = {t for t in body_tokens if t in anchors}
        if not specific:
            return {"portable_count": len(probes), "portable_fraction": 1.0}
        count = sum(1 for q in probes if specific <= _tokens(q))
        return {"portable_count": count, "portable_fraction": count / len(probes)}


def _near_duplicates(
    criteria: Sequence[str], token_sets: Sequence[set[str]], threshold: float = 0.6
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i in range(len(criteria)):
        for j in range(i + 1, len(criteria)):
            score = _jaccard(token_sets[i], token_sets[j])
            if score >= threshold:
                out.append(
                    {
                        "a": i,
                        "b": j,
                        "similarity": round(score, 3),
                        "hint": "merge these or make each check a different thing",
                    }
                )
    return out


# ---------------------------------------------------------------------------
# find_similar_questions
# ---------------------------------------------------------------------------


class FindSimilarQuestionsTool(Tool):
    name = "find_similar_questions"
    description = """
    Retrieve questions from the dataset that resemble a given text. Two uses:
    see what a near-neighbour question looks like so you can tell which parts of
    this one are actually distinctive, and sanity-check a draft criterion by
    reading it against a similar-but-different problem — if it fits that one
    too, it is not specific enough.
    """
    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Text to match against — a question, or a criterion you are drafting.",
            },
            "k": {"type": "integer", "description": "How many neighbours to return. Default 5, max 15."},
            "exclude_self": {
                "type": "boolean",
                "description": "Drop the current question from the results. Default true.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, *, corpus_limit: int = 4000, excerpt_chars: int = 400) -> None:
        self.corpus_limit = corpus_limit
        self.excerpt_chars = excerpt_chars

    async def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        query = str(kwargs.get("query") or "").strip()
        if not query:
            return ToolResult.failure("'query' must be non-empty")
        try:
            k = min(15, max(1, int(kwargs.get("k", 5))))
        except (TypeError, ValueError):
            k = 5
        exclude_self = kwargs.get("exclude_self", True) is not False

        pool = _corpus(ctx.domain or "rar_science", self.corpus_limit)
        if not pool:
            return ToolResult(
                ok=True,
                data={"neighbours": [], "note": "the question corpus is unavailable in this run"},
            )

        own = (ctx.question or "").strip()
        query_tokens = _tokens(query)
        scored: list[tuple[float, str, str]] = []
        for uid, question in pool:
            if exclude_self and question.strip() == own:
                continue
            score = _jaccard(query_tokens, _tokens(question))
            if score > 0:
                scored.append((score, uid, question))
        scored.sort(key=lambda t: -t[0])

        neighbours = [
            {
                "uid": uid,
                "similarity": round(score, 3),
                "question": question[: self.excerpt_chars]
                + ("…" if len(question) > self.excerpt_chars else ""),
            }
            for score, uid, question in scored[:k]
        ]
        return ToolResult(
            ok=True,
            data={
                "n_corpus": len(pool),
                "neighbours": neighbours,
                "note": "similarity is lexical (Jaccard over content words), not semantic",
            },
            meta={"domain": ctx.domain, "k": k},
        )
