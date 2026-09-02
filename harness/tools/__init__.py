"""Executable tools the rubric-drafting agent can call.

Seven capabilities, in two groups.

Deterministic (no LLM, reproducible from the repository alone):

* ``python_eval``            — sandboxed Python for arithmetic and derivations
* ``check_equivalence``      — symbolic/numeric equality of two expressions
* ``check_units``            — dimensional analysis and unit matching
* ``check_specificity``      — anchors, subjective wording, redundancy, and how
                               many unrelated questions a criterion also fits
* ``find_similar_questions`` — nearest neighbours from the dataset

Model-backed:

* ``extract_reference_claims`` — the atomic content a complete answer must have
* ``execute_criterion``        — run criteria against real texts and report
                                 whether they fire
* ``make_counterexample``      — build a wrong answer, verified to be wrong

Network-backed, opt-in (:data:`WEB_TOOLS`, not in :data:`DEFAULT_TOOLS`):

* ``web_search``               — DuckDuckGo, no API key
* ``wikipedia_lookup``         — article summaries, faster and more reliable
                                 than search for textbook facts

The split matters for interpreting results. Five of the seven cost nothing and
are exactly reproducible; the claim that a criterion is question-specific rests
on those, not on the model's assurance.
"""

from __future__ import annotations

from typing import Sequence

from .base import Tool, ToolContext, ToolInvocation, ToolRegistry, ToolResult
from .compute import CheckEquivalenceTool, CheckUnitsTool, PythonEvalTool
from .inspection import CheckSpecificityTool, FindSimilarQuestionsTool, clear_corpus_cache
from .negatives import FLAW_TYPES, MakeCounterexampleTool
from .verification import ExecuteCriterionTool, ExtractReferenceClaimsTool, resolve_targets
from .websearch import WebSearchTool, WikipediaTool, search_cache_stats

__all__ = [
    "Tool",
    "ToolContext",
    "ToolInvocation",
    "ToolRegistry",
    "ToolResult",
    "PythonEvalTool",
    "CheckEquivalenceTool",
    "CheckUnitsTool",
    "CheckSpecificityTool",
    "FindSimilarQuestionsTool",
    "ExtractReferenceClaimsTool",
    "ExecuteCriterionTool",
    "MakeCounterexampleTool",
    "WebSearchTool",
    "WikipediaTool",
    "FLAW_TYPES",
    "DEFAULT_TOOLS",
    "ZERO_LLM_TOOLS",
    "WEB_TOOLS",
    "build_registry",
    "resolve_targets",
    "clear_corpus_cache",
    "search_cache_stats",
]

#: Registration order is also the order the model sees, so the cheap
#: deterministic checks are listed before the ones that cost a call.
DEFAULT_TOOLS: tuple[str, ...] = (
    "python_eval",
    "check_equivalence",
    "check_units",
    "check_specificity",
    "find_similar_questions",
    "extract_reference_claims",
    "execute_criterion",
    "make_counterexample",
)

ZERO_LLM_TOOLS: tuple[str, ...] = (
    "python_eval",
    "check_equivalence",
    "check_units",
    "check_specificity",
    "find_similar_questions",
)

#: Outbound network. Deliberately **not** in :data:`DEFAULT_TOOLS`: putting them
#: there would silently redefine what `agentic-tools` means and invalidate the
#: comparison against the results already recorded for that arm. Ask for them by
#: name, and give the new arm a new name.
WEB_TOOLS: tuple[str, ...] = (
    "web_search",
    "wikipedia_lookup",
)

_FACTORIES = {
    "python_eval": PythonEvalTool,
    "check_equivalence": CheckEquivalenceTool,
    "check_units": CheckUnitsTool,
    "check_specificity": CheckSpecificityTool,
    "find_similar_questions": FindSimilarQuestionsTool,
    "extract_reference_claims": ExtractReferenceClaimsTool,
    "execute_criterion": ExecuteCriterionTool,
    "make_counterexample": MakeCounterexampleTool,
    "web_search": WebSearchTool,
    "wikipedia_lookup": WikipediaTool,
}


def build_registry(
    names: Sequence[str] | None = None,
    *,
    default_timeout_s: float = 90.0,
    sandbox_timeout_s: float = 20.0,
    corpus_limit: int = 4000,
) -> ToolRegistry:
    """Assemble a registry.

    ``names`` defaults to :data:`DEFAULT_TOOLS`. Unknown names raise rather than
    being skipped — a config that asks for a tool which does not exist should
    fail loudly, not produce a run that silently lacks it.
    """
    wanted = list(names) if names is not None else list(DEFAULT_TOOLS)
    if unknown := [n for n in wanted if n not in _FACTORIES]:
        raise KeyError(f"unknown tool(s) {unknown}; known: {sorted(_FACTORIES)}")

    registry = ToolRegistry(default_timeout_s=default_timeout_s)
    for name in wanted:
        if name == "python_eval":
            registry.add(PythonEvalTool(timeout_s=sandbox_timeout_s))
        elif name == "check_specificity":
            registry.add(CheckSpecificityTool(corpus_limit=corpus_limit))
        elif name == "find_similar_questions":
            registry.add(FindSimilarQuestionsTool(corpus_limit=corpus_limit))
        else:
            registry.add(_FACTORIES[name]())
    return registry
