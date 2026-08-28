"""Evaluation suite: one shared judge plus the metric families built on it.

Only the judge is re-exported here. The metric modules are imported lazily by
``scripts/eval_rubrics.py`` so that a broken or missing metric cannot stop the
others from running.
"""

from .judge import (
    AggregationMode,
    CriterionVerdict,
    JudgeResult,
    PolarityMode,
    aggregate,
    judge_many,
    judge_rubric,
    judge_rubric_implicit,
)

__all__ = [
    "AggregationMode",
    "CriterionVerdict",
    "JudgeResult",
    "PolarityMode",
    "aggregate",
    "judge_many",
    "judge_rubric",
    "judge_rubric_implicit",
]
