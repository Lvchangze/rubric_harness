"""Pass-through generator for the rubric that ships with the RaR dataset.

This is the control condition: no model is called, so any difference against it
is attributable to rubric synthesis rather than to the endpoint being used.
"""

from __future__ import annotations

import copy
import time
from typing import Any

from ..llm import LLMEngine
from ..schema import Example
from .base import GenerationResult, RubricGenerator, register

__all__ = ["ShippedRubricGenerator"]


@register
class ShippedRubricGenerator(RubricGenerator):
    """Return ``example.shipped_rubric`` verbatim. Makes zero LLM calls."""

    name = "shipped"

    def __init__(self, engine: LLMEngine | None = None, **kwargs: Any) -> None:
        super().__init__(engine=engine, **kwargs)

    async def generate(self, example: Example) -> GenerationResult:
        started = time.perf_counter()
        # Deep copy: downstream stages mutate weights and categories in place
        # during ablations, and the Example is shared across generators.
        rubric = copy.deepcopy(example.shipped_rubric)
        rubric.meta = dict(rubric.meta)
        rubric.meta["source"] = "shipped"
        rubric.meta["n_items"] = len(rubric)

        error = "shipped rubric is empty" if not len(rubric) else None
        return GenerationResult(
            uid=example.uid,
            source=self.name,
            rubric=rubric,
            trace={"stages": [], "note": "no generation performed; dataset rubric returned as-is"},
            error=error,
            n_llm_calls=0,
            wall_seconds=time.perf_counter() - started,
        )
