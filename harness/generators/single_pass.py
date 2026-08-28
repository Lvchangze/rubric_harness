"""Single-pass RaR baseline: one call with the paper's verbatim prompt.

This is the method the paper describes, reproduced without embellishment, so
that the agentic pipeline is compared against RaR itself rather than against a
weakened paraphrase of it. The only liberties taken are operational: JSON is
recovered defensively and failures are soft.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from ..llm import JSONParseError, LLMEngine
from ..prompts.rar_original import build_rar_user_prompt, system_for_domain
from ..schema import Criterion, Example, Rubric
from .base import GenerationResult, RubricGenerator, register

logger = logging.getLogger(__name__)

__all__ = ["SinglePassGenerator"]

#: The paper's prompt permits up to 20 items with long descriptions; a reasoning
#: model needs headroom well beyond that before the visible answer starts.
DEFAULT_MAX_TOKENS = 16384


@register
class SinglePassGenerator(RubricGenerator):
    """Faithful reproduction of the RaR synthetic-rubric prompt."""

    name = "baseline"

    def __init__(
        self,
        engine: LLMEngine | None = None,
        *,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        temperature: float | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(engine=engine, **kwargs)
        self.max_tokens = max(DEFAULT_MAX_TOKENS, int(max_tokens))
        self.temperature = temperature

    async def generate(self, example: Example) -> GenerationResult:
        started = time.perf_counter()
        system = system_for_domain(example.domain)
        user = build_rar_user_prompt(example.question, example.reference_answer)
        trace: dict[str, Any] = {
            "stages": [{"stage": "single_pass", "input": {"system": system, "user": user}}]
        }

        def finish(rubric: Rubric, error: str | None, n_calls: int) -> GenerationResult:
            rubric.meta.setdefault("source", "baseline")
            rubric.meta["n_items"] = len(rubric)
            if error:
                rubric.meta["error"] = error
            return GenerationResult(
                uid=example.uid,
                source=self.name,
                rubric=rubric,
                trace=trace,
                error=error,
                n_llm_calls=n_calls,
                wall_seconds=time.perf_counter() - started,
            )

        if self.engine is None:
            return finish(Rubric(meta={"source": "baseline"}), "no LLM engine configured", 0)

        try:
            raw = await self.engine.chat_json(
                user,
                system=system,
                expect="array",
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                tag="gen:baseline",
            )
        except (JSONParseError, RuntimeError) as exc:
            message = f"{type(exc).__name__}: {exc}"[:500]
            trace["stages"][0]["error"] = message
            logger.warning("baseline generation failed uid=%s: %s", example.uid, message)
            return finish(Rubric(meta={"source": "baseline"}), message, 1)

        trace["stages"][0]["output"] = raw
        items = _parse_rar_array(raw)
        error = None if items else "model returned no usable rubric items"
        rubric = Rubric(items=items, meta={"source": "baseline"})
        return finish(rubric, error, 1)


def _parse_rar_array(raw: Any) -> list[Criterion]:
    """Turn the model's ``[{title, description, weight}, ...]`` into criteria.

    The category rides inline on ``description`` in the RaR schema and is
    normalised out by :class:`~harness.schema.Criterion`, so the entries pass
    straight through.
    """
    entries: list[Any]
    if isinstance(raw, list):
        entries = raw
    elif isinstance(raw, dict):
        entries = next(
            (v for v in raw.values() if isinstance(v, list)),
            [raw] if "description" in raw else [],
        )
    else:
        return []

    items: list[Criterion] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        description = str(entry.get("description") or "").strip()
        if not description:
            continue
        items.append(
            Criterion(
                title=str(entry.get("title") or "").strip(),
                description=description,
                weight=entry.get("weight", 3),
                category=entry.get("category") or "Important",
                provenance={"stage": "single_pass"},
            )
        )
    return items
