"""Generator interface. Every rubric source implements :class:`RubricGenerator`.

Contract: ``generate`` returns a fully-populated :class:`~harness.schema.Rubric`
plus a JSON-serialisable trace. Failures must be *soft* — a generator that
cannot complete should return its best partial rubric and record the problem in
``meta['error']`` rather than raising, so one bad row never kills a run.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field
from typing import Any

from ..llm import LLMEngine
from ..schema import Example, Rubric

__all__ = ["GenerationResult", "RubricGenerator", "REGISTRY", "register", "build_generator"]


@dataclass
class GenerationResult:
    uid: str
    source: str
    rubric: Rubric
    trace: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    n_llm_calls: int = 0
    wall_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid,
            "source": self.source,
            "rubric": self.rubric.to_dict(),
            "error": self.error,
            "n_llm_calls": self.n_llm_calls,
            "wall_seconds": round(self.wall_seconds, 2),
        }


class RubricGenerator(abc.ABC):
    """Produces a :class:`Rubric` for one :class:`Example`."""

    name: str = "base"

    def __init__(self, engine: LLMEngine | None = None, **kwargs: Any) -> None:
        self.engine = engine
        self.options = kwargs

    @abc.abstractmethod
    async def generate(self, example: Example) -> GenerationResult:
        ...


REGISTRY: dict[str, type[RubricGenerator]] = {}


def register(cls: type[RubricGenerator]) -> type[RubricGenerator]:
    REGISTRY[cls.name] = cls
    return cls


def build_generator(name: str, engine: LLMEngine | None = None, **kwargs: Any) -> RubricGenerator:
    if name not in REGISTRY:
        raise KeyError(f"unknown generator {name!r}; known: {sorted(REGISTRY)}")
    return REGISTRY[name](engine=engine, **kwargs)
