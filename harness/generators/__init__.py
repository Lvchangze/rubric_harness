"""Rubric generators. Importing this package registers every built-in source."""

from .base import GenerationResult, RubricGenerator, REGISTRY, build_generator, register
from . import shipped as _shipped  # noqa: F401 - import for registration side effect
from . import single_pass as _single_pass  # noqa: F401
from . import agentic as _agentic  # noqa: F401

__all__ = [
    "GenerationResult",
    "RubricGenerator",
    "REGISTRY",
    "build_generator",
    "register",
]
