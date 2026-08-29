"""Tool protocol, results, and the registry the agent loop dispatches through.

The agentic generator used to be a fixed chain of prompts: every stage was a
call the orchestrator decided to make. Tools invert that for the stages where
the *model* is better placed to decide what evidence it needs — it can run
arithmetic, test whether a criterion actually fires on a text, or check whether
a draft criterion would apply just as well to an unrelated question.

Three properties matter more than convenience here, because the output feeds a
research claim:

* **Auditable.** Every invocation is recorded with its arguments, result,
  duration and error, and lands in the generation trace. A criterion whose
  justification is "the model said so" is indistinguishable from one backed by a
  computation unless the computation is on disk.
* **Deterministic where possible.** Five of the tools make no LLM call at all.
  Those are reproducible from the repository alone.
* **Bounded.** A tool can fail, time out, or be called with nonsense arguments.
  None of that may abort rubric generation, so every failure is converted into a
  ``ToolResult`` the model can read and react to.
"""

from __future__ import annotations

import abc
import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "ToolResult",
    "ToolContext",
    "Tool",
    "ToolRegistry",
    "ToolInvocation",
    "as_str_list",
]

#: Hard cap on the JSON handed back to the model for one call. Tool output goes
#: into the conversation and is resent on every subsequent turn, so an
#: unbounded result quietly multiplies the cost of the whole loop.
MAX_RESULT_CHARS = 6000


def _truncate(text: str, limit: int = MAX_RESULT_CHARS) -> tuple[str, bool]:
    if len(text) <= limit:
        return text, False
    keep = limit - 80
    return text[:keep] + f"\n…[truncated, {len(text) - keep} chars omitted]", True


@dataclass
class ToolResult:
    """Outcome of one tool invocation.

    ``data`` is what the model sees. ``meta`` never reaches the model and exists
    for the trace — put timings, raw stdout, token counts and other diagnostics
    there rather than inflating the conversation.
    """

    ok: bool
    data: Any = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def failure(cls, error: str, **meta: Any) -> "ToolResult":
        return cls(ok=False, error=error, meta=meta)

    def to_model_payload(self) -> str:
        """Serialise for the ``tool`` message. Always valid JSON."""
        body: dict[str, Any] = {"ok": self.ok}
        if self.error:
            body["error"] = self.error
        if self.data is not None:
            body["result"] = self.data
        try:
            text = json.dumps(body, ensure_ascii=False, default=str)
        except (TypeError, ValueError):
            text = json.dumps({"ok": False, "error": "result was not serialisable"})
        text, truncated = _truncate(text)
        if truncated:
            self.meta["truncated"] = True
        return text

    def to_trace(self) -> dict[str, Any]:
        return {"ok": self.ok, "error": self.error, "data": self.data, "meta": self.meta}


@dataclass
class ToolContext:
    """Per-example state a tool may read.

    Passed at dispatch time rather than bound at construction so one registry
    can serve many concurrently-running examples.
    """

    example: Any = None                     # harness.schema.Example
    engine: Any = None                      # harness.llm.LLMEngine
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def question(self) -> str:
        return getattr(self.example, "question", "") or ""

    @property
    def reference_answer(self) -> str:
        return getattr(self.example, "reference_answer", "") or ""

    @property
    def domain(self) -> str:
        return getattr(self.example, "domain", "") or ""

    @property
    def uid(self) -> str:
        return getattr(self.example, "uid", "") or ""


@dataclass
class ToolInvocation:
    """One recorded call, for the trace."""

    name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    seconds: float
    call_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "tool": self.name,
            "call_id": self.call_id,
            "arguments": self.arguments,
            "seconds": round(self.seconds, 3),
            "result": self.result,
        }


class Tool(abc.ABC):
    """One callable capability exposed to the model.

    Subclasses declare a JSON-schema parameter block and implement
    :meth:`run`. Argument validation, error capture and timing are handled by
    :class:`ToolRegistry`, so ``run`` may assume its declared arguments are
    present and may raise freely.
    """

    #: Function name the model calls. Must be a valid identifier.
    name: str = ""
    #: Shown to the model. Say what the tool *decides*, not how it is built —
    #: this text is the whole basis on which the model chooses to call it.
    description: str = ""
    #: JSON Schema for the arguments object.
    parameters: dict[str, Any] = {"type": "object", "properties": {}}
    #: True when the tool itself calls the LLM. Used to keep zero-LLM runs
    #: honest and to report cost attribution separately.
    uses_llm: bool = False

    @abc.abstractmethod
    async def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        ...

    def schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description.strip(),
                "parameters": self.parameters,
            },
        }


def _required_names(parameters: Mapping[str, Any]) -> list[str]:
    required = parameters.get("required")
    return [str(r) for r in required] if isinstance(required, (list, tuple)) else []


class ToolRegistry:
    """Holds tools, exports their schemas, and dispatches calls safely.

    A registry instance is shared across examples and is stateless with respect
    to them; per-call recording is returned to the caller rather than
    accumulated internally, so concurrent generations cannot interleave traces.
    """

    def __init__(
        self,
        tools: Iterable[Tool] = (),
        *,
        default_timeout_s: float = 60.0,
    ) -> None:
        self._tools: dict[str, Tool] = {}
        self.default_timeout_s = default_timeout_s
        for tool in tools:
            self.add(tool)

    # -- composition ------------------------------------------------------

    def add(self, tool: Tool) -> "ToolRegistry":
        if not tool.name:
            raise ValueError(f"{type(tool).__name__} has no name")
        if tool.name in self._tools:
            raise ValueError(f"duplicate tool name {tool.name!r}")
        self._tools[tool.name] = tool
        return self

    def subset(self, names: Sequence[str]) -> "ToolRegistry":
        """A registry with only ``names``, preserving the given order.

        Unknown names raise: silently returning fewer tools than a config asked
        for would make an ablation look like it ran when it did not.
        """
        missing = [n for n in names if n not in self._tools]
        if missing:
            raise KeyError(f"unknown tools {missing}; known: {sorted(self._tools)}")
        return ToolRegistry(
            (self._tools[n] for n in names), default_timeout_s=self.default_timeout_s
        )

    def __contains__(self, name: object) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    @property
    def names(self) -> list[str]:
        return list(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def schemas(self) -> list[dict[str, Any]]:
        return [t.schema() for t in self._tools.values()]

    def describe(self) -> str:
        """Compact catalogue for embedding in a system prompt."""
        return "\n".join(
            f"- {t.name}: {' '.join(t.description.split())}" for t in self._tools.values()
        )

    # -- dispatch ---------------------------------------------------------

    async def dispatch(
        self,
        name: str,
        arguments: Any,
        ctx: ToolContext,
        *,
        timeout_s: float | None = None,
        call_id: str = "",
    ) -> ToolInvocation:
        """Run one call, converting every failure mode into a readable result.

        ``arguments`` may be a dict or the raw JSON string the model emitted;
        malformed JSON is reported back to the model rather than raised, since
        the model can usually fix it on the next turn.
        """
        started = time.perf_counter()
        parsed, parse_error = _coerce_arguments(arguments)

        if parse_error is not None:
            result = ToolResult.failure(parse_error)
        elif (tool := self._tools.get(name)) is None:
            result = ToolResult.failure(
                f"no such tool {name!r}; available: {', '.join(self._tools)}"
            )
        elif missing := [k for k in _required_names(tool.parameters) if k not in parsed]:
            result = ToolResult.failure(f"missing required argument(s): {', '.join(missing)}")
        else:
            allowed = set((tool.parameters.get("properties") or {}).keys())
            # Extra keys are dropped rather than rejected: models routinely add a
            # stray "reason" field, and failing the call over it wastes a turn.
            kwargs = {k: v for k, v in parsed.items() if k in allowed}
            if dropped := sorted(set(parsed) - allowed):
                logger.debug("tool %s: ignoring unexpected argument(s) %s", name, dropped)
            result = await self._invoke(tool, ctx, kwargs, timeout_s)

        return ToolInvocation(
            name=name,
            arguments=parsed if parse_error is None else {"_raw": str(arguments)[:500]},
            result=result.to_trace(),
            seconds=time.perf_counter() - started,
            call_id=call_id,
        )

    async def _invoke(
        self,
        tool: Tool,
        ctx: ToolContext,
        kwargs: dict[str, Any],
        timeout_s: float | None,
    ) -> ToolResult:
        limit = timeout_s if timeout_s is not None else self.default_timeout_s
        try:
            return await asyncio.wait_for(tool.run(ctx, **kwargs), timeout=limit)
        except asyncio.TimeoutError:
            return ToolResult.failure(f"tool {tool.name!r} timed out after {limit:.0f}s")
        except TypeError as exc:
            # Almost always a bad argument shape from the model, not a bug.
            return ToolResult.failure(f"invalid arguments for {tool.name!r}: {exc}")
        except Exception as exc:  # noqa: BLE001 - a tool must never kill the run
            logger.warning("tool %s raised: %s", tool.name, str(exc)[:300])
            return ToolResult.failure(f"{type(exc).__name__}: {str(exc)[:300]}")


def as_str_list(value: Any) -> list[str]:
    """Coerce a ``string | list[string]`` argument into a list.

    Models routinely send a single criterion as a bare string where the schema
    asks for an array. Rejecting that costs a whole round to fix a mistake with
    exactly one sensible reading, so it is read rather than refused.
    """
    if value is None:
        return []
    if isinstance(value, str):
        return [value] if value.strip() else []
    if isinstance(value, Sequence):
        return [str(v) for v in value if str(v).strip()]
    return []


def _coerce_arguments(arguments: Any) -> tuple[dict[str, Any], str | None]:
    if arguments is None or arguments == "":
        return {}, None
    if isinstance(arguments, Mapping):
        return dict(arguments), None
    if isinstance(arguments, str):
        try:
            value = json.loads(arguments)
        except json.JSONDecodeError as exc:
            return {}, f"arguments were not valid JSON ({exc}); re-emit them as a JSON object"
        if isinstance(value, Mapping):
            return dict(value), None
        return {}, "arguments must be a JSON object"
    return {}, f"arguments must be a JSON object, got {type(arguments).__name__}"
