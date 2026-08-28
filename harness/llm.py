"""LLM access layer: one wrapper around ``contextagent.llm.LLMClient``.

Everything downstream (generators, judges, response builders) goes through
:class:`LLMEngine`. It absorbs the sharp edges of the reasoning endpoint:

* ``chat()`` returns ``{'reasoning_content', 'response'}`` for reasoning models
  and a bare ``str`` for others — :func:`extract_response_text` normalises both.
* An exhausted token budget silently yields an *empty* ``response``; that is
  retried with a larger budget rather than surfaced as valid output.
* Structured stages need JSON, and the model likes to wrap it in ```json fences
  or prepend a sentence — :func:`extract_json` repairs the common cases.
* Every call is content-addressed and cached on disk, so a rerun of a partially
  finished experiment costs nothing and is bit-identical.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import random
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

logger = logging.getLogger(__name__)

_HYCTX_PATH = "/apdcephfs_zwfy6/share_302970870/hunyuan/changzelv/dev/hyctx_data"

__all__ = [
    "LLMEngine",
    "LLMStats",
    "extract_response_text",
    "extract_json",
    "JSONParseError",
    "DEFAULT_MODEL",
]

DEFAULT_MODEL = "hy-t2t-glm-5.2-384k-fp8-L20A-t1-v2"


class JSONParseError(ValueError):
    """Raised when no JSON value can be recovered from a model response."""


# ---------------------------------------------------------------------------
# Response normalisation
# ---------------------------------------------------------------------------


def extract_response_text(out: Any) -> str:
    """Pull the visible answer out of whatever ``LLMClient.chat`` returned."""
    if out is None:
        return ""
    if isinstance(out, str):
        return out
    if isinstance(out, dict):
        for key in ("response", "content", "text", "answer"):
            value = out.get(key)
            if isinstance(value, str) and value.strip():
                return value
        return ""
    return str(out)


def extract_reasoning_text(out: Any) -> str:
    if isinstance(out, dict):
        value = out.get("reasoning_content")
        if isinstance(value, str):
            return value
    return ""


# ---------------------------------------------------------------------------
# JSON recovery
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def _balanced_slice(text: str, opener: str, closer: str) -> str | None:
    """Longest balanced ``opener…closer`` span, ignoring brackets in strings."""
    start = text.find(opener)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _strip_trailing_commas(text: str) -> str:
    return re.sub(r",(\s*[}\]])", r"\1", text)


def extract_json(text: str, *, expect: str | None = None) -> Any:
    """Best-effort JSON extraction from a chatty model response.

    ``expect`` may be ``"object"``, ``"array"`` or ``None`` (either). Tries, in
    order: whole string, fenced blocks, balanced bracket slices, and finally the
    same candidates with trailing commas removed.
    """
    if not text or not text.strip():
        raise JSONParseError("empty response")

    candidates: list[str] = [text.strip()]
    candidates.extend(m.strip() for m in _FENCE_RE.findall(text))

    wanted = (
        [("[", "]")] if expect == "array"
        else [("{", "}")] if expect == "object"
        else [("{", "}"), ("[", "]")]
    )
    for source in list(candidates):
        for opener, closer in wanted:
            span = _balanced_slice(source, opener, closer)
            if span:
                candidates.append(span)

    errors: list[str] = []
    for candidate in candidates:
        for attempt in (candidate, _strip_trailing_commas(candidate)):
            try:
                value = json.loads(attempt)
            except Exception as exc:  # noqa: BLE001 - collected for diagnostics
                errors.append(str(exc))
                continue
            if expect == "object" and not isinstance(value, dict):
                continue
            if expect == "array" and not isinstance(value, list):
                # A common failure: the array wrapped in a single-key object.
                if isinstance(value, dict):
                    for v in value.values():
                        if isinstance(v, list):
                            return v
                continue
            return value
    raise JSONParseError(
        f"no parsable JSON (expect={expect}); first error: {errors[0] if errors else 'n/a'}; "
        f"head={text[:200]!r}"
    )


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------


@dataclass
class LLMStats:
    """Aggregate counters for one engine instance (thread/task safe enough)."""

    calls: int = 0
    cache_hits: int = 0
    api_calls: int = 0
    retries: int = 0
    empty_responses: int = 0
    failures: int = 0
    json_parse_failures: int = 0
    prompt_chars: int = 0
    response_chars: int = 0
    reasoning_chars: int = 0
    wall_seconds: float = 0.0
    by_tag: dict[str, int] = field(default_factory=dict)

    def snapshot(self) -> dict[str, Any]:
        data = {k: v for k, v in self.__dict__.items() if k != "by_tag"}
        data["by_tag"] = dict(self.by_tag)
        data["cache_hit_rate"] = self.cache_hits / self.calls if self.calls else 0.0
        return data


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class LLMEngine:
    """Cached, retrying, JSON-aware facade over one model endpoint.

    Parameters
    ----------
    model:
        Registry key, e.g. ``hy-t2t-glm-5.2-384k-fp8-L20A-t1-v2``.
    concurrency:
        Upper bound on in-flight requests (enforced both by the underlying
        client and by our own semaphore so cache hits stay cheap).
    cache_dir:
        Directory for the on-disk response cache; ``None`` disables caching.
    max_attempts:
        Total tries per logical call, including the first.
    escalate_max_tokens:
        When a reasoning model returns an empty ``response`` the budget is
        multiplied by this factor before retrying.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        *,
        concurrency: int = 16,
        reasoning_effort: str | None = "high",
        cache_dir: str | Path | None = "runs/cache",
        max_attempts: int = 4,
        default_max_tokens: int = 16384,
        escalate_max_tokens: float = 1.5,
        request_timeout_s: int = 1200,
        seed: int = 0,
    ) -> None:
        os.environ.setdefault("LLM_REQUEST_TIMEOUT_S", str(request_timeout_s))
        if _HYCTX_PATH not in sys.path:
            sys.path.insert(0, _HYCTX_PATH)
        from contextagent.llm import LLMClient  # noqa: PLC0415 - late, needs sys.path

        self.model = model
        self.concurrency = concurrency
        self.reasoning_effort = reasoning_effort
        self.default_max_tokens = default_max_tokens
        self.max_attempts = max_attempts
        self.escalate_max_tokens = escalate_max_tokens
        self.stats = LLMStats()

        self._client = LLMClient(model, concurrency=concurrency, reasoning_effort=reasoning_effort)
        self._semaphore = asyncio.Semaphore(concurrency)
        self._cache_dir = Path(cache_dir) if cache_dir else None
        if self._cache_dir is not None:
            self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._io_lock = threading.Lock()
        self._rng = random.Random(seed)

    # -- cache ------------------------------------------------------------

    def _cache_key(self, payload: dict[str, Any]) -> str:
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def _cache_path(self, key: str) -> Path:
        assert self._cache_dir is not None
        # Two-level fan-out keeps directory listings usable at 100k+ entries.
        sub = self._cache_dir / key[:2] / key[2:4]
        sub.mkdir(parents=True, exist_ok=True)
        return sub / f"{key}.json"

    def _cache_read(self, key: str) -> dict[str, Any] | None:
        if self._cache_dir is None:
            return None
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:  # noqa: BLE001 - a corrupt cache entry is just a miss
            return None

    def _cache_write(self, key: str, value: dict[str, Any]) -> None:
        if self._cache_dir is None:
            return
        path = self._cache_path(key)
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        try:
            with self._io_lock:
                with tmp.open("w", encoding="utf-8") as fh:
                    json.dump(value, fh, ensure_ascii=False)
                tmp.replace(path)
        except Exception as exc:  # noqa: BLE001
            logger.debug("cache write failed: %s", exc)

    # -- core call --------------------------------------------------------

    async def chat(
        self,
        messages: list[dict[str, str]] | str,
        *,
        system: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        reasoning_effort: str | None = None,
        tag: str = "misc",
        cache_salt: str | None = None,
        use_cache: bool = True,
        return_reasoning: bool = False,
    ) -> str | tuple[str, str]:
        """Return the model's visible text (and reasoning, if requested).

        ``cache_salt`` lets a caller deliberately draw *different* samples from
        the same prompt (used by the self-consistency stage, where k independent
        rollouts must not collapse onto one cached answer).
        """
        budget = max_tokens or self.default_max_tokens
        # Deliberately absent from the cache key: `budget`. Two calls that differ
        # only in token budget share a cache entry, so raising the budget will
        # not re-draw a reply that was previously truncated. In this run nothing
        # was truncated (verified: zero truncated completions), and including it
        # would invalidate every cached call, but it is a real hazard for anyone
        # who changes a budget expecting longer output.
        params: dict[str, Any] = {
            "model": self.model,
            "reasoning_effort": reasoning_effort or self.reasoning_effort,
            "temperature": temperature,
            "top_p": top_p,
        }
        key_payload = {
            "messages": messages,
            "system": system,
            "params": {k: v for k, v in params.items() if v is not None},
            "salt": cache_salt,
        }
        key = self._cache_key(key_payload)

        self.stats.calls += 1
        self.stats.by_tag[tag] = self.stats.by_tag.get(tag, 0) + 1

        if use_cache:
            cached = self._cache_read(key)
            if cached is not None and cached.get("response"):
                self.stats.cache_hits += 1
                if return_reasoning:
                    return cached["response"], cached.get("reasoning", "")
                return cached["response"]

        kwargs: dict[str, Any] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature

        last_error: Exception | None = None
        text, reasoning = "", ""
        for attempt in range(self.max_attempts):
            if attempt:
                self.stats.retries += 1
                delay = min(30.0, 2.0 * (2 ** (attempt - 1))) * (0.6 + self._rng.random() * 0.8)
                await asyncio.sleep(delay)
            started = time.time()
            try:
                async with self._semaphore:
                    out = await self._client.chat(
                        messages,
                        system=system,
                        max_tokens=int(budget),
                        top_p=top_p,
                        reasoning_effort=params["reasoning_effort"],
                        **kwargs,
                    )
                self.stats.api_calls += 1
                self.stats.wall_seconds += time.time() - started
                text = extract_response_text(out).strip()
                reasoning = extract_reasoning_text(out)
                self.stats.reasoning_chars += len(reasoning)
                if text:
                    self.stats.response_chars += len(text)
                    self._cache_write(key, {"response": text, "reasoning": reasoning, "tag": tag})
                    if return_reasoning:
                        return text, reasoning
                    return text
                # Empty visible answer: the reasoning trace ate the budget.
                self.stats.empty_responses += 1
                budget = int(budget * self.escalate_max_tokens)
                logger.warning(
                    "empty response (tag=%s attempt=%d) — escalating max_tokens to %d",
                    tag, attempt + 1, budget,
                )
            except Exception as exc:  # noqa: BLE001 - transient endpoint errors
                self.stats.wall_seconds += time.time() - started
                last_error = exc
                logger.warning("LLM call failed (tag=%s attempt=%d): %s", tag, attempt + 1, str(exc)[:300])

        self.stats.failures += 1
        raise RuntimeError(
            f"LLM call failed after {self.max_attempts} attempts (tag={tag}): {last_error}"
        )

    async def chat_json(
        self,
        messages: list[dict[str, str]] | str,
        *,
        expect: str | None = None,
        repair_attempts: int = 2,
        **kwargs: Any,
    ) -> Any:
        """Call the model and parse JSON out of the reply.

        On a parse failure the raw text is handed back to the model with an
        explicit "return only JSON" instruction; that recovers nearly every
        case where the model prefixed prose or truncated a fence.
        """
        tag = kwargs.get("tag", "misc")
        text = await self.chat(messages, **kwargs)
        try:
            return extract_json(text, expect=expect)
        except JSONParseError as exc:
            first_error = exc

        self.stats.json_parse_failures += 1
        shape = {"array": "a JSON array", "object": "a JSON object"}.get(expect or "", "valid JSON")
        for i in range(repair_attempts):
            repair_prompt = (
                f"The following text was supposed to contain {shape} but could not be parsed.\n"
                f"Re-emit the content as {shape} and output NOTHING else — no prose, no code fence.\n"
                f"Preserve all information; do not summarise.\n\n<text>\n{text[:60000]}\n</text>"
            )
            try:
                repaired = await self.chat(
                    repair_prompt,
                    max_tokens=kwargs.get("max_tokens") or self.default_max_tokens,
                    tag=f"{tag}:json_repair",
                    cache_salt=f"repair{i}",
                    reasoning_effort="low",
                )
                return extract_json(repaired, expect=expect)
            except (JSONParseError, RuntimeError):
                continue
        raise JSONParseError(f"JSON repair exhausted (tag={tag}): {first_error}")

    # -- convenience ------------------------------------------------------

    async def map_concurrent(
        self,
        coros: Sequence[Any],
        *,
        return_exceptions: bool = True,
    ) -> list[Any]:
        return await asyncio.gather(*coros, return_exceptions=return_exceptions)

    def stats_snapshot(self) -> dict[str, Any]:
        snap = self.stats.snapshot()
        snap["model"] = self.model
        snap["concurrency"] = self.concurrency
        return snap
