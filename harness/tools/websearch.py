"""Web lookup tools: DuckDuckGo search and Wikipedia. No API key required.

Motivation, from `results/rubricbench/REPORT.md` §11.2: an eighth rubric
candidate is only worth building if it brings a *new information source*, and
only three exist. One of them is "be able to solve the task" — §9.8 found that
numbers appearing in the expert rubrics but not in the instruction hit the
human-preferred response 12.4 points more often than chance, which is knowledge
of the right answer, not general task understanding. A generator that can look
a fact up has access to something none of the ten measured candidates had.

Two properties matter as much as the capability itself:

* **Reproducibility.** Search results drift, and every number in this repository
  is meant to be recomputable. Results are cached to disk by query and the cache
  is an artefact, not a temporary file: a rerun replays the same web state, and
  the timestamp of every fetch is recorded so staleness is visible rather than
  silent.
* **Its own concurrency limit.** Generation runs at 64 in-flight LLM calls;
  DuckDuckGo will not take that. A dedicated semaphore (default 8, measured
  clean at 8/8 with no throttling) keeps search from being the thing that gets
  the run banned.

Not in :data:`harness.tools.DEFAULT_TOOLS`. Adding it there would silently
change what `agentic-tools` means and break comparability with the results
already recorded for that arm; ask for it by name instead.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

from .base import Tool, ToolContext, ToolResult

logger = logging.getLogger(__name__)

__all__ = ["WebSearchTool", "WikipediaTool", "search_cache_stats"]

CACHE_DIR = Path(os.environ.get("RUBRIC_WEBCACHE", "runs/websearch_cache"))

#: Measured on this host through the corporate proxy: 8 concurrent queries
#: returned 8/8 with no throttling, ~2 queries/s aggregate. Raise only after
#: re-measuring — a ban costs the whole run, not one call.
_MAX_CONCURRENT = int(os.environ.get("RUBRIC_WEBSEARCH_CONCURRENCY", "8"))
_semaphore: asyncio.Semaphore | None = None
_sem_lock = asyncio.Lock()


async def _get_semaphore() -> asyncio.Semaphore:
    """One semaphore per event loop, created lazily.

    Built on first use rather than at import so it binds to the running loop;
    a module-level Semaphore() would attach to whichever loop imported first.
    """
    global _semaphore
    async with _sem_lock:
        if _semaphore is None:
            _semaphore = asyncio.Semaphore(_MAX_CONCURRENT)
        return _semaphore


def _cache_path(kind: str, payload: dict[str, Any]) -> Path:
    blob = json.dumps({"kind": kind, **payload}, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(blob.encode("utf-8")).hexdigest()
    sub = CACHE_DIR / digest[:2]
    sub.mkdir(parents=True, exist_ok=True)
    return sub / f"{digest}.json"


def _cache_read(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a corrupt entry is a miss
        return None


def _cache_write(path: Path, value: dict[str, Any]) -> None:
    try:
        tmp = path.with_suffix(f".tmp{os.getpid()}")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(path)
    except Exception as exc:  # noqa: BLE001 - caching is best-effort
        logger.debug("web cache write failed: %s", exc)


def search_cache_stats() -> dict[str, Any]:
    """Size and age of the on-disk cache, for run manifests."""
    files = list(CACHE_DIR.rglob("*.json")) if CACHE_DIR.exists() else []
    stamps = []
    for f in files:
        entry = _cache_read(f)
        if entry and entry.get("fetched_at"):
            stamps.append(entry["fetched_at"])
    return {
        "dir": str(CACHE_DIR),
        "entries": len(files),
        "oldest": min(stamps) if stamps else None,
        "newest": max(stamps) if stamps else None,
    }


async def _run_blocking(fn, *args):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, fn, *args)


class WebSearchTool(Tool):
    name = "web_search"
    description = """
    Search the web and get back titles, URLs and snippets. Use it when writing a
    criterion depends on a fact you are not sure of — a constant, a definition, a
    standard, an API signature, who holds a record — so the criterion pins down
    what is actually correct rather than what sounds plausible. Prefer a specific
    query over a broad one, and read the snippets rather than assuming the top
    result is right. Results are cached, so repeating a query is free and returns
    the same thing.
    """
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Search query. Be specific."},
            "max_results": {
                "type": "integer",
                "description": "How many results to return. Default 5, max 10.",
            },
        },
        "required": ["query"],
    }

    def __init__(self, *, default_results: int = 5, timeout_s: float = 45.0,
                 snippet_chars: int = 400) -> None:
        self.default_results = default_results
        self.timeout_s = timeout_s
        self.snippet_chars = snippet_chars

    @staticmethod
    def _search(query: str, n: int) -> list[dict[str, Any]]:
        from ddgs import DDGS  # noqa: PLC0415 - optional dependency

        return list(DDGS().text(query, max_results=n))

    async def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        query = str(kwargs.get("query") or "").strip()
        if not query:
            return ToolResult.failure("'query' must be non-empty")
        try:
            n = min(10, max(1, int(kwargs.get("max_results", self.default_results))))
        except (TypeError, ValueError):
            n = self.default_results

        path = _cache_path("ddg", {"query": query, "n": n})
        if (cached := _cache_read(path)) is not None:
            return ToolResult(
                ok=True,
                data={"query": query, "results": cached["results"],
                      "note": "served from the run's search cache"},
                meta={"cache": "hit", "fetched_at": cached.get("fetched_at")},
            )

        started = time.time()
        sem = await _get_semaphore()
        try:
            async with sem:
                raw = await asyncio.wait_for(
                    _run_blocking(self._search, query, n), timeout=self.timeout_s
                )
        except asyncio.TimeoutError:
            return ToolResult.failure(f"search timed out after {self.timeout_s:.0f}s")
        except Exception as exc:  # noqa: BLE001 - network/parse errors are data
            return ToolResult.failure(f"search failed: {type(exc).__name__}: {str(exc)[:200]}")

        results = [
            {
                "title": str(r.get("title") or "")[:200],
                "url": str(r.get("href") or r.get("url") or "")[:300],
                "snippet": str(r.get("body") or "")[: self.snippet_chars],
            }
            for r in raw
        ]
        if not results:
            return ToolResult(ok=True, data={"query": query, "results": [],
                                             "note": "no results; try a different query"})
        _cache_write(path, {"query": query, "results": results,
                            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        return ToolResult(
            ok=True,
            data={"query": query, "results": results},
            meta={"cache": "miss", "seconds": round(time.time() - started, 2)},
        )


class WikipediaTool(Tool):
    name = "wikipedia_lookup"
    description = """
    Get the opening summary of a Wikipedia article. Faster and more reliable than
    a web search when what you need is a definition, a physical constant, a
    standard formula, or the accepted account of a named thing. Use it first for
    textbook facts; fall back to web_search for anything current, niche, or
    contested. Results are cached.
    """
    parameters = {
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "Article title or search phrase."},
            "sentences": {
                "type": "integer",
                "description": "Sentences of summary to return. Default 4, max 10.",
            },
        },
        "required": ["topic"],
    }

    def __init__(self, *, default_sentences: int = 4, timeout_s: float = 30.0) -> None:
        self.default_sentences = default_sentences
        self.timeout_s = timeout_s

    @staticmethod
    def _lookup(topic: str, sentences: int) -> dict[str, Any]:
        import wikipedia  # noqa: PLC0415 - optional dependency

        try:
            summary = wikipedia.summary(topic, sentences=sentences, auto_suggest=True)
            page = wikipedia.page(topic, auto_suggest=True)
            return {"title": page.title, "url": page.url, "summary": summary}
        except wikipedia.DisambiguationError as exc:
            return {"disambiguation": [str(o) for o in exc.options[:10]]}
        except wikipedia.PageError:
            return {"missing": True}

    async def run(self, ctx: ToolContext, **kwargs: Any) -> ToolResult:
        topic = str(kwargs.get("topic") or "").strip()
        if not topic:
            return ToolResult.failure("'topic' must be non-empty")
        try:
            n = min(10, max(1, int(kwargs.get("sentences", self.default_sentences))))
        except (TypeError, ValueError):
            n = self.default_sentences

        path = _cache_path("wiki", {"topic": topic, "n": n})
        if (cached := _cache_read(path)) is not None:
            return ToolResult(ok=True, data=cached["payload"],
                              meta={"cache": "hit", "fetched_at": cached.get("fetched_at")})

        sem = await _get_semaphore()
        try:
            async with sem:
                payload = await asyncio.wait_for(
                    _run_blocking(self._lookup, topic, n), timeout=self.timeout_s
                )
        except asyncio.TimeoutError:
            return ToolResult.failure(f"wikipedia lookup timed out after {self.timeout_s:.0f}s")
        except Exception as exc:  # noqa: BLE001
            return ToolResult.failure(f"wikipedia lookup failed: {type(exc).__name__}: {str(exc)[:200]}")

        if payload.get("missing"):
            return ToolResult(ok=True, data={"topic": topic, "found": False,
                                             "note": "no such article; try web_search"})
        if payload.get("disambiguation"):
            return ToolResult(ok=True, data={"topic": topic, "found": False,
                                             "ambiguous_options": payload["disambiguation"],
                                             "note": "ambiguous; retry with one of these titles"})
        _cache_write(path, {"payload": {**payload, "found": True},
                            "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
        return ToolResult(ok=True, data={**payload, "found": True}, meta={"cache": "miss"})
