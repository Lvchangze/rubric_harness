"""Run directories, config snapshots and append-safe JSONL artefacts.

A *run* is one directory under ``runs/`` holding everything needed to audit or
resume it: the resolved config, the sampled examples, per-example outputs, and
the full agent trace. Writers are process-local but lock-guarded so many
asyncio tasks can append concurrently.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

__all__ = ["RunDir", "JsonlWriter", "read_jsonl", "jsonable"]


def jsonable(obj: Any) -> Any:
    """Recursively coerce dataclasses / enums / numpy scalars into JSON types."""
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
        if hasattr(obj, "to_dict"):
            return jsonable(obj.to_dict())
        return jsonable(asdict(obj))
    if hasattr(obj, "to_dict") and callable(obj.to_dict):
        return jsonable(obj.to_dict())
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "value") and hasattr(obj, "name"):  # Enum
        return obj.value
    if hasattr(obj, "item"):  # numpy scalar
        try:
            return obj.item()
        except Exception:  # noqa: BLE001
            pass
    return str(obj)


class JsonlWriter:
    """Append-only JSONL sink, safe under concurrent asyncio tasks."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._count = 0

    def write(self, record: Any) -> None:
        line = json.dumps(jsonable(record), ensure_ascii=False)
        with self._lock:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(line + "\n")
            self._count += 1

    @property
    def count(self) -> int:
        return self._count


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


class RunDir:
    """Filesystem layout for one experiment run."""

    def __init__(self, root: str | Path, name: str, *, create: bool = True) -> None:
        self.root = Path(root)
        self.name = name
        self.path = self.root / name
        if create:
            self.path.mkdir(parents=True, exist_ok=True)
            (self.path / "traces").mkdir(exist_ok=True)
        self._writers: dict[str, JsonlWriter] = {}
        self._lock = threading.Lock()

    def snapshot_config(self, config: Any, filename: str = "config.json") -> Path:
        payload = jsonable(config)
        if isinstance(payload, dict):
            payload.setdefault("_snapshot_time", time.strftime("%Y-%m-%d %H:%M:%S"))
        path = self.path / filename
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, ensure_ascii=False, indent=2)
        return path

    def writer(self, name: str) -> JsonlWriter:
        with self._lock:
            if name not in self._writers:
                self._writers[name] = JsonlWriter(self.path / f"{name}.jsonl")
            return self._writers[name]

    def file(self, *parts: str) -> Path:
        path = self.path.joinpath(*parts)
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, name: str, payload: Any) -> Path:
        path = self.file(name)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(jsonable(payload), fh, ensure_ascii=False, indent=2)
        return path

    def read_json(self, name: str, default: Any = None) -> Any:
        path = self.path / name
        if not path.exists():
            return default
        with path.open("r", encoding="utf-8") as fh:
            return json.load(fh)

    def write_trace(self, uid: str, trace: Any) -> Path:
        path = self.path / "traces" / f"{uid}.json"
        with path.open("w", encoding="utf-8") as fh:
            json.dump(jsonable(trace), fh, ensure_ascii=False, indent=2)
        return path

    def existing_uids(self, name: str, key: str = "uid") -> set[str]:
        """UIDs already present in ``<name>.jsonl`` — enables safe resume."""
        return {
            rec[key]
            for rec in read_jsonl(self.path / f"{name}.jsonl")
            if isinstance(rec, dict) and key in rec and not rec.get("error")
        }
