#!/usr/bin/env python3
"""Deduplicate append-only run artifacts by ``uid``, keeping the last record.

The pipeline appends to ``rubrics_<source>.jsonl`` and tolerates duplicate uids
on load, but duplicates inflate line counts and make progress logs misleading.
This rewrites each file in place with one record per uid.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def repair(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0
    seen: dict[str, str] = {}
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        total += 1
        try:
            uid = json.loads(line).get("uid")
        except json.JSONDecodeError:
            continue
        if uid:
            seen[str(uid)] = line
    path.write_text("\n".join(seen.values()) + "\n", encoding="utf-8")
    return total, len(seen)


def main() -> int:
    for arg in sys.argv[1:]:
        p = Path(arg)
        before, after = repair(p)
        flag = "  <-- had duplicates" if before != after else ""
        print(f"{p.name:34s} {before:6d} -> {after:6d}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
