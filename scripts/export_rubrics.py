#!/usr/bin/env python3
"""Export the RaR-corpus rubrics as one JSONL per (domain, source).

Three rubric sources exist for the RaR questions, and the distinction between
them is the whole point of this repository:

``shipped``   what the RaR dataset ships, produced by the paper's single-pass
              pipeline (o3-mini for science, GPT-4o for medicine). Not generated
              here at all.
``baseline``  the paper's prompt, verbatim and domain-matched, run through *our*
              model. This is the honest comparison: it holds the generator model
              fixed so a difference against ``agentic`` is a difference of
              method rather than of model.
``agentic``   this repository's multi-stage pipeline.

The rubrics live across three run directories that overlap (``pilot_v2`` 200
questions, ``rollout_v2``/``tools_v1`` 128, sharing 62). They are merged by uid
here, and the merge **asserts** that a shared uid carries an identical rubric in
every run rather than assuming it — silently preferring one copy would hide a
regeneration that changed the object under study.

Each record carries the question and reference answer alongside the rubric, so a
downstream RL run needs nothing but this file, plus ``rubric_list`` in the
dataset's own category-prefixed string form for drop-in compatibility with RaR
tooling.

Usage::

    python scripts/export_rubrics.py --out-dir exports/rubrics
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.schema import CATEGORY_PREFIXES, Category, Rubric  # noqa: E402

RUNS = ("pilot_v2", "rollout_v2", "tools_v1")
SOURCES = ("shipped", "baseline", "agentic")
DOMAINS = ("rar_science", "rar_medicine")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--out-dir", default="exports/rubrics")
    p.add_argument("--sources", nargs="+", default=list(SOURCES))
    p.add_argument("--verify-shipped", action="store_true", default=True,
                   help="cross-check the shipped export against the parquet it came from")
    return p.parse_args()


def load_examples(runs_dir: Path) -> dict[str, dict[str, Any]]:
    """uid -> question/reference/domain, merged across runs."""
    out: dict[str, dict[str, Any]] = {}
    for run in RUNS:
        path = runs_dir / run / "examples.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            out.setdefault(row["uid"], row)
    return out


def load_rubrics(runs_dir: Path, source: str) -> tuple[dict[str, dict], list[str]]:
    """uid -> rubric dict, merged across runs. Returns (merged, conflicting_uids)."""
    merged: dict[str, dict] = {}
    seen_in: dict[str, str] = {}
    conflicts: list[str] = []
    for run in RUNS:
        path = runs_dir / run / f"rubrics_{source}.jsonl"
        if not path.exists():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            uid, rubric = row["uid"], row.get("rubric") or {}
            if uid in merged:
                # Compare on the criteria only: `meta` legitimately differs
                # between runs (run name, call counts, wall time).
                a = json.dumps(merged[uid].get("items"), sort_keys=True, ensure_ascii=False)
                b = json.dumps(rubric.get("items"), sort_keys=True, ensure_ascii=False)
                if a != b:
                    conflicts.append(f"{uid} ({seen_in[uid]} vs {run})")
                continue
            merged[uid] = rubric
            seen_in[uid] = run
    return merged, conflicts


def to_record(uid: str, example: dict[str, Any], rubric_raw: dict, source: str) -> dict[str, Any]:
    rubric = Rubric.from_dict(rubric_raw)
    items = []
    for c in rubric.items:
        items.append({
            "title": c.title,
            "description": c.description,
            "weight": int(c.weight),
            "category": c.category.value,
            # Explicit because the two corpora disagree about what a negatively
            # weighted criterion means (docs/01_data_forensics.md F6): grade by
            # |weight| and take the good/bad direction from here.
            "polarity": (c.polarity.value if c.polarity else "positive"),
        })
    return {
        "uid": uid,
        "domain": example["domain"],
        "question_source": example.get("question_source", ""),
        "question": example["question"],
        "reference_answer": example["reference_answer"],
        "rubric_source": source,
        "n_criteria": len(items),
        "rubric": items,
        # The dataset's own shape: one category-prefixed sentence per criterion.
        "rubric_list": [
            f"{CATEGORY_PREFIXES[Category.coerce(i['category'])]} {i['description']}".strip()
            for i in items
        ],
    }


def verify_shipped(records: list[dict[str, Any]], runs_dir: Path) -> str:
    """Spot-check exported `shipped` rubrics against the source parquet."""
    try:
        import pandas as pd  # noqa: PLC0415
    except Exception as exc:  # noqa: BLE001
        return f"skipped ({exc})"
    root = Path(__file__).resolve().parent.parent / "data"
    by_domain: dict[str, dict[str, int]] = defaultdict(dict)
    for r in records:
        by_domain[r["domain"]][r["question"].strip()] = r["n_criteria"]
    checked = matched = 0
    for domain, wanted in by_domain.items():
        for split in ("val", "train", "test"):
            path = root / domain / f"{split}-00000-of-00001.parquet"
            if not path.exists():
                continue
            df = pd.read_parquet(path, columns=["question", "rubric_count"])
            for q, n in zip(df["question"].astype(str), df["rubric_count"]):
                key = q.strip()
                if key in wanted:
                    checked += 1
                    matched += int(int(n) == wanted.pop(key))
            if not wanted:
                break
    if not checked:
        return "no overlap found (parquet missing?)"
    return f"{matched}/{checked} criterion counts match the parquet"


def main() -> int:
    args = parse_args()
    runs_dir = Path(args.runs_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    examples = load_examples(runs_dir)
    print(f"examples merged across {len(RUNS)} runs: {len(examples)} unique questions")

    summary: list[tuple[str, str, int, int, float]] = []
    for source in args.sources:
        rubrics, conflicts = load_rubrics(runs_dir, source)
        if conflicts:
            print(f"  !! {source}: {len(conflicts)} uid(s) differ between runs: {conflicts[:3]}")
            raise SystemExit("refusing to export: a shared uid carries different rubrics")
        missing = [u for u in rubrics if u not in examples]
        if missing:
            raise SystemExit(f"{source}: {len(missing)} uid(s) have no example row, e.g. {missing[:3]}")

        by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for uid in sorted(rubrics):
            rec = to_record(uid, examples[uid], rubrics[uid], source)
            by_domain[rec["domain"]].append(rec)

        for domain in DOMAINS:
            rows = by_domain.get(domain, [])
            if not rows:
                continue
            path = out_dir / f"{domain}_{source}.jsonl"
            path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                encoding="utf-8",
            )
            empty = sum(1 for r in rows if r["n_criteria"] == 0)
            mean = sum(r["n_criteria"] for r in rows) / len(rows)
            summary.append((path.name, domain, len(rows), empty, mean))

        if source == "shipped" and args.verify_shipped:
            allrows = [r for rs in by_domain.values() for r in rs]
            print(f"  shipped cross-check vs parquet: {verify_shipped(allrows, runs_dir)}")

    print(f"\n{'file':40}{'n':>6}{'空 rubric':>10}{'平均条数':>10}")
    print("-" * 66)
    for name, _, n, empty, mean in summary:
        print(f"{name:40}{n:>6}{empty:>10}{mean:>10.2f}")
    print(f"\nwrote {len(summary)} files to {out_dir}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
