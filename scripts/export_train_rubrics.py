#!/usr/bin/env python
"""Export one train-split generation run as self-contained JSONL, per domain.

Separate from `export_rubrics.py`, which serves the val-split evaluation runs:
those live across three overlapping run directories and are merged with
assertions that a shared uid carries an identical rubric everywhere. A
train-split run is a single directory with no overlap, so that machinery would
only be a way to get the inputs wrong. The output row shape is identical, so
both exports are consumed the same way.

Self-contained on purpose: each row carries the question and reference answer
next to the rubric, so a downstream RL job needs this file and nothing else.

Rows are keyed by uid and stamped with the generating model. Do not concatenate
exports from different generator models into one training set — `baseline` is a
control only while it shares a model with `agentic-tools` (HANDOFF.md §6.0).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.schema import CATEGORY_PREFIXES, Category, Rubric  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-dir", required=True, help="e.g. runs/train_full_glm53")
    p.add_argument("--source", required=True, help="e.g. baseline, agentic-tools")
    p.add_argument("--out-dir", default="exports/train")
    p.add_argument("--tag", default=None, help="filename suffix; defaults to the run dir name")
    return p.parse_args()


def to_record(uid: str, example: dict[str, Any], rubric_raw: dict, source: str, model: str) -> dict[str, Any]:
    rubric = Rubric.from_dict(rubric_raw)
    items = [
        {
            "title": c.title,
            "description": c.description,
            "weight": int(c.weight),
            "category": c.category.value,
            # Explicit because the two corpora disagree about what a negatively
            # weighted criterion means (docs/01_data_forensics.md F6): grade by
            # |weight| and take the good/bad direction from here.
            "polarity": (c.polarity.value if c.polarity else "positive"),
        }
        for c in rubric.items
    ]
    return {
        "uid": uid,
        "domain": example["domain"],
        "question_source": example.get("question_source", ""),
        "question": example["question"],
        "reference_answer": example["reference_answer"],
        "rubric_source": source,
        "generator_model": model,
        "n_criteria": len(items),
        "rubric": items,
        # The dataset's own shape: one category-prefixed sentence per criterion.
        "rubric_list": [
            f"{CATEGORY_PREFIXES[Category.coerce(i['category'])]} {i['description']}".strip()
            for i in items
        ],
    }


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    tag = args.tag or run_dir.name.replace("train_full_", "")

    examples = {}
    for line in (run_dir / "examples.jsonl").open(encoding="utf-8"):
        if line.strip():
            row = json.loads(line)
            examples[row["uid"]] = row

    src = run_dir / f"rubrics_{args.source}.jsonl"
    by_domain: dict[str, list[dict]] = defaultdict(list)
    models: Counter = Counter()
    skipped_empty = skipped_unknown = 0
    seen: set[str] = set()
    for line in src.open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        uid = row["uid"]
        if uid in seen:
            continue
        seen.add(uid)
        if not row.get("rubric", {}).get("items"):
            # A row with no criteria is a failure record, not a rubric. Exporting
            # it would put an unscoreable question into a training set.
            skipped_empty += 1
            continue
        ex = examples.get(uid)
        if ex is None:
            skipped_unknown += 1
            continue
        model = row.get("model", "<unstamped>")
        models[model] += 1
        by_domain[row.get("domain") or ex["domain"]].append(
            to_record(uid, ex, row["rubric"], args.source, model)
        )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for domain, rows in sorted(by_domain.items()):
        rows.sort(key=lambda r: r["uid"])
        out = out_dir / f"{domain}_{args.source}_{tag}.jsonl"
        tmp = out.with_suffix(".jsonl.tmp")
        with tmp.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        written = sum(1 for line in tmp.open(encoding="utf-8") if line.strip())
        assert written == len(rows), f"wrote {written}, expected {len(rows)}"
        tmp.replace(out)
        mean = sum(r["n_criteria"] for r in rows) / len(rows)
        print(f"  {out}  {len(rows)} 行, 条目数均值 {mean:.1f}")
        total += len(rows)

    print(f"  合计 {total} 行")
    if skipped_empty:
        print(f"  跳过 {skipped_empty} 行空 rubric（失败记录，不是 rubric）")
    if skipped_unknown:
        print(f"  跳过 {skipped_unknown} 行 uid 不在 examples.jsonl 中")
    print(f"  生成模型: {dict(models)}")
    if len(models) > 1:
        print("  注意: 同一文件含多个部署名。GLM-5.3 的 t1/t2/t2-copy 是同权重不同硬件，")
        print("        可以混用；GLM-5.2 或 Kimi 的行不可以（HANDOFF.md §6.0）。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
