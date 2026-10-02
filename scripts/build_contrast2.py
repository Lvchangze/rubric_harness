#!/usr/bin/env python3
"""Build `agentic-tools-contrast2` rubrics from a saved `agentic-tools-contrast` run.

`-contrast2` differs from `-contrast` only in the last step: for STEM and CODE
cases it also keeps the two highest-weight upstream criteria (`keep_upstream`).
Re-running the pipeline would re-draw every upstream call that failed last time
and so change the upstream for most cases; applying the same function to the
saved artefacts keeps the upstream identical and isolates the change.

    python scripts/build_contrast2.py --run-dir results/rubricbench_glm53
    python scripts/rubricbench_run.py --source file:results/rubricbench_glm53/agentic-tools-contrast2_input.json \
        --tag agentic-tools-contrast2 --case-ids results/rubricbench/split.json:dev ...
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from harness import rubricbench as RB  # noqa: E402
from harness.generators.agentic import keep_upstream  # noqa: E402

_LINE_RE = re.compile(r"^\s*\d+\.\s*(?:\[(?P<title>.*?)\])?\s*(?P<desc>.*?)\s*\(importance (?P<w>\d)/5\)\s*$")


def parse(text: str) -> list[SimpleNamespace]:
    items = []
    for line in str(text or "").splitlines():
        if not line.strip():
            continue
        m = _LINE_RE.match(line)
        if m:
            items.append(SimpleNamespace(title=(m["title"] or "").strip(), description=m["desc"].strip(),
                                         weight=int(m["w"])))
        else:
            items.append(SimpleNamespace(title="", description=line.strip(), weight=3))
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", default="results/rubricbench_glm53")
    ap.add_argument("--k", type=int, default=2)
    args = ap.parse_args()
    run = Path(args.run_dir)

    final = json.loads((run / "agentic-tools-contrast_rubrics.json").read_text(encoding="utf-8"))
    pre = {r["case_id"]: r["rubric"] for r in
           json.loads((run / "agentic-tools-contrast_prefinal.json").read_text(encoding="utf-8"))}
    domain = {c.case_id: c.domain for c in RB.load_cases()}
    checkable = RB.DOMAIN_GROUPS["stem"] | RB.DOMAIN_GROUPS["code"]

    out, n_case, n_added = [], 0, 0
    for row in final:
        cid, text = row["case_id"], row["rubric"]
        if domain.get(cid, "").lower() in checkable and cid in pre and str(text).strip():
            kept = parse(text)
            merged = keep_upstream(kept, parse(pre[cid]), args.k)
            n_case += 1
            n_added += len(merged) - len(kept)
            text = RB.rubric_to_text(SimpleNamespace(items=merged))
        out.append({"case_id": cid, "rubric": text})

    dest = run / "agentic-tools-contrast2_input.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(out)} rubrics to {dest}: {n_case} STEM/CODE cases got {n_added} upstream criteria")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
