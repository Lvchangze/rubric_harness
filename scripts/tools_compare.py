#!/usr/bin/env python3
"""Render the `agentic-tools` vs `agentic` contrast from an aggregated summary.

The two arms share every stage, prompt and threshold; `agentic-tools` adds the
`investigate` and `critic_tools` stages and nothing else. So this table answers
one question — does letting the model gather and check its own evidence produce
a better rubric? — and the reference column must be `agentic`, not `baseline`.

Usage::

    python scripts/aggregate_results.py --run-name tools_v1 \\
        --results-dir results/tools_v1/vs_agentic \\
        --sources agentic agentic-tools --reference agentic --expect-metrics 0
    python scripts/tools_compare.py --summary results/tools_v1/vs_agentic/tools_v1/summary.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

REF = "agentic"
ARM = "agentic-tools"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--summary", required=True, help="summary.json written with --reference agentic")
    p.add_argument("--out", default=None, help="also write the rendered table here")
    p.add_argument("--by-domain", action="store_true", default=True)
    p.add_argument("--no-by-domain", dest="by_domain", action="store_false")
    return p.parse_args()


def _fmt(value: Any, spec: str = "+.3f") -> str:
    if value is None:
        return "    —"
    try:
        return format(float(value), spec)
    except (TypeError, ValueError):
        return str(value)


def render(table: Sequence[dict[str, Any]], title: str) -> list[str]:
    lines = [
        f"### {title}",
        "",
        "| metric | dir | n | `agentic` | `agentic-tools` | Δ | 95% CI | p | q |",
        "|---|:--:|--:|--:|--:|--:|---|--:|--:|",
    ]
    for row in table:
        if f"{ARM}__diff_vs_ref" not in row:
            continue
        diff = row.get(f"{ARM}__diff_vs_ref")
        q = row.get(f"{ARM}__q")
        # Bold only what survives multiple-comparison correction. An uncorrected
        # p under 0.05 across 18 metrics is one expected false positive.
        bold = q is not None and q < 0.05
        name = f'{row.get("family", "")}/{row["metric"]}'
        cells = [
            f"**{name}**" if bold else name,
            "↑" if row.get("higher_is_better") else "↓",
            str(row.get("n_paired", "")),
            _fmt(row.get(f"{REF}__mean"), ".3f"),
            _fmt(row.get(f"{ARM}__mean"), ".3f"),
            f"**{_fmt(diff)}**" if bold else _fmt(diff),
            f"[{_fmt(row.get(f'{ARM}__diff_ci_low'))}, {_fmt(row.get(f'{ARM}__diff_ci_high'))}]",
            _fmt(row.get(f"{ARM}__p"), ".3g"),
            _fmt(q, ".3g"),
        ]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    return lines


def verdict(table: Sequence[dict[str, Any]]) -> list[str]:
    wins, losses, flat = [], [], 0
    for row in table:
        q = row.get(f"{ARM}__q")
        diff = row.get(f"{ARM}__diff_vs_ref")
        if diff is None:
            continue
        if q is None or q >= 0.05:
            flat += 1
            continue
        better = (diff > 0) == bool(row.get("higher_is_better"))
        (wins if better else losses).append(f'{row.get("family","")}/{row["metric"]}')
    out = [
        "### 判读",
        "",
        f"- 通过 FDR 的改善：{len(wins)} 项" + (f" — {', '.join(wins)}" if wins else ""),
        f"- 通过 FDR 的退步：{len(losses)} 项" + (f" — {', '.join(losses)}" if losses else ""),
        f"- 无显著差异：{flat} 项",
        "",
    ]
    if not wins and not losses:
        out.append(
            "> 两臂在所有指标上都无法区分。工具没有改善 rubric，也没有损害它 —— "
            "在这些指标上，无工具流水线已经把可测的空间占满了。"
        )
        out.append("")
    return out


def main() -> int:
    args = parse_args()
    args_summary = args.summary
    data = json.loads(Path(args_summary).read_text(encoding="utf-8"))
    if data.get("reference") != REF:
        raise SystemExit(f"summary reference is {data.get('reference')!r}, expected {REF!r}")

    parts = [f"# `{ARM}` vs `{REF}`", "",
             f"来源：`{args_summary}`；参照列 = `{REF}`；粗体 = BH-FDR 校正后 q<0.05。", ""]
    parts += render(data["pooled"], "两域合并")
    if args.by_domain:
        for domain, table in (data.get("by_domain") or {}).items():
            parts += render(table, f"分域：{domain}")
    parts += verdict(data["pooled"])

    text = "\n".join(parts)
    print(text)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
