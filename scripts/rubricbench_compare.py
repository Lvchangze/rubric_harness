#!/usr/bin/env python3
"""Compare RubricBench runs: paired significance, and where the gap actually is.

Every source is judged on the same cases, so the comparisons are paired and
McNemar's exact test is the right instrument — it conditions on the cases the
two sources disagree about and ignores the ones they both get right or both get
wrong, which is where the shared variance lives.

The second half is the part worth reading. Two controls bound the benchmark:
``none`` (no rubric) is the floor a source has to beat to have done anything at
all, and ``expert`` (human-annotated rubrics) is the ceiling reachable under
this judge. A source's score means little on its own; what it captures of the
gap between those two is the actual measurement.

Usage::

    python scripts/rubricbench_compare.py --sources none baseline agentic expert
    python scripts/rubricbench_compare.py --sources none framed contrastive \
        --reference framed --case-ids results/rubricbench/split.json:dev
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RESULTS = Path("results/rubricbench")
GROUPS = ["if", "stem", "code", "safety", "chat"]
LETTER = {"A": 0, "B": 1}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--floor", default="none")
    p.add_argument("--ceiling", default="expert")
    p.add_argument("--reference", default="baseline", help="source the pairwise tests compare against")
    p.add_argument("--results-dir", default=str(RESULTS))
    p.add_argument("--case-ids", default=None,
                   help="restrict every comparison to these cases, e.g. "
                        "results/rubricbench/split.json:dev")
    p.add_argument("--label", default=None, help="name for the case subset, used in the header")
    p.add_argument("--out", default=None)
    return p.parse_args()


def load(source: str, root: Path) -> dict[str, dict[str, Any]]:
    path = root / f"{source}_verdicts.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["case_id"]] = row
    return out


def hit(row: dict[str, Any] | None) -> int | None:
    """1 if the forward-order verdict matched the human label, 0 if not.

    ``None`` for an unparsable verdict. The official evaluator counts those as
    wrong; here they are held out of the paired tests and reported as coverage,
    because scoring a parse failure as a substantive error would attribute a
    judging bug to the rubric.
    """
    if row is None or row.get("forward") is None:
        return None
    return int(LETTER[row["forward"]] == row["label"])


def mcnemar(a: Sequence[int], b: Sequence[int]) -> tuple[int, int, float]:
    """Exact two-sided McNemar on paired 0/1 outcomes. Returns (b_wins, a_wins, p)."""
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)   # b better
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)   # a better
    n = n01 + n10
    if n == 0:
        return 0, 0, 1.0
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return n01, n10, min(1.0, 2 * tail)


def bh_fdr(pvalues: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg q-values, in the input order.

    Candidates are compared against the same reference on the same cases, so the
    per-test p-values are one family and reading them at 0.05 each would let a
    false positive through roughly once per twenty candidates tried. The
    step-up transform is enforced monotone: a q-value may never fall below the
    q-value of a smaller p.
    """
    m = len(pvalues)
    if m == 0:
        return []
    order = sorted(range(m), key=lambda i: pvalues[i])
    qs = [0.0] * m
    running = 1.0
    for rank, idx in reversed(list(enumerate(order, start=1))):
        running = min(running, pvalues[idx] * m / rank)
        qs[idx] = min(1.0, running)
    return qs


def main() -> int:
    args = parse_args()
    root = Path(args.results_dir)
    data = {s: load(s, root) for s in args.sources}
    common = sorted(set.intersection(*(set(d) for d in data.values())))
    subset_label = args.label
    if args.case_ids:
        from rubricbench_split import load_case_ids  # noqa: PLC0415

        wanted = set(load_case_ids(args.case_ids))
        missing = wanted - set(common)
        common = sorted(c for c in common if c in wanted)
        subset_label = subset_label or args.case_ids
        if missing:
            print(f"note: {len(missing)} requested cases are absent from at least one source")
    if not common:
        raise SystemExit("the requested sources share no cases")

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    where = f"（{subset_label}）" if subset_label else ""
    emit(f"# RubricBench{where}: {len(common)} cases judged identically across "
         f"{len(args.sources)} rubric sources")
    emit()
    emit("判定器与生成同模型（`hy-t2t-glm-5.2-384k-fp8-L20A-t1-v2`）；ACC 为官方口径的"
         "单序（forward）准确率，未答计为错。")
    emit()

    # -- headline table ---------------------------------------------------
    domains = {cid: data[args.sources[0]][cid]["domain"] for cid in common}
    from harness.rubricbench import group_of  # noqa: PLC0415

    emit("## 总表")
    emit()
    emit("| source | " + " | ".join(g.upper() for g in GROUPS) + " | Overall | 覆盖率 | 位置一致率 |")
    emit("|---|" + "---|" * (len(GROUPS) + 3))
    acc: dict[str, float] = {}
    for source in args.sources:
        rows = [data[source][cid] for cid in common]
        hits = [hit(r) for r in rows]
        per_group: dict[str, list[int]] = {}
        for cid, h in zip(common, hits):
            g = group_of(domains[cid])
            if g:
                per_group.setdefault(g, []).append(0 if h is None else h)
        overall = sum(0 if h is None else h for h in hits) / len(hits)
        acc[source] = overall
        cov = sum(1 for h in hits if h is not None) / len(hits)
        # A run made with --single-order has no swapped pass, so consistency is
        # undefined rather than zero. The forward call is byte-identical either
        # way, so ACC stays comparable; only this diagnostic goes missing.
        has_swap = any(r.get("swapped") is not None for r in rows)
        cons = (f"{sum(1 for r in rows if r.get('consistent')) / len(rows):.3f}"
                if has_swap else "—")
        cells = [f"{sum(per_group.get(g, [0])) / max(1, len(per_group.get(g, [1]))):.4f}" for g in GROUPS]
        emit(f"| `{source}` | " + " | ".join(cells) +
             f" | **{overall:.4f}** | {cov:.3f} | {cons} |")
    emit()

    # -- headroom ---------------------------------------------------------
    floor, ceiling = args.floor, args.ceiling
    if floor in acc and ceiling in acc:
        span = acc[ceiling] - acc[floor]
        emit("## 相对可用空间")
        emit()
        emit(f"地板 `{floor}`（不给 rubric）= {acc[floor]:.4f}；"
             f"天花板 `{ceiling}`（专家标注）= {acc[ceiling]:.4f}；"
             f"**可用空间 = {span:+.4f}**。")
        emit()
        emit("| source | ACC | 相对地板 | 吃到的空间 |")
        emit("|---|--:|--:|--:|")
        for source in args.sources:
            if source in (floor, ceiling):
                continue
            gain = acc[source] - acc[floor]
            emit(f"| `{source}` | {acc[source]:.4f} | {gain:+.4f} | "
                 f"**{gain / span * 100:.1f}%** |" if span else "| — |")
        emit()

    # -- pairwise ---------------------------------------------------------
    ref = args.reference
    if ref in data:
        others = [s for s in args.sources if s != ref]
        rows_out = []
        for source in others:
            pairs = [(hit(data[ref][c]), hit(data[source][c])) for c in common]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            a_hits, b_hits = [p[0] for p in pairs], [p[1] for p in pairs]
            wins, losses, p = mcnemar(a_hits, b_hits)
            rows_out.append((source, wins, losses, (sum(b_hits) - sum(a_hits)) / len(pairs), p))
        qs = bh_fdr([r[4] for r in rows_out])

        emit(f"## 配对显著性（McNemar 精确检验，参照 `{ref}`；BH-FDR 校正）")
        emit()
        emit(f"| source | 仅该源答对 | 仅 `{ref}` 答对 | Δ ACC | p | q (BH) |")
        emit("|---|--:|--:|--:|--:|--:|")
        for (source, wins, losses, delta, p), q in zip(rows_out, qs):
            star = " **" if q < 0.05 else ""
            emit(f"| `{source}` | {wins} | {losses} | {delta:+.4f} | {p:.3g} | {q:.3g}{star} |")
        emit()
        emit(f"q 为在这 {len(rows_out)} 个对照上做 Benjamini-Hochberg 校正后的值；"
             "`**` 表示 q < 0.05。")
        emit()

        # -- per-group deltas against the reference -----------------------
        # The overall number hides which domain moved: a candidate can gain on
        # SAFETY (80 cases) and lose on CHAT (422) and still look flat.
        emit(f"## 分域 Δ（相对 `{ref}`）")
        emit()
        emit("| source | " + " | ".join(f"{g.upper()}" for g in GROUPS) + " |")
        emit("|---|" + "--:|" * len(GROUPS))
        for source in others:
            cells = []
            for g in GROUPS:
                pairs = [(hit(data[ref][c]), hit(data[source][c])) for c in common
                         if group_of(domains[c]) == g]
                pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
                if not pairs:
                    cells.append("—")
                    continue
                d = (sum(p[1] for p in pairs) - sum(p[0] for p in pairs)) / len(pairs)
                cells.append(f"{d:+.4f}")
            emit(f"| `{source}` | " + " | ".join(cells) + " |")
        emit()

    # -- where the ceiling wins -------------------------------------------
    if ceiling in data and ref in data:
        emit(f"## `{ceiling}` 赢在哪里（相对 `{ref}`）")
        emit()
        emit(f"| 域 | n | `{ref}` | `{ceiling}` | Δ | 仅 `{ceiling}` 答对 |")
        emit("|---|--:|--:|--:|--:|--:|")
        by_group: dict[str, list[tuple[int, int]]] = {}
        for cid in common:
            g = group_of(domains[cid])
            a, b = hit(data[ref][cid]), hit(data[ceiling][cid])
            if g and a is not None and b is not None:
                by_group.setdefault(g, []).append((a, b))
        for g in GROUPS:
            pairs = by_group.get(g) or []
            if not pairs:
                continue
            a_acc = sum(p[0] for p in pairs) / len(pairs)
            b_acc = sum(p[1] for p in pairs) / len(pairs)
            only = sum(1 for a, b in pairs if a == 0 and b == 1)
            emit(f"| {g.upper()} | {len(pairs)} | {a_acc:.4f} | {b_acc:.4f} | "
                 f"{b_acc - a_acc:+.4f} | {only} |")
        emit()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    raise SystemExit(main())
