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
    p.add_argument("--results-dir", nargs="+", default=[str(RESULTS)],
                   help="one or more directories to search for <source>_verdicts.jsonl; "
                        "candidate runs live under opt/runs, the shared controls in the root")
    p.add_argument("--case-ids", default=None,
                   help="restrict every comparison to these cases, e.g. "
                        "results/rubricbench/split.json:dev")
    p.add_argument("--label", default=None, help="name for the case subset, used in the header")
    p.add_argument("--out", default=None)
    return p.parse_args()


def load(source: str, roots: Sequence[Path]) -> dict[str, dict[str, Any]]:
    for root in roots:
        path = Path(root) / f"{source}_verdicts.jsonl"
        if path.exists():
            break
    else:
        raise SystemExit(f"no {source}_verdicts.jsonl in any of {[str(r) for r in roots]}")
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["case_id"]] = row
    return out


def hit(row: dict[str, Any] | None, mode: str = "forward") -> int | None:
    """1 if the verdict matched the human label, 0 if not, ``None`` if unusable.

    ``forward`` is the official single-pass protocol. ``swap`` is the
    position-controlled reading: a case counts as correct only if the same
    response wins in both presentation orders, and a case the judge flips on
    counts as **wrong** rather than being dropped -- the same convention
    VERIFY.md used, and the conservative one, since a verdict that depends on
    which response was shown first is not evidence about the rubric.

    Unparsable verdicts stay ``None``. The official evaluator counts them wrong;
    here they are held out of the paired tests and reported as coverage, because
    scoring a parse failure as a substantive error would attribute a judging bug
    to the rubric.
    """
    if row is None or row.get("forward") is None:
        return None
    if mode == "forward":
        return int(LETTER[row["forward"]] == row["label"])
    if mode == "swap":
        if row.get("swapped") is None:
            return None
        if row["forward"] != row["swapped"]:
            return 0
        return int(LETTER[row["forward"]] == row["label"])
    raise ValueError(f"unknown mode {mode!r}")


def mde(n_discordant: int, n_total: int) -> float:
    """Smallest |Δ ACC| a two-sided exact McNemar could call significant here.

    Reported because it is the difference between "these two are the same" and
    "this benchmark cannot tell them apart at this sample size". The published
    leaderboard's entire span is smaller than its own MDE.
    """
    if n_discordant == 0 or n_total == 0:
        return float("nan")
    for k in range(n_discordant // 2, -1, -1):
        tail = sum(math.comb(n_discordant, i) for i in range(k + 1)) / (2 ** n_discordant)
        if min(1.0, 2 * tail) < 0.05:
            return (n_discordant - 2 * k) / n_total
    return float("nan")


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
    roots = [Path(r) for r in args.results_dir]
    data = {s: load(s, roots) for s in args.sources}
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

    # -- the two readings the headline number hides -----------------------
    # SAFETY is 7% of the benchmark and carried 69% of `framed`'s net gain over
    # `baseline`, and part of that gain is a fight with our own judge prompt. A
    # score reported without the ex-SAFETY figure beside it can be dominated by
    # 80 cases. The swap-controlled column is here for the same reason in the
    # other direction: an effect that only exists in forward order is fragile.
    safety_ids = {c for c in common if group_of(domains[c]) == "safety"}
    nonsafety = [c for c in common if c not in safety_ids]
    emit("## 三种口径")
    emit()
    emit("| source | forward (全部) | forward (不含 SAFETY) | 位置受控 (全部) | 位置受控 (不含 SAFETY) |")
    emit("|---|--:|--:|--:|--:|")
    for source in args.sources:
        cells = []
        for mode in ("forward", "swap"):
            for ids in (common, nonsafety):
                hs = [hit(data[source].get(c), mode) for c in ids]
                if all(h is None for h in hs):
                    cells.append("—")
                else:
                    cells.append(f"{sum(0 if h is None else h for h in hs) / len(hs):.4f}")
        emit(f"| `{source}` | " + " | ".join(cells) + " |")
    emit()
    emit(f"不含 SAFETY 为 {len(nonsafety)} 题（SAFETY {len(safety_ids)} 题）。"
         "位置受控：正反两序判给同一个回答才算对，翻转计为错（与 VERIFY.md 同口径）；"
         "只有单序运行的来源在该列为 —。")
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

        # -- the same tests, restricted and re-scored -------------------------
        emit(f"## 同样的对照，换口径（参照 `{ref}`）")
        emit()
        emit("| source | Δ 不含 SAFETY | p | Δ 位置受控 | p | 判定 |")
        emit("|---|--:|--:|--:|--:|---|")
        for source in others:
            def paired(ids, mode):
                pr = [(hit(data[ref].get(c), mode), hit(data[source].get(c), mode)) for c in ids]
                pr = [(a, b) for a, b in pr if a is not None and b is not None]
                if not pr:
                    return None, None
                a_h, b_h = [x[0] for x in pr], [x[1] for x in pr]
                return (sum(b_h) - sum(a_h)) / len(pr), mcnemar(a_h, b_h)[2]

            d_ns, p_ns = paired(nonsafety, "forward")
            d_sw, p_sw = paired(common, "swap")
            full_p = dict((r[0], r[4]) for r in rows_out)[source]
            # "Fragile" is not a hedge, it is a specific finding: the effect
            # exists in the headline reading and dissolves in a stricter one.
            if d_ns is None:
                note = "—"
            elif full_p < 0.05 and (p_ns >= 0.05 or (p_sw is not None and p_sw >= 0.05)):
                note = "**脆弱**（总体显著，换口径后不显著）"
            elif full_p < 0.05:
                note = "稳健"
            else:
                note = "不显著"
            fmt = lambda d, p: ("—", "—") if d is None else (f"{d:+.4f}", f"{p:.3g}")
            a1, a2 = fmt(d_ns, p_ns)
            b1, b2 = fmt(d_sw, p_sw)
            emit(f"| `{source}` | {a1} | {a2} | {b1} | {b2} | {note} |")
        emit()

        # -- what this sample size can even see -------------------------------
        emit("## 最小可检出差（本样本量下）")
        emit()
        emit("| 对比 | 不一致对数 | 最小可检出 \\|Δ\\| | 实际 Δ |")
        emit("|---|--:|--:|--:|")
        for source in others:
            pr = [(hit(data[ref].get(c)), hit(data[source].get(c))) for c in common]
            pr = [(a, b) for a, b in pr if a is not None and b is not None]
            nd = sum(1 for a, b in pr if a != b)
            delta = (sum(x[1] for x in pr) - sum(x[0] for x in pr)) / len(pr)
            emit(f"| `{ref}` vs `{source}` | {nd}/{len(pr)} | {mde(nd, len(pr)):.4f} | {delta:+.4f} |")
        emit()
        emit("比这个下限小的 Δ 不该当成信号，无论它的符号看起来多顺眼。")
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
