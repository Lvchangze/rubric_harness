#!/usr/bin/env python
"""Render the rollout-oriented (RL-faithful) evaluation tables.

Reads the per-question and per-rollout rows written by
``harness/eval/rollout_metrics.py`` and produces, pooled and per domain:

* the main metric table with paired contrasts and BH-FDR q-values,
* best-of-n selection accuracy against its random floor and oracle ceiling,
* the real reward-hacking probe (score vs length among oracle-incorrect rollouts),
* the oracle reliability and label-balance summary.

The FDR family is declared here, in code, rather than described in prose:
:data:`FDR_FAMILY` fixes which metrics are corrected together, and the family
size is written into the output so a reader can check it.

    python scripts/rollout_report.py --run-name pilot_v2 --reference baseline
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.eval.rollouts import load_rollout_sets, oracle_report  # noqa: E402
from harness.eval.stats import bh_fdr, mean_ci, paired_diff  # noqa: E402

logger = logging.getLogger("rollout_report")

SOURCES = [
    "shipped", "baseline", "agentic-noval", "agentic",
    "agentic-goldonly", "agentic-negonly", "agentic-realneg",
]

#: The BH-FDR family, fixed in code. Correcting over "whatever landed in the
#: table" makes every q-value depend on which metrics happened to be run, so the
#: same comparison can report different significance on a re-run. These are the
#: metrics that carry a claim; descriptive columns are excluded deliberately.
FDR_FAMILY: list[tuple[str, str, bool]] = [
    ("auc", "AUC(rubric score, oracle correctness)", True),
    ("best_of_n_accuracy", "Best-of-n 选择准确率", True),
    ("best_of_n_lift", "Best-of-n 相对随机基线的提升", True),
    ("spearman", "Spearman(score, correctness)", True),
    ("separation", "分数间隔（正确 − 错误）", True),
    ("z_separation", "尺度归一化间隔", True),
]

DESCRIPTIVE: list[tuple[str, str]] = [
    ("mean_correct", "正确 rollout 的平均分"),
    ("mean_incorrect", "错误 rollout 的平均分"),
    ("score_std", "题内分数标准差"),
    ("random_baseline", "随机选择基线（= 该题正确率）"),
]


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise SystemExit(f"missing {path}; run scripts/eval_rubrics.py --metrics rollout")
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def cohort(rows: Sequence[Mapping[str, Any]], sources: Sequence[str], metric: str) -> list[str]:
    """Question ids with a finite value for this metric in *every* source."""
    by_source: dict[str, set[str]] = {s: set() for s in sources}
    for row in rows:
        source = row.get("rubric_source")
        if source not in by_source:
            continue
        value = row.get(metric)
        if isinstance(value, (int, float)) and math.isfinite(value):
            by_source[source].add(str(row["uid"]))
    if not all(by_source.values()):
        return []
    return sorted(set.intersection(*by_source.values()))


def contrast_table(
    rows: Sequence[Mapping[str, Any]],
    *,
    sources: Sequence[str],
    reference: str,
    iters: int,
    seed: int,
) -> dict[str, Any]:
    values: dict[str, dict[str, float]] = {}
    for row in rows:
        source = str(row.get("rubric_source"))
        values.setdefault(source, {})[str(row["uid"])] = row

    out: list[dict[str, Any]] = []
    for metric, label, higher in FDR_FAMILY + [(m, l, None) for m, l in DESCRIPTIVE]:
        uids = cohort(rows, sources, metric)
        if not uids:
            continue
        record: dict[str, Any] = {
            "metric": metric, "label": label, "higher_is_better": higher,
            "n_paired": len(uids), "reference": reference,
        }
        ref_values = [float(values[reference][u][metric]) for u in uids]
        for source in sources:
            series = [float(values[source][u][metric]) for u in uids]
            record[f"{source}__mean"] = float(np.mean(series))
            stats = mean_ci(series, iters=iters, seed=seed)
            record[f"{source}__ci_low"] = stats["ci_low"]
            record[f"{source}__ci_high"] = stats["ci_high"]
            if source != reference:
                diff = paired_diff(series, ref_values, iters=iters, seed=seed)
                record[f"{source}__diff"] = diff["mean_diff"]
                record[f"{source}__diff_ci_low"] = diff["ci_low"]
                record[f"{source}__diff_ci_high"] = diff["ci_high"]
                record[f"{source}__p"] = diff["p_value"]
        out.append(record)

    # BH within the declared family only; descriptive rows are not corrected
    # because they carry no claim, and folding them in would dilute the family.
    family = {m for m, _, _ in FDR_FAMILY}
    fdr_meta: dict[str, Any] = {"family": sorted(family), "per_source": {}}
    for source in sources:
        if source == reference:
            continue
        idx = [
            i for i, r in enumerate(out)
            if r["metric"] in family
            and isinstance(r.get(f"{source}__p"), float)
            and math.isfinite(r[f"{source}__p"])
        ]
        if not idx:
            continue
        for i, q in zip(idx, bh_fdr([out[i][f"{source}__p"] for i in idx])):
            out[i][f"{source}__q"] = q
        fdr_meta["per_source"][source] = {
            "family_n": len(idx), "metrics": [out[i]["metric"] for i in idx]
        }
    return {"rows": out, "fdr": fdr_meta}


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}"


def render(table: Mapping[str, Any], sources: Sequence[str], reference: str, title: str) -> str:
    rows = table["rows"]
    contrasts = [s for s in sources if s != reference]
    lines = [
        f"### {title}",
        "",
        "| 指标 | 方向 | n | " + " | ".join(f"`{s}`" for s in sources) + " | "
        + " | ".join(f"Δ `{s}`" for s in contrasts) + " |",
        "|---|---|---|" + "---|" * (len(sources) + len(contrasts)),
    ]
    for r in rows:
        arrow = {True: "↑", False: "↓", None: "·"}[r["higher_is_better"]]
        cells = [fmt(r.get(f"{s}__mean")) for s in sources]
        deltas = []
        for s in contrasts:
            d = r.get(f"{s}__diff")
            if d is None or not math.isfinite(d):
                deltas.append("—")
                continue
            q, p = r.get(f"{s}__q"), r.get(f"{s}__p")
            tail = f", q={q:.3g}" if isinstance(q, float) else ""
            body = f"{d:+.3f} (p={p:.2g}{tail})"
            deltas.append(f"**{body}**" if isinstance(q, float) and q < 0.05 else body)
        lines.append(
            f"| {r['label']} | {arrow} | {r['n_paired']} | "
            + " | ".join(cells) + " | " + " | ".join(deltas) + " |"
        )
    lines += [
        "",
        f"粗体 = BH-FDR 校正后 q<0.05；族为 {len(table['fdr']['family'])} 个判别类指标"
        f"（{', '.join(table['fdr']['family'])}），描述性指标不参与校正。"
        f"Δ 为与 `{reference}` 的配对差值。",
    ]
    return "\n".join(lines)


def length_hack(per_rollout: Sequence[Mapping[str, Any]], sources: Sequence[str]) -> str:
    lines = [
        "### Reward hacking 探针（真实版）",
        "",
        "在 **oracle 判错**的 rollout 子集里，rubric 分数与回答长度的相关性。"
        "正相关 = rubric 在奖励\"写得长的错误答案\"。",
        "",
        "| 来源 | Pearson r | n(错误 rollout) |",
        "|---|---|---|",
    ]
    for source in sources:
        pairs = [
            (float(r["score"]), float(r["n_chars"]))
            for r in per_rollout
            if r.get("rubric_source") == source and r.get("correct") is False
        ]
        if len(pairs) < 3:
            lines.append(f"| `{source}` | — | {len(pairs)} |")
            continue
        a = np.array([p[0] for p in pairs])
        b = np.array([p[1] for p in pairs])
        r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        lines.append(f"| `{source}` | {fmt(r)} | {len(pairs)} |")
    return "\n".join(lines)


def best_of_n_block(rows: Sequence[Mapping[str, Any]], sources: Sequence[str]) -> str:
    uids = cohort(rows, sources, "best_of_n_accuracy")
    by_source: dict[str, dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        by_source.setdefault(str(row["rubric_source"]), {})[str(row["uid"])] = row
    lines = [
        "### Best-of-n 选择准确率",
        "",
        "用 rubric 分数从该题的 k 个 rollout 里挑最高分的一个，它实际正确的比例。"
        "并列时按期望值计分，因此\"全部打同分\"的 rubric 恰好落在随机基线上。",
        "",
        f"| 来源 | Best-of-n 准确率 | 相对随机基线 | n={len(uids)} 题 |",
        "|---|---|---|---|",
    ]
    if uids:
        floor = float(np.mean([by_source[sources[0]][u]["random_baseline"] for u in uids]))
        for source in sources:
            acc = [float(by_source[source][u]["best_of_n_accuracy"]) for u in uids]
            stats = mean_ci(acc)
            lines.append(
                f"| `{source}` | {fmt(stats['mean'])} "
                f"[{fmt(stats['ci_low'])}, {fmt(stats['ci_high'])}] | "
                f"{float(np.mean(acc)) - floor:+.3f} | |"
            )
        lines += [
            f"| *随机选择（下界）* | {fmt(floor)} | 0.000 | |",
            "| *oracle（上界）* | 1.000 | " + f"{1.0 - floor:+.3f}" + " | |",
        ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default="pilot_v2")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--sources", nargs="+", default=None)
    p.add_argument("--reference", default="baseline")
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    run_path = Path(args.runs_dir) / args.run_name
    out_dir = Path(args.results_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    per_question = load_rows(run_path / "metrics" / "rollout_per_question_rows.jsonl")
    per_rollout = load_rows(run_path / "metrics" / "rollout_per_rollout_rows.jsonl")
    available = {str(r["rubric_source"]) for r in per_question}
    sources = args.sources or [s for s in SOURCES if s in available]
    logger.info("sources: %s", sources)

    report = oracle_report(load_rollout_sets(run_path))
    payload: dict[str, Any] = {"oracle": report, "sources": sources,
                               "reference": args.reference}

    parts = [
        "# 面向 rollout 的评测（RL 忠实协议）",
        "",
        "## Oracle 与 rollout 质量分布",
        "",
        "| 项 | 值 |",
        "|---|---|",
    ]
    for key, label in [
        ("n_questions", "题数"),
        ("n_rollouts_total", "rollout 总数"),
        ("n_rollouts_usable", "可用 rollout 数"),
        ("rollout_correct_rate", "rollout 总体正确率"),
        ("mean_per_question_correct_rate", "逐题正确率均值"),
        ("oracle_self_agreement", "oracle 二次判定自洽率"),
        ("n_oracle_agreement_probed", "自洽率抽查条数"),
        ("low_confidence_rate", "oracle 低置信占比"),
        ("missing_final_answer_rate", "无法抽出最终答案占比"),
        ("n_informative_questions", "有效题数（对错混合）"),
        ("n_all_correct_questions", "全对题数（排除）"),
        ("n_all_incorrect_questions", "全错题数（排除）"),
    ]:
        value = report.get(key)
        parts.append(f"| {label} | {fmt(value) if isinstance(value, float) else value} |")

    pooled = contrast_table(per_question, sources=sources, reference=args.reference,
                            iters=args.iters, seed=args.seed)
    payload["pooled"] = pooled
    parts += ["", render(pooled, sources, args.reference, "主指标表（两域合并）")]
    parts += ["", best_of_n_block(per_question, sources)]
    parts += ["", length_hack(per_rollout, sources)]

    payload["by_domain"] = {}
    for domain in sorted({str(r.get("domain")) for r in per_question if r.get("domain")}):
        sub = [r for r in per_question if r.get("domain") == domain]
        table = contrast_table(sub, sources=sources, reference=args.reference,
                               iters=args.iters, seed=args.seed)
        payload["by_domain"][domain] = table
        parts += ["", render(table, sources, args.reference, f"分域：`{domain}`")]
        parts += ["", best_of_n_block(sub, sources)]

    (out_dir / "rollout_report.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False, default=str), encoding="utf-8"
    )
    (out_dir / "rollout_report.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    print("\n".join(parts))
    logger.info("wrote %s", out_dir / "rollout_report.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
