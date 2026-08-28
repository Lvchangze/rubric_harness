#!/usr/bin/env python3
"""Collect per-question metric rows into the report's summary tables.

Reads ``runs/<run>/metrics/*_per_question_rows.jsonl``, aligns every metric on
the questions all sources share, and emits to ``results/<run>/``:

``summary.json``
    Every metric with per-source mean, bootstrap CI and the paired contrast
    against the reference source.
``summary.csv`` / ``per_question_rows.csv``
    The same table, and the raw long frame, for spreadsheet use.
``summary_table.md``
    The rendered tables the report quotes: the main metric table, the
    distribution-level table (item-count adaptivity and weight informativeness,
    which have no per-question pairing), and the Pitfall polarity sensitivity
    analysis (forensics F6).

Usage::

    python scripts/aggregate_results.py --run-name pilot_v2
    python scripts/aggregate_results.py --run-name pilot_v2 --prefix cc_
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import harness.eval.stats as st  # noqa: E402
from harness.tracing import read_jsonl  # noqa: E402

logger = logging.getLogger("aggregate_results")

DEFAULT_SOURCES = ["shipped", "baseline", "agentic-noval", "agentic"]

# (family, column, higher_is_better, label). higher_is_better None = descriptive.
METRIC_SPEC: list[tuple[str, str, bool | None, str]] = [
    ("discriminative", "mean_margin", True, "Mean margin (gold - degraded)"),
    ("discriminative", "min_margin", True, "Min margin over degradations"),
    ("discriminative", "rank_acc", True, "Pairwise ranking accuracy"),
    ("discriminative", "auc_good_vs_bad", True, "AUC (good vs bad)"),
    ("discriminative", "auc_gold_vs_degraded", True, "AUC (gold vs all degraded)"),
    ("discriminative", "score_range", True, "Score dynamic range (max-min)"),
    ("discriminative", "score_std", True, "Score std across variants"),
    ("discriminative", "saturation_rate", False, "Saturation (degraded scoring >=0.8)"),
    ("discriminative", "score__gold", None, "Score: gold"),
    ("discriminative", "score__verbose_empty", False, "Score: verbose-but-empty (hack probe)"),
    ("discriminative", "margin__verbose_empty", True, "Margin vs verbose-but-empty"),
    ("discriminative", "score__right_method_wrong_answer", False, "Score: confidently wrong"),
    ("discriminative", "margin__right_method_wrong_answer", True, "Margin vs confidently wrong"),
    ("discriminative", "score__off_topic", False, "Score: off-topic"),
    ("discriminative", "margin__off_topic", True, "Margin vs off-topic"),
    ("grounding", "generic_criterion_rate", False, "Generic criterion rate"),
    ("grounding", "generic_rate_permuted", None, "…permutation control"),
    ("grounding", "weight_mass_on_generic_paper", False, "Reward mass on generic criteria"),
    ("grounding", "mean_anchors_per_criterion", True, "Anchors per criterion"),
    ("grounding", "weak_grounding_rate", False, "Weakly grounded (<2 anchors)"),
    ("grounding", "recyclable_rate", False, "Recyclable criterion rate"),
    ("grounding", "subjective_or_style_rate", False, "Subjective/style criterion rate"),
    ("grounding", "weight_mass_on_subjective", False, "Reward mass on subjective criteria"),
    ("grounding", "specificity_lift", True, "Specificity lift vs permuted"),
    ("transfer", "same_question_pass_rate", None, "Same-question pass rate"),
    ("transfer", "cross_pass_rate", False, "Cross-question pass rate"),
    ("transfer", "specificity_gap", True, "Specificity gap"),
    ("transfer", "boilerplate_fraction", False, "Boilerplate criterion fraction"),
    ("transfer", "specific_fraction", True, "Query-specific criterion fraction"),
    ("coverage", "claim_recall", True, "Gold-claim recall"),
    ("coverage", "claim_recall_core", True, "Core gold-claim recall"),
    ("coverage", "grounded_fraction", True, "Grounded criterion fraction"),
    ("coverage", "generic_fraction", False, "Generic criterion fraction"),
    ("coverage", "unsupported_fraction", False, "Unsupported criterion fraction"),
    ("coverage", "subquestion_coverage", True, "Sub-question coverage"),
    ("intrinsic", "objective_fraction", True, "Objectively checkable fraction"),
    ("intrinsic", "self_agreement_rate", True, "Judge self-agreement rate"),
    ("intrinsic", "kappa", True, "Judge self-agreement kappa"),
    ("intrinsic", "mean_checks_per_criterion", False, "Checks per criterion (atomicity)"),
    ("intrinsic", "atomic_fraction", True, "Atomic criterion fraction"),
    ("intrinsic", "weight_entropy", None, "Weight-distribution entropy"),
    ("lint", "compliance_rate", True, "RaR prompt compliance rate"),
    ("lint", "polarity_single_rate", True, "Single-polarity rubric rate (F6)"),
    ("lint", "unclassified_polarity_rate", False, "Unclassified polarity rate"),
    ("lint", "negative_weight_rate", False, "Negative-weight criterion rate"),
    ("lint", "weight_order_violation_rate", False, "Weight-order violations"),
    ("lint", "pitfall_mirror_rate", False, "Pitfall mirrors a positive item (F7)"),
    ("adaptivity", "n_items", None, "Criteria per rubric"),
    ("adaptivity", "items_per_subpart", None, "Criteria per sub-question"),
    ("adaptivity", "orphan_subpart_rate", False, "Orphan sub-questions"),
]

ARROW = {True: "↑", False: "↓", None: "·"}

#: Below this many paired questions a bootstrap CI is decoration, not evidence,
#: so the row is flagged in the rendered table rather than read at face value.
MIN_TRUSTWORTHY_N = 30

#: Metrics defined only on questions with a particular shape, so their paired
#: cohort is legitimately smaller than the run's. They are exempt from the
#: cohort assertion — see :func:`check_cohorts`.
SPARSE_BY_CONSTRUCTION = frozenset({
    "adaptivity/orphan_subpart_rate",
    "coverage/subquestion_coverage",
})


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default="pilot_v2")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--sources", nargs="+", default=None)
    p.add_argument("--reference", default="baseline")
    p.add_argument("--prefix", default="", help="'cc_' to aggregate the count-controlled pass")
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--expect-paired", type=int, default=None,
                   help="required paired-cohort size for every metric; guards against "
                        "a newly added source silently changing the cohort")
    p.add_argument("--expect-metrics", type=int, default=len(METRIC_SPEC),
                   help="minimum number of metric rows; catches a metric family "
                        "whose per-question rows were lost")
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero when a cohort or metric-count check fails")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def load_rows(run_path: Path, prefix: str) -> pd.DataFrame:
    """Concatenate each metric family's per-question rows into one long frame."""
    frames: list[pd.DataFrame] = []
    metrics_dir = run_path / "metrics"
    for path in sorted(metrics_dir.glob("*_per_question_rows.jsonl")):
        name = path.name
        # With no prefix, skip the count-controlled files, and vice versa.
        if prefix:
            if not name.startswith(prefix):
                continue
            family = name[len(prefix):].replace("_per_question_rows.jsonl", "")
        else:
            if name.startswith("cc_"):
                continue
            family = name.replace("_per_question_rows.jsonl", "")
        rows = list(read_jsonl(path))
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["_family"] = family
        frames.append(df)
        logger.info("loaded %s: %d rows", name, len(df))
    if not frames:
        raise SystemExit(f"no per-question metric rows under {metrics_dir}")
    return pd.concat(frames, ignore_index=True, sort=False)


def compute_table(
    df: pd.DataFrame,
    *,
    sources: list[str],
    reference: str,
    iters: int,
    seed: int,
    domain: str | None = None,
) -> list[dict[str, Any]]:
    """Paired contrasts for every metric in :data:`METRIC_SPEC`.

    Only rows belonging to ``sources`` are handed to the pairing step. This is
    load-bearing rather than cosmetic: ``summarize_metric`` intersects question
    ids across *every* group it is given, so an extra source present in the file
    but absent from the table would silently shrink the paired cohort for the
    sources that are displayed — no error, just quietly different numbers. An
    ablation evaluated on a slightly different question set is exactly the case
    that triggers it.

    ``domain`` restricts the analysis to one corpus. Pooled results can be
    carried by a single domain, so every table is also emitted per domain.
    """
    if domain:
        df = df[df.get("domain") == domain]
    out: list[dict[str, Any]] = []
    for family, column, higher_better, label in METRIC_SPEC:
        sub = df[(df["_family"] == family) & (df["rubric_source"].isin(sources))]
        if sub.empty or column not in sub.columns:
            continue
        present = [s for s in sources if s in set(sub["rubric_source"])]
        if not present:
            continue
        rows = sub[["uid", "rubric_source", column]].dropna(subset=[column]).to_dict("records")
        if not rows:
            continue
        ref = reference if reference in present else present[0]
        try:
            summary = st.summarize_metric(
                rows, value=column, group="rubric_source", key="uid",
                baseline_group=ref, iters=iters, seed=seed,
            )
        except Exception:  # noqa: BLE001 - never lose the whole table to one metric
            logger.exception("summarize_metric failed for %s/%s", family, column)
            continue
        record: dict[str, Any] = {
            "family": family,
            "metric": column,
            "label": label,
            "higher_is_better": higher_better,
            "reference": ref,
            "domain": domain or "all",
            "sources_in_cohort": present,
            "n_paired": summary.get("n_paired"),
        }
        per_group = summary.get("per_group") or {}
        for source in present:
            entry = per_group.get(source) or {}
            # The displayed mean is the *paired* mean, over the same questions
            # the contrast is computed on. Showing the marginal mean instead
            # makes the row fail to add up whenever a source is missing
            # questions the others have (kappa is undefined wherever a judge
            # marks every criterion alike), so the reader sees a delta that is
            # not the difference of the two numbers printed beside it.
            record[f"{source}__mean"] = entry.get("paired_mean", entry.get("mean"))
            record[f"{source}__marginal_mean"] = entry.get("mean")
            record[f"{source}__ci_low"] = entry.get("ci_low")
            record[f"{source}__ci_high"] = entry.get("ci_high")
            record[f"{source}__se"] = entry.get("se")
            record[f"{source}__n"] = entry.get("n")
            diff = entry.get("vs_baseline") or {}
            record[f"{source}__diff_vs_ref"] = diff.get("mean_diff")
            record[f"{source}__diff_ci_low"] = diff.get("ci_low")
            record[f"{source}__diff_ci_high"] = diff.get("ci_high")
            record[f"{source}__p"] = diff.get("p_value")
        out.append(record)
    return out


def check_cohorts(
    table: list[dict[str, Any]], *, expect: int | None, label: str = "main"
) -> list[str]:
    """Flag any metric whose paired cohort differs from the rest of the table.

    A silent change in cohort size is the failure mode that adding a source with
    a different question set produces, so it is raised as an error rather than
    left for a reader to notice in a footnote.

    Metrics that are only defined on a subset of questions are exempt.
    ``orphan_subpart_rate`` needs a question with multiple sub-parts and lands on
    a handful; flagging it every single time would train the reader to ignore the
    message, and an alarm that is always on is not an alarm. Those rows already
    carry a ``⚠`` and are declared unusable for conclusions. Exempting them is
    what lets the caller pass ``--strict`` and have a real drift abort the run.
    """
    problems: list[str] = []
    sizes = {r["metric"]: r.get("n_paired") for r in table}
    if expect is not None:
        for metric, n in sorted(sizes.items()):
            if n == expect or metric in SPARSE_BY_CONSTRUCTION:
                continue
            problems.append(f"[{label}] {metric}: n_paired={n}, expected {expect}")
    return problems


def apply_fdr(
    table: list[dict[str, Any]], sources: Iterable[str], reference: str
) -> dict[str, Any]:
    """BH-correct the paired p-values, over a family fixed by :data:`METRIC_SPEC`.

    The family is declared as "every metric in ``METRIC_SPEC`` that this source
    produced a p-value for". Pinning it to the declared spec — rather than to
    whatever happened to land in the table — keeps q-values stable when a run is
    repeated with a different metric subset, which would otherwise silently
    change every q in the report.
    """
    meta: dict[str, Any] = {"declared_family": len(METRIC_SPEC), "per_source": {}}
    for source in sources:
        if source == reference:
            continue
        key = f"{source}__p"
        idx = [
            i for i, r in enumerate(table)
            if isinstance(r.get(key), (int, float)) and not math.isnan(r[key])
        ]
        if not idx:
            continue
        corrected = st.bh_fdr([table[i][key] for i in idx])
        for i, q in zip(idx, corrected):
            table[i][f"{source}__q"] = q
            table[i][f"{source}__fdr_family_n"] = len(idx)
        meta["per_source"][source] = {
            "family_n": len(idx),
            "metrics": [table[i]["metric"] for i in idx],
        }
    return meta


def fmt(value: Any, digits: int = 3) -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    return f"{value:.{digits}f}"


def render_main(table: list[dict[str, Any]], sources: list[str], reference: str) -> str:
    contrasts = [s for s in sources if s != reference]
    head = "| Metric | 方向 | n | " + " | ".join(f"`{s}`" for s in sources) + " | "
    head += " | ".join(f"Δ `{s}` − `{reference}` (95% CI, p)" for s in contrasts) + " |"
    lines = [head, "|---|---|---|" + "---|" * (len(sources) + len(contrasts))]
    current = None
    for r in table:
        if r["family"] != current:
            current = r["family"]
            lines.append(f"| **{current}** | | | " + " | " * (len(sources) + len(contrasts)) + "|")
        cells = [fmt(r.get(f"{s}__mean")) for s in sources]
        deltas = []
        for s in contrasts:
            d = r.get(f"{s}__diff_vs_ref")
            if d is None or (isinstance(d, float) and not math.isfinite(d)):
                deltas.append("—")
                continue
            q = r.get(f"{s}__q")
            body = (f"{d:+.3f} [{fmt(r.get(f'{s}__diff_ci_low'))}, "
                    f"{fmt(r.get(f'{s}__diff_ci_high'))}], p={r.get(f'{s}__p'):.2g}")
            deltas.append(f"**{body}**" if isinstance(q, float) and q < 0.05 else body)
        # A cohort far below the table's norm cannot support a bootstrap CI, and
        # the reader has no other way to tell from the row itself.
        n = r.get("n_paired")
        n_cell = f"{n}" if isinstance(n, int) else "—"
        if isinstance(n, int) and n < MIN_TRUSTWORTHY_N:
            n_cell = f"**{n}** ⚠"
        lines.append(
            f"| {r['label']} | {ARROW[r['higher_is_better']]} | {n_cell} | "
            + " | ".join(cells) + " | " + " | ".join(deltas) + " |"
        )
    lines.append("")
    lines.append(
        "方向: ↑ 越大越好, ↓ 越小越好, · 描述性指标。粗体表示 BH-FDR 校正后 q<0.05。"
        f"括号内为 bootstrap 95% 置信区间；Δ 列为与 `{reference}` 的配对差值。"
        f"各来源列为**配对均值**（在同一批 n 题上），因此 Δ 恰为两列之差。"
        f"n 列标 ⚠ 表示配对题数 < {MIN_TRUSTWORTHY_N}，该行的置信区间不可用于下结论。"
    )
    return "\n".join(lines)


def render_distribution(df: pd.DataFrame, resp: pd.DataFrame, sources: list[str]) -> str:
    """Item-count adaptivity (F1) and weight informativeness (F5).

    These are properties of a source's *distribution* over questions rather than
    of any single question, so there is no paired contrast to report.
    """
    adapt = df[df["_family"] == "adaptivity"]
    rows: list[tuple[str, str, list[str]]] = []

    def per_source(fn) -> list[str]:
        out = []
        for s in sources:
            sub = adapt[adapt["rubric_source"] == s]
            out.append("—" if sub.empty else fn(sub))
        return out

    def cv(sub: pd.DataFrame) -> str:
        n = pd.to_numeric(sub["n_items"], errors="coerce")
        return fmt(n.std(ddof=0) / n.mean())

    def modal(sub: pd.DataFrame) -> str:
        n = pd.to_numeric(sub["n_items"], errors="coerce")
        return fmt(n.value_counts(normalize=True).max())

    def corr(col: str):
        def inner(sub: pd.DataFrame) -> str:
            a = pd.to_numeric(sub["n_items"], errors="coerce")
            b = pd.to_numeric(sub.get(col), errors="coerce")
            ok = a.notna() & b.notna()
            if ok.sum() < 3 or b[ok].std() == 0:
                return "—"
            return fmt(float(np.corrcoef(a[ok], b[ok])[0, 1]))
        return inner

    rows.append(("Item-count CV (shipped science 0.107)", "↑", per_source(cv)))
    rows.append(("Modal item-count share (shipped science 0.646)", "↓", per_source(modal)))
    rows.append(("r(items, question length) (shipped science 0.19)", "↑",
                 per_source(corr("question_words"))))
    rows.append(("r(items, #sub-questions)", "↑", per_source(corr("n_subparts"))))

    # F5: do the weights carry information beyond the category label? Correlate
    # the paper-explicit score against the equal-weight score on real verdicts.
    wcorr, wdiff = [], []
    for s in sources:
        sub = resp[resp["rubric_source"] == s] if not resp.empty else pd.DataFrame()
        if sub.empty or "score_unweighted" not in sub.columns:
            wcorr.append("—")
            wdiff.append("—")
            continue
        a = pd.to_numeric(sub["score"], errors="coerce")
        b = pd.to_numeric(sub["score_unweighted"], errors="coerce")
        ok = a.notna() & b.notna()
        wcorr.append(fmt(float(np.corrcoef(a[ok], b[ok])[0, 1])) if ok.sum() > 2 else "—")
        wdiff.append(fmt(float((a[ok] - b[ok]).abs().mean())))
    rows.append(("Weight vs uniform correlation (shipped ~0.94)", "↓", wcorr))
    rows.append(("Mean \\|weighted − uniform\\| score", "↑", wdiff))

    lines = [
        "### 分布层面指标（无逐题对应值，故不报置信区间）",
        "",
        "| Metric | 方向 | " + " | ".join(f"`{s}`" for s in sources) + " |",
        "|---|---|" + "---|" * len(sources),
    ]
    for label, arrow, cells in rows:
        lines.append(f"| {label} | {arrow} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def render_polarity(resp: pd.DataFrame, sources: list[str]) -> str:
    """F6 sensitivity: rescore the same verdicts under each polarity convention."""
    if resp.empty or "scores_by_polarity_mode" not in resp.columns:
        return ""
    probes = ["gold", "verbose_empty", "off_topic"]
    modes: list[str] = []
    for entry in resp["scores_by_polarity_mode"].dropna():
        if isinstance(entry, dict):
            modes = sorted(entry)
            break
    if not modes:
        return ""
    # `detected` is the configured scoring mode, so it lands in `score` rather
    # than in the sensitivity map. List it first: it is the correct convention.
    if "detected" not in modes:
        modes = ["detected", *modes]

    def score(source: str, response_id: str, mode: str) -> float:
        sub = resp[(resp["rubric_source"] == source) & (resp["response_id"] == response_id)]
        if sub.empty:
            return float("nan")
        if mode == "detected":
            vals = pd.to_numeric(sub["score"], errors="coerce").dropna().tolist()
        else:
            vals = [
                e.get(mode) for e in sub["scores_by_polarity_mode"]
                if isinstance(e, dict) and isinstance(e.get(mode), (int, float))
            ]
        return float(np.mean(vals)) if vals else float("nan")

    lines = [
        "### Pitfall 极性敏感性分析（F6）",
        "",
        "同一批 judge 判定，按不同极性约定重新聚合。`detected` 是正确口径；"
        "其余为强制单一约定，用于确认结论不依赖于极性选择。",
        "",
        "| Response | Polarity 约定 | " + " | ".join(f"`{s}`" for s in sources) + " |",
        "|---|---|" + "---|" * len(sources),
    ]
    for rid in probes:
        for mode in modes:
            cells = [fmt(score(s, rid, mode)) for s in sources]
            lines.append(f"| {rid} | {mode} | " + " | ".join(cells) + " |")

    lines += [
        "",
        "> 注：曾经存在的 `PolarityMode.FAVOURABLE`（\"对照组从宽\"）经审计确认是"
        "`detected` 的恒等别名——`detect_polarity` 对 `unclassified` 本就返回 POSITIVE，"
        "而 `category_default` 只出现在非 Pitfall 上并提前返回。实测 12,484 条 criterion "
        "中 0 条不同、18,849 条判定 max abs diff = 0.0。该分支已删除；"
        "**真正的\"从宽\"就是下面这张表**：逐个来源取对它最有利的整体约定。",
        "",
        "**判别 margin（gold − off_topic），每个来源取对它最有利的极性约定：**",
        "",
        "| 来源 | 最有利约定 | 该约定下 margin | `detected` 下 margin |",
        "|---|---|---|---|",
    ]
    best: dict[str, tuple[str, float]] = {}
    for s in sources:
        margins = {m: score(s, "gold", m) - score(s, "off_topic", m) for m in modes}
        margins = {m: v for m, v in margins.items() if math.isfinite(v)}
        if not margins:
            continue
        mode = max(margins, key=margins.__getitem__)
        best[s] = (mode, margins[mode])
        det = margins.get("detected", float("nan"))
        lines.append(f"| `{s}` | {mode} | {fmt(margins[mode])} | {fmt(det)} |")

    comparators = [s for s in ("shipped", "baseline") if s in best]
    if "agentic" in best and comparators:
        ours = best["agentic"][1]
        worst_case = max(best[c][1] for c in comparators)
        detail = ", ".join(f"`{c}` {fmt(best[c][1])}" for c in comparators)
        lines.append("")
        if ours > worst_case:
            lines.append(
                f"> `agentic` 在**每个对照组的最优极性口径下**仍然领先"
                f"（margin {fmt(ours)} vs {detail}）"
            )
        else:
            lines.append(
                f"> `agentic` 未能击败对照组的最优口径（margin {fmt(ours)} vs {detail}）"
            )
    return "\n".join(lines)


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )
    run_path = Path(args.runs_dir) / args.run_name
    out_dir = Path(args.results_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)

    df = load_rows(run_path, args.prefix)
    sources = args.sources or [s for s in DEFAULT_SOURCES if s in set(df["rubric_source"])]
    logger.info("sources: %s", sources)

    resp_path = run_path / "metrics" / f"{args.prefix}discriminative_per_response_rows.jsonl"
    resp = pd.DataFrame(list(read_jsonl(resp_path))) if resp_path.exists() else pd.DataFrame()

    table = compute_table(
        df, sources=sources, reference=args.reference, iters=args.iters, seed=args.seed
    )
    fdr_meta = apply_fdr(table, sources, args.reference)
    logger.info("computed %d metric rows (declared family: %d metrics)",
                len(table), fdr_meta["declared_family"])

    if len(table) < args.expect_metrics:
        logger.error(
            "only %d/%d declared metrics produced rows — a metric family is "
            "missing from runs/%s/metrics/ (re-run the zero-LLM metrics)",
            len(table), args.expect_metrics, args.run_name,
        )
        if args.strict:
            return 2
    problems = check_cohorts(table, expect=args.expect_paired)
    for problem in problems:
        logger.error("cohort drift: %s", problem)
    if problems and args.strict:
        return 2

    domains = sorted({d for d in df.get("domain", pd.Series(dtype=str)).dropna().unique()})
    by_domain: dict[str, list[dict[str, Any]]] = {}
    for domain in domains:
        sub_table = compute_table(
            df, sources=sources, reference=args.reference,
            iters=args.iters, seed=args.seed, domain=domain,
        )
        apply_fdr(sub_table, sources, args.reference)
        by_domain[domain] = sub_table
        logger.info("domain %s: %d metric rows", domain, len(sub_table))

    stem = f"{args.prefix}summary" if args.prefix else "summary"
    (out_dir / f"{stem}.json").write_text(
        json.dumps(
            {"pooled": table, "by_domain": by_domain, "fdr": fdr_meta,
             "sources": sources, "reference": args.reference,
             "cohort_problems": problems},
            indent=1, ensure_ascii=False, default=str,
        ),
        encoding="utf-8",
    )
    pd.DataFrame(table).to_csv(out_dir / f"{stem}.csv", index=False)
    df.to_csv(out_dir / f"{args.prefix}per_question_rows.csv", index=False)

    parts = ["## 合并（两域 pooled）", "",
             render_main(table, sources, args.reference)]
    for domain in domains:
        if by_domain[domain]:
            parts += ["", f"## 分域：`{domain}`", "",
                      render_main(by_domain[domain], sources, args.reference)]
    parts += ["", render_distribution(df, resp, sources)]
    polarity = render_polarity(resp, sources)
    if polarity:
        parts += ["", polarity]
    name = f"{args.prefix}summary_table.md" if args.prefix else "summary_table.md"
    (out_dir / name).write_text("\n".join(parts) + "\n", encoding="utf-8")
    logger.info("wrote %s", out_dir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
