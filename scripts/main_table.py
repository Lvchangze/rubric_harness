#!/usr/bin/env python3
"""Render the pilot's headline tables from the per-question metric rows.

``aggregate_results.py`` produces the full machine-readable sweep; this renders
the specific contrasts the report argues from, with paired uncertainty against a
chosen reference column. Reads only ``runs/<run>/metrics/*.jsonl``, so it can be
run while later metrics are still being computed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.eval.stats import paired_diff  # noqa: E402
from harness.tracing import read_jsonl  # noqa: E402

SOURCES = ["shipped", "baseline", "agentic-noval", "agentic"]

# (column, label, direction) — direction is only used to annotate the header.
DISCRIMINATIVE = [
    ("mean_margin", "Mean margin (gold - degraded)", "up"),
    ("rank_acc", "Pairwise ranking accuracy", "up"),
    ("auc_gold_vs_degraded", "AUC gold vs degraded", "up"),
    ("auc_good_vs_bad", "AUC good vs bad", "up"),
    ("score_range", "Score dynamic range", "up"),
    ("saturation_rate", "Saturation rate", "down"),
    ("hack_verbose", "Reward hack: verbose-empty score", "down"),
    ("hack_confident_wrong", "Reward hack: confident-wrong score", "down"),
    ("margin__off_topic", "Margin vs off-topic", "up"),
    ("margin__verbose_empty", "Margin vs verbose-empty", "up"),
    ("margin__numeric_error", "Margin vs numeric-error", "up"),
    ("margin__missing_step", "Margin vs missing-step", "up"),
    ("margin__right_method_wrong_answer", "Margin vs right-method-wrong-answer", "up"),
    ("margin__terse_correct", "Margin vs terse-correct", "up"),
    ("score__gold", "Score on gold", "flat"),
    ("score__off_topic", "Score on off-topic", "down"),
]

GROUNDING = [
    ("generic_criterion_rate", "Generic (recyclable) criterion rate", "down"),
    ("weight_mass_on_generic_paper", "Weight mass on generic criteria", "down"),
    ("subjective_or_style_rate", "Subjective/style criterion rate", "down"),
    ("weak_grounding_rate", "Weakly grounded criterion rate", "down"),
    ("specificity_lift", "Specificity lift vs permuted", "up"),
]

# Counts, not rates, so they must not be scaled by the section's percentage factor.
GROUNDING_COUNTS = [
    ("mean_anchors_per_criterion", "Anchors per criterion", "up"),
]

LINT = [
    ("compliance_rate", "RaR prompt-constraint compliance", "up"),
    ("polarity_single_rate", "Single-polarity rubric rate", "up"),
    ("unclassified_polarity_rate", "Unclassified-polarity rate", "down"),
    ("pitfall_mirror_rate", "Pitfall-mirror (double-count) rate", "down"),
]

COVERAGE = [
    ("claim_recall", "Gold-claim recall", "up"),
    ("claim_recall_core", "Core gold-claim recall", "up"),
    ("grounded_fraction", "Criterion groundedness (precision)", "up"),
    ("unsupported_fraction", "Unsupported criterion fraction", "down"),
    ("generic_fraction", "Generic criterion fraction (LLM-judged)", "down"),
    ("subquestion_coverage", "Sub-question coverage", "up"),
]

TRANSFER = [
    ("same_question_pass_rate", "Same-question gold pass rate", "up"),
    ("cross_pass_rate", "Cross-question pass rate", "down"),
    ("specificity_gap", "Specificity gap", "up"),
    ("boilerplate_fraction", "Boilerplate criterion fraction", "down"),
    ("specific_fraction", "Query-specific criterion fraction", "up"),
]

INTRINSIC = [
    ("objective_fraction", "Objectively checkable fraction", "up"),
    ("subjective_fraction", "Subjective criterion fraction", "down"),
    ("kappa", "Self-agreement (Cohen's kappa)", "up"),
    ("self_agreement_rate", "Self-agreement raw rate", "up"),
    ("atomic_fraction", "Atomic criterion fraction", "up"),
]

# Counts / entropies rather than rates.
INTRINSIC_SCALAR = [
    ("mean_checks_per_criterion", "Checks per criterion (atomicity)", "down"),
    ("weight_entropy", "Weight distribution entropy", "flat"),
    ("max_weight_share", "Max single-criterion weight share", "flat"),
]

ARROW = {"up": "higher is better", "down": "lower is better", "flat": "descriptive"}


def load(run: Path, name: str) -> pd.DataFrame:
    path = run / "metrics" / name
    if not path.exists():
        return pd.DataFrame()
    return pd.DataFrame(list(read_jsonl(path)))


def stars(p: float) -> str:
    if not np.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


def section(
    df: pd.DataFrame, spec: list[tuple[str, str, str]], title: str, reference: str, scale: float
) -> None:
    if df.empty or "rubric_source" not in df:
        print(f"\n## {title}\n\n  (no rows yet)\n")
        return
    present = [s for s in SOURCES if s in set(df["rubric_source"])]
    by = {s: df[df["rubric_source"] == s].set_index("uid") for s in present}
    uids = sorted(set.intersection(*[set(v.index) for v in by.values()])) if by else []
    print(f"\n## {title}")
    print(f"\n  paired questions n={len(uids)}; reference column = `{reference}`; "
          f"values x{scale:g}\n")

    def col(s: str, c: str) -> np.ndarray:
        if c not in by[s].columns:
            return np.full(len(uids), np.nan)
        return pd.to_numeric(by[s].loc[uids, c], errors="coerce").to_numpy(float)

    for key, label, dirn in spec:
        if not any(key in by[s].columns for s in present):
            continue
        print(f"### {label}  ({ARROW[dirn]})")
        print(f"    {'source':16s}{'mean':>10s}{'SE':>9s}{'Δ vs ref':>12s}"
              f"{'95% CI':>24s}{'p':>10s}")
        ref = col(reference, key)
        for s in present:
            v = col(s, key)
            n = int(np.sum(np.isfinite(v)))
            m = np.nanmean(v) * scale if n else np.nan
            se = (np.nanstd(v) / np.sqrt(n) * scale) if n else np.nan
            if s == reference:
                print(f"    {s:16s}{m:10.3f}{se:9.3f}{'(reference)':>12s}")
                continue
            r = paired_diff(v, ref, iters=10000, seed=7)
            print(
                f"    {s:16s}{m:10.3f}{se:9.3f}"
                f"{r['mean_diff'] * scale:+12.3f}"
                f"  [{r['ci_low'] * scale:+.3f},{r['ci_high'] * scale:+.3f}]".ljust(24)
                + f"{r['p_value']:>10.2g}{stars(r['p_value'])}"
            )
        print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="pilot_v2")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--reference", default="baseline")
    ap.add_argument("--prefix", default="", help="'cc_' for the count-controlled pass")
    args = ap.parse_args()
    run = Path(args.runs_dir) / args.run_name
    p = args.prefix

    section(load(run, f"{p}discriminative_per_question_rows.jsonl"),
            DISCRIMINATIVE, "Discriminative validity (main result)", args.reference, 1.0)
    grounding_rows = load(run, f"{p}grounding_per_question_rows.jsonl")
    section(grounding_rows, GROUNDING,
            "Grounding / query-specificity (zero-LLM)", args.reference, 100.0)
    section(grounding_rows, GROUNDING_COUNTS,
            "Grounding: anchor counts (zero-LLM)", args.reference, 1.0)
    section(load(run, f"{p}lint_per_question_rows.jsonl"),
            LINT, "Schema and prompt-compliance lint (zero-LLM)", args.reference, 100.0)
    section(load(run, f"{p}coverage_per_question_rows.jsonl"),
            COVERAGE, "Coverage of the reference answer", args.reference, 100.0)
    section(load(run, f"{p}transfer_per_question_rows.jsonl"),
            TRANSFER, "Transferability / query-specificity (LLM)", args.reference, 100.0)
    intrinsic_rows = load(run, f"{p}intrinsic_per_question_rows.jsonl")
    section(intrinsic_rows, INTRINSIC, "Intrinsic quality (rates)", args.reference, 100.0)
    section(intrinsic_rows, INTRINSIC_SCALAR, "Intrinsic quality (scalars)", args.reference, 1.0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
