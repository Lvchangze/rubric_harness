#!/usr/bin/env python3
"""Print the run-level facts the report quotes outside the metric tables.

Head-to-head win rates, LLM call/failure counts, rubric sizes, per-domain
structural checks against the forensics baselines, and the polarity swing.
Read-only; safe to run at any time after a pilot.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.tracing import read_jsonl  # noqa: E402

SOURCES = ["shipped", "baseline", "agentic-noval", "agentic"]

# Full-corpus values from docs/01_data_forensics.md (n=45,325 questions).
FORENSICS = {
    ("generic_criterion_rate", "rar_science"): 28.87,
    ("generic_criterion_rate", "rar_medicine"): 21.97,
    ("generic_rate_permuted", "rar_science"): 96.39,
    ("generic_rate_permuted", "rar_medicine"): 98.54,
    ("weight_mass_on_generic_paper", "rar_science"): 24.56,
    ("weight_mass_on_generic_paper", "rar_medicine"): 15.20,
    ("weak_grounding_rate", "rar_science"): 52.61,
    ("weak_grounding_rate", "rar_medicine"): 43.53,
    ("pitfall_mirror_rate", "rar_science"): 0.95,
    ("pitfall_mirror_rate", "rar_medicine"): 16.21,
    ("weight_order_violation_rate", "rar_science"): 41.30,
    ("weight_order_violation_rate", "rar_medicine"): 0.00,
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", default="pilot_v2")
    ap.add_argument("--runs-dir", default="runs")
    args = ap.parse_args()
    run = Path(args.runs_dir) / args.run_name
    M = run / "metrics"
    load = lambda n: pd.DataFrame(list(read_jsonl(M / n))) if (M / n).exists() else pd.DataFrame()

    print("=" * 78)
    print("HEAD-TO-HEAD (blind, position-bias corrected)")
    print("=" * 78)
    summaries = json.loads((run / "metric_summaries.json").read_text())
    h2h = summaries.get("headtohead", {})
    print(f"  verdicts={h2h.get('n_verdicts')} ties={h2h.get('n_tie_verdicts')} "
          f"failed={h2h.get('n_failed_verdicts')} position_bias={h2h.get('position_bias_rate'):.3f}")
    for name, p in (h2h.get("pairs") or {}).items():
        print(f"  {name:34s} raw {p['raw_win_rate_a']:.3f}/{p['raw_win_rate_b']:.3f}"
              f"   corrected {p['corrected_win_rate_a']:.3f}/{p['corrected_win_rate_b']:.3f}"
              f"   flip={p['flip_rate']:.3f}")

    print()
    print("=" * 78)
    print("LLM COST AND RELIABILITY")
    print("=" * 78)
    for f in ("llm_stats_gen.json", "llm_stats_responses.json", "llm_stats_eval.json"):
        p = run / f
        if not p.exists():
            continue
        s = json.loads(p.read_text())
        calls = s.get("calls", 0)
        jf = s.get("json_parse_failures", 0)
        print(f"  {f:26s} calls={calls:6d} api={s.get('api_calls',0):6d} "
              f"cache_hit={s.get('cache_hit_rate',0):.3f} retries={s.get('retries',0):3d} "
              f"empty={s.get('empty_responses',0):3d} json_fail={jf:3d} "
              f"({100*jf/max(calls,1):.2f}%) fail={s.get('failures',0)}")

    print()
    print("=" * 78)
    print("RUBRIC SIZE AND GENERATION FAILURES")
    print("=" * 78)
    for s in SOURCES:
        p = run / f"rubrics_{s}.jsonl"
        if not p.exists():
            continue
        recs = list(read_jsonl(p))
        sizes = [len((r.get("rubric") or {}).get("items") or []) for r in recs]
        empty = sum(1 for n in sizes if n == 0)
        errs = sum(1 for r in recs if r.get("error"))
        calls = [r.get("n_llm_calls") or 0 for r in recs]
        print(f"  {s:15s} n={len(recs):4d} mean_items={np.mean(sizes):5.2f} "
              f"min={min(sizes)} max={max(sizes)} empty={empty} with_error={errs} "
              f"mean_llm_calls={np.mean(calls):5.1f}")

    print()
    print("=" * 78)
    print("SANITY GATE: shipped vs forensics n=45k (per domain)")
    print("=" * 78)
    g, l = load("grounding_per_question_rows.jsonl"), load("lint_per_question_rows.jsonl")
    print(f"  {'metric':30s}{'domain':14s}{'ours':>9s}{'forensics':>11s}{'diff':>8s}")
    for df, keys in ((g, ["generic_criterion_rate", "generic_rate_permuted",
                          "weight_mass_on_generic_paper", "weak_grounding_rate"]),
                     (l, ["pitfall_mirror_rate", "weight_order_violation_rate"])):
        if df.empty:
            continue
        sub = df[df["rubric_source"] == "shipped"]
        for k in keys:
            for dom in ("rar_science", "rar_medicine"):
                d = sub[sub["domain"] == dom]
                if d.empty or k not in d.columns:
                    continue
                v = float(np.nanmean(pd.to_numeric(d[k], errors="coerce"))) * 100
                ref = FORENSICS.get((k, dom))
                if ref is None:
                    continue
                print(f"  {k:30s}{dom:14s}{v:9.2f}{ref:11.2f}{v-ref:+8.2f}")

    print()
    print("=" * 78)
    print("PER-DOMAIN MAIN RESULT (mean margin)")
    print("=" * 78)
    d = load("discriminative_per_question_rows.jsonl")
    if not d.empty:
        print(f"  {'source':16s}" + "".join(f"{x:>18s}" for x in ("rar_science", "rar_medicine")))
        for s in SOURCES:
            cells = []
            for dom in ("rar_science", "rar_medicine"):
                sub = d[(d["rubric_source"] == s) & (d["domain"] == dom)]
                v = pd.to_numeric(sub["mean_margin"], errors="coerce")
                cells.append(f"{v.mean():>11.3f}±{v.std()/np.sqrt(len(v)):.3f}")
            print(f"  {s:16s}" + "".join(f"{c:>18s}" for c in cells))

    print()
    print("=" * 78)
    print("POLARITY SWING (max - min across conventions, per source)")
    print("=" * 78)
    r = load("discriminative_per_response_rows.jsonl")
    if not r.empty and "scores_by_polarity_mode" in r.columns:
        for s in SOURCES:
            for rid in ("gold", "off_topic"):
                sub = r[(r["rubric_source"] == s) & (r["response_id"] == rid)]
                if sub.empty:
                    continue
                vals = {"detected": pd.to_numeric(sub["score"], errors="coerce").mean()}
                for e in sub["scores_by_polarity_mode"]:
                    if isinstance(e, dict):
                        for m, v in e.items():
                            vals.setdefault(m, [])
                for m in list(vals):
                    if m == "detected":
                        continue
                    xs = [e.get(m) for e in sub["scores_by_polarity_mode"]
                          if isinstance(e, dict) and isinstance(e.get(m), (int, float))]
                    vals[m] = float(np.mean(xs)) if xs else float("nan")
                fin = {k: v for k, v in vals.items() if isinstance(v, float) and np.isfinite(v)}
                swing = (max(fin.values()) - min(fin.values())) * 100
                gold_gt = "yes"
                print(f"  {s:15s} {rid:10s} swing={swing:5.2f}pp  " +
                      " ".join(f"{k}={v:.3f}" for k, v in sorted(fin.items())))
        # Does gold > off_topic survive every convention, for every source?
        print()
        broken = []
        for s in SOURCES:
            for m in ("detected", "favourable", "all_avoidance", "all_failure"):
                def sc(rid):
                    sub = r[(r["rubric_source"] == s) & (r["response_id"] == rid)]
                    if sub.empty:
                        return float("nan")
                    if m == "detected":
                        return pd.to_numeric(sub["score"], errors="coerce").mean()
                    xs = [e.get(m) for e in sub["scores_by_polarity_mode"]
                          if isinstance(e, dict) and isinstance(e.get(m), (int, float))]
                    return float(np.mean(xs)) if xs else float("nan")
                if not (sc("gold") > sc("off_topic")):
                    broken.append((s, m))
        print(f"  'gold > off_topic' sanity check fails in: {broken or 'NONE (passes everywhere)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
