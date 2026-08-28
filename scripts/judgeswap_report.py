#!/usr/bin/env python
"""Main judge-swap result: the discriminative table under both judges, paired.

Reuses ``scripts/confound_audit.py`` wholesale for the metric definitions, so
the swapped-judge numbers are computed by exactly the code that produced the
incumbent's numbers — no reimplementation to disagree about.

What this adds on top of that script:

**BH-FDR.** Each (scope, gold-mode, metric) family is corrected across the five
non-reference sources, because five contrasts are read off every row.

**A judge x source interaction test.** Asking "did the effect shrink?" by
eyeballing two separate deltas is weak: both are estimated with error and the
two judges scored the *same* questions, so the comparison can be made paired.
For each source this reports

    DiD_q = (new_source_q - new_reference_q) - (old_source_q - old_reference_q)

per question, then bootstraps and permutes it like any other paired contrast.
DiD < 0 means the source's advantage over the reference is smaller under the new
judge; a CI that excludes 0 means the change itself is real, not sampling noise.
This is the number that answers "is the conclusion judge-specific".

**The domain contrast, as a contrast.** The headline claim under test is that
z_separation improves in science and not in medicine. Reporting the two domains
side by side invites reading a difference into two noisy estimates, so the
science-minus-medicine gap is also tested directly (unpaired between domains,
since the questions differ, via a bootstrap over questions within each domain).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from confound_audit import (  # noqa: E402
    TIER_ORDER,
    _complete_uids,
    _order_sources,
    _read_jsonl,
    load_scores,
    per_question_discrimination,
)
from harness.eval.stats import bh_fdr, mean_ci, paired_diff  # noqa: E402

METRICS = ["ranking_accuracy", "auc", "separation", "z_separation", "spread"]
GOLD_MODES = {"with_gold": (), "without_gold": ("gold",)}
NAN = float("nan")


def per_question_matrix(
    scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    sources: Sequence[str],
    uids: Sequence[str],
    exclude: Sequence[str],
) -> dict[str, dict[str, dict[str, float]]]:
    """``out[source][uid][metric]`` — the per-question vectors every test uses."""
    out: dict[str, dict[str, dict[str, float]]] = {s: {} for s in sources}
    for uid in uids:
        for source in sources:
            out[source][uid] = per_question_discrimination(
                scores[uid][source], exclude=exclude
            )
    return out


def _vec(mat: Mapping[str, Mapping[str, float]], uids: Sequence[str], metric: str) -> list[float]:
    return [mat[u].get(metric, NAN) for u in uids]


def contrast_block(
    mat: Mapping[str, Mapping[str, Mapping[str, float]]],
    sources: Sequence[str],
    uids: Sequence[str],
    reference: str,
) -> dict[str, Any]:
    """Per-source mean plus paired Δ against the reference, BH-corrected."""
    block: dict[str, Any] = {}
    for metric in METRICS:
        entry: dict[str, Any] = {}
        for source in sources:
            entry[source] = mean_ci(_vec(mat[source], uids, metric))
        pvals, keys = [], []
        for source in sources:
            if source == reference:
                continue
            d = paired_diff(
                _vec(mat[source], uids, metric), _vec(mat[reference], uids, metric)
            )
            entry[f"delta__{source}"] = d
            p = d.get("p_permutation", d.get("p_value", NAN))
            if math.isfinite(p):
                pvals.append(p)
                keys.append(source)
        for source, q in zip(keys, bh_fdr(pvals)):
            entry[f"delta__{source}"]["q_bh"] = float(q)
        block[metric] = entry
    return block


def did_block(
    new_mat: Mapping[str, Mapping[str, Mapping[str, float]]],
    old_mat: Mapping[str, Mapping[str, Mapping[str, float]]],
    sources: Sequence[str],
    uids: Sequence[str],
    reference: str,
) -> dict[str, Any]:
    """judge x source interaction: did each source's edge over the reference move?"""
    block: dict[str, Any] = {}
    for metric in METRICS:
        entry: dict[str, Any] = {}
        pvals, keys = [], []
        for source in sources:
            if source == reference:
                continue
            new_edge = [
                a - b
                for a, b in zip(
                    _vec(new_mat[source], uids, metric),
                    _vec(new_mat[reference], uids, metric),
                )
            ]
            old_edge = [
                a - b
                for a, b in zip(
                    _vec(old_mat[source], uids, metric),
                    _vec(old_mat[reference], uids, metric),
                )
            ]
            d = paired_diff(new_edge, old_edge)
            d["edge_new"] = float(np.nanmean(np.asarray(new_edge, dtype=float)))
            d["edge_old"] = float(np.nanmean(np.asarray(old_edge, dtype=float)))
            entry[source] = d
            p = d.get("p_permutation", d.get("p_value", NAN))
            if math.isfinite(p):
                pvals.append(p)
                keys.append(source)
        for source, q in zip(keys, bh_fdr(pvals)):
            entry[source]["q_bh"] = float(q)
        block[metric] = entry
    return block


def domain_gap(
    mat_by_domain: Mapping[str, Mapping[str, Mapping[str, Mapping[str, float]]]],
    uids_by_domain: Mapping[str, Sequence[str]],
    sources: Sequence[str],
    reference: str,
    metric: str,
    *,
    iters: int = 10000,
    seed: int = 0,
) -> dict[str, Any]:
    """science-minus-medicine difference in each source's edge over the reference.

    The two domains hold different questions, so this cannot be paired. It is
    bootstrapped by resampling questions independently within each domain, which
    is the same resampling unit the rest of the report uses.
    """
    rng = np.random.default_rng(seed)
    out: dict[str, Any] = {}
    domains = sorted(mat_by_domain)
    if len(domains) != 2:
        return out
    d_hi, d_lo = "rar_science", "rar_medicine"
    if d_hi not in domains or d_lo not in domains:
        d_hi, d_lo = domains[1], domains[0]

    for source in sources:
        if source == reference:
            continue
        edges: dict[str, np.ndarray] = {}
        for dom in (d_hi, d_lo):
            uids = uids_by_domain[dom]
            mat = mat_by_domain[dom]
            e = np.asarray(
                [
                    mat[source][u].get(metric, NAN) - mat[reference][u].get(metric, NAN)
                    for u in uids
                ],
                dtype=float,
            )
            edges[dom] = e[np.isfinite(e)]
        if edges[d_hi].size < 2 or edges[d_lo].size < 2:
            continue
        obs = float(edges[d_hi].mean() - edges[d_lo].mean())
        draws = np.empty(iters, dtype=float)
        for i in range(iters):
            a = rng.choice(edges[d_hi], edges[d_hi].size, replace=True)
            b = rng.choice(edges[d_lo], edges[d_lo].size, replace=True)
            draws[i] = a.mean() - b.mean()
        lo, hi = np.percentile(draws, [2.5, 97.5])
        # Two-sided bootstrap p: how much of the resampled distribution sits on
        # the far side of zero, doubled. Cheap and adequate at these n.
        frac = float(np.mean(draws <= 0.0)) if obs > 0 else float(np.mean(draws >= 0.0))
        out[source] = {
            "gap": obs,
            f"edge_{d_hi}": float(edges[d_hi].mean()),
            f"edge_{d_lo}": float(edges[d_lo].mean()),
            "ci_low": float(lo),
            "ci_high": float(hi),
            "p_boot": min(1.0, 2.0 * frac),
            "n_hi": int(edges[d_hi].size),
            "n_lo": int(edges[d_lo].size),
        }
    return out


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def _f(x: Any, nd: int = 3) -> str:
    return "—" if not (isinstance(x, (int, float)) and math.isfinite(float(x))) else f"{float(x):.{nd}f}"


def _s(x: Any, nd: int = 3) -> str:
    """Signed variant of :func:`_f`; sign is what makes a delta column readable."""
    return "—" if not (isinstance(x, (int, float)) and math.isfinite(float(x))) else f"{float(x):+.{nd}f}"


def _sig(p: Any, q: Any = None) -> str:
    v = q if (isinstance(q, (int, float)) and math.isfinite(float(q))) else p
    if not (isinstance(v, (int, float)) and math.isfinite(float(v))):
        return ""
    v = float(v)
    return "***" if v < 0.001 else "**" if v < 0.01 else "*" if v < 0.05 else ""


def render_contrast(title: str, block: Mapping[str, Any], sources: Sequence[str],
                    reference: str, metrics: Sequence[str] = METRICS) -> list[str]:
    out = [f"  {title}", ""]
    for metric in metrics:
        entry = block[metric]
        out.append(f"    {metric}")
        out.append(
            f"      {'source':<20}{'mean':>8}{'SE':>8}{'Δ vs ref':>11}"
            f"{'95% CI':>22}{'p':>9}{'q(BH)':>9}"
        )
        for s in sources:
            cell = entry.get(s, {})
            if s == reference:
                out.append(f"      {s:<20}{_f(cell.get('mean')):>8}{_f(cell.get('se')):>8}"
                           f"{'(reference)':>11}")
                continue
            d = entry.get(f"delta__{s}", {})
            p = d.get("p_permutation", d.get("p_value", NAN))
            ci = f"[{_f(d.get('ci_low'))},{_f(d.get('ci_high'))}]"
            out.append(
                f"      {s:<20}{_f(cell.get('mean')):>8}{_f(cell.get('se')):>8}"
                f"{_s(d.get('mean_diff')):>11}{ci:>22}{_f(p, 4):>9}"
                f"{_f(d.get('q_bh'), 4):>9}{_sig(p, d.get('q_bh'))}"
            )
        out.append("")
    return out


def render_did(title: str, block: Mapping[str, Any], sources: Sequence[str],
               reference: str, metrics: Sequence[str] = METRICS) -> list[str]:
    out = [f"  {title}", ""]
    for metric in metrics:
        entry = block[metric]
        out.append(f"    {metric}")
        out.append(
            f"      {'source':<20}{'edge_old':>10}{'edge_new':>10}{'DiD':>10}"
            f"{'95% CI':>22}{'p':>9}{'q(BH)':>9}"
        )
        for s in sources:
            if s == reference:
                continue
            d = entry.get(s, {})
            p = d.get("p_permutation", d.get("p_value", NAN))
            ci = f"[{_f(d.get('ci_low'))},{_f(d.get('ci_high'))}]"
            out.append(
                f"      {s:<20}{_s(d.get('edge_old')):>10}{_s(d.get('edge_new')):>10}"
                f"{_s(d.get('mean_diff')):>10}{ci:>22}{_f(p, 4):>9}"
                f"{_f(d.get('q_bh'), 4):>9}{_sig(p, d.get('q_bh'))}"
            )
        out.append("")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-run", default="runs/judgeswap")
    ap.add_argument("--old-run", default="runs/oldjudge_sub")
    ap.add_argument("--new-judge", default="api_azure_openai_gpt-5.1")
    ap.add_argument("--old-judge", default="hy-t2t-glm-5.2-384k-fp8-L20A-t1-v2")
    ap.add_argument("--reference", default="baseline")
    ap.add_argument("--score-field", default="score")
    ap.add_argument("--out-prefix", default="results/judgeswap/main")
    args = ap.parse_args()

    loaded: dict[str, Any] = {}
    for label, run in (("new", args.new_run), ("old", args.old_run)):
        rows = _read_jsonl(Path(run) / "metrics" / "discriminative_per_response_rows.jsonl")
        if not rows:
            raise SystemExit(f"no per-response rows under {run}/metrics/")
        loaded[label] = load_scores(rows, score_field=args.score_field)

    (new_scores, new_domains) = loaded["new"]
    (old_scores, old_domains) = loaded["old"]
    sources = _order_sources(
        {s for by in new_scores.values() for s in by} & {s for by in old_scores.values() for s in by}
    )

    # Only questions with a complete ladder under BOTH judges, so every number in
    # this report is on one common sample.
    new_uids = set(_complete_uids(new_scores, sources, TIER_ORDER))
    old_uids = set(_complete_uids(old_scores, sources, TIER_ORDER))
    uids = sorted(new_uids & old_uids)
    if not uids:
        raise SystemExit("no question has a complete ladder under both judges")
    domains = {**old_domains, **new_domains}
    uids_by_domain: dict[str, list[str]] = defaultdict(list)
    for u in uids:
        uids_by_domain[domains.get(u, "?")].append(u)

    report: dict[str, Any] = {
        "new_judge": args.new_judge,
        "old_judge": args.old_judge,
        "reference": args.reference,
        "score_field": args.score_field,
        "sources": sources,
        "n_questions": len(uids),
        "n_by_domain": {k: len(v) for k, v in sorted(uids_by_domain.items())},
        "n_new_complete": len(new_uids),
        "n_old_complete": len(old_uids),
        "dropped_incomplete": sorted((new_uids | old_uids) - set(uids)),
        "scopes": {},
    }

    scopes: dict[str, list[str]] = {"all": uids}
    for dom, sub in sorted(uids_by_domain.items()):
        if len(sub) >= 5:
            scopes[dom] = sub

    mats: dict[str, dict[str, dict]] = {}
    for gold_mode, exclude in GOLD_MODES.items():
        mats[gold_mode] = {
            "new": per_question_matrix(new_scores, sources, uids, exclude),
            "old": per_question_matrix(old_scores, sources, uids, exclude),
        }

    for scope, sub in scopes.items():
        block: dict[str, Any] = {"n_questions": len(sub)}
        for gold_mode in GOLD_MODES:
            block[gold_mode] = {
                "new": contrast_block(mats[gold_mode]["new"], sources, sub, args.reference),
                "old": contrast_block(mats[gold_mode]["old"], sources, sub, args.reference),
                "did": did_block(
                    mats[gold_mode]["new"], mats[gold_mode]["old"], sources, sub, args.reference
                ),
            }
        report["scopes"][scope] = block

    # Domain contrast on z_separation (and ranking_accuracy for context).
    report["domain_gap"] = {}
    dom_scopes = {d: uids_by_domain[d] for d in uids_by_domain if len(uids_by_domain[d]) >= 5}
    if len(dom_scopes) == 2:
        for gold_mode in GOLD_MODES:
            report["domain_gap"][gold_mode] = {}
            for judge in ("new", "old"):
                mat = mats[gold_mode][judge]
                by_dom = {d: mat for d in dom_scopes}
                report["domain_gap"][gold_mode][judge] = {
                    metric: domain_gap(by_dom, dom_scopes, sources, args.reference, metric)
                    for metric in ("z_separation", "ranking_accuracy", "auc")
                }

    # ---------------- render ----------------
    lines: list[str] = [
        "=" * 104,
        "JUDGE-SWAP REPLICATION — discriminative validity under a different-family judge",
        "=" * 104,
        "",
        f"old judge  : {args.old_judge}   (wrote the agentic rubrics too — the confound)",
        f"new judge  : {args.new_judge}",
        f"reference  : `{args.reference}`      aggregation: paper_explicit ({args.score_field})",
        f"questions  : n={len(uids)}  " +
        "  ".join(f"{k}={v}" for k, v in report["n_by_domain"].items()),
        "",
        "Rubrics and the degraded response ladder are REUSED from the pilot byte-for-byte;",
        "judge prompt, polarity handling, aggregation and shuffle seed are unchanged. Both",
        "judges scored the same questions, so every contrast below is paired by question.",
        "",
    ]

    for scope, block in report["scopes"].items():
        lines += ["=" * 104, f"SCOPE: {scope}   (n={block['n_questions']})", "=" * 104, ""]
        for gold_mode in ("with_gold", "without_gold"):
            note = (
                "gold tier removed — an advantage here cannot come from fitting the gold text"
                if gold_mode == "without_gold"
                else "as originally reported"
            )
            lines += [f"  --- {gold_mode.upper()}  ({note}) ---", ""]
            lines += render_contrast(
                f"NEW judge ({args.new_judge})", block[gold_mode]["new"], sources, args.reference
            )
            lines += render_contrast(
                f"OLD judge ({args.old_judge}), same {block['n_questions']} questions",
                block[gold_mode]["old"], sources, args.reference,
            )
            lines += render_did(
                "JUDGE x SOURCE INTERACTION — DiD = (new edge) - (old edge); "
                "negative = effect shrank under the new judge",
                block[gold_mode]["did"], sources, args.reference,
            )

    if report["domain_gap"]:
        lines += ["=" * 104,
                  "DOMAIN CONTRAST — is the effect science-specific under each judge?",
                  "=" * 104, ""]
        for gold_mode in ("with_gold", "without_gold"):
            for judge in ("old", "new"):
                lines.append(f"  {gold_mode} / {judge} judge — "
                             f"edge over `{args.reference}`, science minus medicine")
                for metric, table in report["domain_gap"][gold_mode][judge].items():
                    if not table:
                        continue
                    lines.append(f"    {metric}")
                    lines.append(
                        f"      {'source':<20}{'science':>10}{'medicine':>10}{'gap':>10}"
                        f"{'95% CI':>22}{'p':>9}"
                    )
                    for s in sources:
                        if s == args.reference or s not in table:
                            continue
                        c = table[s]
                        ci = f"[{_f(c.get('ci_low'))},{_f(c.get('ci_high'))}]"
                        lines.append(
                            f"      {s:<20}{_s(c.get('edge_rar_science')):>10}"
                            f"{_s(c.get('edge_rar_medicine')):>10}{_s(c.get('gap')):>10}"
                            f"{ci:>22}{_f(c.get('p_boot'), 4):>9}{_sig(c.get('p_boot'))}"
                        )
                    lines.append("")

    text = "\n".join(lines)
    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".txt").write_text(text)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str))
    print(text)
    print(f"\nwrote {out.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
