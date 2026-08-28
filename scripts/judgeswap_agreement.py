#!/usr/bin/env python
"""Cross-judge agreement, broken down by rubric source — the self-preference test.

The threat this addresses: the ``agentic`` rubrics were written by the same model
that judged them in the pilot, so their wording may suit that judge specifically.
The direct signature of such a shared idiosyncrasy is *differential* agreement —
a swapped-in judge from another family should disagree with the incumbent more on
``agentic`` than on the comparators, because only ``agentic``'s phrasing was
(implicitly) tuned to the incumbent.

Measured on the identical (question, source, response) cells, so the two judges
differ in nothing but themselves:

``pearson`` / ``spearman``
    Correlation of the aggregated score. Spearman is reported because the score
    is bounded and lumpy, so a few saturated cells can dominate Pearson.
``mean_abs_diff`` / ``mean_signed_diff``
    Level agreement. The signed term separates "the new judge is harsher
    everywhere" (a main effect, harmless to the comparison) from "the new judge
    is harsher on this source specifically" (which is the interesting case).
``pass_rate_old`` / ``pass_rate_new``
    Share of criteria each judge called literally true, from ``n_literally_true``.
``min_criterion_disagreement``
    ``|pass_rate_old - pass_rate_new|`` per cell. Per-criterion verdicts were not
    persisted by the pilot, so exact per-criterion agreement cannot be recovered;
    the difference of the two marginal pass rates is the largest quantity that IS
    exactly recoverable, and it is a strict *lower* bound on the per-criterion
    disagreement rate (verdicts that swap in opposite directions cancel in the
    marginal). It is therefore reported as a bound, never as "the" rate.
``rank_agreement_within_question``
    The measure that matters for a reward signal: over pairs of responses of
    *different* known quality within one question, how often do the two judges
    order the pair the same way. Insensitive to any monotone recalibration of the
    scale, so it isolates genuine ordering disagreement.

Every difference between sources is tested paired *by question*, since a question
is the unit that repeats across sources, with BH-FDR over the family of sources
compared against the reference.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.eval.stats import bh_fdr, mean_ci, paired_diff  # noqa: E402

QUALITY_LEVELS: dict[str, int] = {
    "gold": 5, "terse_correct": 4, "missing_step": 3,
    "numeric_error": 2, "right_method_wrong_answer": 2,
    "verbose_empty": 1, "off_topic": 0,
}

SOURCE_ORDER = ["shipped", "baseline", "agentic-noval", "agentic",
                "agentic-goldonly", "agentic-negonly"]


def read_rows(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def index(rows: Sequence[Mapping[str, Any]], score_field: str = "score") -> tuple[dict, dict]:
    """``cells[(uid, source, variant)] = {...}`` for usable rows only."""
    cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    domains: dict[str, str] = {}
    for row in rows:
        if row.get("error") or not row.get("usable", True):
            continue
        value = row.get(score_field)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        uid, src, variant = row.get("uid"), row.get("rubric_source"), row.get("variant")
        if not (uid and src and variant):
            continue
        n_items = row.get("n_items") or 0
        n_true = row.get("n_literally_true")
        cells[(uid, src, variant)] = {
            "score": float(value),
            "n_items": int(n_items),
            "pass_rate": (float(n_true) / n_items) if n_items and isinstance(n_true, int) else math.nan,
        }
        if row.get("domain"):
            domains[uid] = str(row["domain"])
    return cells, domains


def _spearman(x: Sequence[float], y: Sequence[float]) -> float:
    if len(x) < 3:
        return math.nan
    def ranks(v):
        a = np.asarray(v, dtype=float)
        order = np.argsort(a, kind="mergesort")
        ordered = a[order]
        starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
        ends = np.r_[starts[1:], ordered.size]
        rs = np.empty(ordered.size)
        for s, e in zip(starts, ends):
            rs[s:e] = 0.5 * (s + e - 1) + 1.0
        out = np.empty(a.size)
        out[order] = rs
        return out
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _pearson(x: Sequence[float], y: Sequence[float]) -> float:
    a, b = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if a.size < 3 or a.std() == 0 or b.std() == 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def rank_agreement(old: Mapping[str, float], new: Mapping[str, float]) -> float:
    """Do the two judges order same-question response pairs the same way?

    Only pairs of *different* construction-time quality are counted, matching the
    convention used by the discriminative metric, so the deliberate level-2 tie
    is never scored as a disagreement. A pair where exactly one judge ties gets
    half credit; a pair both judges tie counts as agreement.
    """
    variants = [v for v in old if v in new and v in QUALITY_LEVELS]
    hits = 0.0
    n = 0
    for i in range(len(variants)):
        for j in range(i + 1, len(variants)):
            a, b = variants[i], variants[j]
            if QUALITY_LEVELS[a] == QUALITY_LEVELS[b]:
                continue
            n += 1
            so = np.sign(old[a] - old[b])
            sn = np.sign(new[a] - new[b])
            if so == sn:
                hits += 1.0
            elif so == 0 or sn == 0:
                hits += 0.5
    return hits / n if n else math.nan


def per_source_cells(
    old: Mapping[tuple, dict], new: Mapping[tuple, dict], source: str
) -> dict[str, list[float]]:
    keys = [k for k in old if k[1] == source and k in new]
    out: dict[str, list[float]] = defaultdict(list)
    for k in keys:
        o, n = old[k], new[k]
        out["old"].append(o["score"])
        out["new"].append(n["score"])
        out["abs_diff"].append(abs(o["score"] - n["score"]))
        out["signed_diff"].append(n["score"] - o["score"])
        if math.isfinite(o["pass_rate"]) and math.isfinite(n["pass_rate"]):
            out["pass_old"].append(o["pass_rate"])
            out["pass_new"].append(n["pass_rate"])
            out["min_crit_disagree"].append(abs(o["pass_rate"] - n["pass_rate"]))
    return out


def per_question_vectors(
    old: Mapping[tuple, dict], new: Mapping[tuple, dict], source: str, uids: Sequence[str]
) -> dict[str, dict[str, float]]:
    """Per-question scalars, which is the unit the paired tests bootstrap over."""
    out: dict[str, dict[str, float]] = {}
    for uid in uids:
        o = {k[2]: v["score"] for k, v in old.items() if k[0] == uid and k[1] == source}
        n = {k[2]: v["score"] for k, v in new.items() if k[0] == uid and k[1] == source}
        shared = set(o) & set(n)
        if len(shared) < 2:
            continue
        o = {k: v for k, v in o.items() if k in shared}
        n = {k: v for k, v in n.items() if k in shared}
        po = [v["pass_rate"] for k, v in old.items() if k[0] == uid and k[1] == source and k[2] in shared]
        pn = [v["pass_rate"] for k, v in new.items() if k[0] == uid and k[1] == source and k[2] in shared]
        out[uid] = {
            "abs_diff": float(np.mean([abs(o[v] - n[v]) for v in shared])),
            "signed_diff": float(np.mean([n[v] - o[v] for v in shared])),
            "rank_agreement": rank_agreement(o, n),
            "min_crit_disagree": float(np.nanmean(np.abs(np.array(po) - np.array(pn))))
            if po and pn else math.nan,
            "pearson_within": _pearson([o[v] for v in sorted(shared)], [n[v] for v in sorted(shared)]),
        }
    return out


def analyse(
    old: Mapping[tuple, dict],
    new: Mapping[tuple, dict],
    sources: Sequence[str],
    uids: Sequence[str],
    reference: str,
    retest: Mapping[tuple, dict] | None = None,
) -> dict[str, Any]:
    """Cross-judge agreement, and — when ``retest`` is given — the same quantities
    for a second sample from the *incumbent* judge.

    The retest arm is what makes the cross-judge numbers interpretable. Sources
    differ in how self-consistently they can be judged at all (a vaguer criterion
    is noisier for any judge), so raw cross-judge agreement confounds "this
    source's wording suits the incumbent" with "this source is simply easier to
    judge reproducibly". Subtracting the within-judge agreement removes that:
    the residual gap is the part attributable to the judges being different
    models, which is the only part self-preference could explain.
    """
    pooled: dict[str, Any] = {}
    per_q: dict[str, dict[str, dict[str, float]]] = {}
    pooled_retest: dict[str, Any] = {}
    per_q_retest: dict[str, dict[str, dict[str, float]]] = {}
    for source in sources:
        c = per_source_cells(old, new, source)
        pooled[source] = {
            "n_cells": len(c["old"]),
            "pearson": _pearson(c["old"], c["new"]),
            "spearman": _spearman(c["old"], c["new"]),
            "mean_abs_diff": float(np.mean(c["abs_diff"])) if c["abs_diff"] else math.nan,
            "mean_signed_diff": float(np.mean(c["signed_diff"])) if c["signed_diff"] else math.nan,
            "mean_old": float(np.mean(c["old"])) if c["old"] else math.nan,
            "mean_new": float(np.mean(c["new"])) if c["new"] else math.nan,
            "pass_rate_old": float(np.mean(c["pass_old"])) if c["pass_old"] else math.nan,
            "pass_rate_new": float(np.mean(c["pass_new"])) if c["pass_new"] else math.nan,
            "min_criterion_disagreement": float(np.mean(c["min_crit_disagree"]))
            if c["min_crit_disagree"] else math.nan,
        }
        per_q[source] = per_question_vectors(old, new, source, uids)
        if retest is not None:
            cr = per_source_cells(old, retest, source)
            pooled_retest[source] = {
                "n_cells": len(cr["old"]),
                "pearson": _pearson(cr["old"], cr["new"]),
                "spearman": _spearman(cr["old"], cr["new"]),
                "mean_abs_diff": float(np.mean(cr["abs_diff"])) if cr["abs_diff"] else math.nan,
                "mean_signed_diff": float(np.mean(cr["signed_diff"])) if cr["signed_diff"] else math.nan,
            }
            per_q_retest[source] = per_question_vectors(old, retest, source, uids)

    # Paired-by-question contrasts against the reference source.
    metrics = ["abs_diff", "signed_diff", "rank_agreement", "min_crit_disagree", "pearson_within"]
    families: dict[str, dict[str, dict[str, dict[str, float]]]] = {"cross_judge": per_q}
    if per_q_retest:
        families["within_judge_retest"] = per_q_retest
        gap: dict[str, dict[str, dict[str, float]]] = {}
        for source in sources:
            shared = set(per_q[source]) & set(per_q_retest[source])
            gap[source] = {
                u: {
                    m: per_q[source][u][m] - per_q_retest[source][u][m]
                    for m in metrics
                }
                for u in shared
            }
        families["cross_minus_retest"] = gap

    contrasts: dict[str, Any] = {}
    for family, vectors in families.items():
        fam: dict[str, Any] = {}
        for metric in metrics:
            entry: dict[str, Any] = {}
            for source in sources:
                vals = [v[metric] for v in vectors[source].values() if math.isfinite(v[metric])]
                entry[source] = mean_ci(vals)
            pvals: list[float] = []
            keys: list[str] = []
            for source in sources:
                if source == reference:
                    continue
                shared = [u for u in vectors[source] if u in vectors[reference]]
                a = [vectors[source][u][metric] for u in shared]
                b = [vectors[reference][u][metric] for u in shared]
                mask = [i for i in range(len(a)) if math.isfinite(a[i]) and math.isfinite(b[i])]
                d = paired_diff([a[i] for i in mask], [b[i] for i in mask])
                entry[f"delta__{source}"] = d
                p = d.get("p_permutation", d.get("p_value", math.nan))
                if math.isfinite(p):
                    pvals.append(p)
                    keys.append(source)
            if pvals:
                for source, q in zip(keys, bh_fdr(pvals)):
                    entry[f"delta__{source}"]["q_bh"] = q
            fam[metric] = entry
        contrasts[family] = fam
    return {"pooled": pooled, "pooled_retest": pooled_retest,
            "contrasts": contrasts, "metrics": metrics, "per_question": per_q}


def _fmt(x: Any, nd: int = 3) -> str:
    return "—" if not (isinstance(x, (int, float)) and math.isfinite(float(x))) else f"{float(x):.{nd}f}"


def _sfmt(x: Any, nd: int = 3) -> str:
    """Signed variant of :func:`_fmt`; sign is what makes a delta column readable."""
    return "—" if not (isinstance(x, (int, float)) and math.isfinite(float(x))) else f"{float(x):+.{nd}f}"


def _stars(p: float, q: float | None = None) -> str:
    v = q if (q is not None and math.isfinite(q)) else p
    if not math.isfinite(v):
        return ""
    return "***" if v < 0.001 else "**" if v < 0.01 else "*" if v < 0.05 else ""


def render(report: Mapping[str, Any]) -> str:
    sources = report["sources"]
    ref = report["reference"]
    out = [
        "=" * 100,
        "CROSS-JUDGE AGREEMENT BY RUBRIC SOURCE",
        "=" * 100,
        "",
        f"old judge : {report['old_judge']}",
        f"new judge : {report['new_judge']}",
        f"cells     : identical (question, source, response) triples; "
        f"n_questions={report['n_questions']}",
        "",
        "If the agentic rubrics merely suit the incumbent judge's taste, the new judge",
        "should agree with it LESS on `agentic` than on the comparators. Equal agreement",
        "across sources is evidence against that story.",
        "",
    ]
    for scope, block in report["scopes"].items():
        out += ["=" * 100, f"SCOPE: {scope}", "=" * 100, ""]
        p = block["pooled"]
        out.append("  cross-judge, pooled over cells")
        out.append(
            f"    {'source':<20}{'n':>7}{'pearson':>9}{'spearman':>10}{'|Δ|':>8}"
            f"{'signed Δ':>10}{'mean_old':>10}{'mean_new':>10}"
            f"{'pass_old':>10}{'pass_new':>10}{'min_crit_dis':>14}"
        )
        for s in sources:
            c = p.get(s, {})
            out.append(
                f"    {s:<20}{c.get('n_cells', 0):>7}{_fmt(c.get('pearson')):>9}"
                f"{_fmt(c.get('spearman')):>10}{_fmt(c.get('mean_abs_diff')):>8}"
                f"{_sfmt(c.get('mean_signed_diff')):>10}{_fmt(c.get('mean_old')):>10}"
                f"{_fmt(c.get('mean_new')):>10}{_fmt(c.get('pass_rate_old')):>10}"
                f"{_fmt(c.get('pass_rate_new')):>10}"
                f"{_fmt(c.get('min_criterion_disagreement')):>14}"
            )
        out.append("")
        pr = block.get("pooled_retest") or {}
        if pr:
            out.append("  within-judge retest (incumbent vs itself), pooled over cells "
                       "— the floor the row above must be read against")
            out.append(
                f"    {'source':<20}{'n':>7}{'pearson':>9}{'spearman':>10}{'|Δ|':>8}"
                f"{'r_cross-r_retest':>18}{'|Δ|cross-|Δ|retest':>20}"
            )
            for s in sources:
                c, cr = p.get(s, {}), pr.get(s, {})
                dr = (c.get("pearson", math.nan) - cr.get("pearson", math.nan))
                da = (c.get("mean_abs_diff", math.nan) - cr.get("mean_abs_diff", math.nan))
                out.append(
                    f"    {s:<20}{cr.get('n_cells', 0):>7}{_fmt(cr.get('pearson')):>9}"
                    f"{_fmt(cr.get('spearman')):>10}{_fmt(cr.get('mean_abs_diff')):>8}"
                    f"{_sfmt(dr):>18}{_sfmt(da):>20}"
                )
            out.append("")
        for family, fam in block["contrasts"].items():
            out.append(f"  paired by question — {family}, Δ vs `{ref}` "
                       f"(BH-FDR over sources within each metric)")
            for metric in block["metrics"]:
                entry = fam[metric]
                out.append(f"    {metric}")
                out.append(
                    f"      {'source':<20}{'mean':>8}{'SE':>8}{'Δ vs ref':>11}"
                    f"{'95% CI':>22}{'p':>10}{'q(BH)':>9}"
                )
                for s in sources:
                    cell = entry.get(s, {})
                    if s == ref:
                        out.append(
                            f"      {s:<20}{_fmt(cell.get('mean')):>8}"
                            f"{_fmt(cell.get('se')):>8}{'(reference)':>11}"
                        )
                        continue
                    d = entry.get(f"delta__{s}", {})
                    pv = d.get("p_permutation", d.get("p_value", math.nan))
                    q = d.get("q_bh")
                    ci = f"[{_fmt(d.get('ci_low'))},{_fmt(d.get('ci_high'))}]"
                    out.append(
                        f"      {s:<20}{_fmt(cell.get('mean')):>8}{_fmt(cell.get('se')):>8}"
                        f"{_sfmt(d.get('mean_diff')):>11}{ci:>22}{_fmt(pv, 4):>10}"
                        f"{_fmt(q, 4):>9}{_stars(pv, q)}"
                    )
                out.append("")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-rows", default="runs/oldjudge_sub/metrics/discriminative_per_response_rows.jsonl")
    ap.add_argument("--new-rows", default="runs/judgeswap/metrics/discriminative_per_response_rows.jsonl")
    ap.add_argument("--old-judge", default="hy-t2t-glm-5.2-384k-fp8-L20A-t1-v2")
    ap.add_argument("--new-judge", default="api_azure_openai_gpt-5.1")
    ap.add_argument(
        "--retest-rows",
        default=None,
        help="second sample from the INCUMBENT judge; supplies the within-judge "
             "agreement floor that cross-judge agreement must be read against",
    )
    ap.add_argument("--reference", default="baseline")
    ap.add_argument("--sources", nargs="*", default=None)
    ap.add_argument("--out-prefix", default="results/judgeswap/cross_judge_agreement")
    args = ap.parse_args()

    old_rows = read_rows(Path(args.old_rows))
    new_rows = read_rows(Path(args.new_rows))
    old, domains = index(old_rows)
    new, _ = index(new_rows)
    retest = None
    if args.retest_rows and Path(args.retest_rows).exists():
        retest, _ = index(read_rows(Path(args.retest_rows)))

    present = {k[1] for k in old} & {k[1] for k in new}
    sources = args.sources or [s for s in SOURCE_ORDER if s in present] + sorted(present - set(SOURCE_ORDER))
    uids = sorted({k[0] for k in old} & {k[0] for k in new})

    report: dict[str, Any] = {
        "old_judge": args.old_judge,
        "new_judge": args.new_judge,
        "reference": args.reference,
        "sources": sources,
        "n_questions": len(uids),
        "scopes": {},
    }
    scopes = {"all": uids}
    for dom in sorted({domains.get(u, "?") for u in uids}):
        sub = [u for u in uids if domains.get(u) == dom]
        if len(sub) >= 5:
            scopes[dom] = sub
    for scope, sub in scopes.items():
        keep = set(sub)
        sub_old = {k: v for k, v in old.items() if k[0] in keep}
        sub_new = {k: v for k, v in new.items() if k[0] in keep}
        sub_retest = (
            {k: v for k, v in retest.items() if k[0] in keep} if retest is not None else None
        )
        block = analyse(sub_old, sub_new, sources, sub, args.reference, retest=sub_retest)
        block.pop("per_question", None)
        report["scopes"][scope] = block

    text = render(report)
    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".txt").write_text(text)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str))
    print(text)
    print(f"\nwrote {out.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
