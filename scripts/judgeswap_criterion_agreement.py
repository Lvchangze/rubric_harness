#!/usr/bin/env python
"""Per-criterion cross-judge agreement, by rubric source.

The sharpest form of the self-preference test. Aggregated scores can agree by
cancellation — two judges failing different criteria can land on the same number
— so agreement is measured on the atom the judge actually reports: whether one
criterion is literally true of one response.

Rows join on ``(uid, rubric_source, variant, criterion_index)``.
``criterion_index`` refers to the *unshuffled* rubric, so the two judges having
seen the criteria in different presentation orders does not matter.

Reported per source:

``agreement``
    Share of criteria where both judges gave the same verdict.
``kappa``
    Cohen's kappa on the 2x2 table. Reported with its base rates because kappa
    collapses toward 0 when verdicts are one-sided even if the judges never
    disagree, which is the regime these rubrics live in (pass rates near 0.8).
``pabak``
    ``2 * agreement - 1``, the prevalence- and bias-adjusted reading. Use this,
    not kappa, when the pass rates below differ a lot between sources.
``pass_rate_a`` / ``pass_rate_b``
    Each judge's marginal rate of calling a criterion true.

The interpretation that matters: if ``agreement`` on ``agentic`` is materially
BELOW ``agreement`` on ``baseline``/``shipped``, the incumbent judge was reading
agentic criteria in a way the new judge does not share — the signature of
wording tuned to that judge. Equal or higher agreement on ``agentic`` is
evidence against the self-preference story.

A ``--retest`` file (a second sample from the *same* judge) supplies the floor
this has to be read against: cross-judge agreement below within-judge agreement
is expected and uninformative on its own; what matters is whether the *gap*
between the two is larger for ``agentic`` than for the comparators.
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

from harness.eval.stats import bh_fdr, mean_ci, paired_diff  # noqa: E402

SOURCE_ORDER = ["shipped", "baseline", "agentic-noval", "agentic",
                "agentic-goldonly", "agentic-negonly"]
NAN = float("nan")

Key = tuple[str, str, str, int]


def load_verdicts(path: Path) -> tuple[dict[Key, bool], dict[str, str]]:
    verdicts: dict[Key, bool] = {}
    domains: dict[str, str] = {}
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            uid = row.get("uid")
            src = row.get("rubric_source")
            variant = row.get("variant")
            idx = row.get("criterion_index")
            if uid is None or src is None or variant is None or idx is None:
                continue
            verdicts[(uid, src, variant, int(idx))] = bool(row.get("literally_true"))
            if row.get("domain"):
                domains[uid] = str(row["domain"])
    return verdicts, domains


def kappa_from_counts(n11: int, n10: int, n01: int, n00: int) -> float:
    n = n11 + n10 + n01 + n00
    if n == 0:
        return NAN
    po = (n11 + n00) / n
    pe = ((n11 + n10) * (n11 + n01) + (n01 + n00) * (n10 + n00)) / (n * n)
    if abs(1.0 - pe) < 1e-12:
        return NAN
    return (po - pe) / (1.0 - pe)


def pooled_stats(a: Mapping[Key, bool], b: Mapping[Key, bool], source: str) -> dict[str, Any]:
    n11 = n10 = n01 = n00 = 0
    for key, va in a.items():
        if key[1] != source:
            continue
        vb = b.get(key)
        if vb is None:
            continue
        if va and vb:
            n11 += 1
        elif va and not vb:
            n10 += 1
        elif (not va) and vb:
            n01 += 1
        else:
            n00 += 1
    n = n11 + n10 + n01 + n00
    if n == 0:
        return {"n": 0}
    agreement = (n11 + n00) / n
    return {
        "n": n,
        "agreement": agreement,
        "kappa": kappa_from_counts(n11, n10, n01, n00),
        "pabak": 2.0 * agreement - 1.0,
        "pass_rate_a": (n11 + n10) / n,
        "pass_rate_b": (n11 + n01) / n,
        "n_a_true_b_false": n10,
        "n_a_false_b_true": n01,
    }


def per_question_agreement(
    a: Mapping[Key, bool], b: Mapping[Key, bool], source: str
) -> dict[str, float]:
    """Agreement within each question — the resampling unit for the paired tests."""
    hits: dict[str, list[float]] = defaultdict(list)
    for key, va in a.items():
        if key[1] != source:
            continue
        vb = b.get(key)
        if vb is None:
            continue
        hits[key[0]].append(1.0 if va == vb else 0.0)
    return {uid: float(np.mean(v)) for uid, v in hits.items() if v}


def analyse(
    cross: tuple[Mapping[Key, bool], Mapping[Key, bool]],
    retest: tuple[Mapping[Key, bool], Mapping[Key, bool]] | None,
    sources: Sequence[str],
    uids: Sequence[str],
    reference: str,
) -> dict[str, Any]:
    out: dict[str, Any] = {"pooled": {}, "pooled_retest": {}, "contrasts": {}}
    per_q_cross: dict[str, dict[str, float]] = {}
    per_q_retest: dict[str, dict[str, float]] = {}
    keep = set(uids)

    def scope(m: Mapping[Key, bool]) -> dict[Key, bool]:
        return {k: v for k, v in m.items() if k[0] in keep}

    ca, cb = scope(cross[0]), scope(cross[1])
    for source in sources:
        out["pooled"][source] = pooled_stats(ca, cb, source)
        per_q_cross[source] = per_question_agreement(ca, cb, source)
    if retest is not None:
        ra, rb = scope(retest[0]), scope(retest[1])
        for source in sources:
            out["pooled_retest"][source] = pooled_stats(ra, rb, source)
            per_q_retest[source] = per_question_agreement(ra, rb, source)

    families: dict[str, dict[str, dict[str, float]]] = {"cross_judge": per_q_cross}
    if per_q_retest:
        families["within_judge_retest"] = per_q_retest
        gap: dict[str, dict[str, float]] = {}
        for source in sources:
            shared = set(per_q_cross[source]) & set(per_q_retest[source])
            gap[source] = {
                u: per_q_cross[source][u] - per_q_retest[source][u] for u in shared
            }
        families["cross_minus_retest"] = gap

    for name, per_q in families.items():
        entry: dict[str, Any] = {}
        for source in sources:
            entry[source] = mean_ci(list(per_q[source].values()))
        pvals, keys = [], []
        for source in sources:
            if source == reference:
                continue
            shared = sorted(set(per_q[source]) & set(per_q[reference]))
            d = paired_diff(
                [per_q[source][u] for u in shared],
                [per_q[reference][u] for u in shared],
            )
            entry[f"delta__{source}"] = d
            p = d.get("p_permutation", d.get("p_value", NAN))
            if math.isfinite(p):
                pvals.append(p)
                keys.append(source)
        for source, q in zip(keys, bh_fdr(pvals)):
            entry[f"delta__{source}"]["q_bh"] = float(q)
        out["contrasts"][name] = entry
    return out


def _f(x: Any, nd: int = 3) -> str:
    return "—" if not (isinstance(x, (int, float)) and math.isfinite(float(x))) else f"{float(x):.{nd}f}"


def _s(x: Any, nd: int = 3) -> str:
    return "—" if not (isinstance(x, (int, float)) and math.isfinite(float(x))) else f"{float(x):+.{nd}f}"


def _sig(p: Any, q: Any = None) -> str:
    v = q if (isinstance(q, (int, float)) and math.isfinite(float(q))) else p
    if not (isinstance(v, (int, float)) and math.isfinite(float(v))):
        return ""
    v = float(v)
    return "***" if v < 0.001 else "**" if v < 0.01 else "*" if v < 0.05 else ""


def render(report: Mapping[str, Any]) -> str:
    sources, ref = report["sources"], report["reference"]
    lines = [
        "=" * 104,
        "PER-CRITERION CROSS-JUDGE AGREEMENT BY RUBRIC SOURCE",
        "=" * 104,
        "",
        f"judge A (incumbent) : {report['judge_a']}",
        f"judge B (swapped in): {report['judge_b']}",
        f"retest              : {report.get('retest_label') or '(not supplied)'}",
        "",
        "Joined on (uid, source, variant, criterion_index) — the unshuffled index, so",
        "differing presentation order between judges does not affect the join.",
        "",
        "Self-preference reads as: agreement on `agentic` materially BELOW agreement on",
        "`baseline`/`shipped`. Equal or higher agreement on `agentic` argues against it.",
        "",
    ]
    for scope, block in report["scopes"].items():
        lines += ["=" * 104, f"SCOPE: {scope}  (n_questions={block['n_questions']})",
                  "=" * 104, ""]
        for key, title in (("pooled", "cross-judge, pooled over criteria"),
                           ("pooled_retest", "within-judge retest, pooled over criteria")):
            table = block.get(key) or {}
            if not table or all(t.get("n", 0) == 0 for t in table.values()):
                continue
            lines.append(f"  {title}")
            lines.append(
                f"    {'source':<20}{'n_crit':>9}{'agreement':>11}{'kappa':>8}"
                f"{'PABAK':>8}{'pass_A':>9}{'pass_B':>9}{'A+B-':>7}{'A-B+':>7}"
            )
            for s in sources:
                c = table.get(s) or {}
                lines.append(
                    f"    {s:<20}{c.get('n', 0):>9}{_f(c.get('agreement')):>11}"
                    f"{_f(c.get('kappa')):>8}{_f(c.get('pabak')):>8}"
                    f"{_f(c.get('pass_rate_a')):>9}{_f(c.get('pass_rate_b')):>9}"
                    f"{c.get('n_a_true_b_false', 0):>7}{c.get('n_a_false_b_true', 0):>7}"
                )
            lines.append("")
        for name, entry in block["contrasts"].items():
            lines.append(f"  paired by question — {name}, Δ vs `{ref}` (BH-FDR across sources)")
            lines.append(
                f"    {'source':<20}{'mean':>8}{'SE':>8}{'Δ vs ref':>11}"
                f"{'95% CI':>22}{'p':>9}{'q(BH)':>9}"
            )
            for s in sources:
                cell = entry.get(s, {})
                if s == ref:
                    lines.append(f"    {s:<20}{_f(cell.get('mean')):>8}"
                                 f"{_f(cell.get('se')):>8}{'(reference)':>11}")
                    continue
                d = entry.get(f"delta__{s}", {})
                p = d.get("p_permutation", d.get("p_value", NAN))
                ci = f"[{_f(d.get('ci_low'))},{_f(d.get('ci_high'))}]"
                lines.append(
                    f"    {s:<20}{_f(cell.get('mean')):>8}{_f(cell.get('se')):>8}"
                    f"{_s(d.get('mean_diff')):>11}{ci:>22}{_f(p, 4):>9}"
                    f"{_f(d.get('q_bh'), 4):>9}{_sig(p, d.get('q_bh'))}"
                )
            lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="runs/oldjudge_rerun/discriminative_verdicts.jsonl",
                    help="incumbent judge verdicts")
    ap.add_argument("--b", default="runs/judgeswap/discriminative_verdicts.jsonl",
                    help="swapped-in judge verdicts")
    ap.add_argument("--retest", default=None,
                    help="second sample from the incumbent judge (same-judge floor)")
    ap.add_argument("--judge-a", default="hy-t2t-glm-5.2-384k-fp8-L20A-t1-v2")
    ap.add_argument("--judge-b", default="api_azure_openai_gpt-5.1")
    ap.add_argument("--retest-label", default=None)
    ap.add_argument("--reference", default="baseline")
    ap.add_argument("--out-prefix", default="results/judgeswap/criterion_agreement")
    args = ap.parse_args()

    a, domains = load_verdicts(Path(args.a))
    b, dom_b = load_verdicts(Path(args.b))
    domains = {**dom_b, **domains}
    retest = None
    if args.retest and Path(args.retest).exists():
        r, _ = load_verdicts(Path(args.retest))
        retest = (a, r)

    present = {k[1] for k in a} & {k[1] for k in b}
    sources = [s for s in SOURCE_ORDER if s in present] + sorted(present - set(SOURCE_ORDER))
    uids = sorted({k[0] for k in a} & {k[0] for k in b})

    report: dict[str, Any] = {
        "judge_a": args.judge_a,
        "judge_b": args.judge_b,
        "retest_label": args.retest_label or (args.retest if args.retest else None),
        "reference": args.reference,
        "sources": sources,
        "n_questions": len(uids),
        "n_criteria_joined": sum(1 for k in a if k in b),
        "scopes": {},
    }
    scopes = {"all": uids}
    for dom in sorted({domains.get(u, "?") for u in uids}):
        sub = [u for u in uids if domains.get(u) == dom]
        if len(sub) >= 5:
            scopes[dom] = sub
    for scope, sub in scopes.items():
        block = analyse((a, b), retest, sources, sub, args.reference)
        block["n_questions"] = len(sub)
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
