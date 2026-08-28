#!/usr/bin/env python
"""Audit the gold-leakage confound in the discriminative main result.

Why this exists
---------------
The agentic generator's Stage 6 validates candidate criteria against
``example.reference_answer`` and discards the ones that answer fails
(``drop_gold_fail``) or contradicts (``drop_contradicted``). The discriminative
metric then uses *that same* ``reference_answer`` as its ``gold`` positive. So
Stage 6 optimises, directly, the quantity the headline metric rewards. Any
margin gain it produces is therefore ambiguous between

  (a) the rubric genuinely became a better reward signal, and
  (b) the rubric was merely fitted to this particular gold text.

Everything here runs off already-written judge verdicts and costs no LLM calls.
Three analyses, in increasing order of how much they cost the confound:

``decomposition``
    Absolute score per response tier per source. A rubric that got *better*
    should push degraded tiers down; a rubric that merely got *looser* lifts
    every tier, gold most of all. This tells the two apart by inspection.

``gold_excluded``
    Ranking accuracy and AUC recomputed with the gold tier deleted entirely,
    ordering the surviving degraded tiers by their construction-time quality
    level. This is the decisive number: it removes the leaked positive from the
    metric, so an advantage that survives here cannot be circular.

``normalised``
    Separation rescaled per source (margin / gold score, and within-source
    z-scored spread), which removes the "everything scores higher" level effect
    that raw margins confound with genuine discrimination.

Usage
-----
    python scripts/confound_audit.py --run-name pilot_v2 --reference baseline
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Sequence

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.eval.stats import auc, bh_fdr, mean_ci, paired_diff  # noqa: E402

#: Construction-time quality ordering of the response ladder (higher = better).
#: Mirrors ``harness.eval.responses.QUALITY_LEVELS``; two tiers deliberately
#: share level 2 because neither is clearly worse than the other, and equal
#: levels are excluded from ranking comparisons rather than scored as errors.
QUALITY_LEVELS: dict[str, int] = {
    "gold": 5,
    "terse_correct": 4,
    "missing_step": 3,
    "numeric_error": 2,
    "right_method_wrong_answer": 2,
    "verbose_empty": 1,
    "off_topic": 0,
}

#: Tier ordering used for the decomposition table (best first).
TIER_ORDER: list[str] = [
    "gold",
    "missing_step",
    "terse_correct",
    "right_method_wrong_answer",
    "numeric_error",
    "verbose_empty",
    "off_topic",
]

#: Above this level a response counts as "good" for the AUC split.
GOOD_QUALITY_THRESHOLD = 3

SOURCE_ORDER = ["shipped", "baseline", "agentic-noval", "agentic", "agentic-goldonly", "agentic-negonly"]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _order_sources(sources: Iterable[str]) -> list[str]:
    present = set(sources)
    ordered = [s for s in SOURCE_ORDER if s in present]
    return ordered + sorted(present - set(ordered))


def _fmt(x: float, nd: int = 3) -> str:
    return "—" if not (isinstance(x, (int, float)) and math.isfinite(x)) else f"{x:.{nd}f}"


def _stars(p: float) -> str:
    if not math.isfinite(p):
        return ""
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""


# --------------------------------------------------------------------------
# scores[uid][source][variant] = score
# --------------------------------------------------------------------------
def load_scores(
    rows: Sequence[Mapping[str, Any]],
    *,
    score_field: str = "score",
) -> tuple[dict[str, dict[str, dict[str, float]]], dict[str, str]]:
    """Index usable per-response judge scores and each question's domain."""
    scores: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    domains: dict[str, str] = {}
    for row in rows:
        if row.get("error") or not row.get("usable", True):
            continue
        value = row.get(score_field)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            continue
        uid, source, variant = row.get("uid"), row.get("rubric_source"), row.get("variant")
        if not (uid and source and variant):
            continue
        scores[uid][source][variant] = float(value)
        if row.get("domain"):
            domains[uid] = str(row["domain"])
    return scores, domains


def _complete_uids(
    scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    sources: Sequence[str],
    variants: Sequence[str],
) -> list[str]:
    """Questions where every source has every variant — the paired sample."""
    keep = []
    for uid, by_source in scores.items():
        if all(
            source in by_source and all(v in by_source[source] for v in variants)
            for source in sources
        ):
            keep.append(uid)
    return sorted(keep)


# --------------------------------------------------------------------------
# 1. absolute score decomposition
# --------------------------------------------------------------------------
def decomposition(
    scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    sources: Sequence[str],
    uids: Sequence[str],
    *,
    reference: str,
) -> dict[str, Any]:
    """Mean absolute score per (tier, source), plus paired Δ against reference."""
    tiers = [t for t in TIER_ORDER if any(t in scores[u].get(sources[0], {}) for u in uids)]
    table: dict[str, dict[str, Any]] = {}
    for tier in tiers:
        entry: dict[str, Any] = {}
        for source in sources:
            vals = [scores[u][source][tier] for u in uids if tier in scores[u].get(source, {})]
            entry[source] = mean_ci(vals)
        if reference in sources:
            for source in sources:
                if source == reference:
                    continue
                a = [scores[u][source][tier] for u in uids]
                b = [scores[u][reference][tier] for u in uids]
                entry[f"delta__{source}"] = paired_diff(a, b)
        table[tier] = entry
    return {"tiers": tiers, "table": table, "n": len(uids)}


# --------------------------------------------------------------------------
# 2. gold-excluded discrimination — the decisive analysis
# --------------------------------------------------------------------------
def _ranking_accuracy(pairs: Sequence[tuple[float, float]]) -> float:
    """Fraction of ordered pairs ranked correctly; ties score 0.5."""
    if not pairs:
        return math.nan
    hits = sum(1.0 if hi > lo else 0.5 if hi == lo else 0.0 for hi, lo in pairs)
    return hits / len(pairs)


def per_question_discrimination(
    by_variant: Mapping[str, float],
    *,
    exclude: Sequence[str] = (),
) -> dict[str, float]:
    """Ranking accuracy / AUC / spread for one question's ladder.

    ``exclude`` drops tiers before anything is computed, which is how the gold
    tier is removed. Only pairs whose construction-time quality levels *differ*
    contribute to ranking accuracy, so the deliberate 2–2 tie is not counted as
    an error against any source.
    """
    items = [
        (v, s, QUALITY_LEVELS[v])
        for v, s in by_variant.items()
        if v in QUALITY_LEVELS and v not in exclude
    ]
    if len(items) < 2:
        return {"ranking_accuracy": math.nan, "auc": math.nan, "spread": math.nan}

    pairs = [
        (a_score, b_score)
        for _, a_score, a_lvl in items
        for _, b_score, b_lvl in items
        if a_lvl > b_lvl
        for a_score, b_score in [(a_score, b_score)]
    ]
    good = [s for _, s, lvl in items if lvl >= GOOD_QUALITY_THRESHOLD]
    bad = [s for _, s, lvl in items if lvl < GOOD_QUALITY_THRESHOLD]
    all_scores = [s for _, s, _ in items]

    auc_value = math.nan
    if good and bad:
        auc_value = auc(good + bad, [1] * len(good) + [0] * len(bad))

    separation = float(np.mean(good) - np.mean(bad)) if good and bad else math.nan
    # Scale-normalised separation computed on the *same* (possibly gold-free)
    # ladder. Raw separation and spread both grow when a rubric simply scores
    # everything further apart, so this is the version that isolates ordering
    # quality from dynamic range.
    sd = float(np.std(all_scores, ddof=1)) if len(all_scores) > 1 else math.nan
    z_separation = (
        separation / sd if math.isfinite(sd) and sd > 1e-9 and math.isfinite(separation) else math.nan
    )

    return {
        "ranking_accuracy": _ranking_accuracy(pairs),
        "auc": auc_value,
        "separation": separation,
        "z_separation": z_separation,
        "spread": float(max(all_scores) - min(all_scores)),
        "mean_good": float(np.mean(good)) if good else math.nan,
        "mean_bad": float(np.mean(bad)) if bad else math.nan,
    }


def gold_excluded(
    scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    sources: Sequence[str],
    uids: Sequence[str],
    *,
    reference: str,
) -> dict[str, Any]:
    """Recompute discrimination with and without the leaked gold tier."""
    out: dict[str, Any] = {}
    for label, exclude in (("with_gold", ()), ("without_gold", ("gold",))):
        per_source: dict[str, dict[str, list[float]]] = {
            s: defaultdict(list) for s in sources
        }
        for uid in uids:
            for source in sources:
                stats = per_question_discrimination(scores[uid][source], exclude=exclude)
                for k, v in stats.items():
                    per_source[source][k].append(v)

        block: dict[str, Any] = {}
        metrics = ["ranking_accuracy", "auc", "separation", "z_separation", "spread"]
        for metric in metrics:
            entry: dict[str, Any] = {}
            for source in sources:
                entry[source] = mean_ci(per_source[source][metric])
            for source in sources:
                if source == reference:
                    continue
                entry[f"delta__{source}"] = paired_diff(
                    per_source[source][metric], per_source[reference][metric]
                )
            block[metric] = entry
        # BH-FDR over the declared family: every (metric x non-reference source)
        # contrast in this block. The report quotes q-values, so they have to be
        # computed here and stored, not recomputed by hand at writing time --
        # a number that exists only in prose cannot be checked against anything.
        out[label] = {
            "metrics": metrics,
            "table": block,
            "raw": per_source,
            "fdr": _apply_fdr(block, metrics, sources, reference=reference),
        }
    return out


def _apply_fdr(
    block: MutableMapping[str, Any],
    metrics: Sequence[str],
    sources: Sequence[str],
    *,
    reference: str,
) -> dict[str, Any]:
    """Correct one block of contrasts in place; return the family definition.

    The family is every metric crossed with every non-reference source, which is
    the whole set of claims the block makes. Defining it any more narrowly would
    understate the multiplicity.
    """
    keys: list[tuple[str, str]] = []
    pvalues: list[float] = []
    for metric in metrics:
        for source in sources:
            if source == reference:
                continue
            diff = (block.get(metric) or {}).get(f"delta__{source}") or {}
            p = diff.get("p_value", diff.get("p_permutation"))
            if isinstance(p, (int, float)) and math.isfinite(float(p)):
                keys.append((metric, source))
                pvalues.append(float(p))
    if not pvalues:
        return {"family_n": 0, "members": []}
    for (metric, source), q in zip(keys, bh_fdr(pvalues)):
        block[metric][f"delta__{source}"]["q_value"] = q
    return {
        "family_n": len(pvalues),
        "definition": f"{len(metrics)} metrics x {len(keys) // max(len(metrics), 1)} "
                      f"non-reference sources (reference={reference})",
        "members": [f"{m}:{s}" for m, s in keys],
    }


# --------------------------------------------------------------------------
# 3. scale-normalised separation
# --------------------------------------------------------------------------
def normalised(
    scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    sources: Sequence[str],
    uids: Sequence[str],
    *,
    reference: str,
) -> dict[str, Any]:
    """Separation measures that are invariant to a rubric being uniformly looser.

    ``margin_over_gold`` divides the gold-minus-degraded margin by the gold
    score, so a rubric that lifts everything proportionally gains nothing.
    ``z_separation`` z-scores each question's ladder within a source before
    measuring good-vs-bad separation, removing both level and scale.
    ``cohens_d`` is the same idea expressed as a standardised effect size.
    """
    per_source: dict[str, dict[str, list[float]]] = {s: defaultdict(list) for s in sources}
    for uid in uids:
        for source in sources:
            ladder = scores[uid][source]
            items = [(v, s, QUALITY_LEVELS[v]) for v, s in ladder.items() if v in QUALITY_LEVELS]
            gold = ladder.get("gold", math.nan)
            degraded = [s for v, s, _ in items if v != "gold"]
            good = [s for _, s, lvl in items if lvl >= GOOD_QUALITY_THRESHOLD]
            bad = [s for _, s, lvl in items if lvl < GOOD_QUALITY_THRESHOLD]

            margin = gold - float(np.mean(degraded)) if degraded else math.nan
            per_source[source]["margin"].append(margin)
            per_source[source]["margin_over_gold"].append(
                margin / gold if gold and math.isfinite(gold) and gold > 1e-9 else math.nan
            )

            vals = np.array([s for _, s, _ in items], dtype=float)
            sd = float(vals.std(ddof=1)) if vals.size > 1 else math.nan
            if good and bad and math.isfinite(sd) and sd > 1e-9:
                z_sep = (float(np.mean(good)) - float(np.mean(bad))) / sd
            else:
                z_sep = math.nan
            per_source[source]["z_separation"].append(z_sep)

            if len(good) > 1 and len(bad) > 1:
                pooled = math.sqrt(
                    (np.var(good, ddof=1) * (len(good) - 1) + np.var(bad, ddof=1) * (len(bad) - 1))
                    / (len(good) + len(bad) - 2)
                )
                d = (
                    (float(np.mean(good)) - float(np.mean(bad))) / pooled
                    if pooled > 1e-9
                    else math.nan
                )
            else:
                d = math.nan
            per_source[source]["cohens_d"].append(d)

    metrics = ["margin", "margin_over_gold", "z_separation", "cohens_d"]
    table: dict[str, Any] = {}
    for metric in metrics:
        entry: dict[str, Any] = {}
        for source in sources:
            entry[source] = mean_ci(per_source[source][metric])
        for source in sources:
            if source == reference:
                continue
            entry[f"delta__{source}"] = paired_diff(
                per_source[source][metric], per_source[reference][metric]
            )
        table[metric] = entry
    return {"metrics": metrics, "table": table}


# --------------------------------------------------------------------------
# 4. reward geometry — what the score distribution means for RL
# --------------------------------------------------------------------------
def reward_geometry(
    scores: Mapping[str, Mapping[str, Mapping[str, float]]],
    sources: Sequence[str],
    uids: Sequence[str],
    *,
    reference: str,
) -> dict[str, Any]:
    """Distribution shape that matters downstream, not just separation.

    A policy-gradient method with a group-relative baseline (GRPO and
    relatives) forms its advantage from the *spread* of rewards inside a group
    of sampled rollouts. Sampled rollouts from a partly-trained policy are
    mostly plausible attempts, not off-topic text, so what drives learning is
    how far apart the reward function spaces the *good* responses — not how far
    gold sits above an off-topic answer.

    ``good_spread`` is the within-question standard deviation over the good
    tiers, i.e. the usable advantage signal. ``headroom`` is ``1 - gold``, the
    room left before the reward saturates. ``saturation`` is the share of
    *degraded* responses already at or above 0.8, where further improvement
    stops being rewarded.
    """
    per_source: dict[str, dict[str, list[float]]] = {s: defaultdict(list) for s in sources}
    for uid in uids:
        for source in sources:
            ladder = scores[uid][source]
            items = [(v, s, QUALITY_LEVELS[v]) for v, s in ladder.items() if v in QUALITY_LEVELS]
            good = [s for _, s, lvl in items if lvl >= GOOD_QUALITY_THRESHOLD]
            degraded = [s for v, s, _ in items if v != "gold"]
            gold = ladder.get("gold", math.nan)

            per_source[source]["good_spread"].append(
                float(np.std(good, ddof=1)) if len(good) > 1 else math.nan
            )
            per_source[source]["good_range"].append(
                float(max(good) - min(good)) if len(good) > 1 else math.nan
            )
            per_source[source]["headroom"].append(1.0 - gold if math.isfinite(gold) else math.nan)
            per_source[source]["saturation"].append(
                float(np.mean([s >= 0.8 for s in degraded])) if degraded else math.nan
            )
            per_source[source]["mean_level"].append(
                float(np.mean([s for _, s, _ in items])) if items else math.nan
            )

    metrics = ["good_spread", "good_range", "headroom", "saturation", "mean_level"]
    table: dict[str, Any] = {}
    for metric in metrics:
        entry: dict[str, Any] = {}
        for source in sources:
            entry[source] = mean_ci(per_source[source][metric])
        for source in sources:
            if source == reference:
                continue
            entry[f"delta__{source}"] = paired_diff(
                per_source[source][metric], per_source[reference][metric]
            )
        table[metric] = entry
    return {"metrics": metrics, "table": table}


# --------------------------------------------------------------------------
# 5. judge-stability base rates (the kappa question)
# --------------------------------------------------------------------------
def kappa_base_rates(run_dir: Path, sources: Sequence[str]) -> dict[str, Any]:
    """Pass-rate base rates alongside kappa, plus the prevalence-adjusted variant.

    Cohen's kappa penalises class imbalance: when nearly every criterion gets
    the same verdict, expected chance agreement approaches observed agreement
    and kappa collapses even though the raters never disagree. PABAK
    (``2 * p_observed - 1``) is the standard prevalence- and bias-adjusted
    reading, and is reported next to raw agreement so a drop in kappa can be
    attributed to prevalence rather than to instability.
    """
    per_q = _read_jsonl(run_dir / "metrics" / "intrinsic_per_question_rows.jsonl")
    per_resp = _read_jsonl(run_dir / "metrics" / "discriminative_per_response_rows.jsonl")

    agreement: dict[str, list[float]] = defaultdict(list)
    kappa: dict[str, list[float]] = defaultdict(list)
    pabak: dict[str, list[float]] = defaultdict(list)
    for row in per_q:
        source = row.get("rubric_source")
        if source not in sources:
            continue
        p_o = row.get("self_agreement_rate")
        if isinstance(p_o, (int, float)) and math.isfinite(float(p_o)):
            agreement[source].append(float(p_o))
            pabak[source].append(2.0 * float(p_o) - 1.0)
        k = row.get("kappa")
        if isinstance(k, (int, float)) and math.isfinite(float(k)):
            kappa[source].append(float(k))

    pass_rate: dict[str, list[float]] = defaultdict(list)
    gold_pass_rate: dict[str, list[float]] = defaultdict(list)
    for row in per_resp:
        source = row.get("rubric_source")
        if source not in sources or row.get("error"):
            continue
        n_items, n_met = row.get("n_items"), row.get("n_met")
        if not isinstance(n_items, int) or not n_items:
            continue
        rate = float(n_met) / n_items
        pass_rate[source].append(rate)
        if row.get("variant") == "gold":
            gold_pass_rate[source].append(rate)

    return {
        source: {
            "self_agreement": mean_ci(agreement[source]),
            "kappa": mean_ci(kappa[source]),
            "pabak": mean_ci(pabak[source]),
            "criterion_pass_rate": mean_ci(pass_rate[source]),
            "gold_criterion_pass_rate": mean_ci(gold_pass_rate[source]),
            "n_kappa_defined": len(kappa[source]),
            "n_questions": len(agreement[source]),
        }
        for source in sources
    }


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------
def _render_block(
    title: str,
    metrics: Sequence[str],
    table: Mapping[str, Any],
    sources: Sequence[str],
    reference: str,
    *,
    note: str = "",
) -> list[str]:
    lines = [f"### {title}", ""]
    if note:
        lines += [note, ""]
    for metric in metrics:
        entry = table[metric]
        lines.append(f"  {metric}")
        lines.append(
            f"    {'source':<20}{'mean':>8}{'SE':>8}{'Δ vs ref':>11}{'95% CI':>22}"
            f"{'p':>10}{'q(BH)':>10}"
        )
        for source in sources:
            cell = entry.get(source, {})
            mean, se = cell.get("mean", math.nan), cell.get("se", math.nan)
            if source == reference:
                lines.append(
                    f"    {source:<20}{_fmt(mean):>8}{_fmt(se):>8}{'(reference)':>11}"
                )
                continue
            d = entry.get(f"delta__{source}", {})
            md, lo, hi = d.get("mean_diff", math.nan), d.get("ci_low", math.nan), d.get("ci_high", math.nan)
            p = d.get("p_permutation", d.get("p_value", math.nan))
            q = d.get("q_value", math.nan)
            ci = f"[{_fmt(lo)},{_fmt(hi)}]"
            lines.append(
                f"    {source:<20}{_fmt(mean):>8}{_fmt(se):>8}{md:>+11.3f}{ci:>22}"
                f"{_fmt(p, 4):>10}{_fmt(q, 4):>10}{_stars(q)}"
            )
        lines.append("")
    return lines


def render(report: Mapping[str, Any]) -> str:
    sources: list[str] = report["sources"]
    reference: str = report["reference"]
    out: list[str] = [
        "=" * 78,
        "CONFOUND AUDIT — gold leakage in the discriminative main result",
        "=" * 78,
        "",
        f"paired questions n={report['n_questions']}; reference = `{reference}`",
        "",
        "Stage 6 of the agentic generator filters criteria against "
        "example.reference_answer,",
        "which is the same text the discriminative metric uses as its `gold` "
        "positive. Any",
        "margin gain is therefore ambiguous between a better reward signal and a "
        "rubric",
        "fitted to this particular gold. The analyses below separate the two.",
        "",
    ]

    # 1. decomposition
    dec = report["decomposition"]
    out += [
        "=" * 78,
        "1. ABSOLUTE SCORE DECOMPOSITION",
        "=" * 78,
        "",
        "If a rubric genuinely discriminates better, degraded tiers should move "
        "DOWN.",
        "If it is merely looser, every tier moves UP and gold moves up most.",
        "",
        f"  {'tier':<28}" + "".join(f"{s:>16}" for s in sources),
    ]
    for tier in dec["tiers"]:
        entry = dec["table"][tier]
        cells = "".join(f"{_fmt(entry[s]['mean']):>16}" for s in sources)
        out.append(f"  {tier:<28}{cells}")
    out.append("")
    out.append(f"  Paired Δ vs `{reference}` (positive = higher score under that source):")
    out.append(f"  {'tier':<28}" + "".join(f"{s:>16}" for s in sources if s != reference))
    for tier in dec["tiers"]:
        entry = dec["table"][tier]
        cells = ""
        for s in sources:
            if s == reference:
                continue
            d = entry.get(f"delta__{s}", {})
            p = d.get("p_permutation", d.get("p_value", math.nan))
            cells += f"{d.get('mean_diff', math.nan):>+13.3f}{_stars(p):<3}"
        out.append(f"  {tier:<28}{cells}")
    out += ["", ""]

    # 2. gold-excluded
    ge = report["gold_excluded"]
    out += [
        "=" * 78,
        "2. DISCRIMINATION WITH THE LEAKED GOLD TIER REMOVED  (decisive)",
        "=" * 78,
        "",
    ]
    out += _render_block(
        "2a. WITH gold (as originally reported)",
        ge["with_gold"]["metrics"],
        ge["with_gold"]["table"],
        sources,
        reference,
    )
    out += _render_block(
        "2b. WITHOUT gold — degraded tiers only, ordered by construction quality",
        ge["without_gold"]["metrics"],
        ge["without_gold"]["table"],
        sources,
        reference,
        note="  An advantage that survives here cannot come from fitting the gold text.",
    )

    # 3. normalised
    nm = report["normalised"]
    out += ["=" * 78, "3. SCALE-NORMALISED SEPARATION", "=" * 78, ""]
    out += _render_block(
        "3a. level- and scale-invariant separation",
        nm["metrics"],
        nm["table"],
        sources,
        reference,
        note="  Removes the 'everything scores higher' effect that raw margin confounds.",
    )

    # 4. reward geometry
    rg = report.get("reward_geometry")
    if rg:
        out += ["=" * 78, "4. REWARD GEOMETRY — implications for RL", "=" * 78, ""]
        out += _render_block(
            "4a. distribution shape",
            rg["metrics"],
            rg["table"],
            sources,
            reference,
            note=(
                "  good_spread is the usable advantage signal inside a group of\n"
                "  plausible rollouts; headroom is 1-gold; saturation is the share of\n"
                "  degraded responses already >= 0.8."
            ),
        )

    # 5. kappa base rates
    kb = report.get("kappa_base_rates") or {}
    if kb:
        out += ["=" * 78, "5. JUDGE STABILITY — kappa vs its base rate", "=" * 78, ""]
        out.append(
            f"  {'source':<20}{'agreement':>11}{'kappa':>9}{'PABAK':>9}"
            f"{'pass rate':>11}{'gold pass':>11}{'n_kappa':>9}"
        )
        for source in sources:
            c = kb.get(source)
            if not c:
                continue
            out.append(
                f"  {source:<20}"
                f"{_fmt(c['self_agreement']['mean']):>11}"
                f"{_fmt(c['kappa']['mean']):>9}"
                f"{_fmt(c['pabak']['mean']):>9}"
                f"{_fmt(c['criterion_pass_rate']['mean']):>11}"
                f"{_fmt(c['gold_criterion_pass_rate']['mean']):>11}"
                f"{c['n_kappa_defined']:>9}"
            )
        out += [
            "",
            "  Kappa falls when verdicts are one-sided even if raters never disagree.",
            "  Read agreement and PABAK alongside the criterion pass rate before "
            "concluding",
            "  that judge stability actually degraded.",
            "",
        ]
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-name", default="pilot_v2")
    ap.add_argument("--runs-dir", default="runs")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--reference", default="baseline")
    ap.add_argument("--prefix", default="", help="metric-file prefix, e.g. cc_ for count-controlled")
    ap.add_argument("--sources", nargs="*", default=None)
    # On by default. A pooled-only run once let a science-specific effect be
    # written up as a general one, so the split has to be opt-out, not opt-in.
    ap.add_argument(
        "--no-by-domain",
        dest="by_domain",
        action="store_false",
        help="suppress the per-domain breakdown (it is emitted by default)",
    )
    ap.set_defaults(by_domain=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    run_dir = Path(args.runs_dir) / args.run_name
    rows = _read_jsonl(run_dir / "metrics" / f"{args.prefix}discriminative_per_response_rows.jsonl")
    if not rows:
        raise SystemExit(f"no per-response rows under {run_dir}/metrics/")

    scores, domains = load_scores(rows)
    sources = args.sources or _order_sources({r["rubric_source"] for r in rows if r.get("rubric_source")})
    variants = [v for v in TIER_ORDER]
    uids = _complete_uids(scores, sources, variants)
    if not uids:
        raise SystemExit("no question has a complete ladder across all requested sources")

    report: dict[str, Any] = {
        "run_name": args.run_name,
        "prefix": args.prefix,
        "reference": args.reference,
        "sources": sources,
        "n_questions": len(uids),
        "decomposition": decomposition(scores, sources, uids, reference=args.reference),
        "gold_excluded": gold_excluded(scores, sources, uids, reference=args.reference),
        "normalised": normalised(scores, sources, uids, reference=args.reference),
        "reward_geometry": reward_geometry(scores, sources, uids, reference=args.reference),
        "kappa_base_rates": kappa_base_rates(run_dir, sources) if not args.prefix else {},
    }

    if args.by_domain:
        by_domain: dict[str, Any] = {}
        for domain in sorted({domains.get(u, "?") for u in uids}):
            sub = [u for u in uids if domains.get(u) == domain]
            if len(sub) < 5:
                continue
            by_domain[domain] = {
                "n_questions": len(sub),
                "decomposition": decomposition(scores, sources, sub, reference=args.reference),
                "gold_excluded": gold_excluded(scores, sources, sub, reference=args.reference),
                "normalised": normalised(scores, sources, sub, reference=args.reference),
            }
        report["by_domain"] = by_domain

    # `raw` holds per-question vectors; useful on disk, too noisy for JSON summary
    for block in report["gold_excluded"].values():
        block.pop("raw", None)
    for domain_block in report.get("by_domain", {}).values():
        for block in domain_block["gold_excluded"].values():
            block.pop("raw", None)

    text = render(report)
    if args.by_domain:
        for domain, block in report.get("by_domain", {}).items():
            text += "\n\n" + "=" * 78
            text += f"\nPER-DOMAIN: {domain} (n={block['n_questions']})\n" + "=" * 78 + "\n\n"
            text += "\n".join(
                _render_block(
                    "gold-excluded discrimination",
                    block["gold_excluded"]["without_gold"]["metrics"],
                    block["gold_excluded"]["without_gold"]["table"],
                    sources,
                    args.reference,
                )
            )

    out_dir = Path(args.results_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.prefix}confound_audit"
    (out_dir / f"{stem}.txt").write_text(text)
    (out_dir / f"{stem}.json").write_text(json.dumps(report, indent=2, default=str))
    print(text)
    print(f"\nwrote {out_dir / (stem + '.txt')}")


if __name__ == "__main__":
    main()
