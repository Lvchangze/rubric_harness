"""Section 4/4: composite failure-mode rates plus verbatim evidence.

Joins the per-criterion signals computed by the other scripts into two headline
numbers -- the "recyclable criterion" rate (generic text under a mass-produced
title: the RaR-PREDEFINED failure mode the paper itself shows is catastrophic)
and the multiple-choice padding rate -- and dumps quotable examples for the
write-up into analysis/out/examples.md.
"""

from __future__ import annotations

import collections
import json
import re

import numpy as np
import pandas as pd

from common import DOMAINS, OUT, content_tokens, load_domain, pct
from rubric_grounding import NUMBER_RE, detect_subparts, instance_anchors
from rubric_similarity import normalise_title
from rubric_stats import STYLE_TERMS, SUBJECTIVE_TERMS

OPTION_RE = re.compile(r"\(\s*[a-eA-E]\s*\)|\boption\b|\banswer\b|\bchoice\b", re.IGNORECASE)
TITLE_MASS_PRODUCED = 10

# A Pitfall item can be written two opposite ways, and the sign of the reward
# depends on which one it is:
#   "failure-describing"  -> satisfying the sentence means the response is BAD
#   "avoidance-describing"-> satisfying the sentence means the response is GOOD
_SUBJECT = r"(?:the\s+(?:response|answer|explanation)\s+)?"
FAILURE_PHRASING_RE = re.compile(
    rf"^\s*{_SUBJECT}("
    r"does\s+not\s+(?:mention|state|recommend|include|identify|address|note|specify|suggest"
    r"|list|provide|discuss|explain)"
    r"|fails?\s+to|omits?|neglects?\s+to|overlooks?|ignores?|misses"
    r"|recommends?|suggests?|claims?|misidentifies|confuses|mistakes)",
    re.IGNORECASE,
)
AVOIDANCE_PHRASING_RE = re.compile(
    rf"^\s*{_SUBJECT}("
    r"avoid(?:s|ing|ed)?"
    r"|does\s+not\s+(?:incorrectly|wrongly|falsely|erroneously|mistakenly)"
    r"|(?:must|should)\s+(?:not|avoid|never|refrain|steer)"
    r"|do\s+not|refrains?|never|warns?\s+against|cautions?\s+against|ensures?|prevents?)",
    re.IGNORECASE,
)


def build_flags(domain: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_domain(domain)
    crit = pd.read_parquet(OUT / f"criteria_{domain}.parquet")

    doc_freq: collections.Counter = collections.Counter()
    for q, r in zip(df["question"], df["reference_answer"]):
        doc_freq.update(set(content_tokens(f"{q} {r}")))
    rare_cut = max(2, int(0.01 * len(df)))
    rare = {t for t, c in doc_freq.items() if c <= rare_cut}
    anchors_by_qid = {
        row.qid: instance_anchors(row.question, row.reference_answer, rare)
        for row in df.itertuples(index=False)
    }

    grounded = []
    for qid, body in zip(crit["qid"], crit["body"]):
        tokens = set(content_tokens(body)) | set(NUMBER_RE.findall(body.lower()))
        grounded.append(bool(tokens & anchors_by_qid.get(qid, set())))
    crit["grounded"] = grounded

    norm_titles = crit["title"].map(normalise_title)
    title_counts = collections.Counter(norm_titles)
    crit["norm_title"] = norm_titles
    crit["title_reuse"] = norm_titles.map(title_counts)
    crit["mass_produced_title"] = crit["title_reuse"] >= TITLE_MASS_PRODUCED

    lowered = crit["body"].str.lower()
    crit["subjective"] = lowered.apply(lambda s: any(t in s for t in SUBJECTIVE_TERMS))
    crit["style"] = lowered.apply(lambda s: any(t in s for t in STYLE_TERMS))

    # Recyclable: no anchor to this prompt AND phrased in a mould used elsewhere.
    crit["recyclable"] = (~crit["grounded"]) & crit["mass_produced_title"]
    return df, crit


def composite_metrics(df: pd.DataFrame, crit: pd.DataFrame) -> dict:
    n = len(crit)
    per_q = crit.groupby("qid")["recyclable"].agg(["sum", "mean"])

    kinds = {}
    for row in df.itertuples(index=False):
        _spans, kind = detect_subparts(row.question)
        kinds[row.qid] = kind
    crit = crit.assign(qkind=crit["qid"].map(kinds))
    mcq = crit[crit["qkind"] == "mcq"]
    mcq_answer_targeted = mcq["body"].str.contains(OPTION_RE) if len(mcq) else pd.Series(dtype=bool)

    return {
        "recyclable_criterion_rate_pct": pct(int(crit["recyclable"].sum()), n),
        "mean_recyclable_items_per_question": round(float(per_q["sum"].mean()), 3),
        "questions_with_ge2_recyclable_items_pct": pct(int((per_q["sum"] >= 2).sum()), len(per_q)),
        "questions_with_ge3_recyclable_items_pct": pct(int((per_q["sum"] >= 3).sum()), len(per_q)),
        "questions_with_zero_recyclable_items_pct": pct(int((per_q["sum"] == 0).sum()), len(per_q)),
        "generic_and_subjective_rate_pct": pct(
            int(((~crit["grounded"]) & (crit["subjective"] | crit["style"])).sum()), n
        ),
        "mass_produced_title_rate_pct": pct(int(crit["mass_produced_title"].sum()), n),
        "multiple_choice": {
            "n_questions": int(sum(1 for k in kinds.values() if k == "mcq")),
            "pct_of_questions": pct(sum(1 for k in kinds.values() if k == "mcq"), len(df)),
            "mean_criteria": round(float(len(mcq) / max(1, sum(1 for k in kinds.values() if k == "mcq"))), 2),
            "criteria_referring_to_an_answer_option_pct": pct(
                int(mcq_answer_targeted.sum()), max(1, len(mcq))
            ),
            "non_answer_criteria_per_mcq_question": round(
                float(
                    (len(mcq) - int(mcq_answer_targeted.sum()))
                    / max(1, sum(1 for k in kinds.values() if k == "mcq"))
                ),
                2,
            ),
        },
        "reward_share_of_generic_criteria": _reward_share(crit),
        "pitfall_polarity": _pitfall_polarity(crit),
        "weight_order_violations": _weight_order_violations(crit),
    }


def _weight_order_violations(crit: pd.DataFrame) -> dict:
    """Inside one rubric, does a lower-priority item ever weigh as much as a higher one?

    Essential > Important > Optional is the stated semantics; if the numbers do
    not respect it, the weighted sum of Eq. 1 contradicts the category labels.
    """
    ranks = {"Essential": 3, "Important": 2, "Optional": 1}
    sub = crit[crit["category"].isin(ranks)].copy()
    sub["rank"] = sub["category"].map(ranks)
    sub["weight"] = sub["weight"].astype(float)

    violations, comparisons, q_bad = 0, 0, set()
    for qid, group in sub.groupby("qid", sort=False):
        r = group["rank"].to_numpy()
        w = group["weight"].to_numpy()
        higher = r[:, None] > r[None, :]
        comparisons += int(higher.sum())
        bad = higher & (w[:, None] <= w[None, :])
        violations += int(bad.sum())
        if bad.any():
            q_bad.add(qid)
    n_q = sub["qid"].nunique()
    return {
        "n_ordered_pairs": comparisons,
        "pairs_where_lower_category_weighs_ge_higher_pct": pct(violations, comparisons),
        "questions_with_at_least_one_violation_pct": pct(len(q_bad), n_q),
    }


def _pitfall_polarity(crit: pd.DataFrame) -> dict:
    """Do Pitfall items describe the failure, or the avoidance of the failure?

    Eq. 1 multiplies each criterion by a single weight, so a rubric that mixes the
    two phrasings has no consistent sign convention: for one item c_j=1 means the
    response is good, for the next it means the response is bad.
    """
    pit = crit[crit["category"] == "Pitfall"]
    if pit.empty:
        return {}
    failure = pit["body"].str.match(FAILURE_PHRASING_RE)
    avoidance = pit["body"].str.match(AVOIDANCE_PHRASING_RE) & ~failure
    per_q = pd.DataFrame({"f": failure, "a": avoidance}).groupby(pit["qid"].to_numpy()).any()
    return {
        "n_pitfall_criteria": int(len(pit)),
        "failure_describing_pct": pct(int(failure.sum()), len(pit)),
        "avoidance_describing_pct": pct(int(avoidance.sum()), len(pit)),
        "unclassified_pct": pct(int((~failure & ~avoidance).sum()), len(pit)),
        "questions_mixing_both_phrasings_pct": pct(
            int((per_q["f"] & per_q["a"]).sum()), len(per_q)
        ),
        "negative_weight_on_failure_describing_pct": pct(
            int((failure & (pit["weight"] < 0)).sum()), max(1, int(failure.sum()))
        ),
        "negative_weight_on_avoidance_describing_pct": pct(
            int((avoidance & (pit["weight"] < 0)).sum()), max(1, int(avoidance.sum()))
        ),
    }


def _reward_share(crit: pd.DataFrame) -> dict:
    """Under Eq. 1, how much of the available reward mass sits on generic items?"""
    w = crit["weight"].fillna(0).astype(float).abs()
    total = w.sum()
    generic = w[~crit["grounded"]].sum()
    style = w[crit["subjective"] | crit["style"]].sum()
    return {
        "pct_of_total_absolute_weight_on_generic_criteria": pct(float(generic), float(total)),
        "pct_of_total_absolute_weight_on_subjective_or_style_criteria": pct(
            float(style), float(total)
        ),
    }


def _fmt(items: list[str], limit: int = 3) -> str:
    return "\n".join(f"  - `{s.strip()}`" for s in items[:limit])


def write_examples(all_flags: dict[str, tuple[pd.DataFrame, pd.DataFrame]]) -> None:
    lines = ["# Verbatim evidence from RaR-Science / RaR-Medicine", ""]
    lines.append("Auto-generated by `analysis/collect_examples.py`. All text is copied ")
    lines.append("verbatim from the released parquet files.\n")

    sim = json.loads((OUT / "similarity_stats.json").read_text())

    for domain, (df, crit) in all_flags.items():
        lines.append(f"\n## {domain}\n")

        lines.append("### A. Recyclable criteria (generic text + mass-produced title)\n")
        rec = crit[crit["recyclable"]].sample(n=min(8, int(crit["recyclable"].sum())), random_state=0)
        for row in rec.itertuples(index=False):
            lines.append(f"- (title reused {row.title_reuse}x) `{row.description.strip()}`")
        lines.append("")

        lines.append("### B. Highest-frequency criterion openers\n")
        for item in sim[domain]["template_boilerplate"]["top20_openers_6w"][:6]:
            lines.append(f"- {item['count']}x ({item['pct']}%) `{item['opener']} ...`")
        lines.append("")

        lines.append("### C. Identical criterion text on different questions\n")
        for item in sim[domain]["cross_question_near_duplicates"]["examples_top40"][:6]:
            if item["sim"] >= 0.95:
                lines.append(f"- sim={item['sim']} `{item['a'].strip()}`")
                lines.append(f"  - also on `{item['b_qid']}`: `{item['b'].strip()}`")
        lines.append("")

        lines.append("### D. Within-rubric redundancy (anchor-aware: same numbers/symbols)\n")
        for item in sim[domain]["within_question_redundancy"].get(
            "examples_anchor_aware_top25", []
        )[:5]:
            lines.append(f"- sim={item['sim']} on `{item['qid']}`")
            lines.append(f"  - A: `{item['a'].strip()}`")
            lines.append(f"  - B: `{item['b'].strip()}`")
        lines.append("")

        lines.append("### E. Subjective / unverifiable criteria\n")
        subj = crit[crit["subjective"] & ~crit["grounded"]].sample(
            n=min(6, int((crit["subjective"] & ~crit["grounded"]).sum())), random_state=1
        )
        for row in subj.itertuples(index=False):
            lines.append(f"- (w={row.weight}) `{row.description.strip()}`")
        lines.append("")

        lines.append("### F. Whole rubrics for a multi-part question\n")
        multi = [
            r for r in df.itertuples(index=False) if detect_subparts(r.question)[1] == "subquestions"
        ]
        for row in multi[:2]:
            lines.append(f"**Question** (`{row.qid}`, {row.rubric_count} criteria):\n")
            lines.append("```")
            lines.append(row.question.strip()[:900])
            lines.append("```")
            lines.append(f"**Reference answer** ({len(str(row.reference_answer).split())} words):\n")
            lines.append("```")
            lines.append(str(row.reference_answer).strip()[:600])
            lines.append("```")
            lines.append("**Rubric:**\n")
            for item in row.rubric:
                lines.append(f"- (w={item['weight']}) **{item['title']}** — {item['description']}")
            lines.append("")

        lines.append("### G. Rubrics generated from an (almost) empty reference answer\n")
        short = df[df["reference_answer"].map(lambda s: len(str(s).split())) <= 3]
        for row in short.head(2).itertuples(index=False):
            lines.append(f"**Question** (`{row.qid}`):\n")
            lines.append("```")
            lines.append(row.question.strip()[:600])
            lines.append("```")
            lines.append(f"**Reference answer:** `{str(row.reference_answer).strip()[:200]}`\n")
            lines.append("**Rubric:**\n")
            for item in row.rubric:
                lines.append(f"- (w={item['weight']}) **{item['title']}** — {item['description']}")
            lines.append("")

        lines.append("### H. Pitfall criteria with the sign problem (negative weight)\n")
        pit = crit[(crit["category"] == "Pitfall")].sample(n=6, random_state=2)
        for row in pit.itertuples(index=False):
            lines.append(f"- (w={row.weight}) `{row.description.strip()}`")
        lines.append("")

    (OUT / "examples.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    results, all_flags = {}, {}
    for domain in DOMAINS:
        df, crit = build_flags(domain)
        all_flags[domain] = (df, crit)
        results[domain] = composite_metrics(df, crit)
        print(f"[{domain}] {json.dumps(results[domain])}")

    (OUT / "composite_stats.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    write_examples(all_flags)


if __name__ == "__main__":
    main()
