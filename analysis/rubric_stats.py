"""Section 1/4 of the RaR rubric forensics: descriptive stats, schema compliance,
weight informativeness, verifiability and self-containment heuristics.

Runs over the full train+val+test of both domains. Outputs JSON + CSV to analysis/out/.
"""

from __future__ import annotations

import collections
import json
import re

import numpy as np
import pandas as pd

from common import (
    DOMAINS,
    OUT,
    SENT_SPLIT_RE,
    explode_criteria,
    load_domain,
    pct,
    words,
)

# Weight bands the generation prompt asks for, per category. Only the *medicine*
# prompt spells these out; the science prompt just says "use 1-5", so for science
# this measures deviation from the medicine convention rather than a violation.
PROMPT_WEIGHT_RULE = {
    "Essential": {5},
    "Important": {3, 4},
    "Optional": {1, 2},
    "Pitfall": {-1, -2},
}
# The looser hard constraint stated in the prompt ("use 1-5" / "use -1 or -2").
PROMPT_WEIGHT_RANGE = {
    "Essential": set(range(1, 6)),
    "Important": set(range(1, 6)),
    "Optional": set(range(1, 6)),
    "Pitfall": {-1, -2},
}

# Subjective / non-verifiable vocabulary: a judge cannot check these without
# making a taste call, which is exactly what gets reward-hacked under RL.
SUBJECTIVE_TERMS = [
    "clearly", "clear ", "concise", "concisely", "thorough", "thoroughly",
    "appropriate", "appropriately", "well-organized", "well organized",
    "well-structured", "well structured", "readability", "readable", "coherent",
    "coherently", "logical flow", "easy to follow", "comprehensive",
    "comprehensively", "adequate", "adequately", "sufficient", "sufficiently",
    "properly", "effectively", "effective ", "robust", "insightful", "elegant",
    "nuanced", "professional", "engaging", "accessible", "succinct", "succinctly",
    "rigorous", "high-quality", "articulate", "in-depth", "well-reasoned",
    "seamlessly", "meaningful", "helpful", "user-friendly", "enhancing",
    "enhances", "enhance", "smooth", "polished", "elegantly", "aptly",
]

# Criteria about presentation rather than substance.
STYLE_TERMS = [
    "tone", "format", "formatting", "structure", "structured", "organiz",
    "readab", "concise", "brevity", "verbos", "jargon", "empath", "presentation",
    "layout", "bullet", "heading", "wording", "phrasing", "style", "narrative",
    "flow", "language accessible", "plain language", "prose",
]

# Phrases that assert correctness without saying what "correct" is: the judge
# cannot evaluate them in isolation, breaking the paper's self-containment goal.
NON_SELF_CONTAINED_TERMS = [
    "the correct answer", "correct answer", "the right answer", "correct value",
    "correct result", "correct solution", "correct final", "accurate answer",
    "the reference", "reference answer", "expected answer", "expected value",
    "appropriate value", "as expected", "the proper answer", "final answer is correct",
    "correct numerical", "correct conclusion", "arrives at the correct",
]

MODALS = {
    "must": re.compile(r"\bmust\b", re.I),
    "should": re.compile(r"\bshould\b", re.I),
    "may": re.compile(r"\bmay\b|\bmight\b|\bcould\b", re.I),
}


def _hist(series: pd.Series, name: str, max_bins: int = 40) -> dict:
    counts = series.value_counts().sort_index()
    if len(counts) > max_bins:
        binned = pd.cut(series, bins=max_bins).value_counts().sort_index()
        counts = pd.Series(
            binned.values, index=[f"{int(i.left)}-{int(i.right)}" for i in binned.index]
        )
    return {
        "name": name,
        "n": int(series.notna().sum()),
        "mean": round(float(series.mean()), 3),
        "std": round(float(series.std()), 3),
        "min": float(series.min()),
        "p25": float(series.quantile(0.25)),
        "median": float(series.median()),
        "p75": float(series.quantile(0.75)),
        "p95": float(series.quantile(0.95)),
        "max": float(series.max()),
        "histogram": {str(k): int(v) for k, v in counts.items()},
    }


def _contains_any(text: str, terms: list[str]) -> bool:
    lowered = text.lower()
    return any(t in lowered for t in terms)


def _entropy(counts: collections.Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    probs = np.array([c / total for c in counts.values()])
    return float(-(probs * np.log2(probs)).sum())


def weight_informativeness(crit: pd.DataFrame) -> dict:
    """How much signal do the numeric weights actually carry?

    Two views: (a) information-theoretic -- entropy of weight given the category
    label; (b) operational -- does swapping the weighted sum of Eq. 1 for a plain
    unweighted mean change the ranking of responses?
    """
    valid = crit[crit["weight"].notna()].copy()
    valid["weight"] = valid["weight"].astype(int)

    overall = collections.Counter(valid["weight"])
    per_category = {}
    conditional_entropy = 0.0
    for category, group in valid.groupby("category"):
        counts = collections.Counter(group["weight"])
        total = len(group)
        modal_weight, modal_count = counts.most_common(1)[0]
        per_category[category] = {
            "n": total,
            "share_of_all_criteria_pct": pct(total, len(valid)),
            "weight_distribution": {str(k): int(v) for k, v in sorted(counts.items())},
            "modal_weight": int(modal_weight),
            "modal_weight_share_pct": pct(modal_count, total),
            "entropy_bits": round(_entropy(counts), 3),
            "prompt_recommended_band_compliance_pct": pct(
                sum(1 for w in group["weight"] if w in PROMPT_WEIGHT_RULE.get(category, set())),
                total,
            ),
        }
        conditional_entropy += (total / len(valid)) * _entropy(counts)

    # Operational test: simulate random binary criterion outcomes and compare the
    # paper's weighted reward (Eq. 1) with an unweighted mean over the same items.
    rng = np.random.default_rng(0)
    per_question = valid.groupby("qid")["weight"].apply(list)
    sampled = per_question.sample(n=min(4000, len(per_question)), random_state=0)
    weighted_scores, uniform_scores = [], []
    # Also evaluate the paper's own category->weight remap used by RaR-Explicit.
    remap = {"Essential": 1.0, "Important": 0.7, "Optional": 0.3, "Pitfall": 0.9}
    cat_per_question = valid.groupby("qid")["category"].apply(list)
    remap_scores = []
    for qid, weight_list in sampled.items():
        w = np.asarray(weight_list, dtype=float)
        cats = cat_per_question[qid]
        wr = np.asarray([remap.get(c, 0.5) for c in cats], dtype=float)
        for _ in range(8):
            c = rng.integers(0, 2, size=len(w)).astype(float)
            if w.sum() != 0:
                weighted_scores.append(float((w * c).sum() / w.sum()))
                uniform_scores.append(float(c.mean()))
                remap_scores.append(float((wr * c).sum() / wr.sum()))
    weighted = np.asarray(weighted_scores)
    uniform = np.asarray(uniform_scores)
    remapped = np.asarray(remap_scores)

    # Negative weights break Eq. 1: sum(w) shrinks and the reward stops being
    # monotone in the number of satisfied criteria.
    sums = valid.groupby("qid")["weight"].sum()
    any_negative = valid.groupby("qid")["weight"].apply(lambda s: (s < 0).any())

    return {
        "overall_weight_distribution": {str(k): int(v) for k, v in sorted(overall.items())},
        "overall_entropy_bits": round(_entropy(overall), 3),
        "conditional_entropy_given_category_bits": round(conditional_entropy, 3),
        "mutual_information_category_weight_bits": round(
            _entropy(overall) - conditional_entropy, 3
        ),
        "per_category": per_category,
        "simulation": {
            "n_questions": int(len(sampled)),
            "n_draws": int(len(weighted)),
            "pearson_weighted_vs_uniform": round(float(np.corrcoef(weighted, uniform)[0, 1]), 4),
            "spearman_weighted_vs_uniform": round(
                float(
                    np.corrcoef(
                        pd.Series(weighted).rank(), pd.Series(uniform).rank()
                    )[0, 1]
                ),
                4,
            ),
            "mean_abs_diff_weighted_vs_uniform": round(
                float(np.abs(weighted - uniform).mean()), 4
            ),
            "pearson_paperremap_vs_uniform": round(
                float(np.corrcoef(remapped, uniform)[0, 1]), 4
            ),
        },
        "negative_weight_pathology": {
            "questions_with_any_negative_weight_pct": pct(
                int(any_negative.sum()), len(any_negative)
            ),
            "criteria_with_negative_weight_pct": pct(
                int((valid["weight"] < 0).sum()), len(valid)
            ),
            "mean_sum_of_weights": round(float(sums.mean()), 3),
            "mean_sum_of_abs_weights": round(
                float(valid.groupby("qid")["weight"].apply(lambda s: s.abs().sum()).mean()), 3
            ),
            "questions_with_nonpositive_weight_sum": int((sums <= 0).sum()),
        },
    }


def analyse_domain(domain: str) -> dict:
    df = load_domain(domain)
    crit = explode_criteria(df, domain)
    crit.to_parquet(OUT / f"criteria_{domain}.parquet", index=False)

    n_q, n_c = len(df), len(crit)
    desc_words = crit["description"].map(lambda s: len(words(s)))
    body_words = crit["body"].map(lambda s: len(words(s)))
    title_words = crit["title"].map(lambda s: len(words(s)))
    n_sentences = crit["body"].map(lambda s: len(SENT_SPLIT_RE.split(s.strip())) if s.strip() else 0)

    category_counts = crit["category"].value_counts()

    # --- prompt-constraint compliance ------------------------------------
    valid_w = crit["weight"].notna()
    pairs = list(zip(crit["category"], crit["weight"]))
    in_range = pd.Series(
        [
            (not pd.isna(w)) and int(w) in PROMPT_WEIGHT_RANGE.get(c, set(range(-2, 6)))
            for c, w in pairs
        ],
        index=crit.index,
    )
    in_band = pd.Series(
        [(not pd.isna(w)) and int(w) in PROMPT_WEIGHT_RULE.get(c, set()) for c, w in pairs],
        index=crit.index,
    )
    pitfall = crit[crit["category"] == "Pitfall"]
    pitfall_opener_ok = pitfall["body"].str.match(r"\s*(Does not mention|Recommends)", case=False)

    compliance = {
        "items_per_question_in_7_20_pct": pct(
            int(((df["rubric_count"] >= 7) & (df["rubric_count"] <= 20)).sum()), n_q
        ),
        "title_2_to_4_words_pct": pct(int(((title_words >= 2) & (title_words <= 4)).sum()), n_c),
        "title_gt_4_words_pct": pct(int((title_words > 4).sum()), n_c),
        "description_single_sentence_pct": pct(int((n_sentences <= 1).sum()), n_c),
        "has_category_prefix_pct": pct(int(crit["has_prefix"].sum()), n_c),
        "canonical_category_pct": pct(
            int(crit["category"].isin(list(PROMPT_WEIGHT_RULE)).sum()), n_c
        ),
        "malformed_or_missing_category_n": int(
            (~crit["category"].isin(list(PROMPT_WEIGHT_RULE))).sum()
        ),
        "malformed_category_values": {
            str(k): int(v)
            for k, v in collections.Counter(
                crit.loc[~crit["category"].isin(list(PROMPT_WEIGHT_RULE)), "raw_category"]
            ).most_common(12)
        },
        "weight_present_pct": pct(int(valid_w.sum()), n_c),
        "weight_in_prompt_range_pct": pct(int(in_range.sum()), n_c),
        "weight_in_prompt_recommended_band_pct": pct(int(in_band.sum()), n_c),
        "pitfall_opener_compliant_pct": pct(int(pitfall_opener_ok.sum()), len(pitfall)),
    }

    # --- verifiability / style / self-containment -------------------------
    subj = crit["body"].map(lambda s: _contains_any(s, SUBJECTIVE_TERMS))
    style = crit["body"].map(lambda s: _contains_any(s, STYLE_TERMS))
    nsc = crit["body"].map(lambda s: _contains_any(s, NON_SELF_CONTAINED_TERMS))
    modal_hits = {name: crit["body"].str.contains(rx) for name, rx in MODALS.items()}

    # Importance is encoded three times (category, weight, modal verb). Do they agree?
    essential = crit["category"] == "Essential"
    optional = crit["category"] == "Optional"
    modal_conflicts = {
        "essential_but_hedged_may_might_could_pct": pct(
            int((essential & modal_hits["may"]).sum()), int(essential.sum())
        ),
        "essential_using_must_pct": pct(
            int((essential & modal_hits["must"]).sum()), int(essential.sum())
        ),
        "optional_using_must_pct": pct(
            int((optional & modal_hits["must"]).sum()), int(optional.sum())
        ),
    }

    verifiability = {
        "subjective_term_rate_pct": pct(int(subj.sum()), n_c),
        "style_or_presentation_rate_pct": pct(int(style.sum()), n_c),
        "subjective_or_style_rate_pct": pct(int((subj | style).sum()), n_c),
        "non_self_contained_correctness_rate_pct": pct(int(nsc.sum()), n_c),
        "subjective_rate_by_category_pct": {
            str(cat): pct(int(subj[crit["category"] == cat].sum()), int((crit["category"] == cat).sum()))
            for cat in category_counts.index
        },
        "modal_verb_rate_pct": {k: pct(int(v.sum()), n_c) for k, v in modal_hits.items()},
        "modal_vs_category_conflicts": modal_conflicts,
        "top_subjective_terms": {
            term.strip(): int(crit["body"].str.lower().str.contains(re.escape(term)).sum())
            for term in sorted(
                SUBJECTIVE_TERMS,
                key=lambda t: -int(crit["body"].str.lower().str.contains(re.escape(t)).sum()),
            )[:15]
        },
    }

    # --- degeneracy of the item count ------------------------------------
    q_words = df["question"].map(lambda s: len(words(s)))
    ref_words = df["reference_answer"].map(lambda s: len(words(s)))
    count_vs_complexity = {
        "pearson_count_vs_question_words": round(
            float(np.corrcoef(df["rubric_count"], q_words)[0, 1]), 4
        ),
        "pearson_count_vs_reference_words": round(
            float(np.corrcoef(df["rubric_count"], ref_words)[0, 1]), 4
        ),
        "count_coefficient_of_variation": round(
            float(df["rubric_count"].std() / df["rubric_count"].mean()), 4
        ),
        "modal_count_share_pct": pct(
            int(df["rubric_count"].value_counts().iloc[0]), n_q
        ),
        "top2_counts_share_pct": pct(
            int(df["rubric_count"].value_counts().iloc[:2].sum()), n_q
        ),
        "share_of_prompt_allowed_range_used_pct": pct(
            int(df["rubric_count"].max() - df["rubric_count"].min() + 1), 14
        ),
    }

    # --- duplicate questions / cross-split contamination -------------------
    split_sets = df.groupby("question")["split"].apply(lambda s: frozenset(s))
    dup = {
        "duplicate_question_rows": int(df["question"].duplicated().sum()),
        "duplicate_question_rate_pct": pct(int(df["question"].duplicated().sum()), n_q),
        "questions_appearing_in_multiple_splits": int(sum(1 for v in split_sets if len(v) > 1)),
        "train_test_overlap_questions": int(
            sum(1 for v in split_sets if {"train", "test"} <= set(v))
        ),
        "test_rows_whose_question_also_in_train_pct": pct(
            int(
                df[df["split"] == "test"]["question"]
                .isin(set(df[df["split"] == "train"]["question"]))
                .sum()
            ),
            int((df["split"] == "test").sum()),
        ),
    }

    return {
        "domain": domain,
        "n_questions": n_q,
        "n_criteria": n_c,
        "rows_per_split": {k: int(v) for k, v in df["split"].value_counts().items()},
        "question_sources": {k: int(v) for k, v in df["question_source"].value_counts().items()},
        "items_per_question": _hist(df["rubric_count"], "rubric_count"),
        "item_count_degeneracy": count_vs_complexity,
        "category_distribution": {
            str(k): {"n": int(v), "pct": pct(int(v), n_c)} for k, v in category_counts.items()
        },
        "description_words": _hist(desc_words, "description_words"),
        "body_words_excl_prefix": _hist(body_words, "body_words"),
        "title_words": _hist(title_words, "title_words"),
        "question_words": _hist(q_words, "question_words"),
        "reference_answer_words": _hist(ref_words, "reference_answer_words"),
        "prompt_constraint_compliance": compliance,
        "verifiability": verifiability,
        "weight_informativeness": weight_informativeness(crit),
        "duplication": dup,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = {d: analyse_domain(d) for d in DOMAINS}
    (OUT / "basic_stats.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    summary_rows = []
    for domain, res in results.items():
        summary_rows.append(
            {
                "domain": domain,
                "questions": res["n_questions"],
                "criteria": res["n_criteria"],
                "items_mean": res["items_per_question"]["mean"],
                "items_std": res["items_per_question"]["std"],
                "items_median": res["items_per_question"]["median"],
                "items_max": res["items_per_question"]["max"],
                "desc_words_mean": res["description_words"]["mean"],
                "title_words_mean": res["title_words"]["mean"],
                "pct_essential": res["category_distribution"].get("Essential", {}).get("pct"),
                "pct_important": res["category_distribution"].get("Important", {}).get("pct"),
                "pct_optional": res["category_distribution"].get("Optional", {}).get("pct"),
                "pct_pitfall": res["category_distribution"].get("Pitfall", {}).get("pct"),
                "subjective_rate": res["verifiability"]["subjective_term_rate_pct"],
                "style_rate": res["verifiability"]["style_or_presentation_rate_pct"],
                "non_self_contained_rate": res["verifiability"][
                    "non_self_contained_correctness_rate_pct"
                ],
            }
        )
    pd.DataFrame(summary_rows).to_csv(OUT / "basic_summary.csv", index=False)
    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
