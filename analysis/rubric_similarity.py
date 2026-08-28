"""Section 2/4: how boilerplate are the rubrics, and how redundant are the items
inside a single rubric?

Three notions of "not query-specific":
  * title reuse           -- the same 2-4 word title recycled across questions
  * template reuse        -- identical description once rare tokens are masked
  * fuzzy cross-question  -- TF-IDF cosine near-duplicates between questions

Plus within-question redundancy (pairwise cosine / Jaccard between the items of
one rubric), which is what makes the weighted sum of Eq. 1 double-count.
"""

from __future__ import annotations

import argparse
import collections
import json
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer

from common import DOMAINS, OUT, content_tokens, pct

NUM_RE = re.compile(r"\d+(?:[.,]\d+)?")
NON_WORD_RE = re.compile(r"[^a-z0-9\s]")
WS_RE = re.compile(r"\s+")

# Similarity thresholds used throughout.
NEAR_DUP = 0.80
STRONG_DUP = 0.90
REDUNDANT_PAIR = 0.50


def normalise_title(title: str) -> str:
    return WS_RE.sub(" ", NON_WORD_RE.sub(" ", (title or "").lower())).strip()


def template_of(body: str, rare_tokens: set[str]) -> str:
    """Mask numbers and corpus-rare tokens; what remains is the reusable skeleton."""
    lowered = NUM_RE.sub("<NUM>", (body or "").lower())
    lowered = NON_WORD_RE.sub(" ", lowered.replace("<num>", " <NUM> "))
    out = []
    for token in lowered.split():
        if token == "NUM":
            out.append("<NUM>")
        elif token in rare_tokens:
            out.append("<X>")
        else:
            out.append(token)
    return " ".join(out)


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def title_boilerplate(crit: pd.DataFrame) -> dict:
    titles = crit["title"].map(normalise_title)
    counts = collections.Counter(titles)
    n = len(crit)
    per_title_share = {
        f"criteria_whose_title_reused_ge_{k}x_pct": pct(
            sum(c for c in counts.values() if c >= k), n
        )
        for k in (2, 10, 50, 100, 500, 1000)
    }
    top = counts.most_common(50)
    return {
        "n_criteria": n,
        "n_unique_titles": len(counts),
        "unique_title_ratio_pct": pct(len(counts), n),
        "titles_used_once_pct": pct(sum(1 for c in counts.values() if c == 1), len(counts)),
        **per_title_share,
        "top50_titles": [
            {"title": t, "count": c, "pct_of_criteria": pct(c, n)} for t, c in top
        ],
        "top50_cumulative_pct_of_criteria": pct(sum(c for _, c in top), n),
    }


def skeleton_of(body: str, common_tokens: set[str]) -> str:
    """Keep only corpus-frequent tokens: what remains is the reusable phrasing mould."""
    lowered = NON_WORD_RE.sub(" ", NUM_RE.sub(" ", (body or "").lower()))
    return " ".join(t for t in lowered.split() if t in common_tokens)


def template_boilerplate(crit: pd.DataFrame, n_common: int = 200) -> dict:
    """Exact / masked-template / skeleton duplication, computed on the FULL corpus."""
    n = len(crit)
    bodies = crit["body"].str.lower().str.strip()

    exact_counts = collections.Counter(bodies)
    exact_dup_criteria = sum(c for c in exact_counts.values() if c > 1)
    # Only cross-question repeats prove the text is not query-specific.
    qids_per_body: dict[str, set] = collections.defaultdict(set)
    for body, qid in zip(bodies, crit["qid"]):
        qids_per_body[body].add(qid)
    cross_q_bodies = {b for b, qs in qids_per_body.items() if len(qs) > 1}
    cross_q_criteria = int(bodies.isin(cross_q_bodies).sum())

    # Document frequency over criteria; "rare" == appears in <0.1% of criteria.
    doc_freq: collections.Counter = collections.Counter()
    for body in bodies:
        doc_freq.update(set(content_tokens(body)))
    rare_cut = max(2, int(0.001 * n))
    rare_tokens = {t for t, c in doc_freq.items() if c < rare_cut}

    templates = [template_of(b, rare_tokens) for b in bodies]
    template_counts = collections.Counter(templates)
    template_dup = sum(c for c in template_counts.values() if c > 1)

    # Skeleton: drop everything outside the N most frequent tokens of the corpus
    # (function words included), leaving only the phrasing mould.
    all_freq: collections.Counter = collections.Counter()
    for body in bodies:
        all_freq.update(NON_WORD_RE.sub(" ", NUM_RE.sub(" ", body)).split())
    common_tokens = {t for t, _ in all_freq.most_common(n_common)}
    skeletons = [skeleton_of(b, common_tokens) for b in bodies]
    skeleton_counts = collections.Counter(skeletons)

    # Opening 6 words: the strongest "one-mould" signal.
    openers = [" ".join(b.split()[:6]) for b in bodies]
    opener_counts = collections.Counter(openers)

    return {
        "exact_duplicate_description_pct": pct(exact_dup_criteria, n),
        "exact_duplicate_across_different_questions_pct": pct(cross_q_criteria, n),
        "n_description_strings_reused_across_questions": len(cross_q_bodies),
        "n_distinct_descriptions": len(exact_counts),
        "masked_template_distinct": len(template_counts),
        "masked_template_compression_ratio": round(len(template_counts) / n, 4),
        "criteria_sharing_a_masked_template_pct": pct(template_dup, n),
        "criteria_in_template_reused_ge_10x_pct": pct(
            sum(c for c in template_counts.values() if c >= 10), n
        ),
        "rare_token_cutoff_docfreq": rare_cut,
        "skeleton_top_n_tokens": n_common,
        "skeleton_distinct": len(skeleton_counts),
        "skeleton_compression_ratio": round(len(skeleton_counts) / n, 4),
        "criteria_sharing_a_skeleton_pct": pct(
            sum(c for c in skeleton_counts.values() if c > 1), n
        ),
        "criteria_in_skeleton_reused_ge_10x_pct": pct(
            sum(c for c in skeleton_counts.values() if c >= 10), n
        ),
        "top15_skeletons": [
            {"skeleton": t, "count": c, "pct": pct(c, n)}
            for t, c in skeleton_counts.most_common(15)
        ],
        "top20_masked_templates": [
            {"template": t, "count": c, "pct": pct(c, n)}
            for t, c in template_counts.most_common(20)
        ],
        "top20_openers_6w": [
            {"opener": t, "count": c, "pct": pct(c, n)}
            for t, c in opener_counts.most_common(20)
        ],
        "top1_opener_share_pct": pct(opener_counts.most_common(1)[0][1], n),
        "top10_openers_share_pct": pct(
            sum(c for _, c in opener_counts.most_common(10)), n
        ),
    }


def cross_question_near_duplicates(crit: pd.DataFrame, sample: int, seed: int = 0) -> dict:
    """Fuzzy near-duplicate rate between criteria belonging to *different* questions.

    Searching only inside a random sample can only miss matches, never invent
    them, so the reported rates are lower bounds on the full-corpus rate.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(crit), size=min(sample, len(crit)), replace=False)
    sub = crit.iloc[np.sort(idx)].reset_index(drop=True)

    vec = TfidfVectorizer(
        lowercase=True, sublinear_tf=True, min_df=3, max_features=60000, stop_words="english"
    )
    X = vec.fit_transform(sub["body"]).astype(np.float32)
    X = sp.csr_matrix(X)
    qids = sub["qid"].to_numpy()

    best_sim = np.zeros(X.shape[0], dtype=np.float32)
    best_j = np.full(X.shape[0], -1, dtype=np.int64)
    chunk = 512
    for start in range(0, X.shape[0], chunk):
        stop = min(start + chunk, X.shape[0])
        block = (X[start:stop] @ X.T).toarray()
        # Mask out self and same-question neighbours.
        same_q = qids[start:stop][:, None] == qids[None, :]
        block[same_q] = -1.0
        arg = block.argmax(axis=1)
        best_sim[start:stop] = block[np.arange(stop - start), arg]
        best_j[start:stop] = arg

    order = np.argsort(-best_sim)[:40]
    examples = [
        {
            "sim": round(float(best_sim[i]), 3),
            "a_qid": str(sub["qid"][i]),
            "a_title": sub["title"][i],
            "a": sub["description"][i],
            "b_qid": str(sub["qid"][best_j[i]]),
            "b_title": sub["title"][best_j[i]],
            "b": sub["description"][best_j[i]],
        }
        for i in order
    ]
    return {
        "sample_size": int(X.shape[0]),
        "vocab": int(X.shape[1]),
        "mean_max_cross_question_cosine": round(float(best_sim.mean()), 4),
        "median_max_cross_question_cosine": round(float(np.median(best_sim)), 4),
        "pct_with_cross_question_cosine_ge_0.80": pct(int((best_sim >= NEAR_DUP).sum()), len(best_sim)),
        "pct_with_cross_question_cosine_ge_0.90": pct(int((best_sim >= STRONG_DUP).sum()), len(best_sim)),
        "pct_with_cross_question_cosine_ge_0.99": pct(int((best_sim >= 0.99).sum()), len(best_sim)),
        "pct_with_cross_question_cosine_ge_0.60": pct(int((best_sim >= 0.60).sum()), len(best_sim)),
        "examples_top40": examples,
    }


# Openers that turn a positive criterion into its negative twin.
PITFALL_OPENER_RE = re.compile(
    r"^\s*(?:the\s+(?:response|answer)\s+)?(?:must\s+not|should\s+not|does\s+not|do\s+not|fails?\s+to"
    r"|neglects?\s+to|omits?|avoids?|avoiding|recommends?)\s*"
    r"(?:mention(?:ing)?|state|recommend|include|identify|note)?\s*",
    re.IGNORECASE,
)
POSITIVE_OPENER_RE = re.compile(
    r"^\s*(?:the\s+)?(?:response|answer)?\s*(?:must|should|may|will)?\s*"
    r"(?:clearly|explicitly|correctly|accurately)?\s*"
    r"(?:state[sd]?|identif(?:y|ies)|mention[sd]?|explain[sd]?|includ(?:e|es)|note[sd]?|provide[sd]?)\s*"
    r"(?:that\s+)?",
    re.IGNORECASE,
)


def polarity_stripped(body: str) -> str:
    """Remove the polarity wrapper so a Pitfall and its positive twin look alike."""
    out = PITFALL_OPENER_RE.sub("", body or "")
    return POSITIVE_OPENER_RE.sub("", out).strip()


def discriminative_marks(text: str, rare: set[str] | None = None) -> frozenset[str]:
    """What tells two lexically parallel criteria apart: numbers, symbols, entities.

    Two criteria such as "... every 3 months for the first 2 years" and
    "... every 6 months for the next 2 years", or "identifies Pimozide ..." and
    "identifies Penfluridol ...", are near-identical to TF-IDF but check different
    facts, so they must not be counted as redundant.
    """
    lowered = (text or "").lower()
    marks = set(NUM_RE.findall(lowered))
    marks.update(re.findall(r"[^\w\s]", lowered))
    marks.update(re.findall(r"\(([a-e])\)", lowered))
    if rare:
        marks.update(t for t in content_tokens(lowered) if t in rare)
    return frozenset(marks)


def within_question_redundancy(crit: pd.DataFrame, sample_questions: int, seed: int = 0) -> dict:
    """Pairwise similarity between the criteria of the same rubric.

    Reports both the raw lexical rate and an anchor-aware rate that only counts a
    pair as redundant when the two items also carry the same numbers/symbols.
    """
    rng = np.random.default_rng(seed)
    all_qids = crit["qid"].unique()
    chosen = rng.choice(all_qids, size=min(sample_questions, len(all_qids)), replace=False)
    sub = crit[crit["qid"].isin(set(chosen))].reset_index(drop=True)

    # Token pattern keeps single characters, digits and math symbols so that
    # "K+K-" and "pi+pi-" do not collapse onto the same vector.
    vec = TfidfVectorizer(
        lowercase=True,
        sublinear_tf=True,
        min_df=2,
        max_features=120000,
        token_pattern=r"(?u)\w+|[^\w\s]",
    )
    X = sp.csr_matrix(vec.fit_transform(sub["body"]).astype(np.float32))
    token_sets = [set(content_tokens(b)) for b in sub["body"]]

    # Entity-level marks need a notion of "rare in this corpus".
    body_df: collections.Counter = collections.Counter()
    for body in sub["body"]:
        body_df.update(set(content_tokens(body)))
    rare = {t for t, c in body_df.items() if c < max(2, int(0.001 * len(sub)))}
    marks = [discriminative_marks(b, rare) for b in sub["body"]]

    # Separate space for polarity-stripped text, used to spot Pitfall items that
    # merely negate a positive criterion of the same rubric.
    vec_pol = TfidfVectorizer(lowercase=True, sublinear_tf=True, min_df=2, max_features=120000)
    Xp = sp.csr_matrix(
        vec_pol.fit_transform(sub["body"].map(polarity_stripped)).astype(np.float32)
    )
    is_pitfall = (sub["category"] == "Pitfall").to_numpy()

    rows, worst, worst_true, mirrors = [], [], [], []
    for qid, group in sub.groupby("qid", sort=False):
        pos = group.index.to_numpy()
        if len(pos) < 2:
            continue
        sim = (X[pos] @ X[pos].T).toarray()
        np.fill_diagonal(sim, 0.0)
        iu = np.triu_indices(len(pos), k=1)
        cos_pairs = sim[iu]
        jac_pairs = np.array(
            [jaccard(token_sets[pos[i]], token_sets[pos[j]]) for i, j in zip(*iu)]
        )
        same_marks = np.array(
            [marks[pos[i]] == marks[pos[j]] for i, j in zip(*iu)], dtype=bool
        )
        true_pairs = cos_pairs * same_marks

        # Pitfall x non-Pitfall similarity after removing the polarity wrapper.
        p_idx = pos[is_pitfall[pos]]
        n_idx = pos[~is_pitfall[pos]]
        mirror_max, mirror_n = 0.0, 0
        if len(p_idx) and len(n_idx):
            block = (Xp[p_idx] @ Xp[n_idx].T).toarray()
            mirror_max = float(block.max())
            mirror_n = int((block >= 0.8).sum())
            if mirror_max >= 0.8:
                a, b = np.unravel_index(int(block.argmax()), block.shape)
                mirrors.append(
                    (mirror_max, qid, sub["description"][p_idx[a]], sub["description"][n_idx[b]])
                )

        best = int(cos_pairs.argmax())
        rows.append(
            {
                "qid": qid,
                "n_items": len(pos),
                "pitfall_mirror_max": mirror_max,
                "n_pitfall_mirrors": mirror_n,
                "max_cos": float(cos_pairs.max()),
                "mean_cos": float(cos_pairs.mean()),
                "max_jaccard": float(jac_pairs.max()),
                "mean_jaccard": float(jac_pairs.mean()),
                "max_cos_same_marks": float(true_pairs.max()),
                "n_pairs": int(len(cos_pairs)),
                "n_pairs_ge_0.5": int((cos_pairs >= REDUNDANT_PAIR).sum()),
                "n_pairs_ge_0.7": int((cos_pairs >= 0.7).sum()),
                "n_true_pairs_ge_0.5": int((true_pairs >= REDUNDANT_PAIR).sum()),
                "n_true_pairs_ge_0.7": int((true_pairs >= 0.7).sum()),
            }
        )
        i, j = iu[0][best], iu[1][best]
        worst.append((float(cos_pairs.max()), qid, sub["description"][pos[i]], sub["description"][pos[j]]))
        if true_pairs.max() > 0:
            bt = int(true_pairs.argmax())
            i2, j2 = iu[0][bt], iu[1][bt]
            worst_true.append(
                (float(true_pairs.max()), qid, sub["description"][pos[i2]], sub["description"][pos[j2]])
            )

    stats = pd.DataFrame(rows)
    stats.to_csv(OUT / f"within_question_redundancy_{crit['domain'].iloc[0]}.csv", index=False)
    worst.sort(key=lambda t: -t[0])
    worst_true.sort(key=lambda t: -t[0])
    mirrors.sort(key=lambda t: -t[0])
    total_pairs = int(stats["n_pairs"].sum())

    return {
        "sample_questions": int(len(stats)),
        "mean_max_pairwise_cosine": round(float(stats["max_cos"].mean()), 4),
        "median_max_pairwise_cosine": round(float(stats["max_cos"].median()), 4),
        "mean_mean_pairwise_cosine": round(float(stats["mean_cos"].mean()), 4),
        "mean_max_pairwise_jaccard": round(float(stats["max_jaccard"].mean()), 4),
        "mean_mean_pairwise_jaccard": round(float(stats["mean_jaccard"].mean()), 4),
        "pct_questions_with_a_pair_ge_0.5": pct(int((stats["max_cos"] >= 0.5).sum()), len(stats)),
        "pct_questions_with_a_pair_ge_0.7": pct(int((stats["max_cos"] >= 0.7).sum()), len(stats)),
        "pct_of_all_pairs_ge_0.5": pct(int(stats["n_pairs_ge_0.5"].sum()), total_pairs),
        "pct_of_all_pairs_ge_0.7": pct(int(stats["n_pairs_ge_0.7"].sum()), total_pairs),
        "mean_redundant_pairs_per_question": round(float(stats["n_pairs_ge_0.5"].mean()), 3),
        "anchor_aware": {
            "note": "pair counted only if the two items carry identical numbers/symbols",
            "pct_questions_with_a_true_pair_ge_0.5": pct(
                int((stats["max_cos_same_marks"] >= 0.5).sum()), len(stats)
            ),
            "pct_questions_with_a_true_pair_ge_0.7": pct(
                int((stats["max_cos_same_marks"] >= 0.7).sum()), len(stats)
            ),
            "pct_of_all_pairs_true_ge_0.5": pct(
                int(stats["n_true_pairs_ge_0.5"].sum()), total_pairs
            ),
            "mean_true_redundant_pairs_per_question": round(
                float(stats["n_true_pairs_ge_0.5"].mean()), 3
            ),
        },
        "pitfall_mirrors_a_positive_criterion": {
            "note": "Pitfall item whose polarity-stripped text matches a non-Pitfall item "
            "of the same rubric at cosine >= 0.8: the same fact is scored twice, "
            "once positively and once with a negative weight",
            **{
                f"pct_questions_with_a_mirror_ge_{t}": pct(
                    int((stats["pitfall_mirror_max"] >= t).sum()), len(stats)
                )
                for t in (0.6, 0.7, 0.8, 0.9, 0.99)
            },
            "mean_mirrors_per_question": round(float(stats["n_pitfall_mirrors"].mean()), 3),
            "mean_max_mirror_similarity": round(float(stats["pitfall_mirror_max"].mean()), 4),
            "examples_top10": [
                {"sim": round(s, 3), "qid": str(q), "pitfall": a, "positive": b}
                for s, q, a, b in mirrors[:10]
            ],
        },
        "examples_top25": [
            {"sim": round(s, 3), "qid": str(q), "a": a, "b": b} for s, q, a, b in worst[:25]
        ],
        "examples_anchor_aware_top25": [
            {"sim": round(s, 3), "qid": str(q), "a": a, "b": b} for s, q, a, b in worst_true[:25]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-criteria", type=int, default=40000)
    parser.add_argument("--sample-questions", type=int, default=6000)
    args = parser.parse_args()

    results = {}
    for domain in DOMAINS:
        crit = pd.read_parquet(OUT / f"criteria_{domain}.parquet")
        print(f"[{domain}] {len(crit)} criteria")
        results[domain] = {
            "title_boilerplate": title_boilerplate(crit),
            "template_boilerplate": template_boilerplate(crit),
            "cross_question_near_duplicates": cross_question_near_duplicates(
                crit, args.sample_criteria
            ),
            "within_question_redundancy": within_question_redundancy(
                crit, args.sample_questions
            ),
        }
        r = results[domain]
        print(
            f"  unique titles {r['title_boilerplate']['unique_title_ratio_pct']}% | "
            f"template dup {r['template_boilerplate']['criteria_sharing_a_masked_template_pct']}% | "
            f"cross-q cos>=.8 {r['cross_question_near_duplicates']['pct_with_cross_question_cosine_ge_0.80']}% | "
            f"within-q pairs>=.5 {r['within_question_redundancy']['pct_of_all_pairs_ge_0.5']}%"
        )

    (OUT / "similarity_stats.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Flat CSV of the top titles for the write-up.
    rows = []
    for domain, res in results.items():
        for rank, item in enumerate(res["title_boilerplate"]["top50_titles"], start=1):
            rows.append({"domain": domain, "rank": rank, **item})
    pd.DataFrame(rows).to_csv(OUT / "top_titles.csv", index=False)


if __name__ == "__main__":
    main()
