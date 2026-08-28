"""Section 3/4: is a criterion actually *about* its own question?

Three families of measurement:
  1. anchor grounding      -- does the criterion cite any rare token / number /
                              symbol that occurs in its own question or reference?
  2. attribution retrieval -- given a criterion, can TF-IDF pick its own question
                              out of a pool of distractors? (query-specificity)
  3. reference coupling    -- n-gram overlap with the reference answer, and
                              outright leakage of the reference's final numbers.
Plus sub-question coverage for multi-part prompts.
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
from sklearn.preprocessing import normalize

from common import DOMAINS, OUT, content_tokens, load_domain, ngrams, pct

NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
SYMBOL_RE = re.compile(r"[^\x00-\x7F]|\\[a-zA-Z]+|[=<>+^_/]")
# Enumerated sub-parts: "a)", "(b)", "1.", "(iii)" at a plausible boundary.
LETTER_PART_RE = re.compile(r"(?:^|[\s(])\(?([a-h])[\)\.](?=\s)", re.MULTILINE)
NUMERIC_PART_RE = re.compile(r"(?:^|\n)\s*\(?([1-9])[\)\.](?=\s)")
TASK_CUE_RE = re.compile(
    r"\b(find|calculate|determine|compute|show|derive|prove|explain|estimate|evaluate"
    r"|obtain|sketch|describe|discuss|what|how|why|write|give|list|solve|verify)\b",
    re.IGNORECASE,
)


def instance_anchors(question: str, reference: str, rare: set[str]) -> set[str]:
    """Tokens that pin a criterion to *this* prompt: rare words, numbers, symbols."""
    anchors: set[str] = set()
    for text in (question, reference):
        lowered = (text or "").lower()
        anchors.update(t for t in content_tokens(lowered) if t in rare)
        anchors.update(n for n in NUMBER_RE.findall(lowered) if len(n) >= 2 or float(n) > 2)
    return anchors


def anchor_grounding(df: pd.DataFrame, crit: pd.DataFrame, rare: set[str], seed: int = 0) -> dict:
    """Grounded = the criterion body mentions >=1 anchor from its own instance.

    The permutation control re-runs the same test against a *random other*
    instance's anchors: if the two rates are close, "grounding" is an artefact of
    shared vocabulary rather than genuine query-specificity.
    """
    rng = np.random.default_rng(seed)
    anchors_by_qid = {
        row.qid: instance_anchors(row.question, row.reference_answer, rare)
        for row in df.itertuples(index=False)
    }
    qids = list(anchors_by_qid)
    shuffled = {q: anchors_by_qid[qids[int(rng.integers(len(qids)))]] for q in qids}

    n_anchor_hits, hit, shuffled_hit = [], [], []
    for qid, body in zip(crit["qid"], crit["body"]):
        tokens = set(content_tokens(body)) | set(NUMBER_RE.findall(body.lower()))
        own = anchors_by_qid.get(qid, set())
        k = len(tokens & own)
        n_anchor_hits.append(k)
        hit.append(k > 0)
        shuffled_hit.append(len(tokens & shuffled.get(qid, set())) > 0)

    hits = np.asarray(n_anchor_hits)
    hit = np.asarray(hit)
    shuffled_hit = np.asarray(shuffled_hit)
    crit = crit.assign(n_anchors=hits, grounded=hit)

    by_category = {
        str(cat): {
            "n": int(len(g)),
            "generic_rate_pct": pct(int((~g["grounded"]).sum()), len(g)),
            "mean_anchors": round(float(g["n_anchors"].mean()), 3),
        }
        for cat, g in crit.groupby("category")
    }

    # Does grounding collapse when the reference answer is short?
    ref_words = df.set_index("qid")["reference_answer"].map(lambda s: len(str(s).split()))
    bucket = pd.cut(
        crit["qid"].map(ref_words),
        bins=[-1, 0, 5, 20, 60, 150, 10**6],
        labels=["0", "1-5", "6-20", "21-60", "61-150", ">150"],
    )
    by_ref_len = {
        str(b): {
            "n_criteria": int(len(g)),
            "generic_rate_pct": pct(int((~g["grounded"]).sum()), len(g)),
            "mean_anchors": round(float(g["n_anchors"].mean()), 3),
        }
        for b, g in crit.groupby(bucket, observed=True)
    }

    per_question = crit.groupby("qid")["grounded"].mean()
    return {
        "generic_criterion_rate_pct": pct(int((~hit).sum()), len(hit)),
        "generic_rate_permutation_control_pct": pct(int((~shuffled_hit).sum()), len(shuffled_hit)),
        "specificity_lift_pct_points": round(
            pct(int(hit.sum()), len(hit)) - pct(int(shuffled_hit.sum()), len(shuffled_hit)), 2
        ),
        "strictly_generic_rate_ge2_anchors_pct": pct(int((hits < 2).sum()), len(hits)),
        "mean_anchors_per_criterion": round(float(hits.mean()), 3),
        "median_anchors_per_criterion": float(np.median(hits)),
        "questions_with_zero_grounded_criteria_pct": pct(
            int((per_question == 0).sum()), len(per_question)
        ),
        "questions_with_majority_generic_criteria_pct": pct(
            int((per_question < 0.5).sum()), len(per_question)
        ),
        "by_category": by_category,
        "by_reference_answer_length": by_ref_len,
    }


def attribution_retrieval(
    df: pd.DataFrame, crit: pd.DataFrame, sample: int, n_distractors: int, seed: int = 0
) -> dict:
    """Can TF-IDF recover a criterion's own question from a pool of distractors?"""
    rng = np.random.default_rng(seed)
    q_docs = (df["question"].fillna("") + " \n " + df["reference_answer"].fillna("")).tolist()
    qid_to_row = {q: i for i, q in enumerate(df["qid"])}

    vec = TfidfVectorizer(sublinear_tf=True, min_df=3, max_features=120000, stop_words="english")
    Q = normalize(vec.fit_transform(q_docs).astype(np.float32))

    idx = rng.choice(len(crit), size=min(sample, len(crit)), replace=False)
    sub = crit.iloc[np.sort(idx)].reset_index(drop=True)
    C = normalize(vec.transform(sub["body"]).astype(np.float32))

    own_rows = np.asarray([qid_to_row[q] for q in sub["qid"]])
    pool = rng.choice(len(df), size=min(n_distractors, len(df)), replace=False)

    own_sim = np.asarray(C.multiply(Q[own_rows]).sum(axis=1)).ravel()
    distract = (C @ Q[pool].T).toarray()
    # A distractor that happens to be the criterion's own question is not a distractor.
    distract[np.isin(pool, own_rows)[None, :].repeat(len(sub), axis=0) & (pool[None, :] == own_rows[:, None])] = -1.0

    beaten = (distract > own_sim[:, None]).sum(axis=1)
    return {
        "sample_criteria": int(len(sub)),
        "n_distractor_questions": int(len(pool)),
        "top1_attribution_accuracy_pct": pct(int((beaten == 0).sum()), len(beaten)),
        "top5_attribution_accuracy_pct": pct(int((beaten < 5).sum()), len(beaten)),
        "random_baseline_top1_pct": round(100.0 / (len(pool) + 1), 3),
        "mean_own_question_cosine": round(float(own_sim.mean()), 4),
        "mean_best_distractor_cosine": round(float(distract.max(axis=1).mean()), 4),
        "criteria_with_zero_own_question_overlap_pct": pct(
            int((own_sim <= 1e-6).sum()), len(own_sim)
        ),
        "criteria_beaten_by_ge_10_distractors_pct": pct(int((beaten >= 10).sum()), len(beaten)),
    }


def reference_coupling(df: pd.DataFrame, crit: pd.DataFrame) -> dict:
    """N-gram overlap with the reference answer, and leakage of its final numbers."""
    ref_by_qid = dict(zip(df["qid"], df["reference_answer"].fillna("")))
    q_by_qid = dict(zip(df["qid"], df["question"].fillna("")))

    ref_grams: dict[str, tuple[set, set]] = {}
    for qid, ref in ref_by_qid.items():
        toks = content_tokens(ref)
        ref_grams[qid] = (ngrams(toks, 3), ngrams(toks, 4))
    q_grams = {qid: ngrams(content_tokens(q), 4) for qid, q in q_by_qid.items()}

    # Numbers that appear in the reference but NOT in the question: these are the
    # answer values. A criterion repeating one is handing the judge the answer.
    leak_numbers: dict[str, set[str]] = {}
    for qid, ref in ref_by_qid.items():
        q_nums = set(NUMBER_RE.findall(q_by_qid.get(qid, "")))
        r_nums = {n for n in NUMBER_RE.findall(ref) if len(n) >= 2 or float(n) > 2}
        leak_numbers[qid] = r_nums - q_nums

    ov3, ov4, ovq, leak = [], [], [], []
    for qid, body in zip(crit["qid"], crit["body"]):
        toks = content_tokens(body)
        g3, g4 = ngrams(toks, 3), ngrams(toks, 4)
        r3, r4 = ref_grams.get(qid, (set(), set()))
        ov3.append(len(g3 & r3) / len(g3) if g3 else 0.0)
        ov4.append(len(g4 & r4) / len(g4) if g4 else 0.0)
        ovq.append(len(g4 & q_grams.get(qid, set())) / len(g4) if g4 else 0.0)
        nums = set(NUMBER_RE.findall(body))
        leak.append(bool(nums & leak_numbers.get(qid, set())))

    ov3 = np.asarray(ov3)
    ov4 = np.asarray(ov4)
    ovq = np.asarray(ovq)
    leak = np.asarray(leak)
    per_q_leak = pd.Series(leak).groupby(crit["qid"].to_numpy()).any()

    return {
        "mean_trigram_overlap_with_reference": round(float(ov3.mean()), 4),
        "mean_4gram_overlap_with_reference": round(float(ov4.mean()), 4),
        "mean_4gram_overlap_with_question": round(float(ovq.mean()), 4),
        "criteria_with_ge50pct_trigram_overlap_pct": pct(int((ov3 >= 0.5).sum()), len(ov3)),
        "criteria_with_ge30pct_trigram_overlap_pct": pct(int((ov3 >= 0.3).sum()), len(ov3)),
        "criteria_with_zero_trigram_overlap_pct": pct(int((ov3 == 0).sum()), len(ov3)),
        "criteria_leaking_reference_only_number_pct": pct(int(leak.sum()), len(leak)),
        "questions_with_any_leaked_number_pct": pct(int(per_q_leak.sum()), len(per_q_leak)),
    }


def _label_run(question: str, pattern: re.Pattern, alphabet: str) -> list[tuple[str, int]]:
    """Longest prefix of an in-order label run (a, b, c, ... / 1, 2, 3, ...)."""
    seq: list[tuple[str, int]] = []
    for match in pattern.finditer(question or ""):
        if match.group(1) == alphabet[len(seq)]:
            seq.append((match.group(1), match.start()))
        if len(seq) == len(alphabet):
            break
    return seq


def detect_subparts(question: str) -> tuple[list[tuple[str, str]], str]:
    """Split an enumerated prompt into labelled spans and say what the labels are.

    Returns (spans, kind) where kind is "subquestions", "mcq" or "none".
    Multiple-choice option lists look identical to sub-question lists at the regex
    level, so a span only counts as a sub-question when it actually asks for
    something (a question mark or an instruction verb).
    """
    question = question or ""
    seq = _label_run(question, LETTER_PART_RE, "abcdefgh")
    if len(seq) < 2:
        seq = _label_run(question, NUMERIC_PART_RE, "123456789")
    if len(seq) < 2:
        return [], "none"

    spans = []
    for i, (label, pos) in enumerate(seq):
        end = seq[i + 1][1] if i + 1 < len(seq) else len(question)
        spans.append((label, question[pos:end]))

    task_spans = [s for s in spans if "?" in s[1] or TASK_CUE_RE.search(s[1])]
    if len(task_spans) >= 2:
        return task_spans, "subquestions"
    return spans, "mcq"


def subquestion_coverage(df: pd.DataFrame, crit: pd.DataFrame, min_cos: float = 0.08) -> dict:
    """For multi-part prompts, is every sub-part represented in the rubric?"""
    parts_by_qid, n_parts, kinds = {}, {}, {}
    for row in df.itertuples(index=False):
        spans, kind = detect_subparts(row.question)
        kinds[row.qid] = kind
        n_parts[row.qid] = len(spans) if kind == "subquestions" else 0
        if kind == "subquestions":
            parts_by_qid[row.qid] = spans

    counts = pd.Series(n_parts)
    kind_counts = pd.Series(kinds).value_counts()
    rubric_counts = df.set_index("qid")["rubric_count"]
    multi = counts[counts >= 2].index

    by_nparts = {}
    for k in range(2, 7):
        sel = counts[counts == k].index
        if len(sel) >= 20:
            by_nparts[str(k)] = {
                "n_questions": int(len(sel)),
                "mean_rubric_items": round(float(rubric_counts[sel].mean()), 2),
                "mean_items_per_subpart": round(float((rubric_counts[sel] / k).mean()), 2),
            }
    single = counts[counts == 0].index
    baseline_items = float(rubric_counts[single].mean()) if len(single) else float("nan")

    # Semantic coverage: best TF-IDF cosine between each sub-part and any criterion
    # of the same question.
    sub_texts, sub_keys = [], []
    for qid, spans in parts_by_qid.items():
        for label, text in spans:
            sub_texts.append(text)
            sub_keys.append((qid, label))
    orphan_stats = {}
    label_mention = {}
    if sub_texts:
        crit_multi = crit[crit["qid"].isin(set(parts_by_qid))].reset_index(drop=True)
        vec = TfidfVectorizer(sublinear_tf=True, min_df=2, stop_words="english")
        vec.fit(sub_texts + crit_multi["body"].tolist())
        S = normalize(vec.transform(sub_texts).astype(np.float32))
        C = normalize(vec.transform(crit_multi["body"]).astype(np.float32))
        rows_by_qid = collections.defaultdict(list)
        for i, qid in enumerate(crit_multi["qid"]):
            rows_by_qid[qid].append(i)

        best = np.zeros(len(sub_texts), dtype=np.float32)
        for i, (qid, _label) in enumerate(sub_keys):
            rows = rows_by_qid.get(qid, [])
            if rows:
                best[i] = float((C[rows] @ S[i].T).toarray().max())
        orphan = best < min_cos
        orphan_by_q = pd.Series(orphan).groupby([k[0] for k in sub_keys]).mean()
        orphan_stats = {
            "n_subparts_scored": int(len(best)),
            "min_cosine_threshold": min_cos,
            "mean_best_subpart_to_criterion_cosine": round(float(best.mean()), 4),
            "orphan_subpart_rate_pct": pct(int(orphan.sum()), len(orphan)),
            "multi_part_questions_with_any_orphan_subpart_pct": pct(
                int((orphan_by_q > 0).sum()), len(orphan_by_q)
            ),
        }

        # Explicit addressability: does any criterion name the part label at all?
        named = 0
        for qid, spans in parts_by_qid.items():
            bodies = " ".join(crit_multi.loc[rows_by_qid.get(qid, []), "body"]).lower()
            if any(
                re.search(rf"\bpart\s*\(?{lab}\)?|\(\s*{lab}\s*\)", bodies) for lab, _ in spans
            ):
                named += 1
        label_mention = {
            "multi_part_questions_naming_at_least_one_part_label_pct": pct(
                named, len(parts_by_qid)
            )
        }

    return {
        "questions_with_detected_subparts_pct": pct(int((counts >= 2).sum()), len(counts)),
        "questions_that_are_multiple_choice_pct": pct(
            int(kind_counts.get("mcq", 0)), len(counts)
        ),
        "mean_rubric_items_multiple_choice": round(
            float(rubric_counts[[q for q, k in kinds.items() if k == "mcq"]].mean()), 2
        )
        if kind_counts.get("mcq", 0)
        else None,
        "n_multi_part_questions": int(len(multi)),
        "mean_rubric_items_single_part": round(baseline_items, 2),
        "mean_rubric_items_multi_part": round(float(rubric_counts[multi].mean()), 2)
        if len(multi)
        else None,
        # Slope of rubric size on task count, treating a single-part prompt as 1 task.
        "extra_items_per_additional_subpart": round(
            float(
                np.polyfit(
                    counts.clip(lower=1).values,
                    rubric_counts[counts.index].values.astype(float),
                    1,
                )[0]
            ),
            3,
        )
        if len(multi) >= 30
        else None,
        "rubric_items_by_n_subparts": by_nparts,
        **orphan_stats,
        **label_mention,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-criteria", type=int, default=40000)
    parser.add_argument("--distractors", type=int, default=500)
    args = parser.parse_args()

    results = {}
    for domain in DOMAINS:
        df = load_domain(domain)
        crit = pd.read_parquet(OUT / f"criteria_{domain}.parquet")
        print(f"[{domain}] {len(df)} questions / {len(crit)} criteria")

        # Corpus-level document frequency over *questions*, used to decide which
        # tokens count as instance-specific anchors.
        doc_freq: collections.Counter = collections.Counter()
        for q, r in zip(df["question"], df["reference_answer"]):
            doc_freq.update(set(content_tokens(f"{q} {r}")))
        rare_cut = max(2, int(0.01 * len(df)))
        rare = {t for t, c in doc_freq.items() if c <= rare_cut}

        results[domain] = {
            "rare_token_docfreq_cutoff": rare_cut,
            "n_rare_tokens": len(rare),
            "anchor_grounding": anchor_grounding(df, crit, rare),
            "attribution_retrieval": attribution_retrieval(
                df, crit, args.sample_criteria, args.distractors
            ),
            "reference_coupling": reference_coupling(df, crit),
            "subquestion_coverage": subquestion_coverage(df, crit),
        }
        r = results[domain]
        print(
            f"  generic {r['anchor_grounding']['generic_criterion_rate_pct']}% "
            f"(control {r['anchor_grounding']['generic_rate_permutation_control_pct']}%) | "
            f"attribution top1 {r['attribution_retrieval']['top1_attribution_accuracy_pct']}% | "
            f"ref 3gram overlap {r['reference_coupling']['mean_trigram_overlap_with_reference']} | "
            f"orphan subparts {r['subquestion_coverage'].get('orphan_subpart_rate_pct')}%"
        )

    (OUT / "grounding_stats.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
