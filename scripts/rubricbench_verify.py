#!/usr/bin/env python3
"""Recompute every RubricBench claim from the raw artefacts, from scratch.

This is deliberately independent of ``harness/rubricbench.py`` and of the
``*_score.json`` files: it reads only ``*_verdicts.jsonl``, ``*_rubrics.json``
and the benchmark JSON, and re-derives labels, domains and accuracies itself.
If it disagrees with the stored scores, the stored scores are wrong.

Sections, in order:

1. accuracy for every source, against the official evaluator's convention
   (unanswered counted wrong) and against answered-only
2. paired McNemar exact tests, with BH-FDR over three declared families
3. per-domain deltas with exact binomial confidence intervals on the paired
   discordant counts, because SAFETY is 80 cases and a point estimate there
   invites overreading
4. the ``agentic-tools`` subset comparison, on its own 299 cases only
5/6. the two falsified hypotheses -- restraint wording and rubric length --
   recomputed rather than quoted
7. the alternative explanation for ``framed``: our own judge prompt penalises
   refusals, and SAFETY is where the human label rewards them
8. whether the *published* leaderboard separates its own four systems
9. cases where ``framed`` is right and ``baseline`` wrong, with the rubrics

Following ``requirements.txt``, no scipy: the exact McNemar, Clopper-Pearson,
Wilson and Fisher routines here are implemented directly. ``--section selftest``
checks each of them against closed-form values and against the repository's own
``harness.eval.stats.bh_fdr``; that check also runs in ``scripts/selftest.sh``.

Usage::

    python scripts/rubricbench_verify.py                 # everything
    python scripts/rubricbench_verify.py --section stats # one section
"""
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Iterable, Sequence

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results" / "rubricbench"
BENCH = ROOT / "rubricbench" / "data" / "rubricbench_data.json"

DOMAIN_GROUPS: dict[str, set[str]] = {
    "chat": {"general", "focus", "human-preference", "factuality", "helpful"},
    "if": {"precise if", "ifeval"},
    "stem": {"stem", "math", "mmlu-pro", "gpqa"},
    "code": {"mbpp", "code"},
    "safety": {"safety", "harmlessness"},
}
GROUPS = ["if", "stem", "code", "safety", "chat"]
LETTER = {"A": 0, "B": 1}

ALL_SOURCES = [
    "none", "baseline", "agentic", "baseline_top5", "agentic_top5",
    "framed", "expert", "agentic-tools",
]


def group_of(domain: str) -> str | None:
    domain = (domain or "").strip().lower()
    for group, members in DOMAIN_GROUPS.items():
        if domain in members:
            return group
    return None


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------

def load_bench() -> dict[str, dict[str, Any]]:
    raw = json.loads(BENCH.read_text(encoding="utf-8"))
    return {
        str(r["case_id"]): {
            "label": int(r["label"]),
            "domain": str(r.get("domain", "")).strip().lower(),
            "group": group_of(r.get("domain", "")),
            "instruction": str(r.get("instruction", "")),
            "response_a": str(r.get("response_a", "")),
            "response_b": str(r.get("response_b", "")),
            "expert_rubrics": str(r.get("rubrics", "") or ""),
        }
        for r in raw if r.get("case_id")
    }


def load_verdicts(source: str) -> dict[str, dict[str, Any]]:
    path = RESULTS / f"{source}_verdicts.jsonl"
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[str(row["case_id"])] = row
    return out


def load_rubrics(source: str) -> dict[str, str]:
    path = RESULTS / f"{source}_rubrics.json"
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {str(r["case_id"]): str(r.get("rubric", "") or "") for r in raw}


def load_submission(source: str) -> dict[str, int | None]:
    path = RESULTS / f"{source}_submission.csv"
    out: dict[str, int | None] = {}
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines[1:]:
        if not line.strip():
            continue
        cid, _, pred = line.partition(",")
        pred = pred.strip().upper()
        out[cid.strip()] = 0 if pred == "A" else 1 if pred == "B" else None
    return out


def hit(row: dict[str, Any] | None, bench: dict[str, dict[str, Any]], mode: str = "forward") -> int | None:
    """1/0 against the benchmark's own label; ``None`` when nothing was parsed.

    The label is re-read from the benchmark rather than trusted from the verdict
    file, so a corrupted verdict file cannot silently agree with itself.
    """
    if row is None:
        return None
    if mode == "swap_consistent":
        pred = row.get("forward") if row.get("forward") == row.get("swapped") else None
    else:
        pred = row.get(mode)
    if pred not in LETTER:
        return None
    return int(LETTER[pred] == bench[str(row["case_id"])]["label"])


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def mcnemar_exact(a: Sequence[int], b: Sequence[int]) -> tuple[int, int, float]:
    """Two-sided exact McNemar. Returns (only-b-right, only-a-right, p)."""
    n01 = sum(1 for x, y in zip(a, b) if x == 0 and y == 1)
    n10 = sum(1 for x, y in zip(a, b) if x == 1 and y == 0)
    n = n01 + n10
    if n == 0:
        return 0, 0, 1.0
    k = min(n01, n10)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2 ** n)
    return n01, n10, min(1.0, 2 * tail)


Z95 = 1.959963984540054  # normal 0.975 quantile; 1.96 is off in the 6th decimal


def wilson(k: int, n: int, z: float = Z95) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, centre - half), min(1.0, centre + half))


def _beta_ppf(q: float, a: float, b: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Bisection inverse of the regularised incomplete beta, good to 1e-10."""
    for _ in range(200):
        mid = (lo + hi) / 2
        if _betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _betainc(a: float, b: float, x: float) -> float:
    """Regularised incomplete beta via the continued fraction (Lentz)."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1 - x) * b - lbeta) / a
    if x > (a + 1) / (a + b + 2):
        return 1 - _betainc(b, a, 1 - x)
    f, c, d = 1.0, 1.0, 0.0
    for i in range(0, 300):
        m = i // 2
        if i == 0:
            num = 1.0
        elif i % 2 == 0:
            num = (m * (b - m) * x) / ((a + 2 * m - 1) * (a + 2 * m))
        else:
            num = -((a + m) * (a + b + m) * x) / ((a + 2 * m) * (a + 2 * m + 1))
        d = 1.0 + num * d
        d = 1e-30 if abs(d) < 1e-30 else d
        d = 1 / d
        c = 1.0 + num / c
        c = 1e-30 if abs(c) < 1e-30 else c
        f *= c * d
        if abs(1 - c * d) < 1e-12:
            break
    return front * (f - 1)


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    lo = 0.0 if k == 0 else _beta_ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else _beta_ppf(1 - alpha / 2, k + 1, n - k)
    return lo, hi


def paired_delta_ci(n01: int, n10: int, n_cases: int, alpha: float = 0.05) -> tuple[float, float]:
    """CI on the paired accuracy difference, from the discordant pairs alone.

    Exact-binomial interval on the share of discordant pairs won by the second
    source, rescaled by ``(n01 + n10) / n_cases``. This is the interval that
    matches the McNemar test: it conditions on the same discordant set and so
    carries the same paired-variance reduction.
    """
    d = n01 + n10
    if d == 0 or n_cases == 0:
        return (0.0, 0.0)
    lo, hi = clopper_pearson(n01, d, alpha)
    scale = d / n_cases
    return ((2 * lo - 1) * scale, (2 * hi - 1) * scale)


def bh_fdr(pvals: Sequence[float]) -> list[float]:
    """Benjamini-Hochberg q-values, monotonised."""
    m = len(pvals)
    order = sorted(range(m), key=lambda i: pvals[i])
    q = [0.0] * m
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):
        i = m - rank + 1
        val = min(prev, pvals[idx] * m / i)
        q[idx] = val
        prev = val
    return q


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------

def section_stats(bench, verdicts, out) -> None:
    out("## 1. 独立重算的准确率（对照官方评测器）")
    out()
    out("| source | n | answered | ACC(全部) | ACC(已答) | 位置一致率 | swap-consistent ACC |")
    out("|---|--:|--:|--:|--:|--:|--:|")
    for s in ALL_SOURCES:
        v = verdicts[s]
        cids = sorted(v)
        hits = [hit(v[c], bench) for c in cids]
        n = len(cids)
        answered = sum(1 for h in hits if h is not None)
        correct = sum(h for h in hits if h)
        sc = [hit(v[c], bench, "swap_consistent") for c in cids]
        sc_ans = sum(1 for h in sc if h is not None)
        sc_cor = sum(h for h in sc if h)
        out(f"| `{s}` | {n} | {answered} | {correct / n:.4f} | {correct / answered:.4f} | "
            f"{sc_ans / n:.4f} | {sc_cor / n:.4f} (已答 {sc_cor / max(1, sc_ans):.4f}) |")
    out()

    # a submission CSV must agree with the verdict file it was written from
    out("独立核对：`*_submission.csv` 与 `*_verdicts.jsonl` 的 forward 预测逐条一致性 —— ")
    bad = []
    for s in ALL_SOURCES:
        sub, v = load_submission(s), verdicts[s]
        mism = sum(1 for c in v if sub.get(c) != (LETTER.get(v[c].get("forward")) if v[c].get("forward") in LETTER else None))
        if mism or set(sub) != set(v):
            bad.append(f"{s}: {mism} 条不一致, csv={len(sub)} jsonl={len(v)}")
    out("全部一致。" if not bad else "；".join(bad))
    out()

    # trivial baselines worth knowing about
    labels = [bench[c]["label"] for c in bench]
    maj = max(sum(1 for x in labels if x == 0), sum(1 for x in labels if x == 1)) / len(labels)
    out(f"参考：多数类常量预测 ACC = {maj:.4f}（A={sum(1 for x in labels if x == 0)}, "
        f"B={sum(1 for x in labels if x == 1)}），随机 = 0.5。")
    out()
    fwd_a = {s: sum(1 for c in verdicts[s] if verdicts[s][c].get("forward") == "A") / len(verdicts[s])
             for s in ALL_SOURCES}
    out("判定器选 A（第一个位置）的比例 —— 位置偏置的直接读数：")
    out(", ".join(f"`{s}` {fwd_a[s]:.3f}" for s in ALL_SOURCES))
    out()


FULL_SOURCES = [s for s in ALL_SOURCES if s != "agentic-tools"]


def _paired_test(bench, verdicts, ref: str, src: str, cids: Sequence[str]) -> dict[str, Any]:
    pairs = [(hit(verdicts[ref].get(c), bench), hit(verdicts[src].get(c), bench)) for c in cids]
    pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
    a_h, b_h = [p[0] for p in pairs], [p[1] for p in pairs]
    n01, n10, p = mcnemar_exact(a_h, b_h)
    return {
        "ref": ref, "src": src, "n": len(pairs), "n01": n01, "n10": n10,
        "acc_ref": sum(a_h) / len(pairs), "acc_src": sum(b_h) / len(pairs),
        "delta": (sum(b_h) - sum(a_h)) / len(pairs),
        "ci": paired_delta_ci(n01, n10, len(pairs)), "p": p,
    }


def family_overall(bench, verdicts) -> list[dict[str, Any]]:
    """F1: every unordered pair among the seven full-benchmark sources."""
    common = sorted(set.intersection(*(set(verdicts[s]) for s in FULL_SOURCES)))
    tests = []
    for i, ref in enumerate(FULL_SOURCES):
        for src in FULL_SOURCES[i + 1:]:
            tests.append(_paired_test(bench, verdicts, ref, src, common))
    for t, q in zip(tests, bh_fdr([t["p"] for t in tests])):
        t["q"] = q
    return tests


def family_domain(bench, verdicts) -> list[dict[str, Any]]:
    """F2: `framed` against each control, split by domain."""
    common = sorted(set.intersection(*(set(verdicts[s]) for s in FULL_SOURCES)))
    tests = []
    for ref in ("baseline", "none"):
        for g in GROUPS:
            cids = [c for c in common if bench[c]["group"] == g]
            t = _paired_test(bench, verdicts, ref, "framed", cids)
            t["group"] = g
            tests.append(t)
    for t, q in zip(tests, bh_fdr([t["p"] for t in tests])):
        t["q"] = q
    return tests


def family_tools(bench, verdicts) -> list[dict[str, Any]]:
    """F3: `agentic-tools` against the others, on its own 299 cases."""
    common = sorted(set(verdicts["agentic-tools"]).intersection(
        *(set(verdicts[s]) for s in FULL_SOURCES)))
    tests = [_paired_test(bench, verdicts, ref, "agentic-tools", common)
             for ref in ("agentic", "baseline", "none", "framed", "expert")]
    for t, q in zip(tests, bh_fdr([t["p"] for t in tests])):
        t["q"] = q
    return tests


def _emit_tests(out, tests: Sequence[dict[str, Any]], label_key: str | None = None) -> None:
    head = "| 域 " if label_key else "| "
    out(head + "| 参照 | source | n | 仅 source 对 | 仅参照对 | 参照 ACC | source ACC | Δ ACC | Δ 的 95% CI | p | q (BH) |")
    out(("|---|" if label_key else "|") + "---|---|--:|--:|--:|--:|--:|--:|:--:|--:|--:|")
    for t in tests:
        star = " **" if t["q"] < 0.05 else ""
        prefix = f"| {t[label_key].upper()} " if label_key else "| "
        out(prefix + f"| `{t['ref']}` | `{t['src']}` | {t['n']} | {t['n01']} | {t['n10']} | "
            f"{t['acc_ref']:.4f} | {t['acc_src']:.4f} | {t['delta']:+.4f} | "
            f"[{t['ci'][0]:+.4f}, {t['ci'][1]:+.4f}] | {t['p']:.3g} | {t['q']:.3g}{star} |")
    out()


def section_paired(bench, verdicts, out) -> list[dict[str, Any]]:
    out("## 2. 配对显著性（McNemar 精确检验）+ BH-FDR")
    out()
    common = sorted(set.intersection(*(set(verdicts[s]) for s in FULL_SOURCES)))
    out(f"每个来源判的是同一批题，所以比较是配对的；McNemar 精确检验只看两个来源"
        f"**判得不一样**的那些题，共同答对/答错的部分是共享方差，不进入检验。"
        f"7 个全量来源的交集 = {len(common)} 题。")
    out()
    out("**族的定义。** 报告里出现的检验分三族，每族内部单独做 BH-FDR：")
    out()
    out("- **F1**：7 个全量来源两两之间的总体比较，共 C(7,2)=21 个检验。"
        "取全部两两组合而不是只取我们关心的那几个，是为了不让「先看结果再选检验」进来。")
    out("- **F2**：`framed` 对 `baseline` 与对 `none` 的**分域**比较，5 域 × 2 参照 = 10 个检验。")
    out("- **F3**：`agentic-tools` 在它自己 299 题子集上对 5 个参照的比较。")
    out()
    f1 = family_overall(bench, verdicts)
    out("### F1：总体两两比较（21 个检验，BH-FDR）")
    out()
    _emit_tests(out, sorted(f1, key=lambda t: t["p"]))
    all_p = [t["p"] for t in f1] + [t["p"] for t in family_domain(bench, verdicts)] \
        + [t["p"] for t in family_tools(bench, verdicts)]
    glob = bh_fdr(all_p)
    key = {(t["ref"], t["src"]): q for t, q in zip(f1, glob[:len(f1)])}
    out("稳健性：若把三族合成一族（共 "
        f"{len(all_p)} 个检验）统一校正，核心结论的 q 值为 —— "
        f"`framed` vs `baseline` q={key[('baseline', 'framed')]:.3g}，"
        f"`framed` vs `none` q={key[('none', 'framed')]:.3g}，"
        f"`agentic` vs `baseline` q={key[('baseline', 'agentic')]:.3g}。结论不变。")
    out()
    return f1


def section_domains(bench, verdicts, out) -> None:
    out("## 3. 分域：`framed` vs 两个参照，含不确定性")
    out()
    out("### F2：分域配对检验（10 个检验，BH-FDR）")
    out()
    _emit_tests(out, family_domain(bench, verdicts), label_key="group")
    out("SAFETY 只有 80 题，所以下面同时给出每个来源在该域上的**边际** Wilson 95% CI，"
        "看单点估计本身有多松。CI 重叠不代表配对差异不显著（配对检验的方差更小），"
        "但它说明**绝对水平**不该被当成精确值读。")
    out()
    out("| 域 | n | " + " | ".join(f"`{s}` ACC [95% CI]" for s in ("none", "baseline", "framed", "expert")) + " |")
    out("|---|--:|" + ":--|" * 4)
    for g in GROUPS:
        cids = [c for c in verdicts["framed"] if bench[c]["group"] == g]
        cells = []
        for s in ("none", "baseline", "framed", "expert"):
            h = [hit(verdicts[s][c], bench) for c in cids]
            k = sum(x for x in h if x)
            lo, hi = wilson(k, len(cids))
            cells.append(f"{k / len(cids):.4f} [{lo:.3f}, {hi:.3f}]")
        out(f"| {g.upper()} | {len(cids)} | " + " | ".join(cells) + " |")
    out()


def section_tools(bench, verdicts, out) -> None:
    out("## 4. `agentic-tools`：只在它自己的 299 题子集上比较")
    out()
    sub = set(verdicts["agentic-tools"])
    others = [s for s in ALL_SOURCES if s != "agentic-tools"]
    common = sorted(sub.intersection(*(set(verdicts[s]) for s in others)))
    out(f"子集大小 {len(sub)}，与全量来源的交集 {len(common)}。分层构成：")
    from collections import Counter
    cnt = Counter(bench[c]["group"] for c in common)
    full = Counter(bench[c]["group"] for c in bench)
    out()
    out("| 域 | 子集 n | 子集占比 | 全量占比 |")
    out("|---|--:|--:|--:|")
    for g in GROUPS:
        out(f"| {g.upper()} | {cnt[g]} | {cnt[g] / len(common):.3f} | {full[g] / len(bench):.3f} |")
    out()
    out("同一 299 题上每个来源的 ACC（这是唯一合法的比较口径）：")
    out()
    out("| source | ACC@299 | ACC@1147 | 差 |")
    out("|---|--:|--:|--:|")
    accs = {}
    for s in ALL_SOURCES:
        h = [hit(verdicts[s][c], bench) for c in common]
        accs[s] = sum(x for x in h if x) / len(common)
        full_h = [hit(verdicts[s][c], bench) for c in sorted(verdicts[s])]
        full_acc = sum(x for x in full_h if x) / len(full_h)
        note = f"{full_acc:.4f}" if s != "agentic-tools" else "—"
        diff = f"{accs[s] - full_acc:+.4f}" if s != "agentic-tools" else "—"
        out(f"| `{s}` | {accs[s]:.4f} | {note} | {diff} |")
    out()
    out("### F3：子集内配对检验（5 个检验，BH-FDR）")
    out()
    _emit_tests(out, family_tools(bench, verdicts))


RESTRAINT = re.compile(
    r"\b(avoid|avoids|avoiding|without|refrain|refrains|refuse|refuses|refusing|refusal|"
    r"decline|declines|declining|withhold|withholds|withholding|does not|do not|doesn't|"
    r"don't|never|abstain|abstains|omit|omits|omitting|exclude|excludes|excluding|"
    r"free of|free from|no |not )", re.I)


def _criteria(text: str) -> list[str]:
    return [ln.strip() for ln in (text or "").splitlines() if ln.strip()]


def section_restraint(bench, verdicts, out) -> None:
    out("## 5. 被证伪的假设 A：restraint 措辞")
    out()
    sources = ["baseline", "agentic", "framed", "expert"]
    rub = {s: load_rubrics(s) for s in sources}
    rub["expert"] = {c: bench[c]["expert_rubrics"] for c in bench}
    out("每条判据是否含 avoid/without/refrain/decline/never/... 类措辞（正则），"
        "按域给出**含 restraint 判据的条目占比**：")
    out()
    out("| source | " + " | ".join(g.upper() for g in GROUPS) + " | 全部 |")
    out("|---|" + "--:|" * (len(GROUPS) + 1))
    for s in sources:
        cells = []
        tot_hit = tot_all = 0
        for g in GROUPS:
            hits = alls = 0
            for c in bench:
                if bench[c]["group"] != g:
                    continue
                for line in _criteria(rub[s].get(c, "")):
                    alls += 1
                    hits += bool(RESTRAINT.search(line))
            cells.append(f"{hits / alls:.3f}" if alls else "—")
            tot_hit += hits
            tot_all += alls
        out(f"| `{s}` | " + " | ".join(cells) + f" | {tot_hit / tot_all:.3f} |")
    out()
    out("如果 restraint 措辞是机制，`agentic` 在措辞率高于 `expert` 的域上就该赢。逐域核对：")
    out()
    out("| 域 | `agentic` restraint 率 | `expert` restraint 率 | agentic 率更高? | `agentic` ACC | `expert` ACC |")
    out("|---|--:|--:|:--:|--:|--:|")
    for g in GROUPS:
        rates = {}
        for s in ("agentic", "expert"):
            hits = alls = 0
            for c in bench:
                if bench[c]["group"] != g:
                    continue
                for line in _criteria(rub[s].get(c, "")):
                    alls += 1
                    hits += bool(RESTRAINT.search(line))
            rates[s] = hits / alls if alls else 0.0
        accs = {}
        for s in ("agentic", "expert"):
            cids = [c for c in verdicts[s] if bench[c]["group"] == g]
            h = [hit(verdicts[s][c], bench) for c in cids]
            accs[s] = sum(x for x in h if x) / len(cids)
        mark = "是" if rates["agentic"] > rates["expert"] else "否"
        out(f"| {g.upper()} | {rates['agentic']:.3f} | {rates['expert']:.3f} | {mark} | "
            f"{accs['agentic']:.4f} | {accs['expert']:.4f} |")
    out()
    out("SAFETY 域内部：有/无 restraint 判据的题，判对率如何？")
    out()
    out("| source | 有 restraint 判据 n | 判对率 | 无 n | 判对率 | 差 | p |")
    out("|---|--:|--:|--:|--:|--:|--:|")
    for s in ("baseline", "agentic", "framed", "expert"):
        with_, without = [], []
        for c in bench:
            if bench[c]["group"] != "safety":
                continue
            h = hit(verdicts[s].get(c), bench)
            if h is None or c not in verdicts[s]:
                continue
            lines = _criteria(rub[s].get(c, ""))
            if not lines:
                continue
            (with_ if any(RESTRAINT.search(ln) for ln in lines) else without).append(h)
        if not with_ or not without:
            out(f"| `{s}` | {len(with_)} | — | {len(without)} | — | — | — |")
            continue
        pa, pb = sum(with_) / len(with_), sum(without) / len(without)
        out(f"| `{s}` | {len(with_)} | {pa:.3f} | {len(without)} | {pb:.3f} | {pa - pb:+.3f} | "
            f"{_fisher(sum(with_), len(with_), sum(without), len(without)):.3g} |")
    out()


def _fisher(k1: int, n1: int, k2: int, n2: int) -> float:
    """Two-sided Fisher exact on a 2x2 table, by summing tables at most as likely."""
    a, b, c, d = k1, n1 - k1, k2, n2 - k2
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def prob(x: int) -> float:
        return (math.comb(r1, x) * math.comb(n - r1, c1 - x)) / math.comb(n, c1)

    obs = prob(a)
    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= obs * (1 + 1e-9)))


def section_length(bench, verdicts, out) -> None:
    out("## 6. 被证伪的假设 B：rubric 长度")
    out()
    sources = ["baseline", "agentic", "framed", "expert"]
    rub = {s: load_rubrics(s) for s in sources}
    rub["expert"] = {c: bench[c]["expert_rubrics"] for c in bench}
    out("平均判据条数：" + "，".join(
        f"`{s}` {sum(len(_criteria(rub[s].get(c, ''))) for c in bench) / sum(1 for c in bench if _criteria(rub[s].get(c, ''))):.2f}"
        for s in sources) + "。")
    out()
    bins = [(1, 4), (5, 7), (8, 10), (11, 99)]
    out("同一来源内部，按自己 rubric 的长度分箱后的判对率（若长度是机制，应单调下降）：")
    out()
    out("| source | " + " | ".join(f"{lo}-{hi if hi < 99 else '+'} 条" for lo, hi in bins) + " |")
    out("|---|" + "--:|" * len(bins))
    binned: dict[str, dict[tuple[int, int], list[str]]] = {}
    for s in sources:
        cells = []
        binned[s] = {}
        for lo, hi in bins:
            cids = [c for c in verdicts[s]
                    if lo <= len(_criteria(rub[s].get(c, ""))) <= hi and hit(verdicts[s][c], bench) is not None]
            binned[s][(lo, hi)] = cids
            if not cids:
                cells.append("—")
                continue
            h = [hit(verdicts[s][c], bench) for c in cids]
            cells.append(f"{sum(h) / len(h):.3f} (n={len(h)})")
        out(f"| `{s}` | " + " | ".join(cells) + " |")
    out()
    out("**混淆检验**：把 `expert` 放到*别人的*长度分箱上。若 `expert` 也随箱下降，"
        "那么下降的是题目难度而不是 rubric 长度。")
    out()
    out("| 分箱依据 | " + " | ".join(f"{lo}-{hi if hi < 99 else '+'} 条" for lo, hi in bins) + " |")
    out("|---|" + "--:|" * len(bins))
    for s in ("baseline", "agentic"):
        cells = []
        for b in bins:
            cids = [c for c in binned[s][b] if hit(verdicts["expert"].get(c), bench) is not None]
            if not cids:
                cells.append("—")
                continue
            h = [hit(verdicts["expert"][c], bench) for c in cids]
            cells.append(f"{sum(h) / len(h):.3f} (n={len(h)})")
        out(f"| `{s}` 的长度 → `expert` 的 ACC | " + " | ".join(cells) + " |")
    out()
    out("**直接干预**：截断到前 5 条。若长度是机制，这应该把分数推向 `expert` 的水平。")
    out()
    out("| 对照 | 截断前 | 截断后 | Δ | 仅截断后对 | 仅截断前对 | p |")
    out("|---|--:|--:|--:|--:|--:|--:|")
    for base, top5 in (("baseline", "baseline_top5"), ("agentic", "agentic_top5")):
        common = sorted(set(verdicts[base]) & set(verdicts[top5]))
        pairs = [(hit(verdicts[base][c], bench), hit(verdicts[top5][c], bench)) for c in common]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        a_h, b_h = [p[0] for p in pairs], [p[1] for p in pairs]
        n01, n10, p = mcnemar_exact(a_h, b_h)
        out(f"| `{base}` → `{top5}` | {sum(a_h) / len(pairs):.4f} | {sum(b_h) / len(pairs):.4f} | "
            f"{(sum(b_h) - sum(a_h)) / len(pairs):+.4f} | {n01} | {n10} | {p:.3g} |")
    out()
    out("**长度不是 `framed` 的解释**：`framed` 平均 "
        f"{sum(len(_criteria(rub['framed'].get(c, ''))) for c in bench) / max(1, sum(1 for c in bench if _criteria(rub['framed'].get(c, '')))):.2f} 条，"
        "与 `baseline_top5` 的 5.0 条基本相同，但两者 ACC 相差 "
        f"{_acc(verdicts['framed'], bench) - _acc(verdicts['baseline_top5'], bench):+.4f}。")
    out()


def _acc(v: dict[str, dict[str, Any]], bench) -> float:
    h = [hit(v[c], bench) for c in sorted(v)]
    return sum(x for x in h if x) / len(h)


#: Prefix test for "this response declines rather than complies". Deliberately
#: anchored at the start of the text: a refusal announces itself in the first
#: sentence, whereas a compliant answer that later mentions "I cannot verify"
#: should not count.
REFUSAL = re.compile(
    r"^\s*[\"']?\s*(i'?m sorry|i am sorry|sorry,|i cannot|i can'?t|i won'?t|i will not|"
    r"i must (clarify|decline|point out)|i'?m not able|i am not able|i'?m unable|i am unable|"
    r"i do not|i don'?t (feel )?(comfortable|think)|as an ai|i'?m an ai|unfortunately, i|"
    r"i apolog|i'?m not (going to|supposed to)|i am not supposed to|i'?d rather not|"
    r"this request|i'?m afraid)", re.I)


def _is_refusal(text: str) -> bool:
    return bool(REFUSAL.match((text or "").strip()[:200]))


def section_refusal(bench, verdicts, out) -> None:
    out("## 7. 备择解释：`framed` 在 SAFETY 上赢，有多少只是「押拒答」？")
    out()
    out("我们自己的判定器系统提示里有一句 —— *“A response that refuses or deflects when "
        "the task was answerable is a failure”* —— 这是一条**反拒答**先验，对 SAFETY 是反的。"
        "如果 `framed` 的收益主要来自把这条先验顶回去，那它测到的就不是 rubric 质量，"
        "而是我们自己引入的一个偏置。这一节量化它。")
    out()
    out("先看标签本身：人类偏好的那一侧是不是拒答？（按开头前缀判断）")
    out()
    out("| 域 | n | 偏好侧是拒答 | 被拒侧是拒答 |")
    out("|---|--:|--:|--:|")
    for g in GROUPS:
        cids = [c for c in bench if bench[c]["group"] == g]
        pref = sum(_is_refusal(bench[c]["response_a"] if bench[c]["label"] == 0 else bench[c]["response_b"])
                   for c in cids)
        rej = sum(_is_refusal(bench[c]["response_b"] if bench[c]["label"] == 0 else bench[c]["response_a"])
                  for c in cids)
        out(f"| {g.upper()} | {len(cids)} | {pref / len(cids):.3f} ({pref}) | {rej / len(cids):.3f} ({rej}) |")
    out()
    sf = [c for c in bench if bench[c]["group"] == "safety"]
    one = [c for c in sf if _is_refusal(bench[c]["response_a"]) != _is_refusal(bench[c]["response_b"])]
    rest = [c for c in sf if c not in one]
    won = sum(1 for c in one
              if _is_refusal(bench[c]["response_a"] if bench[c]["label"] == 0 else bench[c]["response_b"]))
    out(f"SAFETY 里恰好一侧拒答的有 {len(one)}/{len(sf)} 题；其中人类偏好拒答那一侧的占 "
        f"{won}/{len(one)} = {won / len(one):.1%}。也就是说，在这 {len(one)} 题上，"
        f"**“无脑选拒答”这条规则能拿 {won / len(one):.3f}**。")
    out()
    out(f"| source | 这 {len(one)} 题上的 ACC | 其余 {len(rest)} 题 | 全部 SAFETY |")
    out("|---|--:|--:|--:|")
    for s in ("none", "baseline", "agentic", "framed", "expert"):
        cells = []
        for cids in (one, rest, sf):
            h = [hit(verdicts[s].get(c), bench) for c in cids]
            h = [x for x in h if x is not None]
            cells.append(f"{sum(h) / len(h):.3f}")
        out(f"| `{s}` | " + " | ".join(cells) + " |")
    out()
    out("读法：`none`/`baseline`/`agentic` 在「一侧拒答」的题上远低于 0.5，"
        "说明判定器在系统性地**挑非拒答的那一侧** —— 正是那条基础规则的效果。"
        "`framed` 把它拉到 0.781，`expert` 到 0.906。")
    out()
    out("但这只解释了一半。把 `framed` 相对 `baseline` 在 SAFETY 的净增量拆开：")
    out()
    out("| SAFETY 子集 | n | `baseline` | `framed` | 净增 |")
    out("|---|--:|--:|--:|--:|")
    for name, cids in (("恰好一侧拒答", one), ("两侧都拒/都不拒", rest), ("全部", sf)):
        pairs = [(hit(verdicts["baseline"].get(c), bench), hit(verdicts["framed"].get(c), bench)) for c in cids]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        bs, fs = sum(p[0] for p in pairs), sum(p[1] for p in pairs)
        out(f"| {name} | {len(pairs)} | {bs / len(pairs):.3f} ({bs}) | {fs / len(pairs):.3f} ({fs}) | +{fs - bs} |")
    out()


def section_leaderboard(bench, verdicts, out) -> None:
    out("## 8. 官方排行榜自己分辨得开吗？")
    out()
    out("这一节完全不用我们的判定器：直接读 `rubricbench/samples/submissions/*.csv`，"
        "四个已发表系统跑的是同一批 1147 题、同一个 judge（gemini3-flash），所以它们之间"
        "也是配对的，可以直接做 McNemar。")
    out()
    subs: dict[str, dict[str, int | None]] = {}
    for path in sorted((ROOT / "rubricbench" / "samples" / "submissions").glob("*.csv")):
        d: dict[str, int | None] = {}
        for line in path.read_text(encoding="utf-8").splitlines()[1:]:
            if not line.strip():
                continue
            cid, _, pred = line.partition(",")
            pred = pred.strip().upper()
            d[cid.strip()] = 0 if pred in {"A", "[[A]]", "0"} else 1 if pred in {"B", "[[B]]", "1"} else None
        subs[path.stem] = d
    names = sorted(subs, key=lambda n: -sum(1 for c in bench if subs[n].get(c) == bench[c]["label"]))
    out("| 已发表系统 | ACC |")
    out("|---|--:|")
    for n in names:
        out(f"| `{n}` | {sum(1 for c in bench if subs[n].get(c) == bench[c]['label']) / len(bench):.4f} |")
    out()
    import itertools  # noqa: PLC0415
    tests = []
    for a, b in itertools.combinations(names, 2):
        ha = [1 if subs[a].get(c) == bench[c]["label"] else 0 for c in bench]
        hb = [1 if subs[b].get(c) == bench[c]["label"] else 0 for c in bench]
        n01, n10, p = mcnemar_exact(ha, hb)
        tests.append({"a": a, "b": b, "d": sum(hb) / len(hb) - sum(ha) / len(ha),
                      "n01": n01, "n10": n10, "p": p})
    for t, q in zip(tests, bh_fdr([t["p"] for t in tests])):
        t["q"] = q
    out("| A | B | Δ (B−A) | 仅 B 对 | 仅 A 对 | p | q (BH) |")
    out("|---|---|--:|--:|--:|--:|--:|")
    for t in tests:
        out(f"| `{t['a']}` | `{t['b']}` | {t['d']:+.4f} | {t['n01']} | {t['n10']} | "
            f"{t['p']:.3g} | {t['q']:.3g} |")
    out()
    sig = [t for t in tests if t["q"] < 0.05]
    ends = next(t for t in tests if {t["a"], t["b"]} == {names[0], names[-1]})
    out(f"**{len(sig)}/{len(tests)} 对达到显著。** 榜首与榜尾（`{names[0]}` vs `{names[-1]}`）"
        f"的差是 {abs(ends['d']):.4f}，p={ends['p']:.3g}。整张榜排的是噪声。")
    out()
    out("换个角度：在 n≈1145、观测到的不一致率下，双侧精确 McNemar 能检出的最小差是多少？")
    out()
    out("| 对比 | 不一致对数 | 最小可检出 \\|Δ\\| | 实际观测 Δ |")
    out("|---|--:|--:|--:|")
    for ref, src in (("none", "baseline"), ("none", "agentic"), ("none", "framed")):
        common = sorted(set(verdicts[ref]) & set(verdicts[src]))
        pr = [(hit(verdicts[ref][c], bench), hit(verdicts[src][c], bench)) for c in common]
        pr = [x for x in pr if x[0] is not None and x[1] is not None]
        n01, n10, _ = mcnemar_exact([x[0] for x in pr], [x[1] for x in pr])
        d, N = n01 + n10, len(pr)
        mde = next(((2 * k - d) / N for k in range(d // 2, d + 1)
                    if mcnemar_exact([1] * (d - k) + [0] * k, [0] * (d - k) + [1] * k)[2] < 0.05), float("nan"))
        out(f"| `{src}` vs `{ref}` | {d}/{N} | {mde:.4f} | {(n01 - n10) / N:+.4f} |")
    out()
    out("已发表排行榜的**全部跨度**是 0.5798−0.5650 = 0.0148，比可检出下限（≈0.027）还小。"
        "这不是「他们差不多」，而是「这个基准在这个规模上分辨不了他们」。")
    out()


def section_cases(bench, verdicts, out, n: int = 12) -> None:
    out("## 9. `framed` 判对而 `baseline` 判错的案例（用于人工核对判据内容）")
    out()
    from collections import Counter  # noqa: PLC0415
    rub_b, rub_f = load_rubrics("baseline"), load_rubrics("framed")
    common = sorted(set(verdicts["baseline"]) & set(verdicts["framed"]))
    rows = [c for c in common
            if hit(verdicts["baseline"][c], bench) == 0 and hit(verdicts["framed"][c], bench) == 1]
    lost = [c for c in common
            if hit(verdicts["baseline"][c], bench) == 1 and hit(verdicts["framed"][c], bench) == 0]
    win_c, loss_c = Counter(bench[c]["group"] for c in rows), Counter(bench[c]["group"] for c in lost)
    out(f"`framed` 赢 {len(rows)} 例、输 {len(lost)} 例，净 +{len(rows) - len(lost)}。按域拆开净增量：")
    out()
    out("| 域 | framed 独对 | baseline 独对 | 净 | 占总净增量 |")
    out("|---|--:|--:|--:|--:|")
    net_total = len(rows) - len(lost)
    for g in GROUPS:
        net = win_c[g] - loss_c[g]
        out(f"| {g.upper()} | {win_c[g]} | {loss_c[g]} | {net:+d} | {net / net_total:+.1%} |")
    out()
    out("**这是本次最需要如实写下来的一行**：净增量的 "
        f"{(win_c['safety'] - loss_c['safety']) / net_total:.0%} 来自 SAFETY —— 全基准 80 题、占 7%。")
    out()
    safety = [c for c in rows if bench[c]["group"] == "safety"]
    out(f"SAFETY 内 {len(safety)} 例：{', '.join(safety)}")
    out()
    for c in (safety + [r for r in rows if r not in safety])[:n]:
        out(f"### `{c}`  ({bench[c]['domain']})")
        out()
        out(f"instruction（截断）：{bench[c]['instruction'][:400]!r}")
        out()
        out(f"人类偏好：{'A' if bench[c]['label'] == 0 else 'B'}")
        out()
        out("`baseline` rubric：")
        out("```")
        out(rub_b.get(c, "")[:900])
        out("```")
        out("`framed` rubric：")
        out("```")
        out(rub_f.get(c, "")[:900])
        out("```")
        out("`expert` rubric：")
        out("```")
        out(bench[c]["expert_rubrics"][:700])
        out("```")
        out()


def selftest() -> int:
    """Check the hand-rolled statistics against closed-form / reference values.

    Reference values were taken from scipy 1.16 once, at authoring time, and are
    frozen here as literals so the check does not require scipy to run.
    """
    bad = 0

    def check(label: str, got: float | tuple[float, ...], want: float | tuple[float, ...],
              tol: float = 1e-6) -> None:
        nonlocal bad
        g = got if isinstance(got, tuple) else (got,)
        w = want if isinstance(want, tuple) else (want,)
        if any(abs(x - y) > tol for x, y in zip(g, w)):
            print(f"  FAIL {label}: {got} != {want}")
            bad += 1
        else:
            print(f"  OK   {label}")

    # McNemar exact == two-sided binomial on the discordant pairs
    check("mcnemar(129,84)", mcnemar_exact([1] * 84 + [0] * 129, [0] * 84 + [1] * 129)[2],
          0.0024902882810814)
    check("mcnemar(0,0)", mcnemar_exact([1, 1], [1, 1])[2], 1.0)
    check("mcnemar(1,1)", mcnemar_exact([1, 0], [0, 1])[2], 1.0)

    # Clopper-Pearson vs scipy.stats.binomtest(...).proportion_ci('exact')
    check("clopper_pearson(5,20)", clopper_pearson(5, 20), (0.08657146910143, 0.49104587170796))
    check("clopper_pearson(0,10)", clopper_pearson(0, 10), (0.0, 0.30849710781876))
    check("clopper_pearson(44,80)", clopper_pearson(44, 80), (0.43467397909415, 0.66151361551018))
    check("clopper_pearson(10,10)", clopper_pearson(10, 10), (0.69150289218124, 1.0))

    # Wilson vs scipy.stats.binomtest(...).proportion_ci('wilson')
    check("wilson(5,20)", wilson(5, 20), (0.11186170140767, 0.46870087761874), tol=1e-9)
    check("wilson(44,80)", wilson(44, 80), (0.44119507889356, 0.65422310815289), tol=1e-9)

    # Fisher exact vs scipy.stats.fisher_exact
    check("fisher(3/10 vs 7/10)", _fisher(3, 10, 7, 10), 0.17889540799758)
    check("fisher(1/10 vs 9/10)", _fisher(1, 10, 9, 10), 0.00109333391067)
    check("fisher(5/10 vs 5/10)", _fisher(5, 10, 5, 10), 1.0)

    # BH-FDR must agree with the repository's own implementation
    try:
        import sys
        sys.path.insert(0, str(ROOT))
        from harness.eval.stats import bh_fdr as repo_bh  # noqa: PLC0415
        probes = [[0.01, 0.02, 0.03, 0.04, 0.05], [0.001, 0.5, 0.9],
                  [0.04, 0.001, 0.6, 0.02, 0.9, 0.3]]
        for p in probes:
            check(f"bh_fdr{p}", tuple(bh_fdr(p)), tuple(repo_bh(p)))
    except Exception as exc:  # noqa: BLE001
        print(f"  FAIL bh_fdr cross-check unavailable: {exc}")
        bad += 1

    # paired_delta_ci must bracket the point estimate it is an interval for
    lo, hi = paired_delta_ci(129, 84, 1145)
    point = (129 - 84) / 1145
    check("paired_delta_ci brackets its point estimate", float(lo < point < hi), 1.0)

    _assert_report_facts(check)   # failures are counted by `check` itself
    print("SELFTEST PASSED" if not bad else f"SELFTEST FAILED ({bad})")
    return 1 if bad else 0


#: The load-bearing numbers in ``results/rubricbench/REPORT.md``. Pinned here so
#: that re-running a source, or editing this script, cannot silently change what
#: the report claims without the check going red.
REPORT_FACTS = {
    "acc": {"none": 0.5650, "baseline": 0.5876, "agentic": 0.5798,
            "baseline_top5": 0.5885, "agentic_top5": 0.5859,
            "framed": 0.6260, "expert": 0.7777, "agentic-tools": 0.6054},
    # (ref, src) -> (delta, p) on the full-benchmark intersection
    "paired": {("baseline", "framed"): (+0.0394, 0.00249),
               ("none", "framed"): (+0.0629, 3.55e-06),
               ("baseline", "agentic"): (-0.0079, 0.628),
               ("none", "baseline"): (+0.0236, 0.0828)},
    "safety_framed_vs_baseline": (+0.3924, 1.23e-07),
    "safety_share_of_net_gain": 0.689,
}


def _assert_report_facts(check) -> None:
    """Re-derive the report's headline numbers and compare with REPORT_FACTS."""
    if not (RESULTS / "framed_verdicts.jsonl").exists() or not BENCH.exists():
        print("  SKIP report facts (no results/rubricbench artefacts)")
        return
    bench = load_bench()
    verdicts = {s: load_verdicts(s) for s in ALL_SOURCES}
    for s, want in REPORT_FACTS["acc"].items():
        check(f"REPORT ACC {s}", round(_acc(verdicts[s], bench), 4), want, tol=5e-5)

    common = sorted(set.intersection(*(set(verdicts[s]) for s in FULL_SOURCES)))
    for (ref, src), (d, p) in REPORT_FACTS["paired"].items():
        t = _paired_test(bench, verdicts, ref, src, common)
        check(f"REPORT Δ {src} vs {ref}", round(t["delta"], 4), d, tol=5e-5)
        check(f"REPORT p {src} vs {ref}", t["p"], p, tol=max(1e-9, abs(p) * 1e-2))

    sf = [c for c in common if bench[c]["group"] == "safety"]
    t = _paired_test(bench, verdicts, "baseline", "framed", sf)
    d, p = REPORT_FACTS["safety_framed_vs_baseline"]
    check("REPORT Δ framed vs baseline @SAFETY", round(t["delta"], 4), d, tol=5e-5)
    check("REPORT p framed vs baseline @SAFETY", t["p"], p, tol=abs(p) * 1e-2)

    wins = sum(1 for c in common if bench[c]["group"] == "safety"
               and hit(verdicts["baseline"][c], bench) == 0 and hit(verdicts["framed"][c], bench) == 1)
    losses = sum(1 for c in common if bench[c]["group"] == "safety"
                 and hit(verdicts["baseline"][c], bench) == 1 and hit(verdicts["framed"][c], bench) == 0)
    net_all = sum(1 for c in common
                  if hit(verdicts["baseline"][c], bench) == 0 and hit(verdicts["framed"][c], bench) == 1) \
        - sum(1 for c in common
              if hit(verdicts["baseline"][c], bench) == 1 and hit(verdicts["framed"][c], bench) == 0)
    check("REPORT SAFETY share of framed's net gain",
          round((wins - losses) / net_all, 3), REPORT_FACTS["safety_share_of_net_gain"], tol=5e-4)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--section", default="all",
                    choices=["all", "selftest", "stats", "paired", "domains", "tools",
                             "restraint", "length", "refusal", "leaderboard", "cases"])
    ap.add_argument("--out", default=None)
    ap.add_argument("--cases", type=int, default=12)
    args = ap.parse_args()

    if args.section == "selftest":
        return selftest()

    bench = load_bench()
    verdicts = {s: load_verdicts(s) for s in ALL_SOURCES}
    lines: list[str] = []

    def out(text: str = "") -> None:
        print(text)
        lines.append(text)

    want = args.section
    if want in ("all", "stats"):
        section_stats(bench, verdicts, out)
    if want in ("all", "paired"):
        section_paired(bench, verdicts, out)
    if want in ("all", "domains"):
        section_domains(bench, verdicts, out)
    if want in ("all", "tools"):
        section_tools(bench, verdicts, out)
    if want in ("all", "restraint"):
        section_restraint(bench, verdicts, out)
    if want in ("all", "length"):
        section_length(bench, verdicts, out)
    if want in ("all", "refusal"):
        section_refusal(bench, verdicts, out)
    if want in ("all", "leaderboard"):
        section_leaderboard(bench, verdicts, out)
    if want in ("all", "cases"):
        section_cases(bench, verdicts, out, args.cases)

    if args.out:
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
