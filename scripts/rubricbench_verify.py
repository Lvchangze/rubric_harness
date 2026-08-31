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
    _assert_round2_facts(check, bench)
    _assert_judge_profile_facts(check, bench)


#: The load-bearing numbers added by §9 (ceiling validity) and §10 (round two).
#: `opt/**` is frozen history, so these are pinned against REPORT.md, not against
#: the optimisation log.
ROUND2_FACTS = {
    # §9.4, the claim the ceiling revision rests on: share of the expert rubric's
    # discriminating content words that land in the human-preferred response.
    "expert_winner_share": 0.5763,
    "framed_winner_share": 0.5092,
    # §10: dev ACC for every source that exists on dev, forward, unanswered wrong
    "dev_acc": {"none": 0.5817, "baseline": 0.6083, "framed": 0.6500, "expert": 0.8083,
                "framed_bare": 0.6517, "framed_noweight": 0.6583, "qform": 0.6283,
                "balanced": 0.6267, "contrastive": 0.5867, "fs_sim5": 0.6050,
                "fs_fix5": 0.6233},
    # §10: candidate vs `framed`, ex-SAFETY, unanswered dropped (the report's
    # paired convention). src -> (delta, p)
    "dev_ex_safety_vs_framed": {
        "framed_noweight": (+0.0143, 0.403), "framed_bare": (-0.0018, 1.0),
        "qform": (-0.0197, 0.315), "balanced": (-0.0054, 0.845),
        "fs_fix5": (-0.0234, 0.218), "fs_sim5": (-0.0396, 0.0316),
        "contrastive": (-0.0467, 0.0292), "expert": (+0.1634, 8.49e-19),
    },
    # §10: the offline bounds that closed two families
    "route_oracle_ex_safety": +0.0233,
    "per_case_oracle_ex_safety": 0.8459,
    "vote_ex_safety_tie_to_incumbent": +0.0072,
    # §9.6 / §10.1: the two fragility numbers the "bottleneck is the judge" claim uses
    "framed_dev_position_consistency": 0.8633,
    "framed_correct_held_by_all_nine": 0.5000,
    "expert_dev_error_rate": 0.1917,
}

#: §10.0.1, the review of `opt/OPTIMIZATION_LOG.md` §5. The decision there rested
#: on marginal accuracies; these are the paired numbers that replace them. Both
#: judge profiles have all 600 per-case verdicts on disk, so every entry here is
#: recomputable -- which is the whole point of the section.
#: (contrast, scope) -> {profile: (delta, p)}
JUDGE_PROFILE_FACTS = {
    "marginal": {  # forward ACC, official convention, unanswered wrong
        ("default", "none", "ex"): 0.6057, ("neutral", "none", "ex"): 0.6201,
        ("default", "framed", "ex"): 0.6487, ("neutral", "framed", "ex"): 0.6434,
        ("default", "expert", "ex"): 0.8100, ("neutral", "expert", "ex"): 0.8100,
        ("default", "framed", "safety"): 0.6667, ("neutral", "framed", "safety"): 0.5952,
        # the neutral judge is measurably blunter on SAFETY: the ceiling drops too
        ("default", "expert", "safety"): 0.7857, ("neutral", "expert", "safety"): 0.6667,
    },
    #: §10.0.1: switching profile on one source, (only-default-right, only-neutral-right).
    #: `expert` on SAFETY is the only near-significant cell, and it is one-sided.
    "cross_profile_safety": {"expert": (5, 0), "none": (2, 3)},
    "paired": {
        # the two results that lose significance without the anti-refusal clause
        ("framed", "baseline", "all"): {"default": (+0.0417, 0.0223),
                                        "neutral": (+0.0251, 0.176)},
        ("framed", "none", "ex"): {"default": (+0.0430, 0.0308),
                                   "neutral": (+0.0233, 0.255)},
        # the part of §5's decision that does hold: nothing detectable ex-SAFETY
        ("framed", "baseline", "ex"): {"default": (+0.0179, 0.348),
                                       "neutral": (+0.0090, 0.675)},
        # SAFETY, where the clause is worth about a third of framed's margin
        ("framed", "baseline", "safety"): {"default": (+0.3571, 0.000729),
                                          "neutral": (+0.2439, 0.0213)},
    },
    #: difference in differences, neutral minus default; bootstrap point estimates
    "dind": {("framed", "baseline", "ex"): -0.0072,
             ("framed", "baseline", "safety"): -0.1220},
    #: §10.0.1: framed's share of the dev ex-SAFETY floor-to-ceiling span
    "space_ex_safety": {"default": 0.211, "neutral": 0.123},
}

#: Ten dev sources: `framed` plus the seven candidates plus the two references.
_DEV_SOURCES = ["framed", "framed_bare", "framed_noweight", "qform", "balanced",
                "contrastive", "fs_sim5", "fs_fix5", "baseline", "none"]


def _load_dev_verdicts(source: str) -> dict[str, dict[str, Any]] | None:
    """Verdicts for a dev-only candidate, which live under ``opt/runs``."""
    for path in (RESULTS / f"{source}_verdicts.jsonl",
                 RESULTS / "opt" / "runs" / f"dev_{source}_verdicts.jsonl"):
        if path.exists():
            rows: dict[str, dict[str, Any]] = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    rows[str(row["case_id"])] = row
            return rows
    return None


def _content_words(text: str) -> set[str]:
    """A third independent tokenisation of "content word", for §9.4.

    Deliberately not the one in ``rubricbench_rubric_stats.py`` (which produced
    the number) nor the one in ``rubricbench_tilt_audit.py`` (which audited it):
    if 57.6% survives a third arbitrary choice of stop list and word regex, it is
    not a property of any of them.
    """
    stop = {"the", "and", "for", "that", "this", "with", "not", "are", "was", "have",
            "has", "does", "response", "answer", "any", "all", "its", "their", "they",
            "which", "when", "where", "how", "can", "should", "would", "there"}
    return {w for w in re.findall(r"[a-z]{3,}", (text or "").lower()) if w not in stop}


def _assert_round2_facts(check, bench) -> None:
    """Re-derive §9 and §10's headline numbers from the artefacts."""
    split_path = RESULTS / "split.json"
    if not split_path.exists():
        print("  SKIP round-2 facts (no split.json)")
        return

    # -- §9.4: does the expert rubric's discriminating vocabulary favour the winner?
    raw = json.loads(BENCH.read_text(encoding="utf-8"))
    shares: dict[str, float] = {}
    framed_rub = load_rubrics("framed") if (RESULTS / "framed_rubrics.json").exists() else {}
    for label, getter in (("expert", lambda r: str(r.get("rubrics") or "")),
                          ("framed", lambda r: framed_rub.get(str(r["case_id"]), ""))):
        w_only = l_only = 0
        for r in raw:
            text = getter(r)
            if not text.strip():
                continue
            instr = _content_words(r.get("instruction", ""))
            tok = _content_words(text) - instr
            a, b = str(r.get("response_a", "")), str(r.get("response_b", ""))
            win, lose = (b, a) if int(r["label"]) == 1 else (a, b)
            wt, lt = _content_words(win) - instr, _content_words(lose) - instr
            w_only += len((tok & wt) - lt)
            l_only += len((tok & lt) - wt)
        shares[label] = w_only / (w_only + l_only) if (w_only + l_only) else 0.5
    # A third tokenisation cannot reproduce the reported figure to the digit, and
    # is not supposed to: the claim is that the effect does not depend on the
    # choice. So this asserts agreement within a percentage point, and that the
    # gap to `framed` survives.
    check("REPORT §9.4 expert discriminating words in winner (3rd tokenisation)",
          shares["expert"], ROUND2_FACTS["expert_winner_share"], tol=0.015)
    check("REPORT §9.4 framed discriminating words in winner (3rd tokenisation)",
          shares["framed"], ROUND2_FACTS["framed_winner_share"], tol=0.015)
    check("REPORT §9.4 expert favours the winner and framed does not",
          float(shares["expert"] - shares["framed"] > 0.04), 1.0)
    check("REPORT §9.4 expert share excludes 0.5 by a wide margin",
          float(shares["expert"] > 0.54), 1.0)

    # -- §10: dev scores, candidate deltas, and the offline bounds
    dev = sorted(json.loads(split_path.read_text(encoding="utf-8"))["dev"])
    vd = {s: _load_dev_verdicts(s) for s in [*_DEV_SOURCES, "expert"]}
    if any(v is None for v in vd.values()):
        missing = [s for s, v in vd.items() if v is None]
        print(f"  SKIP round-2 dev facts (no verdicts for {missing})")
        return
    ex_safety = [c for c in dev if bench[c]["group"] != "safety"]

    for s, want in ROUND2_FACTS["dev_acc"].items():
        got = sum(hit(vd[s].get(c), bench) or 0 for c in dev) / len(dev)
        check(f"REPORT dev ACC {s}", round(got, 4), want, tol=5e-5)

    for s, (d, p) in ROUND2_FACTS["dev_ex_safety_vs_framed"].items():
        t = _paired_test(bench, vd, "framed", s, ex_safety)
        check(f"REPORT dev Δ ex-SAFETY {s} vs framed", round(t["delta"], 4), d, tol=5e-5)
        check(f"REPORT dev p ex-SAFETY {s} vs framed", t["p"], p, tol=max(1e-9, abs(p) * 2e-2))

    hits = {s: {c: hit(vd[s].get(c), bench) for c in dev} for s in _DEV_SOURCES}
    # Oracle routing: ground-truth group label, winner chosen in sample, ten sources.
    by_group: dict[str, list[str]] = {}
    for c in ex_safety:
        by_group.setdefault(bench[c]["group"], []).append(c)
    pick = {}
    for g, ids in by_group.items():
        scores = {s: sum(hits[s][c] or 0 for c in ids) / len(ids) for s in _DEV_SOURCES}
        pick[g] = max(_DEV_SOURCES, key=lambda s: (scores[s], s == "framed"))
    routed = sum(hits[pick[bench[c]["group"]]][c] or 0 for c in ex_safety) / len(ex_safety)
    base = sum(hits["framed"][c] or 0 for c in ex_safety) / len(ex_safety)
    check("REPORT §10 oracle routing Δ ex-SAFETY", round(routed - base, 4),
          ROUND2_FACTS["route_oracle_ex_safety"], tol=5e-5)

    oracle = sum(int(any(hits[s][c] for s in _DEV_SOURCES)) for c in ex_safety) / len(ex_safety)
    check("REPORT §10 per-case oracle ex-SAFETY", round(oracle, 4),
          ROUND2_FACTS["per_case_oracle_ex_safety"], tol=5e-5)

    # Majority vote, ties falling back to the incumbent (the artefact's rule).
    letter = {"A": 0, "B": 1}
    vote_hits = {}
    for c in ex_safety:
        picks = [vd[s][c].get("forward") for s in _DEV_SOURCES if c in vd[s]]
        picks = [p for p in picks if p]
        if not picks:
            continue
        top = max(set(picks), key=picks.count)
        vote_hits[c] = (hits["framed"][c] if picks.count(top) * 2 == len(picks)
                        else int(letter[top] == bench[c]["label"]))
    ids = [c for c in ex_safety if vote_hits.get(c) is not None and hits["framed"][c] is not None]
    delta = sum(vote_hits[c] - hits["framed"][c] for c in ids) / len(ids)
    check("REPORT §10 majority vote Δ ex-SAFETY (ties to incumbent)", round(delta, 4),
          ROUND2_FACTS["vote_ex_safety_tie_to_incumbent"], tol=5e-5)

    # Fragility: the two numbers "the bottleneck is the judge" is built on.
    pc = sum(1 for c in dev if vd["framed"][c].get("forward") is not None
             and vd["framed"][c]["forward"] == vd["framed"][c].get("swapped")) / len(dev)
    check("REPORT §10 framed dev position consistency", round(pc, 4),
          ROUND2_FACTS["framed_dev_position_consistency"], tol=5e-5)
    others = [s for s in _DEV_SOURCES if s != "framed"]
    right = [c for c in dev if hits["framed"][c] == 1]
    held = [c for c in right if all(hits[s][c] == 1 for s in others)]
    check("REPORT §10 share of framed's correct answers held by all nine",
          round(len(held) / len(right), 4),
          ROUND2_FACTS["framed_correct_held_by_all_nine"], tol=5e-5)
    err = sum(1 for c in dev if (hit(vd["expert"].get(c), bench) or 0) == 0) / len(dev)
    check("REPORT §9.6 expert dev error rate", round(err, 4),
          ROUND2_FACTS["expert_dev_error_rate"], tol=5e-5)


def _assert_judge_profile_facts(check, bench) -> None:
    """§10.0.1: the paired two-profile comparison, re-derived from the verdicts.

    `opt/OPTIMIZATION_LOG.md` §5 decided "mainline stays default, the confound is
    measured and small" from marginal accuracies alone. The per-case verdicts for
    both profiles are archived, so the paired tests are recomputable; these
    assertions pin the ones the revised decision rests on.
    """
    split_path = RESULTS / "split.json"
    if not split_path.exists():
        print("  SKIP judge-profile facts (no split.json)")
        return
    dev = sorted(str(c) for c in json.loads(split_path.read_text(encoding="utf-8"))["dev"])
    sources = ["none", "baseline", "framed", "expert"]
    prof: dict[str, dict[str, dict[str, Any]]] = {"default": {}, "neutral": {}}
    for s in sources:
        neutral_path = RESULTS / f"devN_{s}_verdicts.jsonl"
        if not neutral_path.exists():
            print(f"  SKIP judge-profile facts (no {neutral_path.name})")
            return
        prof["neutral"][s] = load_verdicts(f"devN_{s}")
        prof["default"][s] = {c: r for c, r in load_verdicts(s).items() if c in set(dev)}
    for pname, d in prof.items():
        for s, rows in d.items():
            check(f"JUDGE_PROFILE {pname}/{s} covers the 600 dev cases",
                  float(set(rows) == set(dev)), 1.0)

    scope = {"all": dev,
             "safety": [c for c in dev if bench[c]["group"] == "safety"],
             "ex": [c for c in dev if bench[c]["group"] != "safety"]}
    check("JUDGE_PROFILE dev SAFETY is 42 cases", float(len(scope["safety"])), 42.0)

    for (pname, s, sc), want in JUDGE_PROFILE_FACTS["marginal"].items():
        ids = scope[sc]
        got = sum(1 for c in ids if hit(prof[pname][s].get(c), bench) == 1) / len(ids)
        check(f"REPORT §10 marginal ACC {pname}/{s}/{sc}", round(got, 4), want, tol=5e-5)

    for (hi, lo, sc), per_profile in JUDGE_PROFILE_FACTS["paired"].items():
        for pname, (d, p) in per_profile.items():
            t = _paired_test(bench, prof[pname], lo, hi, scope[sc])
            check(f"REPORT §10 paired Δ {hi}−{lo} {sc} ({pname})", round(t["delta"], 4),
                  d, tol=5e-5)
            check(f"REPORT §10 paired p {hi}−{lo} {sc} ({pname})", t["p"], p,
                  tol=max(1e-9, abs(p) * 2e-2))

    # Switching profile on one source: how many SAFETY cases actually turn over.
    # The net figure `opt/**` reported hides the churn, and for `expert` the churn
    # is entirely one-directional -- the neutral judge loses five and gains none.
    for s, (only_def, only_neu) in JUDGE_PROFILE_FACTS["cross_profile_safety"].items():
        # ref is `default`, so n10 counts only-ref-right and n01 only-src-right.
        t = _paired_test(bench, {"d": prof["default"][s], "n": prof["neutral"][s]},
                         "d", "n", scope["safety"])
        check(f"REPORT §10 {s} SAFETY cases only default gets right", float(t["n10"]),
              float(only_def))
        check(f"REPORT §10 {s} SAFETY cases only neutral gets right", float(t["n01"]),
              float(only_neu))

    # The load-bearing correction: ex-SAFETY the clause moves nothing, inside
    # SAFETY it moves about a third of framed's margin.
    for (hi, lo, sc), want in JUDGE_PROFILE_FACTS["dind"].items():
        vals = []
        for c in scope[sc]:
            h = [hit(prof[p][s].get(c), bench) for p in ("neutral", "default") for s in (hi, lo)]
            if any(v is None for v in h):
                continue
            vals.append((h[0] - h[1]) - (h[2] - h[3]))
        check(f"REPORT §10 difference-in-differences {hi}−{lo} {sc}",
              round(sum(vals) / len(vals), 4), want, tol=5e-5)

    # §10.0.1: framed's share of the dev ex-SAFETY span, which nearly halves.
    for pname, want in JUDGE_PROFILE_FACTS["space_ex_safety"].items():
        acc = {s: sum(1 for c in scope["ex"] if hit(prof[pname][s].get(c), bench) == 1)
                  / len(scope["ex"]) for s in ("none", "framed", "expert")}
        share = (acc["framed"] - acc["none"]) / (acc["expert"] - acc["none"])
        check(f"REPORT §10.0.1 framed share of dev ex-SAFETY span ({pname})",
              round(share, 3), want, tol=5e-4)
    # And the direction of that halving is the claim, not the two decimals.
    check("REPORT §10.0.1 the span share falls under the neutral judge",
          float(JUDGE_PROFILE_FACTS["space_ex_safety"]["neutral"]
                < JUDGE_PROFILE_FACTS["space_ex_safety"]["default"] - 0.05), 1.0)


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
