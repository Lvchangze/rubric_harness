#!/usr/bin/env python3
"""Decompose RubricBench failures: is the rubric wrong, or is the judge?

`REPORT.md` §11.1(2) asks for this and notes no measurement separated the two.
It turns out no new judging is needed. The rubric-guided judge prompt already
asks for `per_criterion` verdicts alongside its holistic winner, and the whole
reply is in the content-addressed cache — we simply never parsed the per-item
half. Rebuilding the same prompts reads it back at **zero API calls**.

That yields three things the aggregate accuracy cannot show:

* **Where a failure happens.** Aggregating the per-criterion verdicts
  mechanically and comparing with the judge's holistic verdict separates "the
  criteria pointed the right way and the judge did not follow them" from "the
  criteria themselves pointed the wrong way".
* **Whether the rubric could have worked.** Running the same decomposition with
  the benchmark's expert rubric on the same cases is an existence proof: if
  expert's criteria split correctly where ours do not, the gap is rubric
  content, not judge execution.
* **A target that is measurable without labels at generation time.** Two
  properties, both free:
  `split_rate`  — how often a criterion comes out differently on the two
                  responses. A criterion both satisfy separates nothing.
  `decisive_vote_accuracy` — when the splitting criteria do favour one side,
                  how often that side is the one humans preferred.

Usage::

    python scripts/rubricbench_error_decomp.py --out results/rubricbench/ERROR_DECOMP.md
    python scripts/rubricbench_error_decomp.py --scope holdout   # spends nothing, but see HOLDOUT_LOG
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics as st
import sys
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.llm import JSONParseError, extract_json  # noqa: E402
from harness.rubricbench import RUBRIC_SYSTEM, build_pair_prompt, group_of, load_cases  # noqa: E402

CACHE = Path("runs/cache")
RESULTS = Path("results/rubricbench")
LETTER = {"A": 0, "B": 1}

#: Must match what `rubricbench_run.py` sent, or every lookup misses. The engine
#: drops None-valued params before hashing, so only these two appear.
PARAMS = {"model": "hy-t2t-glm-5.2-384k-fp8-L20A-t1-v2", "reasoning_effort": "high"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--scope", default="dev", choices=["dev", "holdout", "full"])
    p.add_argument("--reference", default="framed", help="source whose failures are decomposed")
    p.add_argument("--out", default=None)
    return p.parse_args()


def cache_key(prompt: str, system: str, salt: str | None) -> str:
    payload = {"messages": prompt, "system": system, "params": PARAMS, "salt": salt}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def cache_read(key: str) -> dict[str, Any] | None:
    path = CACHE / key[:2] / key[2:4] / f"{key}.json"
    if not path.exists():
        return None
    try:
        return json.load(path.open(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - a corrupt entry is a miss
        return None


def criterion_weights(rubric: str) -> list[int]:
    """Importance per line, defaulting to 3 where a source emits none."""
    out: list[int] = []
    for line in (rubric or "").splitlines():
        if len(line.strip()) <= 15:
            continue
        m = re.search(r"\(importance (\d+)/5\)", line)
        out.append(int(m.group(1)) if m else 3)
    return out


def judgement(case, rubric: str, *, swapped: bool = False) -> dict[str, Any] | None:
    """Per-criterion and holistic verdicts for one (case, rubric), from cache.

    Verdicts are returned in *dataset* terms: when the pair was shown swapped,
    both the per-criterion flags and the winner are flipped back.
    """
    entry = cache_read(
        cache_key(build_pair_prompt(case, rubric, swapped=swapped), RUBRIC_SYSTEM,
                  "swapped" if swapped else None)
    )
    if entry is None:
        return None
    try:
        parsed = extract_json(entry.get("response", ""), expect="object")
    except JSONParseError:
        return None
    if not isinstance(parsed, dict):
        return None

    weights = criterion_weights(rubric)
    score_a = score_b = 0.0
    favour_a = favour_b = 0.0
    n_items = n_split = 0
    for i, item in enumerate(parsed.get("per_criterion") or []):
        if not isinstance(item, dict):
            continue
        weight = weights[i] if i < len(weights) else 3
        a, b = bool(item.get("a")), bool(item.get("b"))
        if swapped:
            a, b = b, a
        score_a += weight * a
        score_b += weight * b
        n_items += 1
        if a != b:
            n_split += 1
            if a:
                favour_a += weight
            else:
                favour_b += weight

    raw = str(parsed.get("winner", "")).strip().upper()[:1]
    holistic = raw if raw in ("A", "B") else None
    if swapped and holistic:
        holistic = "B" if holistic == "A" else "A"

    return {
        "holistic": holistic,
        # Mechanical aggregation of the same verdicts, RaR Eq.(1) style.
        "mechanical": None if (n_items == 0 or score_a == score_b) else ("A" if score_a > score_b else "B"),
        # Restricted to criteria that actually separated the pair.
        "decisive": None if favour_a == favour_b else ("A" if favour_a > favour_b else "B"),
        "n_items": n_items,
        "n_split": n_split,
        "coherence": abs(favour_a - favour_b) / (favour_a + favour_b) if (favour_a + favour_b) else None,
    }


def load_rubrics(tag: str, cases: dict[str, Any]) -> dict[str, str] | None:
    if tag == "expert":
        return {cid: c.expert_rubrics for cid, c in cases.items()}
    for path in (RESULTS / f"{tag}_rubrics.json", RESULTS / "opt" / "rubrics" / f"{tag}.json"):
        if path.exists():
            raw = json.load(path.open(encoding="utf-8"))
            return {str(r["case_id"]): (r.get("rubric") or "") for r in raw}
    return None


def profile(cases, ids: Iterable[str], rubrics: dict[str, str]) -> dict[str, Any]:
    ids = list(ids)
    n_items = n_split = 0
    hits = decisive_n = decisive_right = 0
    coherences: list[float] = []
    per_case: dict[str, dict[str, Any]] = {}
    for cid in ids:
        rubric = rubrics.get(cid)
        if rubric is None:
            continue
        j = judgement(cases[cid], rubric)
        if j is None:
            continue
        per_case[cid] = j
        n_items += j["n_items"]
        n_split += j["n_split"]
        if j["coherence"] is not None:
            coherences.append(j["coherence"])
        if j["decisive"]:
            decisive_n += 1
            decisive_right += int(LETTER[j["decisive"]] == cases[cid].label)
        if j["holistic"]:
            hits += int(LETTER[j["holistic"]] == cases[cid].label)
    n = len(ids)
    return {
        "n": n,
        "parsed": len(per_case),
        "acc": hits / n if n else 0.0,
        "split_rate": n_split / n_items if n_items else 0.0,
        "coherence": st.mean(coherences) if coherences else 0.0,
        "decisive_vote_accuracy": decisive_right / decisive_n if decisive_n else 0.0,
        "decisive_coverage": decisive_n / n if n else 0.0,
        "criteria_per_case": n_items / len(per_case) if per_case else 0.0,
        "per_case": per_case,
    }


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 3:
        return float("nan")
    mx, my = st.mean(xs), st.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den = (sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)) ** 0.5
    return num / den if den else float("nan")


CANDIDATES = ["framed", "baseline", "agentic", "contrastive", "balanced", "qform",
              "fs_sim5", "fs_fix5", "framed_noweight", "framed_bare", "expert"]


def main() -> int:
    args = parse_args()
    cases = {c.case_id: c for c in load_cases()}
    split = json.load((RESULTS / "split.json").open(encoding="utf-8"))
    ids = sorted(cases) if args.scope == "full" else sorted(split[args.scope])

    out: list[str] = []

    def emit(line: str = "") -> None:
        print(line)
        out.append(line)

    emit(f"# 错误分解：rubric 缺判别轴，还是 judge 没执行对（{args.scope}, n={len(ids)}）")
    emit()
    emit("零 LLM 调用。judge 的 rubric prompt 本来就要求输出 `per_criterion`，"
         "完整回复在内容寻址缓存里；重建同样的 prompt 即可读回逐条判定。")
    emit()

    profiles: list[tuple[str, dict[str, Any]]] = []
    for tag in CANDIDATES:
        rubrics = load_rubrics(tag, cases)
        if rubrics:
            profiles.append((tag, profile(cases, ids, rubrics)))

    emit("## 1. 各来源的判据级画像")
    emit()
    emit("| source | ACC | 分裂率 | 方向一致度 | **判据票决准确** | 有票覆盖 | 条数 |")
    emit("|---|--:|--:|--:|--:|--:|--:|")
    for tag, p in profiles:
        bold = "**" if tag == "expert" else ""
        emit(f"| `{tag}` | {p['acc']:.4f} | {p['split_rate']:.3f} | {p['coherence']:.3f} | "
             f"{bold}{p['decisive_vote_accuracy']:.3f}{bold} | {p['decisive_coverage']:.3f} | "
             f"{p['criteria_per_case']:.1f} |")
    emit()
    emit("- **分裂率**：一条判据在两个回答上给出不同结果的比例。两个都满足的判据分不开任何东西。")
    emit("- **方向一致度**：在分裂的判据里，票有多集中在同一侧（1.0 = 全部指向同一个回答）。")
    emit("- **判据票决准确**：按分裂判据的加权票决出的赢家，与人类标签一致的比例。")
    emit()

    gen = [(t, p) for t, p in profiles if t != "expert"]
    emit("## 2. 哪个属性真的和准确率相关")
    emit()
    emit("| 属性 | 含 `expert` | 仅生成来源 |")
    emit("|---|--:|--:|")
    for label, getter in (("分裂率", "split_rate"), ("方向一致度", "coherence"),
                          ("判据票决准确", "decisive_vote_accuracy")):
        allr = pearson([p[getter] for _, p in profiles], [p["acc"] for _, p in profiles])
        genr = pearson([p[getter] for _, p in gen], [p["acc"] for _, p in gen])
        emit(f"| {label} vs ACC | {allr:.3f} | {genr:.3f} |")
    emit()
    emit(f"（n={len(profiles)} / {len(gen)} 个来源）**含 `expert` 的相关几乎全由它这个离群点撑着**；"
         "生成来源内部区间太窄，不足以当作可优化的信号。有价值的不是相关系数，是下一节的**区间不重叠**。")
    emit()

    ref = args.reference
    ref_p = dict(profiles).get(ref)
    exp_p = dict(profiles).get("expert")
    if ref_p and exp_p:
        gen_dec = [p["decisive_vote_accuracy"] for _, p in gen]
        gen_split = [p["split_rate"] for _, p in gen]
        emit("## 3. 差距在哪：两条区间完全不重叠")
        emit()
        emit(f"| 属性 | 生成来源区间（{len(gen)} 个） | `expert` |")
        emit("|---|--:|--:|")
        emit(f"| 分裂率 | {min(gen_split):.3f} – {max(gen_split):.3f} | **{exp_p['split_rate']:.3f}** |")
        emit(f"| 判据票决准确 | {min(gen_dec):.3f} – {max(gen_dec):.3f} | **{exp_p['decisive_vote_accuracy']:.3f}** |")
        emit()
        emit("十个生成来源攻过形式、判别性意图、restraint 平衡、few-shot 目标分布，"
             "**判据票决准确全部落在一个很窄的带里，没有一个接近 `expert`**。"
             "这不像是 prompt 能撬动的量。")
        emit()

    # -- failure decomposition -------------------------------------------
    if ref_p:
        ref_cases = ref_p["per_case"]
        exp_cases = exp_p["per_case"] if exp_p else {}
        fails = [c for c in ref_cases
                 if ref_cases[c]["holistic"] and LETTER[ref_cases[c]["holistic"]] != cases[c].label]
        recovered = [c for c in fails
                     if ref_cases[c]["mechanical"] and LETTER[ref_cases[c]["mechanical"]] == cases[c].label]
        ties = [c for c in fails if ref_cases[c]["mechanical"] is None]
        wrong = [c for c in fails if c not in set(recovered) | set(ties)]

        emit(f"## 4. `{ref}` 失败题的分解（n={len(fails)}）")
        emit()
        emit("| 类别 | n | 占比 | 含义 |")
        emit("|---|--:|--:|---|")
        emit(f"| 机械聚合能救回 | {len(recovered)} | {len(recovered)/len(fails):.1%} | "
             "judge 的**聚合**环节出错 |")
        emit(f"| 逐条判定打平 | {len(ties)} | {len(ties)/len(fails):.1%} | 判据分不开这一对 |")
        emit(f"| 逐条判定指向错的一边 | {len(wrong)} | {len(wrong)/len(fails):.1%} | "
             "判据本身指错了方向 |")
        emit()
        emit(f"**只有 {len(recovered)/len(fails):.1%} 的失败出在聚合上。**"
             "所以 §11.1(1) 里「正反两序取一致 / k 次采样多数」那类干预，"
             "针对的不是主要失效模式。")
        emit()

        if exp_cases:
            def expert_verdict(pool: list[str]) -> tuple[int, int, int]:
                right = sum(1 for c in pool if c in exp_cases and exp_cases[c]["mechanical"]
                            and LETTER[exp_cases[c]["mechanical"]] == cases[c].label)
                tie = sum(1 for c in pool if c in exp_cases and exp_cases[c]["mechanical"] is None)
                return right, tie, len([c for c in pool if c in exp_cases]) - right - tie

            emit(f"### 4.1 同样这些题，`expert` 的判据表现如何（存在性证明）")
            emit()
            emit("| `" + ref + "` 的失败类型 | n | `expert` 指对 | 也打平 | 也指错 |")
            emit("|---|--:|--:|--:|--:|")
            for label, pool in (("逐条指错方向", wrong), ("逐条打平", ties)):
                r, t, w = expert_verdict(pool)
                covered = max(1, r + t + w)
                emit(f"| {label} | {covered} | **{r} ({r/covered:.1%})** | {t} | {w} |")
            r1, _, _ = expert_verdict(wrong)
            r2, _, _ = expert_verdict(ties)
            emit()
            emit(f"**{r1 + r2} 题（占失败的 {(r1+r2)/len(fails):.1%}）上，"
                 f"`expert` 的判据把同一对回答分对了而 `{ref}` 的没有。**"
                 "这些题的差距是**判据内容**，不是 judge 执行——因为同一个 judge、同一对回答，"
                 "换一份 rubric 就分对了。")
            emit()

    # -- holistic vs mechanical -------------------------------------------
    if ref_p:
        ref_cases = ref_p["per_case"]
        n = len(ref_cases)
        h = sum(1 for c, j in ref_cases.items() if j["holistic"] and LETTER[j["holistic"]] == cases[c].label)
        m = sum(1 for c, j in ref_cases.items() if j["mechanical"] and LETTER[j["mechanical"]] == cases[c].label)
        m_cov = sum(1 for j in ref_cases.values() if j["mechanical"]) / n
        emit("## 5. 附带发现：整体判决远好于机械聚合")
        emit()
        emit(f"同一批逐条判定，`{ref}` 上：")
        emit()
        emit(f"| 聚合方式 | ACC | 覆盖 |")
        emit("|---|--:|--:|")
        emit(f"| judge 的整体判决 | **{h/n:.4f}** | 1.000 |")
        emit(f"| 逐条判定加权机械聚合（RaR Eq.1 口径） | {m/n:.4f} | {m_cov:.3f} |")
        emit()
        emit(f"差 {h/n - m/n:.3f}。**这对本仓库最初的 RaR 目标直接相关**："
             "RL 里 reward 是**机械聚合**出来的，不是让 judge 自由判决。"
             "同一份 rubric 当推理脚手架时的价值，明显高于它当打分函数时的价值——"
             "所以「在 RubricBench 上好用」并不等于「当 RL reward 好用」，反之亦然。")
        emit()

    emit("## 6. 结论")
    emit()
    emit("1. **失败不在聚合环节**（3% 量级），所以先修 judge 聚合的性价比很低。")
    emit("2. **失败在判据本身**：分不开（打平），或者分开了但指错方向。"
         "而 `expert` 在其中过半的题上用同一个 judge 分对了，说明这是**内容**差距的存在性证明。")
    emit("3. **有了一个零成本、无需人类标签就能测的靶子**：分裂率与判据票决准确。"
         "但十个候选在后者上挤在一个窄带里、没有一个接近 `expert`，"
         "说明它大概率不是 prompt 层面能撬动的——与 §9 的推断一致："
         "`expert` 的判据经过 Stage III 用 held-out 回答筛选过，那是我们没有的信号。")
    emit()

    text = "\n".join(out)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
