#!/usr/bin/env python
"""Reward-hacking probe on real failures: does a rubric pay for length?

Among rollouts an independent oracle marked **incorrect**, correlate the rubric's
score with the response length. A positive correlation means the rubric rewards
writing more while being wrong, which is the failure mode that matters in RL:
the policy can raise its reward without becoming more correct.

This measures the same thing as the synthetic ``verbose_empty`` tier in
``discriminative.py``, but on responses the model actually produced rather than
on padding written to order, so it cannot be gamed by how convincingly the
degradation prompt asked for waffle.

Two details that change the answer:

* **Cluster bootstrap by question.** Scores for rollouts of the same question
  share a rubric and a difficulty, so they are not independent. Resampling
  individual rollouts would understate the interval substantially.
* **Per domain.** The pooled correlation hides a sign reversal between the two
  corpora, and the pooled number on its own would state the opposite of what
  holds in one of them.

    python scripts/reward_hacking_probe.py --run-name pilot_v2
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

logger = logging.getLogger("reward_hacking_probe")

SOURCES = [
    "shipped", "baseline", "agentic-noval", "agentic",
    "agentic-goldonly", "agentic-negonly", "agentic-realneg",
]

NAN = float("nan")


def pooled_r(records: Sequence[Mapping[str, Any]]) -> float:
    if len(records) < 3:
        return NAN
    scores = np.array([r["score"] for r in records], dtype=float)
    lengths = np.array([r["n_chars"] for r in records], dtype=float)
    if scores.std() == 0 or lengths.std() == 0:
        return NAN
    return float(np.corrcoef(scores, lengths)[0, 1])


def probe(
    rows: Sequence[Mapping[str, Any]],
    sources: Sequence[str],
    *,
    reference: str,
    keep: Callable[[Mapping[str, Any]], bool],
    iters: int,
    seed: int,
) -> dict[str, Any]:
    wrong = {
        s: [r for r in rows if r["rubric_source"] == s and r["correct"] is False and keep(r)]
        for s in sources
    }
    if not wrong.get(reference):
        return {}
    uids = sorted({r["uid"] for r in wrong[reference]})
    observed = {s: pooled_r(wrong[s]) for s in sources}

    rng = np.random.default_rng(seed)
    boots: dict[str, list[float]] = {s: [] for s in sources}
    for _ in range(iters):
        counts: dict[str, int] = {}
        for i in rng.choice(len(uids), size=len(uids), replace=True):
            uid = uids[i]
            counts[uid] = counts.get(uid, 0) + 1
        for source in sources:
            resampled: list[Mapping[str, Any]] = []
            for record in wrong[source]:
                resampled.extend([record] * counts.get(record["uid"], 0))
            boots[source].append(pooled_r(resampled))

    out: dict[str, Any] = {"n_questions": len(uids), "reference": reference, "by_source": {}}
    for source in sources:
        series = np.array([v for v in boots[source] if np.isfinite(v)])
        entry: dict[str, Any] = {
            "r": observed[source],
            "n_incorrect_rollouts": len(wrong[source]),
            "ci_low": float(np.percentile(series, 2.5)) if series.size else NAN,
            "ci_high": float(np.percentile(series, 97.5)) if series.size else NAN,
        }
        if source != reference:
            diffs = np.array([
                boots[source][i] - boots[reference][i]
                for i in range(len(boots[source]))
                if np.isfinite(boots[source][i]) and np.isfinite(boots[reference][i])
            ])
            entry["delta"] = observed[source] - observed[reference]
            entry["delta_ci_low"] = float(np.percentile(diffs, 2.5))
            entry["delta_ci_high"] = float(np.percentile(diffs, 97.5))
            entry["p_value"] = float(2 * min((diffs >= 0).mean(), (diffs <= 0).mean()))
        out["by_source"][source] = entry
    return out


def render(block: Mapping[str, Any], sources: Sequence[str], title: str) -> str:
    if not block:
        return ""
    reference = block["reference"]
    lines = [
        f"### {title}（n={block['n_questions']} 题）",
        "",
        "| 来源 | r(score, length) | 95% CI | Δ vs `baseline` | Δ 的 95% CI | p |",
        "|---|---|---|---|---|---|",
    ]
    for source in sources:
        e = block["by_source"].get(source)
        if not e:
            continue

        def f(x: Any) -> str:
            return "—" if x is None or not np.isfinite(x) else f"{x:+.3f}"

        if source == reference:
            lines.append(
                f"| `{source}` | {f(e['r'])} | [{f(e['ci_low'])}, {f(e['ci_high'])}] "
                f"| （参照） | | |"
            )
            continue
        p = e.get("p_value", NAN)
        mark = "**" if p < 0.05 else ""
        lines.append(
            f"| `{source}` | {f(e['r'])} | [{f(e['ci_low'])}, {f(e['ci_high'])}] "
            f"| {mark}{f(e['delta'])}{mark} "
            f"| [{f(e['delta_ci_low'])}, {f(e['delta_ci_high'])}] | {p:.3f} |"
        )
    lines += ["", "越低越好（负相关 = 越长的错误答案得分越低）。"
                  "置信区间为**按题聚类**的 bootstrap。"]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run-name", default="pilot_v2")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--results-dir", default="results")
    p.add_argument("--reference", default="baseline")
    p.add_argument("--iters", type=int, default=4000)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()
    path = Path(args.runs_dir) / args.run_name / "metrics" / "rollout_per_rollout_rows.jsonl"
    if not path.exists():
        raise SystemExit(f"missing {path}")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    sources = [s for s in SOURCES if any(r["rubric_source"] == s for r in rows)]

    payload: dict[str, Any] = {}
    parts = ["## Reward hacking 探针（真实失败 rollout）", ""]
    payload["pooled"] = probe(rows, sources, reference=args.reference,
                              keep=lambda r: True, iters=args.iters, seed=args.seed)
    parts += [render(payload["pooled"], sources, "两域合并"), ""]
    for domain in sorted({str(r.get("domain")) for r in rows if r.get("domain")}):
        block = probe(rows, sources, reference=args.reference,
                      keep=lambda r, d=domain: r.get("domain") == d,
                      iters=args.iters, seed=args.seed)
        payload[domain] = block
        parts += [render(block, sources, f"分域：`{domain}`"), ""]

    out_dir = Path(args.results_dir) / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "reward_hacking_probe.json").write_text(
        json.dumps(payload, indent=1, ensure_ascii=False, default=str), encoding="utf-8"
    )
    text = "\n".join(parts)
    (out_dir / "reward_hacking_probe.md").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
