#!/usr/bin/env python
"""Head-to-head preference under both judges, on the same questions.

The pilot ran head-to-head over every question it had; the swap ran it over the
114-question subset. Comparing the two summaries directly would mix a judge
change with a sample change, so this recomputes both on the shared uids.

The pilot's counter-intuitive finding is the thing being rechecked: the
incumbent judge preferred the single-pass `baseline` rubric to the `agentic`
one, which points the opposite way to every discriminative metric. If that
preference is an idiosyncrasy of the incumbent, a different-family judge should
not reproduce it.

Per question the corrected verdict — the one that survives showing the two
rubrics in both slot orders — is coded

    +1  source_b preferred      0  tie      -1  source_a preferred

and the two judges are compared with a paired sign-flip permutation test over
questions, which is the same machinery the rest of the report uses.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness.eval.stats import mean_ci, paired_diff  # noqa: E402

NAN = float("nan")


def read_rows(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def index(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        if row.get("n_verdicts") != 2:
            continue
        out[row["pair"]][row["uid"]] = row
    return out


def code(row: dict[str, Any]) -> float:
    return {"a": -1.0, "b": 1.0}.get(row.get("corrected_winner"), 0.0)


def summarise(rows: dict[str, dict[str, Any]], uids: list[str]) -> dict[str, Any]:
    sel = [rows[u] for u in uids if u in rows]
    if not sel:
        return {"n": 0}
    n = len(sel)
    a = sum(1 for r in sel if r["corrected_winner"] == "a")
    b = sum(1 for r in sel if r["corrected_winner"] == "b")
    decisive = sum(int(r["n_decisive"]) for r in sel)
    first = sum(int(r["n_first_slot_wins"]) for r in sel)
    votes_a = sum(int(r["raw_wins_a"]) for r in sel)
    votes_b = sum(int(r["raw_wins_b"]) for r in sel)
    return {
        "n": n,
        "corrected_a": a / n,
        "corrected_b": b / n,
        "corrected_tie": (n - a - b) / n,
        "raw_a": votes_a / (2 * n),
        "raw_b": votes_b / (2 * n),
        "flip_rate": sum(1 for r in sel if r["verdict_order1"] != r["verdict_order2"]) / n,
        "position_bias": (first / decisive) if decisive else NAN,
        "margin_b_minus_a": (b - a) / n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new-rows", default="runs/judgeswap/metrics/headtohead_per_pair_rows.jsonl")
    ap.add_argument("--old-rows", default="runs/pilot_v2/metrics/headtohead_per_pair_rows.jsonl")
    ap.add_argument("--new-judge", default="api_azure_openai_gpt-5.1")
    ap.add_argument("--old-judge", default="hy-t2t-glm-5.2-384k-fp8-L20A-t1-v2")
    ap.add_argument("--out-prefix", default="results/judgeswap/headtohead_compare")
    args = ap.parse_args()

    new = index(read_rows(Path(args.new_rows)))
    old = index(read_rows(Path(args.old_rows)))

    report: dict[str, Any] = {
        "new_judge": args.new_judge,
        "old_judge": args.old_judge,
        "pairs": {},
    }
    lines = [
        "=" * 108,
        "HEAD-TO-HEAD BLIND PREFERENCE — does the incumbent's preference survive a judge swap?",
        "=" * 108,
        "",
        f"old judge : {args.old_judge}",
        f"new judge : {args.new_judge}",
        "",
        "Rates are position-bias CORRECTED: a rubric wins a question only if it wins with",
        "the slots in both orders, otherwise the question is a tie. Both judges are scored",
        "on the same questions.",
        "",
        f"  {'pair':<32}{'judge':<6}{'n':>5}{'win_A':>8}{'win_B':>8}{'tie':>8}"
        f"{'B-A':>8}{'flip':>7}{'posbias':>9}",
        "  " + "-" * 100,
    ]

    for pair in new:
        if pair not in old:
            continue
        uids = sorted(set(new[pair]) & set(old[pair]))
        if not uids:
            continue
        s_new = summarise(new[pair], uids)
        s_old = summarise(old[pair], uids)
        src_a = new[pair][uids[0]]["source_a"]
        src_b = new[pair][uids[0]]["source_b"]

        vec_new = [code(new[pair][u]) for u in uids]
        vec_old = [code(old[pair][u]) for u in uids]
        shift = paired_diff(vec_new, vec_old)

        # Did the sign of the corrected preference flip between judges?
        flipped = (
            math.copysign(1, s_new["margin_b_minus_a"]) != math.copysign(1, s_old["margin_b_minus_a"])
            and abs(s_new["margin_b_minus_a"]) > 1e-9
            and abs(s_old["margin_b_minus_a"]) > 1e-9
        )
        report["pairs"][pair] = {
            "source_a": src_a, "source_b": src_b, "n": len(uids),
            "old": s_old, "new": s_new,
            "preference_shift_toward_b": shift,
            "sign_flipped": bool(flipped),
            "agreement_rate": float(
                np.mean([1.0 if a == b else 0.0 for a, b in zip(vec_new, vec_old)])
            ),
        }
        for lbl, s in (("old", s_old), ("new", s_new)):
            lines.append(
                f"  {pair:<32}{lbl:<6}{s['n']:>5}{s['corrected_a']:>8.3f}"
                f"{s['corrected_b']:>8.3f}{s['corrected_tie']:>8.3f}"
                f"{s['margin_b_minus_a']:>+8.3f}{s['flip_rate']:>7.3f}{s['position_bias']:>9.3f}"
            )
        p = shift.get("p_permutation", shift.get("p_value", NAN))
        star = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
        lines.append(
            f"  {'':<32}{'shift':<6}{'':>5}  toward `{src_b}`: "
            f"{shift['mean_diff']:+.3f} "
            f"[{shift['ci_low']:+.3f},{shift['ci_high']:+.3f}] p={p:.4f}{star}"
            f"   per-question agreement={report['pairs'][pair]['agreement_rate']:.3f}"
            f"   SIGN {'FLIPPED' if flipped else 'unchanged'}"
        )
        lines.append("")

    text = "\n".join(lines)
    out = Path(args.out_prefix)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.with_suffix(".txt").write_text(text)
    out.with_suffix(".json").write_text(json.dumps(report, indent=2, default=str))
    print(text)
    print(f"wrote {out.with_suffix('.txt')}")


if __name__ == "__main__":
    main()
