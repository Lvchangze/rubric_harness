#!/usr/bin/env python3
"""Measure what a rubric variant changed: its form, or what its criteria say.

Why this is separate from the score
-----------------------------------
OPTIMIZATION_LOG §3 established that surface form is not a mechanism here:
rewriting `framed`'s rubrics into the expert rubrics' exact shape moved dev ACC
by 0.0017. So "the output looks more like an expert rubric" is not evidence of
anything, and a variant that only changes shape is reproducing a known null.

The interesting question about a variant is whether its criteria say different
things, and in particular whether they say the things the *expert* criteria say.
That needs a content measure, not a formatting one:

``expert_recall``
    fraction of the expert rubric's content tokens for that case which appear
    somewhere in the candidate's rubric.
``expert_recall_novel``
    the same, restricted to expert tokens that do **not** occur in the
    instruction. This is the column to read. Tokens shared with the instruction
    are free — any generator that echoes the prompt collects them — so they
    inflate the plain overlap and hide whether the candidate independently named
    the specific thing the expert names.
``framed_f1``
    token F1 against `framed`'s own rubric for the same case, i.e. how far the
    content moved from the incumbent at all. A variant with high framed_f1 and
    an unchanged score has changed nothing that matters; one with low framed_f1
    and an unchanged score has changed plenty and none of it mattered.

Usage::

    python scripts/rubricbench_rubric_stats.py --sources framed fs_sim5 fs_fix5
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import rubricbench as RB  # noqa: E402

RUBRIC_DIRS = ["results/rubricbench/opt/rubrics", "results/rubricbench"]

# Small closed-class list only. A big stop list would start removing the words
# that carry the content ("avoid", "without", "exactly").
_STOP = frozenset("""a an the and or but if of to in on for with without by as at from into is
are was were be been being does do did doesn don not no it its this that these those response
responses answer answers there their they them then than so such any all each other more most
which who whom whose what when where why how can could should would will shall may might must
have has had having i you he she we us our your his her provide provides provided include
includes included given give gives""".split())
_WORD = re.compile(r"[a-z][a-z'-]{2,}")
_NEG = re.compile(r"\b(avoid|avoids|without|refrain|decline|declines|omit|omits|"
                  r"rather than|instead of|fail|fails|incorrect|not\b|no\b|never)", re.I)


def tokens(text: str) -> set[str]:
    return {w for w in _WORD.findall((text or "").lower()) if w not in _STOP}


def resolve(source: str) -> dict[str, str]:
    """``case_id -> rubric text`` for a variant name, an explicit path, or `expert`."""
    if source == "expert":
        return {c.case_id: c.expert_rubrics for c in RB.load_cases()}
    for candidate in (Path(source),
                      *(Path(d) / f"{source}.json" for d in RUBRIC_DIRS),
                      *(Path(d) / f"{source}_rubrics.json" for d in RUBRIC_DIRS)):
        if candidate.exists():
            raw = json.loads(candidate.read_text(encoding="utf-8"))
            return {str(r["case_id"]): str(r.get("rubric") or "") for r in raw}
    raise SystemExit(f"no rubric file found for {source!r}")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", nargs="+", required=True)
    p.add_argument("--reference", default="framed")
    p.add_argument("--case-ids", default="results/rubricbench/split.json:dev")
    p.add_argument("--out", default=None)
    args = p.parse_args()

    from rubricbench_split import load_case_ids  # noqa: PLC0415

    wanted = set(load_case_ids(args.case_ids))
    cases = {c.case_id: c for c in RB.load_cases() if c.case_id in wanted}
    sources = list(dict.fromkeys([args.reference, *args.sources]))
    rubrics = {s: resolve(s) for s in sources}
    ids = sorted(cid for cid in cases
                 if all(rubrics[s].get(cid, "").strip() for s in sources)
                 and cases[cid].expert_rubrics.strip())

    expert_tok = {cid: tokens(cases[cid].expert_rubrics) for cid in ids}
    instr_tok = {cid: tokens(cases[cid].instruction) for cid in ids}
    novel_tok = {cid: expert_tok[cid] - instr_tok[cid] for cid in ids}

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit(f"# Rubric form and content, {len(ids)} dev cases with an expert rubric")
    emit()
    emit("| source | criteria | words/crit | ends `?` | negated | expert_recall | "
         "**expert_recall_novel** | framed_f1 |")
    emit("|---|--:|--:|--:|--:|--:|--:|--:|")
    for source in sources:
        n_crit = n_words = n_q = n_neg = n_lines = 0
        recall = novel = f1 = 0.0
        for cid in ids:
            text = rubrics[source][cid]
            crit = [ln for ln in text.splitlines() if ln.strip()]
            n_crit += len(crit)
            n_lines += 1
            for ln in crit:
                n_words += len(ln.split())
                n_q += ln.strip().endswith("?")
                n_neg += bool(_NEG.search(ln))
            tok = tokens(text)
            recall += len(tok & expert_tok[cid]) / max(1, len(expert_tok[cid]))
            novel += len(tok & novel_tok[cid]) / max(1, len(novel_tok[cid]))
            ref = tokens(rubrics[args.reference][cid])
            f1 += 2 * len(tok & ref) / max(1, len(tok) + len(ref))
        emit(f"| `{source}` | {n_crit / n_lines:.2f} | {n_words / max(1, n_crit):.1f} | "
             f"{n_q / max(1, n_crit):.2f} | {n_neg / max(1, n_crit):.2f} | "
             f"{recall / n_lines:.3f} | **{novel / n_lines:.3f}** | {f1 / n_lines:.3f} |")
    emit()
    emit("`expert_recall_novel` excludes expert tokens that already occur in the "
         "instruction, so it does not reward echoing the prompt. `framed_f1` is "
         f"token F1 against `{args.reference}`; 1.000 means the content did not move.")
    emit()

    # -- does the rubric already know which response won? -----------------
    # A rubric generated from the instruction alone cannot know which of the two
    # responses the humans preferred, so its vocabulary should sit roughly
    # symmetrically between them. A rubric written by an annotator who had the
    # pair in front of them can name the thing only the preferred response does.
    # That asymmetry is measurable, and it decides whether `expert` is a target
    # an instruction-only generator could ever hit or a different kind of object.
    emit("## Is the rubric's vocabulary tilted toward the winning response?")
    emit()
    emit("For each case, rubric tokens that do not occur in the instruction are "
         "looked up in each response. A generator that never sees the responses "
         "has no way to tilt; an annotator who saw them does.")
    emit()
    import numpy as np  # noqa: PLC0415

    resp_tok = {}
    for cid in ids:
        case = cases[cid]
        chosen, other = ((case.response_b, case.response_a) if case.label == 1
                         else (case.response_a, case.response_b))
        resp_tok[cid] = (tokens(chosen) - instr_tok[cid], tokens(other) - instr_tok[cid])

    emit("| source | in preferred | in rejected | **tilt** | 95% CI (bootstrap) |")
    emit("|---|--:|--:|--:|---|")
    rng = np.random.default_rng(20260831)
    for source in sources:
        pref, rej = [], []
        for cid in ids:
            tok = tokens(rubrics[source][cid]) - instr_tok[cid]
            if not tok:
                continue
            c_tok, r_tok = resp_tok[cid]
            pref.append(len(tok & c_tok) / len(tok))
            rej.append(len(tok & r_tok) / len(tok))
        tilt = np.array(pref) - np.array(rej)
        draws = rng.integers(0, len(tilt), size=(2000, len(tilt)))
        lo, hi = np.percentile(tilt[draws].mean(axis=1), [2.5, 97.5])
        emit(f"| `{source}` | {np.mean(pref):.3f} | {np.mean(rej):.3f} | "
             f"**{tilt.mean():+.3f}** | [{lo:+.3f}, {hi:+.3f}] |")
    emit()
    emit("2000 case-level bootstrap resamples, seed 20260831.")
    emit()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
