#!/usr/bin/env python3
"""Self-evolve the final rubric-writing stage of agentic-tools on RubricBench dev.

The upstream (decompose, rollouts, tools, critic, calibrate) is frozen: every
candidate reads the same verified checklist per case, taken from one
agentic-tools run. Differences between candidates are therefore differences in
the writer, not the pipeline's run-to-run noise, which on dev is about as large
as the effects being chased (REPORT.md §12.2).

Each round an optimizer model reads the incumbent writer's failures on the
train split — the instruction, both responses with their lengths, the human
preference, the rubric it wrote, the judge's stated reason, the expert rubric —
and proposes revisions of the writer's guidance. The output contract is fixed
and appended here, so a proposal can change what the writer aims for but not
the format the pipeline parses. Every candidate is scored on the train split;
the best replaces the incumbent only if it beats it there. The val split is
reported and never selected on. The holdout is not read here at all.

    python3 scripts/rubricbench_evolve.py --rounds 5 --candidates 3
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import rubricbench as RB  # noqa: E402
from harness.generators.agentic import _parse_candidates  # noqa: E402
from harness.llm import LLMEngine  # noqa: E402
from harness.prompts.agentic import DISTILL_SYSTEM, build_distill_user  # noqa: E402
from rubricbench_split import load_case_ids  # noqa: E402

logger = logging.getLogger("rubricbench_evolve")

_CONTRACT_MARK = "\n\nOutput ONLY a JSON array:"
assert _CONTRACT_MARK in DISTILL_SYSTEM
SEED_GUIDANCE, _tail = DISTILL_SYSTEM.split(_CONTRACT_MARK, 1)
CONTRACT = _CONTRACT_MARK + _tail
_LINE_RE = re.compile(r"^\s*\d+\.\s*(?:\[(?P<title>.*?)\])?\s*(?P<desc>.*?)\s*\(importance (?P<w>\d)/5\)\s*$")

OPTIMIZER_SYSTEM = """You improve the instructions given to a rubric writer.

The setting. For each task, an analysis pipeline has already solved the instruction and verified facts; it hands the rubric writer the instruction and a verified checklist. The writer produces a short rubric. A judge model then uses that rubric to decide which of two responses to the instruction is better, and the decision is scored against a human preference. The writer never sees the responses: its rubric must follow from the instruction and the checklist alone.

You get the writer's current guidance and cases from a training set where the judge, using the writer's rubric, picked the response the human did NOT prefer (plus a few where it got right what a simpler method got wrong). For each case you see the instruction, both responses (excerpts, with lengths), which one the human preferred, the rubric the writer produced, the judge's stated reason, and a rubric an expert wrote for the same instruction.

Your job: find the GENERAL patterns behind the failures and rewrite the guidance so the writer produces rubrics that lead the judge to the human-preferred response more often, without breaking the cases it already gets right.

Hard constraints on the guidance you write:
- General rules only. Never mention a specific case, topic, entity, number or wording from the cases you were shown; the guidance will be used on new instructions.
- The writer never sees responses. Do not instruct it to compare or inspect responses.
- Do not specify the output format; a fixed JSON contract is appended after your text.
- Keep it under 700 words.

Propose exactly three revisions that differ in approach, each a complete replacement for the current guidance. Output ONLY a JSON array:
[{"diagnosis": "<the failure pattern this revision targets, 1-2 sentences>", "guidance": "<the full revised guidance>"}]"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--upstream", default="results/rubricbench_glm53/agentic-tools-distill_predistill.json",
                   help="frozen per-case verified checklists (the undistilled output of one agentic-tools run)")
    p.add_argument("--baseline", default="results/rubricbench_glm53/baseline_verdicts.jsonl")
    p.add_argument("--dev", default="results/rubricbench/split.json:dev")
    p.add_argument("--out", default="results/rubricbench_glm53/evolve")
    p.add_argument("--model", default="GLM-5.3-H20-t2-copy")
    p.add_argument("--concurrency", type=int, default=128)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--candidates", type=int, default=3)
    p.add_argument("--train-frac", type=float, default=2 / 3)
    p.add_argument("--seed", type=int, default=20261001)
    p.add_argument("--n-fail", type=int, default=14, help="failure cases shown to the optimizer per round")
    p.add_argument("--n-win", type=int, default=4, help="cases the incumbent wins over baseline, shown so they are kept")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def split_dev(cases: list[RB.BenchCase], frac: float, seed: int) -> tuple[list[str], list[str]]:
    by_group: dict[str, list[str]] = defaultdict(list)
    for c in cases:
        by_group[RB.group_of(c.domain)].append(c.case_id)
    train, val = [], []
    for g in sorted(by_group):
        ids = sorted(by_group[g])
        random.Random(f"{seed}:{g}").shuffle(ids)
        k = round(len(ids) * frac)
        train += ids[:k]
        val += ids[k:]
    return sorted(train), sorted(val)


def load_checklists(path: str) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in json.loads(Path(path).read_text(encoding="utf-8")):
        items = []
        for line in str(row["rubric"]).splitlines():
            if not line.strip():
                continue
            m = _LINE_RE.match(line)
            if m:
                items.append({"title": (m["title"] or "").strip(), "description": m["desc"].strip(),
                              "weight": int(m["w"])})
            else:
                items.append({"title": "", "description": line.strip(), "weight": 3})
        out[str(row["case_id"])] = items
    return out


def load_forward_right(path: str) -> dict[str, bool]:
    out = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            v = json.loads(line)
            out[v["case_id"]] = v.get("forward") == "AB"[int(v["label"])]
    return out


# ---------------------------------------------------------------------------
# evaluating one writer guidance
# ---------------------------------------------------------------------------

async def write_rubrics(engine: LLMEngine, cases, checklists, guidance: str, tag: str) -> dict[str, str]:
    system = guidance.rstrip() + CONTRACT

    async def one(case) -> tuple[str, str]:
        user = build_distill_user(case.instruction, checklists.get(case.case_id, []), min_items=3, max_items=7)
        try:
            raw = await engine.chat_json(user, system=system, expect="array", max_tokens=12288, tag=tag)
            crits = [c.to_criterion() for c in _parse_candidates(raw)[:7]]
            return case.case_id, RB.rubric_to_text(SimpleRubric(crits))
        except Exception as exc:  # noqa: BLE001 - a failed write is scored, not fatal
            logger.warning("writer failed on %s: %s", case.case_id, str(exc)[:120])
            return case.case_id, ""

    return dict(await asyncio.gather(*(one(c) for c in cases)))


class SimpleRubric:
    def __init__(self, items):
        self.items = items


async def evaluate(engine, cases, checklists, guidance, tag) -> dict[str, Any]:
    rubrics = await write_rubrics(engine, cases, checklists, guidance, tag)
    verdicts = await RB.judge_all(engine, cases, rubrics, both_orders=True, max_tokens=8192, progress_every=10**9)
    right = {v.case_id: v.forward == "AB"[v.label] for v in verdicts}
    by_group: dict[str, list[int]] = defaultdict(list)
    for c in cases:
        by_group[RB.group_of(c.domain)].append(int(right[c.case_id]))
    return {
        "acc": sum(right.values()) / len(right),
        "by_group": {g: round(sum(v) / len(v), 4) for g, v in sorted(by_group.items())},
        "right": right,
        "rubrics": rubrics,
        "empty": sum(1 for t in rubrics.values() if not t.strip()),
    }


# ---------------------------------------------------------------------------
# the optimizer's dossier
# ---------------------------------------------------------------------------

def _clip(text: str, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[:n] + " …"


async def judge_reason(engine, case, rubric_text: str) -> str:
    """The judge's own explanation for the forward verdict; a cache hit."""
    no_rubric, with_rubric = RB.judge_systems("default")
    try:
        parsed = await engine.chat_json(
            RB.build_pair_prompt(case, rubric_text, swapped=False),
            system=with_rubric if rubric_text.strip() else no_rubric,
            expect="object", max_tokens=8192, tag="rbench:judge", cache_salt=None,
        )
        return _clip(parsed.get("why", ""), 320) if isinstance(parsed, dict) else ""
    except Exception:  # noqa: BLE001
        return ""


async def build_dossier(engine, cases_by_id, train_ids, result, base_right, args, rnd) -> str:
    rng = random.Random(f"{args.seed}:dossier:{rnd}")
    fails = [cid for cid in train_ids if not result["right"][cid]]
    lost = [cid for cid in fails if base_right.get(cid)]          # baseline got these right
    other = [cid for cid in fails if not base_right.get(cid)]
    wins = [cid for cid in train_ids if result["right"][cid] and not base_right.get(cid)]
    pick_fail = _stratified(lost, cases_by_id, rng, args.n_fail * 2 // 3) + \
        _stratified(other, cases_by_id, rng, args.n_fail - args.n_fail * 2 // 3)
    pick_win = _stratified(wins, cases_by_id, rng, args.n_win)

    async def render(cid, kind):
        c = cases_by_id[cid]
        pref, other_r = (c.response_a, c.response_b) if c.label == 0 else (c.response_b, c.response_a)
        rubric = result["rubrics"].get(cid, "")
        why = await judge_reason(engine, c, rubric)
        expert = "\n".join(_clip(x, 200) for x in c.expert_rubrics.splitlines()[:6] if x.strip())
        rub = "\n".join(_clip(x, 220) for x in rubric.splitlines()[:8]) or "(empty)"
        return (f"### {kind} | domain: {RB.group_of(c.domain)}\n"
                f"INSTRUCTION: {_clip(c.instruction, 600)}\n"
                f"HUMAN-PREFERRED RESPONSE ({len(pref)} chars): {_clip(pref, 450)}\n"
                f"OTHER RESPONSE ({len(other_r)} chars): {_clip(other_r, 450)}\n"
                f"WRITER'S RUBRIC:\n{rub}\n"
                f"JUDGE'S REASON: {why or '(none)'}\n"
                f"EXPERT RUBRIC:\n{expert}")

    blocks = await asyncio.gather(*([render(c, "FAILURE") for c in pick_fail] +
                                    [render(c, "KEEP (writer right, simpler method wrong)") for c in pick_win]))
    return "\n\n".join(blocks)


def _stratified(ids, cases_by_id, rng, k):
    by_g: dict[str, list[str]] = defaultdict(list)
    for cid in ids:
        by_g[RB.group_of(cases_by_id[cid].domain)].append(cid)
    for g in by_g:
        rng.shuffle(by_g[g])
    out: list[str] = []
    while len(out) < k and any(by_g.values()):
        for g in sorted(by_g):
            if by_g[g] and len(out) < k:
                out.append(by_g[g].pop())
    return out


async def propose(engine, guidance, dossier, summary, k, rnd) -> list[dict[str, str]]:
    user = (f"CURRENT GUIDANCE:\n<<<\n{guidance}\n>>>\n\n"
            f"TRAINING-SET ACCURACY (agreement with the human, by domain):\n{summary}\n\n"
            f"CASES:\n\n{dossier}\n\nPropose {k} revisions now.")
    raw = await engine.chat_json(user, system=OPTIMIZER_SYSTEM, expect="array", max_tokens=16384,
                                 tag="evolve:optimizer", cache_salt=f"round{rnd}")
    out = []
    for item in raw if isinstance(raw, list) else []:
        g = str((item or {}).get("guidance") or "").strip()
        if 200 <= len(g) <= 7000:
            out.append({"diagnosis": str(item.get("diagnosis") or "")[:400], "guidance": g})
    return out[:k]


# ---------------------------------------------------------------------------

async def main_async() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    out = Path(args.out)
    (out / "prompts").mkdir(parents=True, exist_ok=True)

    dev_ids = set(load_case_ids(args.dev))
    cases = [c for c in RB.load_cases() if c.case_id in dev_ids]
    cases_by_id = {c.case_id: c for c in cases}
    train_ids, val_ids = split_dev(cases, args.train_frac, args.seed)
    (out / "split.json").write_text(json.dumps({"seed": args.seed, "train": train_ids, "val": val_ids}, indent=1))
    train = [cases_by_id[i] for i in train_ids]
    val = [cases_by_id[i] for i in val_ids]
    checklists = load_checklists(args.upstream)
    base_right = load_forward_right(args.baseline)
    base_train = sum(base_right[i] for i in train_ids) / len(train_ids)
    base_val = sum(base_right[i] for i in val_ids) / len(val_ids)
    logger.info("train %d / val %d; baseline acc train %.4f val %.4f", len(train), len(val), base_train, base_val)

    engine = LLMEngine(args.model, concurrency=args.concurrency, cache_dir="runs/cache")
    log_path = out / "log.jsonl"

    def log(rec):
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    incumbent = SEED_GUIDANCE
    (out / "prompts" / "r0_seed.txt").write_text(incumbent + CONTRACT, encoding="utf-8")
    t0 = time.time()
    inc_train = await evaluate(engine, train, checklists, incumbent, "evolve:writer")
    inc_val = await evaluate(engine, val, checklists, incumbent, "evolve:writer")
    logger.info("seed: train %.4f val %.4f (%.0fs)", inc_train["acc"], inc_val["acc"], time.time() - t0)
    log({"round": 0, "name": "seed", "train": inc_train["acc"], "val": inc_val["acc"],
         "train_by_group": inc_train["by_group"], "val_by_group": inc_val["by_group"]})
    history = [("seed", inc_train["acc"], inc_val["acc"])]

    for rnd in range(1, args.rounds + 1):
        base_by_g: dict[str, list[int]] = defaultdict(list)
        for c in train:
            base_by_g[RB.group_of(c.domain)].append(int(base_right[c.case_id]))
        summary = "\n".join(
            f"- {g}: writer {inc_train['by_group'][g]:.3f} vs simpler method {sum(v)/len(v):.3f} (n={len(v)})"
            for g, v in sorted(base_by_g.items()))
        summary += f"\n- overall: writer {inc_train['acc']:.3f} vs simpler method {base_train:.3f}"
        dossier = await build_dossier(engine, cases_by_id, train_ids, inc_train, base_right, args, rnd)
        proposals = await propose(engine, incumbent, dossier, summary, args.candidates, rnd)
        logger.info("round %d: %d proposals", rnd, len(proposals))
        best = None
        for k, prop in enumerate(proposals, start=1):
            name = f"r{rnd}c{k}"
            (out / "prompts" / f"{name}.txt").write_text(prop["guidance"] + CONTRACT, encoding="utf-8")
            res = await evaluate(engine, train, checklists, prop["guidance"], "evolve:writer")
            logger.info("  %s train %.4f (incumbent %.4f) empty=%d | %s", name, res["acc"], inc_train["acc"],
                        res["empty"], prop["diagnosis"][:120])
            log({"round": rnd, "name": name, "train": res["acc"], "train_by_group": res["by_group"],
                 "empty": res["empty"], "diagnosis": prop["diagnosis"]})
            if best is None or res["acc"] > best[1]["acc"]:
                best = (prop, res, name)
        if best and best[1]["acc"] > inc_train["acc"]:
            incumbent, inc_train = best[0]["guidance"], best[1]
            inc_val = await evaluate(engine, val, checklists, incumbent, "evolve:writer")
            logger.info("round %d: %s promoted — train %.4f val %.4f", rnd, best[2], inc_train["acc"], inc_val["acc"])
            log({"round": rnd, "promoted": best[2], "train": inc_train["acc"], "val": inc_val["acc"],
                 "val_by_group": inc_val["by_group"]})
            history.append((best[2], inc_train["acc"], inc_val["acc"]))
        else:
            logger.info("round %d: no candidate beat the incumbent", rnd)
            log({"round": rnd, "promoted": None})

    (out / "final_prompt.txt").write_text(incumbent + CONTRACT, encoding="utf-8")
    lines = ["# Self-evolution of the rubric writer (RubricBench dev, GLM-5.3)", "",
             f"Frozen upstream: `{args.upstream}`. Train {len(train)} / val {len(val)} (seed {args.seed}).",
             f"Baseline: train {base_train:.4f}, val {base_val:.4f}.", "",
             "| incumbent | train | val |", "|---|--:|--:|"]
    lines += [f"| `{n}` | {tr:.4f} | {va:.4f} |" for n, tr, va in history]
    lines += ["", "Selection used the train split only; val is reported, never selected on. "
              "The holdout was not read.", ""]
    (out / "EVOLVE.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
