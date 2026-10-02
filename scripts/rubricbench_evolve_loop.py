#!/usr/bin/env python3
"""Keep evolving the agentic-tools writer until one candidate beats baseline on the holdout.

"Loop until it beats baseline" is only meaningful if the final check is one the
loop never selected on; otherwise enough noisy candidates guarantee a dev "win".
So the loop has three layers, and only the last can declare success:

1. Search on the train slice of dev (selection happens here). Beam of the best
   two guidances; each round the optimizer proposes revisions of each.
2. Gate on the val slice: a beam member is sent onward only if it leads
   baseline by --train-margin on train AND by --val-margin on val.
3. Verdict on the holdout. Before any holdout number is computed, the spend is
   appended to results/rubricbench/opt/HOLDOUT_LOG.md. The candidate's writer
   runs over the holdout's frozen agentic-tools checklists (generated earlier
   with rubricbench_run.py --no-judge) and is judged against baseline with an
   exact McNemar test at alpha / budget (Bonferroni across the looks). A win
   stops the loop; otherwise the look is spent and the search continues.

The loop also stops when the holdout budget is used up or after --max-rounds.

    python3 scripts/rubricbench_evolve_loop.py
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import random
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
from harness.prompts.agentic import build_distill_user  # noqa: E402
from rubricbench_evolve import CONTRACT, SEED_GUIDANCE, _clip, _stratified, load_checklists, split_dev  # noqa: E402
from rubricbench_run import FRAMED_SYSTEM  # noqa: E402
from rubricbench_split import load_case_ids  # noqa: E402

logger = logging.getLogger("rubricbench_evolve_loop")

_FRAMED_GUIDANCE = FRAMED_SYSTEM.split("\n\nOutput ONLY a JSON array:", 1)[0]
#: `framed` was the strongest single-call prompt measured; here it also gets the
#: verified checklist, for facts only.
SEED_FRAMED = _FRAMED_GUIDANCE + """

You are also given a verified checklist from an analysis that already solved the task and checked facts. Use it for what is true - the correct answer, the real constraints, the facts a response must not get wrong - but do not turn it into an answer key: never require details, examples or sub-points just because the checklist mentions them."""

OPTIMIZER_SYSTEM = """You improve the instructions given to a rubric writer.

The setting. For each task, an analysis pipeline has already solved the instruction and verified facts; it hands the rubric writer the instruction and a verified checklist. The writer produces a short rubric. A judge model then uses that rubric to decide which of two responses to the instruction is better, and the decision is scored against a human preference. The writer never sees the responses: its rubric must follow from the instruction and the checklist alone.

You get the writer's current guidance and training cases where the judge, using the writer's rubric, picked the response the human did NOT prefer, plus a few it gets right that a simpler method gets wrong. For each case you see the instruction, both responses (excerpts, with lengths), which one the human preferred, the writer's rubric with how the judge marked each criterion for each response, the judge's reason, and a rubric an expert wrote for the same instruction.

Find the GENERAL patterns behind the failures and revise the guidance so the judge reaches the human-preferred response more often, without losing the cases it already gets right. Prefer focused edits to the current guidance over rewriting it from scratch: what it does well should survive.

Hard constraints on the guidance you write:
- General rules only. Never mention a specific case, topic, entity, number or wording from the cases you were shown; the guidance will be used on new instructions.
- The writer never sees responses. Do not instruct it to compare or inspect responses.
- Do not specify the output format; a fixed JSON contract is appended after your text.
- Keep it under 700 words.

Propose exactly the requested number of revisions, each taking a different approach and each a complete replacement for the current guidance. Output ONLY a JSON array:
[{"diagnosis": "<the failure pattern this revision targets, 1-2 sentences>", "guidance": "<the full revised guidance>"}]"""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--upstream", default="results/rubricbench_glm53/agentic-tools-distill_predistill.json")
    p.add_argument("--baseline", default="results/rubricbench_glm53/baseline_verdicts.jsonl")
    p.add_argument("--dev", default="results/rubricbench/split.json:dev")
    p.add_argument("--split", default="results/rubricbench_glm53/evolve/split.json",
                   help="reuse the train/val slices of the first evolution run")
    p.add_argument("--seeds", nargs="+", default=["results/rubricbench_glm53/evolve/prompts/r1c2.txt"],
                   help="guidance files to start from (contract suffix, if present, is stripped); "
                        "the v1 seed and the framed-with-checklist seed are always added")
    p.add_argument("--holdout", default="results/rubricbench/split.json:holdout")
    p.add_argument("--holdout-checklists", default="runs/rubricbench_holdout_inputs/agentic-tools-distill_predistill.json")
    p.add_argument("--holdout-baseline", default="runs/rubricbench_holdout_inputs/baseline_rubrics.json")
    p.add_argument("--holdout-log", default="results/rubricbench/opt/HOLDOUT_LOG.md")
    p.add_argument("--budget", type=int, default=3, help="holdout looks available")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--train-margin", type=float, default=0.02)
    p.add_argument("--val-margin", type=float, default=0.03)
    p.add_argument("--out", default="results/rubricbench_glm53/evolve_loop")
    p.add_argument("--model", default="GLM-5.3-H20-t2-copy")
    p.add_argument("--concurrency", type=int, default=80)
    p.add_argument("--max-rounds", type=int, default=20)
    p.add_argument("--beam", type=int, default=2)
    p.add_argument("--per-member", type=int, default=2, help="proposals per beam member per round")
    p.add_argument("--n-fail", type=int, default=18)
    p.add_argument("--n-win", type=int, default=4)
    p.add_argument("--seed", type=int, default=20261001)
    p.add_argument("--resume", action="store_true",
                   help="rebuild the pool, val results and holdout looks from --out/log.jsonl and "
                        "continue after the last completed round")
    p.add_argument("--log-level", default="INFO")
    return p.parse_args()


def _strip_contract(text: str) -> str:
    return text.split(CONTRACT, 1)[0].rstrip() if CONTRACT in text else text.rstrip()


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p from the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


class Evaluator:
    def __init__(self, engine: LLMEngine):
        self.engine = engine
        self.reasons: dict[tuple[str, str], dict] = {}

    async def write(self, cases, checklists, guidance) -> dict[str, str]:
        system = guidance.rstrip() + CONTRACT

        async def one(case):
            user = build_distill_user(case.instruction, checklists.get(case.case_id, []), min_items=3, max_items=7)
            try:
                raw = await self.engine.chat_json(user, system=system, expect="array", max_tokens=12288,
                                                  tag="evolve:writer")
                crits = [c.to_criterion() for c in _parse_candidates(raw)[:7]]
                return case.case_id, RB.rubric_to_text(_Items(crits))
            except Exception as exc:  # noqa: BLE001
                logger.warning("writer failed on %s: %s", case.case_id, str(exc)[:120])
                return case.case_id, ""

        return dict(await asyncio.gather(*(one(c) for c in cases)))

    async def score(self, cases, rubrics, *, both_orders=False) -> dict[str, Any]:
        verdicts = await RB.judge_all(self.engine, cases, rubrics, both_orders=both_orders,
                                      max_tokens=8192, progress_every=10**9)
        right = {v.case_id: v.forward == "AB"[v.label] for v in verdicts}
        by_g: dict[str, list[int]] = defaultdict(list)
        for c in cases:
            by_g[RB.group_of(c.domain)].append(int(right[c.case_id]))
        return {"acc": sum(right.values()) / len(right), "right": right, "rubrics": rubrics,
                "by_group": {g: round(sum(v) / len(v), 4) for g, v in sorted(by_g.items())},
                "verdicts": verdicts}

    async def evaluate(self, cases, checklists, guidance, *, both_orders=False):
        return await self.score(cases, await self.write(cases, checklists, guidance), both_orders=both_orders)

    async def judge_detail(self, case, rubric_text) -> dict:
        no_rubric, with_rubric = RB.judge_systems("default")
        try:
            parsed = await self.engine.chat_json(
                RB.build_pair_prompt(case, rubric_text, swapped=False),
                system=with_rubric if rubric_text.strip() else no_rubric,
                expect="object", max_tokens=8192, tag="rbench:judge", cache_salt=None)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:  # noqa: BLE001
            return {}


class _Items:
    def __init__(self, items):
        self.items = items


async def build_dossier(ev, cases_by_id, train_ids, res, base_right, args, tag) -> str:
    rng = random.Random(f"{args.seed}:loop:{tag}")
    fails = [c for c in train_ids if not res["right"][c]]
    lost = [c for c in fails if base_right.get(c)]
    other = [c for c in fails if not base_right.get(c)]
    wins = [c for c in train_ids if res["right"][c] and not base_right.get(c)]
    n_lost = args.n_fail * 2 // 3
    picks = ([(c, "FAILURE") for c in _stratified(lost, cases_by_id, rng, n_lost)]
             + [(c, "FAILURE") for c in _stratified(other, cases_by_id, rng, args.n_fail - n_lost)]
             + [(c, "KEEP (writer right, simpler method wrong)") for c in _stratified(wins, cases_by_id, rng, args.n_win)])

    async def render(cid, kind):
        c = cases_by_id[cid]
        pref_slot = "A" if c.label == 0 else "B"
        rubric = res["rubrics"].get(cid, "")
        detail = await ev.judge_detail(c, rubric)
        marks = {}
        for e in detail.get("per_criterion") or []:
            if not isinstance(e, dict):
                continue
            try:
                marks[int(e.get("id"))] = (bool(e.get("a")), bool(e.get("b")))
            except (TypeError, ValueError):
                continue
        lines = []
        for i, line in enumerate([x for x in rubric.splitlines() if x.strip()][:8], start=1):
            a, b = marks.get(i, (None, None))
            tick = "" if a is None else f"  [A:{'yes' if a else 'no'} B:{'yes' if b else 'no'}]"
            lines.append(_clip(line, 200) + tick)
        expert = "\n".join(_clip(x, 200) for x in c.expert_rubrics.splitlines()[:6] if x.strip())
        return (f"### {kind} | domain: {RB.group_of(c.domain)} | human preferred: {pref_slot}\n"
                f"INSTRUCTION: {_clip(c.instruction, 600)}\n"
                f"RESPONSE A ({len(c.response_a)} chars): {_clip(c.response_a, 420)}\n"
                f"RESPONSE B ({len(c.response_b)} chars): {_clip(c.response_b, 420)}\n"
                f"WRITER'S RUBRIC, with the judge's marks:\n" + ("\n".join(lines) or "(empty)") + "\n"
                f"JUDGE PICKED: {detail.get('winner', '?')} — {_clip(detail.get('why', ''), 280)}\n"
                f"EXPERT RUBRIC:\n{expert}")

    return "\n\n".join(await asyncio.gather(*(render(c, k) for c, k in picks)))


async def propose(engine, guidance, dossier, summary, k, salt) -> list[dict[str, str]]:
    user = (f"CURRENT GUIDANCE:\n<<<\n{guidance}\n>>>\n\nTRAINING-SET ACCURACY:\n{summary}\n\n"
            f"CASES:\n\n{dossier}\n\nPropose {k} revisions now.")
    try:
        raw = await engine.chat_json(user, system=OPTIMIZER_SYSTEM, expect="array", max_tokens=16384,
                                     tag="evolve:optimizer", cache_salt=salt)
    except Exception as exc:  # noqa: BLE001
        logger.warning("optimizer failed (%s): %s", salt, str(exc)[:160])
        return []
    out = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        g = str(item.get("guidance") or "").strip()
        if 200 <= len(g) <= 7000:
            out.append({"diagnosis": str(item.get("diagnosis") or "")[:400], "guidance": g})
    return out[:k]


async def main_async() -> int:
    args = parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    out = Path(args.out)
    (out / "prompts").mkdir(parents=True, exist_ok=True)
    log_path = out / "log.jsonl"

    def log(rec):
        rec["t"] = time.strftime("%F %T")
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    dev_ids = set(load_case_ids(args.dev))
    all_cases = {c.case_id: c for c in RB.load_cases()}
    split = json.loads(Path(args.split).read_text()) if Path(args.split).exists() else None
    if split is None:
        tr, va = split_dev([all_cases[i] for i in dev_ids], 2 / 3, args.seed)
        split = {"train": tr, "val": va}
    train = [all_cases[i] for i in split["train"]]
    val = [all_cases[i] for i in split["val"]]
    checklists = load_checklists(args.upstream)
    base_right = {}
    for line in Path(args.baseline).read_text().splitlines():
        if line.strip():
            v = json.loads(line)
            base_right[v["case_id"]] = v.get("forward") == "AB"[int(v["label"])]
    base_train = sum(base_right[c.case_id] for c in train) / len(train)
    base_val = sum(base_right[c.case_id] for c in val) / len(val)
    logger.info("train %d val %d | baseline train %.4f val %.4f", len(train), len(val), base_train, base_val)

    engine = LLMEngine(args.model, concurrency=args.concurrency, cache_dir="runs/cache")
    ev = Evaluator(engine)

    pool: dict[str, dict[str, Any]] = {}
    looks_used = 0
    holdout_baseline = None
    sent: set[str] = set()
    val_cache: dict[str, float] = {}
    start_round = 1

    if args.resume and log_path.exists():
        # Every evaluation is cached, so only the members that will be expanded
        # need their full results back; the rest only compete on train accuracy.
        rows = [json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        for r in rows:
            if r.get("event") in ("seed", "candidate"):
                guidance = _strip_contract((out / "prompts" / f"{r['name']}.txt").read_text(encoding="utf-8"))
                pool[r["name"]] = {"guidance": guidance, "train": {"acc": r["train"]}, "parent": r.get("parent")}
            elif r.get("event") == "val":
                val_cache[r["name"]] = r["val"]
            elif r.get("event") == "holdout":
                sent.add(r["name"])
                looks_used = max(looks_used, int(r["looks_used"]))
        start_round = 1 + max((r["round"] for r in rows if r.get("event") == "candidate"), default=0)
        top = sorted(pool, key=lambda n: pool[n]["train"]["acc"], reverse=True)[: args.beam]
        for name in top:
            pool[name]["train"] = await ev.evaluate(train, checklists, pool[name]["guidance"])
        logger.info("resumed: %d in pool, %d val results, %d holdout looks; starting at round %d; beam %s",
                    len(pool), len(val_cache), looks_used, start_round,
                    ", ".join(f"{n}={pool[n]['train']['acc']:.4f}" for n in top))
        log({"event": "resume", "start_round": start_round, "pool": len(pool)})
    else:
        seeds = {"seed_v1": SEED_GUIDANCE, "seed_framed": SEED_FRAMED}
        for path in args.seeds:
            seeds[Path(path).stem] = _strip_contract(Path(path).read_text(encoding="utf-8"))
        for name, guidance in seeds.items():
            (out / "prompts" / f"{name}.txt").write_text(guidance + CONTRACT, encoding="utf-8")
            res = await ev.evaluate(train, checklists, guidance)
            pool[name] = {"guidance": guidance, "train": res}
            logger.info("seed %-14s train %.4f", name, res["acc"])
            log({"event": "seed", "name": name, "train": res["acc"], "by_group": res["by_group"]})

    async def val_acc(name):
        if name not in val_cache:
            r = await ev.evaluate(val, checklists, pool[name]["guidance"])
            val_cache[name] = r["acc"]
            log({"event": "val", "name": name, "val": r["acc"], "by_group": r["by_group"]})
        return val_cache[name]

    async def holdout_check(name) -> bool:
        nonlocal looks_used, holdout_baseline
        for _ in range(720):          # wait up to 12h for the holdout inputs
            if Path(args.holdout_checklists).exists() and Path(args.holdout_baseline).exists():
                break
            logger.info("holdout inputs not ready; waiting")
            await asyncio.sleep(60)
        else:
            logger.error("holdout inputs never appeared; skipping the check")
            return False
        looks_used += 1
        thr = args.alpha / args.budget
        tr, va = pool[name]["train"]["acc"], val_cache[name]
        entry = (f"\n## Spend {looks_used} of {args.budget} ({time.strftime('%F %T')}): `{name}` from the evolution loop\n\n"
                 f"Logged before any holdout number was computed. Candidate writer guidance: "
                 f"`{out}/prompts/{name}.txt`. Dev: train {tr:.4f} (baseline {base_train:.4f}), "
                 f"val {va:.4f} (baseline {base_val:.4f}); it passed the loop's gate (train +{args.train_margin}, "
                 f"val +{args.val_margin}). Test: the writer over the holdout's frozen agentic-tools checklists "
                 f"(`{args.holdout_checklists}`) vs baseline (`{args.holdout_baseline}`), forward ACC, exact McNemar, "
                 f"two-sided, win only if Δ>0 and p < {thr:.4f} (alpha {args.alpha} / {args.budget} looks).\n")
        with Path(args.holdout_log).open("a", encoding="utf-8") as fh:
            fh.write(entry)
        hold_ids = set(load_case_ids(args.holdout))
        hold = [all_cases[i] for i in sorted(hold_ids)]
        hold_lists = load_checklists(args.holdout_checklists)
        if holdout_baseline is None:
            base_rubrics = {r["case_id"]: r["rubric"] for r in json.loads(Path(args.holdout_baseline).read_text())}
            holdout_baseline = await ev.score(hold, base_rubrics, both_orders=True)
        cand = await ev.evaluate(hold, hold_lists, pool[name]["guidance"], both_orders=True)
        b = sum(1 for c in hold if cand["right"][c.case_id] and not holdout_baseline["right"][c.case_id])
        cc = sum(1 for c in hold if holdout_baseline["right"][c.case_id] and not cand["right"][c.case_id])
        delta = cand["acc"] - holdout_baseline["acc"]
        p = mcnemar_exact(b, cc)
        win = delta > 0 and p < thr
        result = (f"Result: candidate {cand['acc']:.4f} vs baseline {holdout_baseline['acc']:.4f} "
                  f"(Δ {delta:+.4f}; only-candidate {b}, only-baseline {cc}; p={p:.4g}, threshold {thr:.4f}) "
                  f"— **{'WIN' if win else 'no win'}**. Used: {looks_used} of {args.budget}.\n")
        with Path(args.holdout_log).open("a", encoding="utf-8") as fh:
            fh.write("\n" + result)
        logger.info("HOLDOUT %s: %s", name, result.strip())
        log({"event": "holdout", "name": name, "cand": cand["acc"], "baseline": holdout_baseline["acc"],
             "delta": delta, "b": b, "c": cc, "p": p, "threshold": thr, "win": win, "looks_used": looks_used,
             "cand_by_group": cand["by_group"], "base_by_group": holdout_baseline["by_group"]})
        (out / f"holdout_{name}_rubrics.json").write_text(json.dumps(
            [{"case_id": k, "rubric": v} for k, v in cand["rubrics"].items()], ensure_ascii=False, indent=1))
        return win

    def beam():
        return sorted(pool, key=lambda n: pool[n]["train"]["acc"], reverse=True)[: args.beam]

    for rnd in range(start_round, args.max_rounds + 1):
        members = beam()
        # gate: any beam member that clears train and val margins goes to the holdout
        for name in members:
            if name in sent:
                continue
            if pool[name]["train"]["acc"] - base_train < args.train_margin:
                continue
            if await val_acc(name) - base_val < args.val_margin:
                continue
            sent.add(name)
            if await holdout_check(name):
                (out / "WINNER.txt").write_text(pool[name]["guidance"] + CONTRACT, encoding="utf-8")
                logger.info("stopping: %s beat baseline on the holdout", name)
                return 0
            if looks_used >= args.budget:
                logger.info("stopping: holdout budget used up without a win")
                return 0
        logger.info("round %d beam: %s", rnd, ", ".join(f"{n}={pool[n]['train']['acc']:.4f}" for n in members))
        new = []
        for mi, name in enumerate(members):
            res = pool[name]["train"]
            base_by_g: dict[str, list[int]] = defaultdict(list)
            for c in train:
                base_by_g[RB.group_of(c.domain)].append(int(base_right[c.case_id]))
            summary = "\n".join(f"- {g}: writer {res['by_group'][g]:.3f} vs simpler method {sum(v)/len(v):.3f} (n={len(v)})"
                                for g, v in sorted(base_by_g.items()))
            summary += f"\n- overall: writer {res['acc']:.3f} vs simpler method {base_train:.3f}"
            dossier = await build_dossier(ev, all_cases, split["train"], res, base_right, args, f"{rnd}:{name}")
            props = await propose(engine, pool[name]["guidance"], dossier, summary, args.per_member,
                                  f"loop:{rnd}:{name}")
            for k, prop in enumerate(props, start=1):
                cname = f"L{rnd}m{mi + 1}c{k}"
                (out / "prompts" / f"{cname}.txt").write_text(prop["guidance"] + CONTRACT, encoding="utf-8")
                r = await ev.evaluate(train, checklists, prop["guidance"])
                pool[cname] = {"guidance": prop["guidance"], "train": r, "parent": name}
                new.append(cname)
                logger.info("  %s (from %s) train %.4f | %s", cname, name, r["acc"], prop["diagnosis"][:110])
                log({"event": "candidate", "round": rnd, "name": cname, "parent": name, "train": r["acc"],
                     "by_group": r["by_group"], "diagnosis": prop["diagnosis"]})
        if not new:
            logger.info("round %d produced no usable proposals", rnd)

    logger.info("stopping: %d rounds without a holdout-confirmed win", args.max_rounds)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
