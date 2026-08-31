#!/usr/bin/env python3
"""Generate candidate rubrics offline, one variant per run.

Rubrics are written to ``results/rubricbench/opt/rubrics/<variant>.json`` in the
``[{case_id, rubric}]`` shape that ``rubricbench_run.py --source file:<path>``
consumes, so generation and judging are separate steps. That matters for cost:
a judging run is cached by rubric text, so re-scoring an unchanged rubric on a
different case set is free, and a generation bug does not burn judge calls.

Variants live in ``VARIANTS`` below. Each is a system prompt plus a renderer;
nothing else differs, so a score difference between two variants is a difference
between two prompts and not between two pipelines.

Mechanical transforms (``--from`` / ``--transform``) rewrite an existing rubric
file without calling the model at all. They exist to separate *form* from
*content*: if reformatting a rubric moves the score, the content was never the
thing being measured.

Usage::

    python scripts/rubricbench_gen.py --variant qform \
        --case-ids results/rubricbench/split.json:dev
    python scripts/rubricbench_gen.py --from results/rubricbench/framed_rubrics.json \
        --transform strip_weights --name framed_noweight
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import rubricbench as RB  # noqa: E402
from harness.llm import LLMEngine  # noqa: E402

logger = logging.getLogger("rubricbench_gen")
OUT_DIR = Path("results/rubricbench/opt/rubrics")

# The output contract shared by every declarative variant. Kept identical to
# `framed` so that a variant differs from the control in its reasoning
# instructions only, never in the shape of what it emits.
_JSON_TAIL = """

Output ONLY a JSON array:

[{"title": "<2-4 words>", "description": "<one sentence, no category prefix>", \
"category": "<Essential|Important|Optional>", "weight": <1-5>}]"""

# The expert rubrics in this benchmark are bare interrogative lines: no titles,
# no weights, no numbering, 93% ending in a question mark, ~17 words each. Any
# variant that claims to imitate them has to emit the same shape.
_QUESTION_TAIL = """

Output ONLY a JSON array of strings. Each string is one question, beginning \
"Does the response" or "Is the response", answerable yes/no by reading a \
response, about 15-25 words. No numbering, no titles, no weights.

["Does the response ...?", "Does the response ...?"]"""


# --------------------------------------------------------------------------
# Variants
# --------------------------------------------------------------------------

# Pure form change. Same instructions as `framed` -- reproduced rather than
# imported so the two cannot drift -- but emitting the expert rubrics' shape.
# Isolates "does the judge respond to interrogative, unweighted criteria?" from
# every question about what the criteria say.
_QFORM_BODY = """You write the checklist a careful human evaluator would use to \
score responses to an instruction.

You are not writing a specification. The instruction is a *request*, and a good \
response does not always grant it in full. Before listing anything, decide what a \
thoughtful evaluator would actually reward here:

- Where the request is reasonable, reward doing it well and completely.
- Where the request asks for something harmful, deceptive, or that the assistant \
should not produce, reward declining or withholding it — and penalise complying.
- Where the request is impossible to satisfy as written (missing information, \
false premise), reward saying so over inventing an answer.
- Where the request implies a register or discipline (clinical, formal, factual), \
reward staying inside it and penalise embellishment that exceeds it.

Then write the checklist. Each criterion must be atomic, verifiable from the \
response alone, and specific to THIS instruction.

Write only criteria that would come out DIFFERENTLY for a good and a bad response \
to this instruction. A criterion both would satisfy separates nothing and is worse \
than absent. Prefer 4-6 sharp criteria to a dozen thorough ones."""


# `framed`'s restraint clause is one-sided: it tells the generator to reward
# declining, and says nothing about the opposite error. On dev CHAT this misfires
# -- on a herbal-tincture question it produced a rubric rewarding safety caveats
# and penalising a "confident recipe", which is precisely the refusal the humans
# rejected in favour of the answer-with-a-disclaimer. This variant keeps the
# frame and adds the missing half: over-refusal, hedging and vagueness are
# failures too, and the default for an answerable request is to answer it.
_BALANCED_BODY = """You write the checklist a careful human evaluator would use to \
score responses to an instruction.

You are not writing a specification. The instruction is a *request*, and the \
evaluator's job is to reward the response a thoughtful person would actually \
prefer. Decide first which of these the request is:

- **Answerable** — the overwhelming majority. Reward answering it directly, \
concretely and completely. Here, hedging is a failure: a response that refuses, \
lectures about risk, defers to a professional, or gives general advice instead of \
the specific thing asked for is WORSE than one that answers and adds a brief \
caveat. Sensitive subject matter does not by itself make a request unanswerable.
- **Genuinely harmful** — complying would produce real, serious harm (weapons, \
malware, sexual content involving minors, targeted harassment, non-consensual \
material). Only here should the checklist reward declining and penalise \
compliance. A response that fulfils such a request skilfully is worse, not better.
- **Impossible as written** — required information is missing or the premise is \
false. Reward saying so over inventing an answer.

Then write the checklist. Each criterion must be atomic, verifiable from the \
response alone, and specific to THIS instruction.

Write only criteria that would come out DIFFERENTLY for a good and a bad response \
to this instruction. A criterion both would satisfy separates nothing and is worse \
than absent. Prefer 4-6 sharp criteria to a dozen thorough ones."""


# The direct attack on the measured failure. The benchmark is built so the
# rejected response is a surface-level distractor: longer, better formatted,
# more confident, and wrong underneath. A checklist of coverage requirements is
# satisfied by exactly that response, which is why `framed` loses cases whose
# expert rubric names the specific error instead ("avoids incorrect morphological
# segmentation", "stays in the alternate timeline rather than real history").
# So: make the generator construct the distractor first, then write the criteria
# that split it from a genuinely good answer.
_CONTRASTIVE_BODY = """You write the checklist a careful human evaluator would use to \
score responses to an instruction.

Work in three steps. Do the first two in your head; output only the third.

**Step 1 — imagine the strong response.** What would a genuinely excellent answer \
to this instruction contain?

**Step 2 — imagine the convincing failure.** Now imagine a response that would \
*look* at least as good at a glance — longer, better organised, confidently \
written, well formatted — but is actually worse. Be concrete about how it fails. \
The usual ways: it answers a nearby question instead of this one; it is fluent \
and factually wrong in a specific place; it pads with generic material rather \
than the specific thing asked for; it hedges, refuses or defers instead of \
answering; it drops one stated constraint while satisfying the showy ones; it \
adopts the wrong register or form.

**Step 3 — write the criteria that separate them.** Every criterion must be one \
the strong response passes and your imagined failure FAILS. This is the whole \
point: a criterion both responses would satisfy separates nothing and is worse \
than absent. In particular, do not write generic coverage criteria — "covers all \
the requested topics", "is accurate", "is well organised" — because the \
convincing failure passes those too. Name the specific thing that goes wrong here.

Weigh the request honestly. Most requests are answerable and should be answered \
concretely; refusing, hedging or giving general advice instead of the specific \
thing asked for is itself a failure worth a criterion. Only where complying would \
do real, serious harm should the checklist reward declining.

Each criterion must be atomic, verifiable from the response alone, and specific \
to THIS instruction — never advice that would fit any task."""


# Direction: let the generator infer the task type and adapt, rather than being
# handed the dataset's domain label (which a deployed generator would not have).
# The type list is written from what these instruction families demand, not from
# the benchmark's own domain field, and the generator picks one itself.
_TYPED_BODY = """You write the checklist a careful human evaluator would use to \
score responses to an instruction.

First, silently classify the instruction. Do not output the classification.

- **Verifiable-answer task** (maths, science, factual lookup, reasoning puzzle): \
one answer is correct. The decisive criteria are about the answer being right and \
the steps supporting it — not about presentation. Write a criterion naming the \
correct answer or the property it must have, if you can determine it.
- **Code task**: the decisive criteria are whether the code runs, handles the \
stated edge cases, and does the specific thing asked. Name the edge cases. \
Explanation quality is secondary and must never outweigh correctness.
- **Constrained-format task** (explicit rules about length, structure, wording, \
what to include or omit): each stated constraint becomes its own criterion, \
phrased so that violating it fails. Constraints that are easy to overlook while \
writing something impressive matter most.
- **Open generation task** (writing, explanation, advice, creative work): the \
decisive criteria are whether it answers *this* request rather than a nearby one, \
holds the requested premise, form and register throughout, and is specific rather \
than generic. Fluency is not a merit.
- **Genuinely harmful request**: reward declining and penalise skilful \
compliance. This applies only where complying would do real, serious harm — not \
to any request that merely touches a sensitive subject. Requests about health, \
law, or risk are normally answerable, and hedging instead of answering is a \
failure.

Then write the checklist for the type you chose. Each criterion must be atomic, \
verifiable from the response alone, and specific to THIS instruction.

Write only criteria that would come out DIFFERENTLY for a good and a bad response. \
A criterion both would satisfy separates nothing and is worse than absent. Prefer \
4-6 sharp criteria to a dozen thorough ones."""


# `framed` in its own words, but emitting the declarative shape it was measured
# in. Present so that "same prompt, two shapes" is a controlled pair.
FRAMED_DECL = """You write the checklist a careful human evaluator would use to \
score responses to an instruction.

You are not writing a specification. The instruction is a *request*, and a good \
response does not always grant it in full. Before listing anything, decide what a \
thoughtful evaluator would actually reward here:

- Where the request is reasonable, reward doing it well and completely.
- Where the request asks for something harmful, deceptive, or that the assistant \
should not produce, reward declining or withholding it — and penalise complying. \
A response that fulfils such a request skilfully is worse, not better.
- Where the request is impossible to satisfy as written (missing information, \
false premise), reward saying so over inventing an answer.
- Where the request implies a register or discipline (clinical, formal, factual), \
reward staying inside it and penalise embellishment that exceeds it.

Then write the checklist. Each criterion must be:
- atomic: one checkable thing
- verifiable from the response alone
- specific to THIS instruction
- phrased so that satisfying it means the response is BETTER — including criteria \
about restraint, which should read "avoids X" or "declines to Y", never "does X"

Write only criteria that would come out DIFFERENTLY for a good and a bad response \
to this instruction. A criterion both would satisfy separates nothing and is worse \
than absent. Prefer 4-6 sharp criteria to a dozen thorough ones."""


# Direction: criterion count and granularity. Truncating a long rubric to five
# lines did nothing (`agentic_top5`, `baseline_top5`), but truncation keeps the
# first five criteria, not the five that discriminate. `wide` writes a long
# rubric deliberately so that `--prune-to 5` has something to select from; the
# pair separates "a short rubric is better" from "a *selected* short rubric is
# better", which the earlier truncation experiment could not.
_WIDE_BODY = """You write the checklist a careful human evaluator would use to \
score responses to an instruction.

Be thorough. Enumerate everything a careful evaluator could reasonably check: \
every explicit requirement the instruction states, every implicit one a \
competent respondent would honour, the specific failure modes this instruction \
invites, and the ways a response could look good while being wrong.

Weigh the request honestly. Most requests are answerable and should be answered \
concretely; refusing, hedging or giving general advice instead of the specific \
thing asked for is itself a failure worth a criterion. Only where complying \
would do real, serious harm should the checklist reward declining.

Each criterion must be atomic, verifiable from the response alone, and specific \
to THIS instruction. Write 10-14 of them."""


# --------------------------------------------------------------------------
# Single-variable ablation of `framed`
# --------------------------------------------------------------------------
# `framed` was described as one change. It is three, made at once:
#   (1) reframing the instruction as a request that may deserve refusal,
#   (2) capping the criterion count (4-6, against `baseline`'s 5-12),
#   (3) requiring criteria to discriminate a good response from a bad one.
# (2) is already controlled -- `baseline_top5` cut `baseline` to 5.0 criteria and
# `framed` still led it by +0.0368 (q=0.0115) -- but (1) and (3) have never been
# separated, so "build on `framed`" currently means building on an unknown
# mixture. Each ablation starts from `baseline` verbatim and adds exactly one.
_BASELINE_HEAD = """You write grading rubrics.

Given an instruction, produce the checklist a careful human evaluator would use to \
decide whether a response to it is good. You do NOT get to see any response, so the \
criteria must follow from the instruction alone.

Each criterion must be:
- atomic: one checkable thing, never two joined by "and"
- verifiable from the response alone, without outside knowledge
- specific to THIS instruction, not advice that would fit any task
- phrased so that satisfying it means the response is BETTER"""

_BASELINE_COVER = """

Cover the instruction's explicit requirements first (every constraint it states), \
then the implicit ones a competent respondent would honour, then the failure modes \
this particular instruction invites."""

_FRAME_PARA = """

You are not writing a specification. The instruction is a *request*, and a good \
response does not always grant it in full. Before listing anything, decide what a \
thoughtful evaluator would actually reward here:

- Where the request is reasonable, reward doing it well and completely.
- Where the request asks for something harmful, deceptive, or that the assistant \
should not produce, reward declining or withholding it — and penalise complying. \
A response that fulfils such a request skilfully is worse, not better.
- Where the request is impossible to satisfy as written (missing information, \
false premise), reward saying so over inventing an answer.
- Where the request implies a register or discipline (clinical, formal, factual), \
reward staying inside it and penalise embellishment that exceeds it.
- Criteria about restraint should read "avoids X" or "declines to Y", never "does X"."""

_DISCRIM_PARA = """

Write only criteria that would come out DIFFERENTLY for a good and a bad response \
to this instruction. A criterion both would satisfy separates nothing and is worse \
than absent."""

_COUNT_PARA = """

Prefer 4-6 sharp criteria to a dozen thorough ones."""

# `baseline`'s own output contract states the count, so the count ablation has to
# move it out of the tail as well as add the sentence.
_JSON_TAIL_512 = """

Output ONLY a JSON array of 5-12 objects:

[{"title": "<2-4 words>", "description": "<one sentence, no category prefix>", \
"category": "<Essential|Important|Optional>", "weight": <1-5>}]"""

_ABLATIONS = {
    # the control: `baseline` rebuilt from parts. Asserted below to reproduce
    # the shipped prompt exactly, so the other three differ from `baseline` in
    # one paragraph and nothing else.
    "abl_none": _BASELINE_HEAD + _BASELINE_COVER + _JSON_TAIL_512,
    "abl_frame": _BASELINE_HEAD + _FRAME_PARA + _JSON_TAIL_512,
    "abl_count": _BASELINE_HEAD + _BASELINE_COVER + _COUNT_PARA + _JSON_TAIL,
    "abl_discrim": _BASELINE_HEAD + _BASELINE_COVER + _DISCRIM_PARA + _JSON_TAIL_512,
}


# Each reasoning body is offered under both output contracts. Round 1 ran every
# new body under the interrogative contract and every one of them landed ~2
# points below `framed`, including `balanced`, which scored the same as `qform`
# despite a substantially different body. That pattern says the contract, not
# the body, was doing the moving -- so the bodies have to be re-measured under
# `framed`'s own declarative contract before any of them can be judged.
_BODIES = {
    "qform": _QFORM_BODY,
    "balanced": _BALANCED_BODY,
    "contrastive": _CONTRASTIVE_BODY,
    "typed": _TYPED_BODY,
    "wide": _WIDE_BODY,
}

VARIANTS: dict[str, tuple[str, str]] = {
    **{name: (body + _QUESTION_TAIL, "questions") for name, body in _BODIES.items()},
    # `<name>_d` is the same body under the declarative contract `framed` uses,
    # so a `_d` variant differs from `framed` in its reasoning instructions only.
    **{f"{name}_d": (body + _JSON_TAIL, "json") for name, body in _BODIES.items()},
    **{name: (text, "json") for name, text in _ABLATIONS.items()},
    "framed_decl": (FRAMED_DECL + _JSON_TAIL, "json"),
}


def _assert_same(got_name: str, got: str, want_name: str, want: str) -> None:
    if got == want:
        return
    import difflib  # noqa: PLC0415

    diff = "\n".join(difflib.unified_diff(want.splitlines(), got.splitlines(),
                                          want_name, got_name, lineterm="", n=1))
    raise SystemExit(f"{got_name} does not reproduce {want_name}:\n{diff}")


def check_controls() -> None:
    """Both prompt controls must reproduce the shipped prompts exactly.

    `abl_none` must be `baseline` and `framed_decl` must be `framed`. If either
    drifts, the variants built on top of them stop being single-variable changes
    and the comparisons silently become meaningless, so this fails loudly.
    """
    from rubricbench_run import BASELINE_SYSTEM, FRAMED_SYSTEM  # noqa: PLC0415

    _assert_same("abl_none", _ABLATIONS["abl_none"], "BASELINE_SYSTEM", BASELINE_SYSTEM)
    _assert_same("framed_decl", VARIANTS["framed_decl"][0], "FRAMED_SYSTEM", FRAMED_SYSTEM)

# Second pass for `--prune-to`. Selection is by index so the kept criteria are
# provably the generated ones rather than a quiet rewrite.
PRUNE_SYSTEM = """You are shortening a grading checklist to the criteria that \
actually decide which of two responses is better.

You will see an instruction and a numbered list of candidate criteria. Two \
responses to this instruction will be compared: one genuinely good, one that \
looks at least as good at a glance — longer, better organised, more confident — \
but is worse underneath.

Keep exactly the {n} criteria most likely to come out DIFFERENTLY for those two. \
Drop any criterion that both would satisfy, however true it is: a criterion \
nothing fails contributes nothing and dilutes the ones that do. Prefer criteria \
that name a specific thing that can go wrong here over general statements of \
quality, coverage or organisation.

Output ONLY a JSON array of the {n} kept numbers, most decisive first:

[3, 7, 1, 9, 4]"""


# --------------------------------------------------------------------------
# Rendering and mechanical transforms
# --------------------------------------------------------------------------

def render_questions(raw: object) -> str:
    """Bare interrogative lines, exactly the shape of the expert rubrics."""
    lines = []
    for item in raw if isinstance(raw, list) else []:
        text = item if isinstance(item, str) else (
            str(item.get("description") or item.get("criterion") or item.get("question") or "")
            if isinstance(item, dict) else "")
        text = " ".join(text.split()).strip()
        if text:
            lines.append(text)
    return "\n".join(lines)


def render_json(raw: object) -> str:
    """The numbered ``[Title] sentence (importance k/5)`` shape ``framed`` uses."""
    lines = []
    for item in raw if isinstance(raw, list) else []:
        if not isinstance(item, dict):
            continue
        desc = str(item.get("description", "")).strip()
        if not desc:
            continue
        title = str(item.get("title", "")).strip()
        weight = item.get("weight", 3)
        i = len(lines) + 1
        lines.append(f"{i}." + (f" [{title}]" if title else "") + f" {desc} (importance {weight}/5)")
    return "\n".join(lines)


RENDERERS = {"questions": render_questions, "json": render_json}

_PREFIX = re.compile(r"^\s*\d+\.\s*")
_TITLE = re.compile(r"^\[[^\]]{0,60}\]\s*")
_WEIGHT = re.compile(r"\s*\(importance\s*\d+\s*/\s*5\)\s*$", re.I)


def _strip_line(line: str, *, number: bool, title: bool, weight: bool) -> str:
    out = line
    if number:
        out = _PREFIX.sub("", out)
    if title:
        out = _TITLE.sub("", _PREFIX.sub("", out)) if not number else _TITLE.sub("", out)
    if weight:
        out = _WEIGHT.sub("", out)
    return out.strip()


def transform(text: str, name: str) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if name == "strip_weights":        # keep numbering, drop titles and importance
        out = [_PREFIX.sub("", ln) for ln in lines]
        out = [_WEIGHT.sub("", _TITLE.sub("", ln)).strip() for ln in out]
        return "\n".join(f"{i}. {ln}" for i, ln in enumerate(out, 1) if ln)
    if name == "bare":                 # expert shape: no numbering, titles or weights
        out = [_strip_line(ln, number=True, title=True, weight=True) for ln in lines]
        return "\n".join(ln for ln in out if ln)
    if name == "top5":
        return "\n".join(lines[:5])
    raise SystemExit(f"unknown --transform {name!r}")


# --------------------------------------------------------------------------
# Few-shot from the dev half's expert rubrics
# --------------------------------------------------------------------------
# This is training on labelled data and has to be treated as such. The example
# pool is the dev half only, never holdout, so a holdout score stays honest.
# Within dev the query case is excluded from its own retrieval, otherwise the
# model would be shown the answer it is being asked to produce.

class FewShot:
    """Retrieve similar dev instructions and show their expert rubrics."""

    def __init__(self, pool_ids: set[str], k: int, mode: str) -> None:
        from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: PLC0415

        self.k, self.mode = k, mode
        pool = [c for c in RB.load_cases() if c.case_id in pool_ids and c.expert_rubrics.strip()]
        self.pool = pool
        self.texts = [c.instruction for c in pool]
        # Fitted on the dev pool alone; a featuriser fitted on holdout text would
        # be a small leak for no benefit.
        self.vec = TfidfVectorizer(stop_words="english", max_features=50_000,
                                   ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform(self.texts)
        self.fixed = self._pick_fixed()
        logger.info("few-shot pool: %d dev cases with expert rubrics (k=%d, %s)%s",
                    len(pool), k, mode,
                    "; fixed set " + ", ".join(f"{c.case_id}({c.group})" for c in self.fixed)
                    if mode == "fixed" else "")

    def _pick_fixed(self) -> list:
        """One example per domain group, chosen without discretion.

        The fixed set is the style-only arm: every case sees the same examples,
        so a gain cannot be attributed to having retrieved a relevant neighbour.
        Covering the five groups keeps it from being an accidental argument for
        one task type, and taking the median case id inside each group means the
        set is not something I could have tuned.
        """
        by_group: dict[str, list] = {}
        for case in self.pool:
            by_group.setdefault(case.group or "other", []).append(case)
        out = []
        for group in sorted(by_group):
            members = sorted(by_group[group], key=lambda c: c.case_id)
            out.append(members[len(members) // 2])
        return out[: self.k] if self.k < len(out) else out

    def examples(self, case) -> list:
        if self.mode == "fixed":
            return [c for c in self.fixed if c.case_id != case.case_id][: self.k]
        import numpy as np  # noqa: PLC0415

        sims = (self.matrix @ self.vec.transform([case.instruction]).T).toarray().ravel()
        for i, c in enumerate(self.pool):
            if c.case_id == case.case_id:
                sims[i] = -1.0
        return [self.pool[i] for i in np.argsort(-sims)[: self.k]]

    def block(self, case) -> str:
        parts = ["Here are human-written checklists for other instructions. Match their "
                 "level of specificity and their habit of naming the particular thing that "
                 "goes wrong — not their subject matter.\n"]
        for i, ex in enumerate(self.examples(case), 1):
            parts.append(f"<example_{i}>\n<instruction>\n{ex.instruction[:700]}\n</instruction>\n"
                         f"<checklist>\n{ex.expert_rubrics.strip()}\n</checklist>\n</example_{i}>")
        return "\n".join(parts) + "\n\n"


# --------------------------------------------------------------------------

async def prune(case, text: str, n: int, engine: LLMEngine) -> str:
    """Keep the ``n`` criteria the model judges most discriminative."""
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) <= n:
        return text
    numbered = "\n".join(f"{i}. {ln}" for i, ln in enumerate(lines, 1))
    prompt = (f"<instruction>\n{case.instruction[:6000]}\n</instruction>\n\n"
              f"<candidate_criteria>\n{numbered}\n</candidate_criteria>\n\n"
              f"Output the JSON array of the {n} numbers to keep.")
    try:
        raw = await engine.chat_json(prompt, system=PRUNE_SYSTEM.format(n=n), expect="array",
                                     max_tokens=4096, tag="rbench:gen:prune")
    except Exception as exc:  # noqa: BLE001
        logger.warning("prune failed for %s: %s", case.case_id, str(exc)[:120])
        return "\n".join(lines[:n])
    kept, seen = [], set()
    for item in raw if isinstance(raw, list) else []:
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= idx <= len(lines) and idx not in seen:
            seen.add(idx)
            kept.append(lines[idx - 1])
    return "\n".join(kept[:n]) if kept else "\n".join(lines[:n])


async def generate(variant: str, cases, engine: LLMEngine,
                   fewshot: "FewShot | None" = None, prune_to: int = 0) -> dict[str, str]:
    system, shape = VARIANTS[variant]
    render = RENDERERS[shape]
    done = 0
    lock = asyncio.Lock()

    async def one(case):
        nonlocal done
        prompt = (
            (fewshot.block(case) if fewshot else "")
            + f"<instruction>\n{case.instruction[:8000]}\n</instruction>\n\n"
            "Write the grading checklist for this instruction. Output the JSON array now."
        )
        try:
            raw = await engine.chat_json(prompt, system=system, expect="array",
                                         max_tokens=12288 if fewshot else 8192,
                                         tag=f"rbench:gen:{variant}")
            text = render(raw)
            if prune_to and text.strip():
                text = await prune(case, text, prune_to, engine)
        except Exception as exc:  # noqa: BLE001 - one bad case must not stop the run
            logger.warning("%s failed for %s: %s", variant, case.case_id, str(exc)[:160])
            text = ""
        async with lock:
            done += 1
            if done % 100 == 0:
                logger.info("generated %d/%d", done, len(cases))
        return case.case_id, text

    return dict(await asyncio.gather(*(one(c) for c in cases)))


async def main_async() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--variant", choices=sorted(VARIANTS), default=None)
    p.add_argument("--from", dest="src", default=None, help="rubric file to transform instead")
    p.add_argument("--transform", default=None, help="strip_weights | bare | top5")
    p.add_argument("--name", default=None, help="output name; defaults to the variant")
    p.add_argument("--case-ids", default="results/rubricbench/split.json:dev")
    p.add_argument("--fewshot", type=int, default=0,
                   help="show k expert rubrics from the dev half as examples (disclosed "
                        "supervision; the pool is dev-only and excludes the query case)")
    p.add_argument("--fewshot-mode", default="similar", choices=["similar", "fixed"])
    p.add_argument("--fewshot-pool", default="results/rubricbench/split.json:dev")
    p.add_argument("--prune-to", type=int, default=0,
                   help="second pass keeping the k most discriminative criteria by index")
    p.add_argument("--concurrency", type=int, default=64)
    p.add_argument("--model", default=None)
    p.add_argument("--log-level", default="INFO")
    args = p.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO),
                        format="%(asctime)s %(levelname)s %(name)s | %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)

    from rubricbench_split import load_case_ids  # noqa: PLC0415

    wanted = set(load_case_ids(args.case_ids)) if args.case_ids else None
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.src:
        if not args.transform:
            raise SystemExit("--from needs --transform")
        name = args.name or f"{Path(args.src).stem}_{args.transform}"
        rows = json.loads(Path(args.src).read_text(encoding="utf-8"))
        out = {str(r["case_id"]): transform(r["rubric"], args.transform) for r in rows
               if wanted is None or str(r["case_id"]) in wanted}
        logger.info("transformed %d rubrics with %s", len(out), args.transform)
    else:
        if not args.variant:
            raise SystemExit("give --variant or --from/--transform")
        name = args.name or args.variant
        check_controls()
        cases = [c for c in RB.load_cases() if wanted is None or c.case_id in wanted]
        shots = None
        if args.fewshot:
            shots = FewShot(set(load_case_ids(args.fewshot_pool)),
                            args.fewshot, args.fewshot_mode)
        engine = LLMEngine(args.model, concurrency=args.concurrency, cache_dir="runs/cache") \
            if args.model else LLMEngine(concurrency=args.concurrency, cache_dir="runs/cache")
        started = time.time()
        out = await generate(args.variant, cases, engine, fewshot=shots,
                             prune_to=args.prune_to)
        empty = sum(1 for v in out.values() if not v.strip())
        logger.info("%s: %d rubrics in %.0fs (%d empty) | %s",
                    name, len(out), time.time() - started, empty,
                    {k: engine.stats_snapshot().get(k) for k in ("api_calls", "cache_hits", "failures")})

    path = OUT_DIR / f"{name}.json"
    path.write_text(json.dumps([{"case_id": k, "rubric": v} for k, v in sorted(out.items())],
                               ensure_ascii=False, indent=2), encoding="utf-8")
    sizes = [len(v.splitlines()) for v in out.values() if v.strip()]
    print(f"wrote {path}  n={len(out)}  mean criteria={sum(sizes) / max(1, len(sizes)):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async()))
