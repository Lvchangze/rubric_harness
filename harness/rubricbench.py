"""RubricBench adapter: rubric quality measured against human preference labels.

Every metric in this repository so far has been a local proxy. RubricBench is
not: 1,147 pairwise comparisons with human labels, where a rubric's value is
whether it helps a judge pick the response people actually preferred.

Two properties make it a better test of this project's claim than the RaR setup:

* **There is no reference answer.** The gold-side leakage that confounded
  protocol 1 (``results/pilot_v2/confound_audit.txt``) cannot occur here, because
  there is no gold text to fit. It also means the reference-dependent half of the
  agentic pipeline has to be switched off rather than merely audited.
* **The ground truth is human.** Not a synthetic degradation ladder, not an
  oracle derived from the same reference the rubric was written from.

Two controls this module adds, which the published leaderboard does not report
and without which no score on it is interpretable:

* ``none`` — judge the pair with no rubric at all. If this matches the
  rubric-guided systems, rubrics are not doing the work and the leaderboard is
  ranking noise.
* ``expert`` — judge with the dataset's own expert-annotated rubrics. This is
  the ceiling available to a perfect generator under this judge, and it bounds
  how much any rubric-quality improvement could possibly be worth.

Position bias is controlled by running each pair in both orders and only
counting a win when it survives the swap; the single-order score is also
reported because that is what the official baselines measure.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

from .llm import JSONParseError, LLMEngine

logger = logging.getLogger(__name__)

__all__ = [
    "BenchCase",
    "PairVerdict",
    "load_cases",
    "DOMAIN_GROUPS",
    "group_of",
    "judge_pair",
    "judge_all",
    "write_submission",
    "rubric_to_text",
]

BENCH_ROOT = Path(__file__).resolve().parent.parent / "rubricbench"

#: Copied from ``rubricbench/eval_submission.py`` so reporting here groups
#: domains exactly as the official evaluator does.
DOMAIN_GROUPS: dict[str, set[str]] = {
    "chat": {"general", "focus", "human-preference", "factuality", "helpful"},
    "if": {"precise if", "ifeval"},
    "stem": {"stem", "math", "mmlu-pro", "gpqa"},
    "code": {"mbpp", "code"},
    "safety": {"safety", "harmlessness"},
}

#: Responses are pasted in pairs; this keeps a case inside the judge's context.
MAX_RESPONSE_CHARS = 14_000
MAX_INSTRUCTION_CHARS = 8_000


def group_of(domain: str) -> str | None:
    domain = (domain or "").strip().lower()
    for group, members in DOMAIN_GROUPS.items():
        if domain in members:
            return group
    return None


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    return text[:head] + f"\n…[{len(text) - limit} chars omitted]…\n" + text[-(limit - head):]


@dataclass
class BenchCase:
    case_id: str
    instruction: str
    response_a: str
    response_b: str
    label: int                 # 0 = A preferred, 1 = B preferred
    domain: str
    source: str = ""
    expert_rubrics: str = ""   # newline-separated, human-annotated

    @property
    def group(self) -> str | None:
        return group_of(self.domain)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "BenchCase":
        return cls(
            case_id=str(raw["case_id"]),
            instruction=str(raw.get("instruction", "")),
            response_a=str(raw.get("response_a", "")),
            response_b=str(raw.get("response_b", "")),
            label=int(raw.get("label", 0)),
            domain=str(raw.get("domain", "")),
            source=str(raw.get("source", "")),
            expert_rubrics=str(raw.get("rubrics", "") or ""),
        )


def load_cases(
    path: str | Path | None = None,
    *,
    limit: int | None = None,
    domains: Sequence[str] | None = None,
    seed: int = 1234,
) -> list[BenchCase]:
    """Load the benchmark, optionally down-sampling reproducibly.

    Sampling is stratified by domain group so a subset keeps the benchmark's
    composition; the official score is a group-weighted average, so an
    unstratified subset would not be comparable to the published numbers.
    """
    path = Path(path) if path else BENCH_ROOT / "data" / "rubricbench_data.json"
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    cases = [BenchCase.from_dict(r) for r in raw if r.get("case_id")]
    if domains:
        wanted = {d.strip().lower() for d in domains}
        cases = [c for c in cases if c.domain.lower() in wanted or (c.group or "") in wanted]
    if limit is not None and limit < len(cases):
        by_group: dict[str, list[BenchCase]] = {}
        for case in cases:
            by_group.setdefault(case.group or "other", []).append(case)
        rng = random.Random(seed)
        picked: list[BenchCase] = []
        for group, members in sorted(by_group.items()):
            share = max(1, round(limit * len(members) / len(cases)))
            picked.extend(rng.sample(members, min(share, len(members))))
        rng.shuffle(picked)
        cases = sorted(picked[:limit], key=lambda c: c.case_id)
    return cases


def rubric_to_text(rubric: Any) -> str:
    """Render whatever a generator produced into checklist lines for the judge."""
    if rubric is None:
        return ""
    if isinstance(rubric, str):
        return rubric.strip()
    items = getattr(rubric, "items", None)
    if items is None:
        return str(rubric).strip()
    lines: list[str] = []
    for i, criterion in enumerate(items, start=1):
        title = getattr(criterion, "title", "") or ""
        desc = getattr(criterion, "description", "") or ""
        weight = abs(int(getattr(criterion, "weight", 3) or 3))
        head = f"{i}." + (f" [{title}]" if title else "")
        lines.append(f"{head} {desc} (importance {weight}/5)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Judging
# ---------------------------------------------------------------------------

_BASE_RULES = """You are comparing two assistant responses to the same instruction and \
deciding which one a careful human evaluator would prefer.

Judge substance, not surface. This benchmark is built specifically from pairs where \
the weaker response looks better at a glance: it may be longer, better formatted, \
more confident, or more polished while failing the instruction's actual requirements. \
Length, formatting and tone are not merits in themselves.

Weigh, in this order: whether the response does what was actually asked, whether it \
is correct, whether it is complete, and only then how well it is presented. A \
response that refuses or deflects when the task was answerable is a failure; a \
response that answers a different question than the one asked is a failure."""

NO_RUBRIC_SYSTEM = f"""{_BASE_RULES}

Output ONLY this JSON object:

{{"winner": "A" or "B", "why": "<one sentence>"}}"""

RUBRIC_SYSTEM = f"""{_BASE_RULES}

You are given a CHECKLIST of criteria for this instruction. Work through it: for each \
criterion decide whether each response satisfies it, then let the pattern of \
satisfied criteria decide the winner, weighting by the stated importance. If the \
checklist misses something decisive, say so in 'why' and still pick the better \
response — the checklist is an aid, not a cage.

Output ONLY this JSON object:

{{"per_criterion": [{{"id": 1, "a": true, "b": false}}], \
"winner": "A" or "B", "why": "<one sentence>"}}"""


def build_pair_prompt(
    case: BenchCase, rubric_text: str, *, swapped: bool
) -> str:
    """Render one comparison. ``swapped`` puts the dataset's B in slot A."""
    first, second = (
        (case.response_b, case.response_a) if swapped else (case.response_a, case.response_b)
    )
    parts = [
        f"<instruction>\n{_clip(case.instruction, MAX_INSTRUCTION_CHARS)}\n</instruction>",
        f"<response_A>\n{_clip(first, MAX_RESPONSE_CHARS)}\n</response_A>",
        f"<response_B>\n{_clip(second, MAX_RESPONSE_CHARS)}\n</response_B>",
    ]
    if rubric_text.strip():
        parts.append(f"<checklist>\n{rubric_text.strip()}\n</checklist>")
    parts.append("Which response is better? Output the JSON object now.")
    return "\n\n".join(parts)


@dataclass
class PairVerdict:
    """One case's outcome, in both presentation orders."""

    case_id: str
    label: int
    domain: str
    forward: str | None = None      # "A"/"B" as the dataset labels them
    swapped: str | None = None
    errors: list[str] = field(default_factory=list)

    @property
    def consistent(self) -> bool:
        return self.forward is not None and self.forward == self.swapped

    def prediction(self, mode: str = "forward") -> str | None:
        """``forward`` matches the official single-pass protocol.

        ``swap_consistent`` returns a verdict only when both orders agree, which
        removes position bias at the cost of leaving ties unresolved; the caller
        decides how to score those.
        """
        if mode == "forward":
            return self.forward
        if mode == "swapped":
            return self.swapped
        if mode == "swap_consistent":
            return self.forward if self.consistent else None
        raise ValueError(f"unknown mode {mode!r}")

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "label": self.label,
            "domain": self.domain,
            "forward": self.forward,
            "swapped": self.swapped,
            "consistent": self.consistent,
            "errors": self.errors,
        }


def _parse_winner(parsed: Any, *, swapped: bool) -> str | None:
    """Read the verdict and undo the swap so it is always in dataset terms."""
    if not isinstance(parsed, dict):
        return None
    raw = str(parsed.get("winner", "")).strip().upper()
    slot = "A" if raw.startswith("A") or raw == "[[A]]" else "B" if raw.startswith("B") or raw == "[[B]]" else None
    if slot is None:
        return None
    if not swapped:
        return slot
    return "B" if slot == "A" else "A"


async def judge_pair(
    engine: LLMEngine,
    case: BenchCase,
    rubric_text: str = "",
    *,
    both_orders: bool = True,
    max_tokens: int = 8192,
    tag: str = "rbench:judge",
) -> PairVerdict:
    """Judge one pair; never raises. A failed call leaves the slot ``None``."""
    verdict = PairVerdict(case_id=case.case_id, label=case.label, domain=case.domain)
    system = RUBRIC_SYSTEM if rubric_text.strip() else NO_RUBRIC_SYSTEM
    orders = [False, True] if both_orders else [False]

    async def one(swapped: bool) -> str | None:
        try:
            parsed = await engine.chat_json(
                build_pair_prompt(case, rubric_text, swapped=swapped),
                system=system,
                expect="object",
                max_tokens=max_tokens,
                tag=tag,
                cache_salt="swapped" if swapped else None,
            )
        except (JSONParseError, RuntimeError) as exc:
            verdict.errors.append(f"{'swapped' if swapped else 'forward'}: {str(exc)[:160]}")
            return None
        return _parse_winner(parsed, swapped=swapped)

    results = await asyncio.gather(*(one(s) for s in orders))
    verdict.forward = results[0]
    verdict.swapped = results[1] if len(results) > 1 else None
    return verdict


async def judge_all(
    engine: LLMEngine,
    cases: Sequence[BenchCase],
    rubrics: dict[str, str] | None = None,
    *,
    both_orders: bool = True,
    max_tokens: int = 8192,
    tag: str = "rbench:judge",
    progress_every: int = 100,
) -> list[PairVerdict]:
    """Judge every case. ``rubrics`` maps ``case_id`` to checklist text."""
    rubrics = rubrics or {}
    done = 0
    lock = asyncio.Lock()

    async def run(case: BenchCase) -> PairVerdict:
        nonlocal done
        out = await judge_pair(
            engine, case, rubrics.get(case.case_id, ""),
            both_orders=both_orders, max_tokens=max_tokens, tag=tag,
        )
        async with lock:
            done += 1
            if progress_every and done % progress_every == 0:
                logger.info("judged %d/%d cases", done, len(cases))
        return out

    return list(await asyncio.gather(*(run(c) for c in cases)))


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_LETTER = {"A": 0, "B": 1}


def score(verdicts: Iterable[PairVerdict], *, mode: str = "forward") -> dict[str, Any]:
    """Accuracy overall and per domain group, matching the official evaluator.

    The official score counts a missing or unparsable prediction as wrong, so it
    is an accuracy over *all* cases rather than over answered ones. Both are
    reported: coverage below 1.0 makes the difference matter.
    """
    verdicts = list(verdicts)
    by_group: dict[str, list[int]] = {}
    correct = answered = 0
    for v in verdicts:
        pred = v.prediction(mode)
        group = group_of(v.domain)
        hit = 0
        if pred is not None:
            answered += 1
            hit = int(_LETTER[pred] == v.label)
            correct += hit
        if group:
            by_group.setdefault(group, []).append(hit)
    total = len(verdicts)
    return {
        "mode": mode,
        "n": total,
        "answered": answered,
        "coverage": answered / total if total else 0.0,
        "acc": correct / total if total else 0.0,
        "acc_answered": correct / answered if answered else 0.0,
        "by_group": {g: sum(v) / len(v) for g, v in sorted(by_group.items())},
        "group_n": {g: len(v) for g, v in sorted(by_group.items())},
        "position_consistency": (
            sum(1 for v in verdicts if v.consistent) / total if total else 0.0
        ),
    }


def write_submission(verdicts: Iterable[PairVerdict], path: str | Path, *, mode: str = "forward") -> Path:
    """Write the official CSV so ``eval_submission.py`` can score it directly."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["case_id,prediction"]
    for v in verdicts:
        pred = v.prediction(mode)
        lines.append(f"{v.case_id},{pred if pred else ''}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
