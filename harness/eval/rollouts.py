"""Fresh model rollouts plus a rubric-blind correctness oracle.

Why this module exists
----------------------
The discriminative metric in ``discriminative.py`` scores a ladder built *from*
``reference_answer``, with the reference itself as the ``gold`` positive. That
makes it structurally unable to give a fair reading of any generation technique
that legitimately consults the reference — most importantly the agentic
generator's Stage 6 gold filter, which drops criteria the reference answer
cannot satisfy.

That filter does not leak in deployment. The real RaR-style pipeline is:

1. **Offline**: build a rubric from ``(question, reference_answer)``. The
   reference is a legitimate, available input at this point.
2. **Training**: score *model rollouts* with that rubric. The reference answer
   is never the thing being scored.

Step 2 is what this module reproduces. Positives and negatives are real
rollouts from the policy model, labelled by an oracle that compares each
rollout's final answer against the reference. The reference answer is therefore
an input to rubric *construction* and to the *oracle*, but never a scored
object — which severs the leak path while staying faithful to how a rubric is
actually used as a reward.

Leakage discipline enforced here
--------------------------------
- Evaluation rollouts are drawn under the ``eval-rollout`` cache-salt namespace
  (:data:`EVAL_ROLLOUT_SALT_NS`), which the agentic generator never uses. They
  are therefore guaranteed-fresh samples, not a replay of the Stage 2 rollouts
  that were fed into reconciliation and drafting. Reusing those would leak to
  ``agentic`` while leaving ``baseline``/``shipped`` unaffected.
- :func:`assert_disjoint_from_generator_rollouts` turns that guarantee into a
  checked postcondition rather than a comment.
- The oracle prompt never sees a rubric, so no rubric source can influence the
  labels every source is scored against.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..llm import LLMEngine
from ..schema import Example
from ..tracing import RunDir
from .responses import CandidateResponse, ResponseSet

logger = logging.getLogger(__name__)

#: Cache-salt namespace for evaluation rollouts. The agentic generator draws its
#: Stage 2 rollouts under "rollout{i}", so nothing here can collide with them.
EVAL_ROLLOUT_SALT_NS = "eval-rollout"

#: Questions between checkpoint writes of rollouts.jsonl.
CHECKPOINT_EVERY = 20

#: Sampling settings, chosen to spread quality rather than to maximise it. A
#: reward model is only useful when it can separate rollouts that differ, so a
#: set where every sample is correct carries no information.
#:
#: Quality is varied through the prompt and through temperature, not through
#: ``reasoning_effort``. The obvious design — high/medium/low effort — was tried
#: and abandoned after measurement: on this endpoint the lower settings are not
#: merely ineffective at reducing quality, they make generation *slower*
#: (medium 245s and low 316s against high's 34s on the same question) and drive
#: empty completions that burn a retry each. ``reasoning_effort`` is routed
#: through ``chat_template_kwargs`` and the served profile pins it to "high", so
#: the lower values are close to inert.
#:
#: What does work is instructing the model how to answer: ``terse`` forbids
#: showing any work and ``rushed`` forbids checking, both of which produce
#: genuinely weaker answers at a latency comparable to the careful settings.
#: Temperature is capped at 1.0. Above that the model tends to wander inside its
#: reasoning trace without ever emitting a reply, which costs several retries at
#: escalating token budgets and yields nothing — the failure is a stall, not a
#: worse answer, so it buys no quality spread.
#: Order matters. ``_draw_plan(k)`` takes the first ``k``, and screening runs at
#: k=4 to decide which questions are worth generating rubrics for. A prefix of
#: three careful settings would call a question "the policy always gets this
#: right" mostly because the screen never tried a weak setting, so the first four
#: are deliberately balanced 2 careful / 2 degraded. The full k=8 set is
#: unchanged, so runs drawn at k=8 before this reordering are unaffected.
ROLLOUT_SETTINGS: list[dict[str, Any]] = [
    {"name": "careful_t02", "reasoning_effort": "high", "temperature": 0.2},
    {"name": "terse_t07", "reasoning_effort": "high", "temperature": 0.7, "style": "terse"},
    {"name": "rushed_t09", "reasoning_effort": "high", "temperature": 0.9, "style": "rushed"},
    {"name": "careful_t07", "reasoning_effort": "high", "temperature": 0.7},
    {"name": "careful_t10", "reasoning_effort": "high", "temperature": 1.0},
    {"name": "terse_t10", "reasoning_effort": "high", "temperature": 1.0, "style": "terse"},
    {"name": "rushed_t05", "reasoning_effort": "high", "temperature": 0.5, "style": "rushed"},
    {"name": "careful_t05", "reasoning_effort": "high", "temperature": 0.5},
]

SOLVER_SYSTEM = """You are answering an exam question in {domain}.

Work the problem and give your answer. Be accurate and complete.

End your reply with a final line of exactly this form:

FINAL ANSWER: <your answer>

The final line must contain the answer itself (a number with units, an
expression, a chosen option, or a one-sentence conclusion) — not a description
of where the answer can be found."""

SOLVER_SYSTEM_TERSE = """You are answering an exam question in {domain}.

State your answer directly. Do NOT show your reasoning, derivation, or working
— give only the conclusion.

End your reply with a final line of exactly this form:

FINAL ANSWER: <your answer>"""

SOLVER_SYSTEM_RUSHED = """You are answering an exam question in {domain} under
severe time pressure.

Give your first-instinct answer. Do not double-check it, do not verify your
arithmetic, and do not reconsider your approach. Keep it to a few sentences.

End your reply with a final line of exactly this form:

FINAL ANSWER: <your answer>"""

SOLVER_SYSTEMS: dict[str, str] = {
    "": SOLVER_SYSTEM,
    "terse": SOLVER_SYSTEM_TERSE,
    "rushed": SOLVER_SYSTEM_RUSHED,
}

ORACLE_SYSTEM = """You are grading whether a candidate answer reached the same
conclusion as a reference answer.

You will be given a question, the reference answer, and a candidate answer.

Judge ONLY whether the candidate's final conclusion agrees with the reference's
final conclusion. Specifically:

- Ignore style, length, formatting, and whether the candidate showed its work.
- Ignore differences in wording, notation, or units that denote the same value
  (e.g. 0.5 and 1/2; 3.0 m and 300 cm; "option B" and "B").
- Accept ordinary rounding differences, but NOT a genuinely different value.
- If the question has several parts, the candidate is correct only if it gets
  every part that the reference states.
- If the candidate gives no usable conclusion, it is incorrect.

Return ONLY this JSON object:

{"reference_final": "<the reference's final answer, quoted briefly>",
 "candidate_final": "<the candidate's final answer, quoted briefly>",
 "correct": true or false,
 "confidence": "high" or "low",
 "why": "<one sentence>"}

Set confidence to "low" if the reference does not state a checkable conclusion,
or if you genuinely cannot tell whether they agree."""

_FINAL_RE = re.compile(r"final\s*answer\s*[:\-]\s*(.+)", re.IGNORECASE)


def extract_final_answer(text: str) -> str:
    """Pull the trailing ``FINAL ANSWER:`` line, if the model produced one."""
    matches = _FINAL_RE.findall(text or "")
    if not matches:
        return ""
    return " ".join(matches[-1].split())[:400]


def _clean(raw: Any) -> str:
    if isinstance(raw, dict):
        raw = raw.get("response", "")
    return str(raw or "").strip()


def _digest(text: str) -> str:
    """Stable fingerprint of a rollout's text, for the disjointness check."""
    return hashlib.sha1(" ".join((text or "").split()).encode("utf-8")).hexdigest()[:16]


@dataclass
class Rollout:
    """One sampled attempt at a question, with its oracle label."""

    uid: str
    rollout_id: str
    text: str
    setting: str
    final_answer: str = ""
    correct: bool | None = None
    oracle_confidence: str = ""
    oracle_why: str = ""
    oracle_agreement: bool | None = None
    n_chars: int = 0
    error: str | None = None

    @property
    def usable(self) -> bool:
        return bool(self.text) and self.correct is not None and not self.error

    def to_dict(self) -> dict[str, Any]:
        return {
            "uid": self.uid, "rollout_id": self.rollout_id, "text": self.text,
            "setting": self.setting, "final_answer": self.final_answer,
            "correct": self.correct, "oracle_confidence": self.oracle_confidence,
            "oracle_why": self.oracle_why, "oracle_agreement": self.oracle_agreement,
            "n_chars": self.n_chars, "error": self.error,
            "text_digest": _digest(self.text),
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Rollout":
        return cls(
            uid=str(raw["uid"]), rollout_id=str(raw["rollout_id"]),
            text=str(raw.get("text", "")), setting=str(raw.get("setting", "")),
            final_answer=str(raw.get("final_answer", "")),
            correct=raw.get("correct"),
            oracle_confidence=str(raw.get("oracle_confidence", "")),
            oracle_why=str(raw.get("oracle_why", "")),
            oracle_agreement=raw.get("oracle_agreement"),
            n_chars=int(raw.get("n_chars", 0) or 0),
            error=raw.get("error"),
        )


@dataclass
class RolloutSet:
    """All evaluation rollouts for one question. Shared across rubric sources."""

    uid: str
    domain: str = ""
    rollouts: list[Rollout] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> list[Rollout]:
        return [r for r in self.rollouts if r.usable]

    @property
    def n_correct(self) -> int:
        return sum(1 for r in self.usable if r.correct)

    @property
    def is_informative(self) -> bool:
        """True when the labels are mixed.

        A question whose rollouts are all correct or all incorrect cannot
        contribute to AUC, best-of-n, or any separation measure: there is no
        pair to order. Such questions are excluded from those metrics and
        counted separately rather than silently averaged in as 0.5 or NaN.
        """
        usable = self.usable
        return len(usable) >= 2 and 0 < self.n_correct < len(usable)

    def as_candidates(self) -> list[CandidateResponse]:
        """Adapt to the shape ``judge_rubric`` already consumes."""
        return [
            CandidateResponse(
                uid=self.uid, response_id=r.rollout_id, text=r.text,
                quality_level=1 if r.correct else 0,
                meta={"variant": "rollout", "setting": r.setting,
                      "correct": r.correct, "n_chars": r.n_chars},
            )
            for r in self.usable
        ]

    def to_dict(self) -> dict[str, Any]:
        return {"uid": self.uid, "domain": self.domain,
                "rollouts": [r.to_dict() for r in self.rollouts], "meta": dict(self.meta)}

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "RolloutSet":
        return cls(uid=str(raw["uid"]), domain=str(raw.get("domain", "")),
                   rollouts=[Rollout.from_dict(r) for r in raw.get("rollouts") or []],
                   meta=dict(raw.get("meta") or {}))


async def _sample_one(
    engine: LLMEngine, example: Example, setting: Mapping[str, Any], *, max_tokens: int
) -> Rollout:
    name = str(setting["name"])
    system = SOLVER_SYSTEMS[str(setting.get("style") or "")].format(
        domain=example.domain.replace("rar_", "")
    )
    rollout = Rollout(uid=example.uid, rollout_id=f"rollout::{name}", text="", setting=name)
    try:
        raw = await engine.chat(
            example.question,
            system=system,
            # A terse rollout must still be given room to think: on this
            # endpoint the reasoning trace draws from the same budget as the
            # reply, so a small cap does not produce a short answer, it produces
            # an empty one. Brevity is enforced by the prompt, not the budget.
            max_tokens=max_tokens,
            temperature=setting.get("temperature"),
            reasoning_effort=setting.get("reasoning_effort"),
            # Namespaced so an evaluation rollout can never collide with the
            # generator's Stage 2 samples, whatever settings coincide.
            cache_salt=f"{EVAL_ROLLOUT_SALT_NS}:{name}",
            tag="eval:rollout",
        )
    except Exception as exc:  # noqa: BLE001 - one bad sample must not kill the question
        rollout.error = f"{type(exc).__name__}: {exc}"
        return rollout
    text = _clean(raw)
    rollout.text = text
    rollout.n_chars = len(text)
    rollout.final_answer = extract_final_answer(text)
    if not text:
        rollout.error = "empty response"
    return rollout


async def _label_one(
    engine: LLMEngine, example: Example, rollout: Rollout, *, repeat: bool = False
) -> dict[str, Any]:
    """Oracle verdict for one rollout. Never sees any rubric."""
    user = (
        f"QUESTION:\n{example.question}\n\n"
        f"REFERENCE ANSWER:\n{example.reference_answer}\n\n"
        f"CANDIDATE ANSWER:\n{rollout.text}"
    )
    try:
        return await engine.chat_json(
            user, system=ORACLE_SYSTEM, max_tokens=2048, reasoning_effort="medium",
            cache_salt=f"oracle{'2' if repeat else ''}", tag="eval:oracle",
        ) or {}
    except Exception as exc:  # noqa: BLE001
        logger.debug("oracle failed for %s/%s: %s", example.uid, rollout.rollout_id, exc)
        return {}


def _draw_plan(k: int) -> list[dict[str, Any]]:
    """``k`` sampling settings, recycling the failure-prone ones past the list.

    Beyond the named settings, extra draws reuse ``terse`` and ``rushed`` under
    fresh salts. Extra draws exist to find a failure on a question the policy
    keeps getting right, so there is no point spending them on the careful
    settings that already succeeded.
    """
    plan = list(ROLLOUT_SETTINGS[:k])
    recycle = [s for s in ROLLOUT_SETTINGS if s.get("style")]
    round_index = 0
    while len(plan) < k:
        base = recycle[len(plan) % len(recycle)]
        round_index = len(plan) // max(len(recycle), 1)
        plan.append({**base, "name": f"{base['name']}_x{round_index}"})
    return plan


async def build_rollout_set(
    engine: LLMEngine,
    example: Example,
    *,
    k: int = 6,
    max_tokens: int = 8192,
    agreement_probe: bool = False,
) -> RolloutSet:
    """Sample ``k`` fresh rollouts for one question and label each one."""
    settings = _draw_plan(k)
    rollouts = list(
        await asyncio.gather(
            *(_sample_one(engine, example, s, max_tokens=max_tokens) for s in settings)
        )
    )

    labelled = [r for r in rollouts if r.text and not r.error]
    verdicts = await asyncio.gather(*(_label_one(engine, example, r) for r in labelled))
    for rollout, verdict in zip(labelled, verdicts):
        if not verdict:
            rollout.error = "oracle failed"
            continue
        correct = verdict.get("correct")
        if not isinstance(correct, bool):
            rollout.error = "oracle returned no boolean verdict"
            continue
        rollout.correct = correct
        rollout.oracle_confidence = str(verdict.get("confidence") or "")
        rollout.oracle_why = str(verdict.get("why") or "")[:300]

    if agreement_probe:
        repeats = await asyncio.gather(
            *(_label_one(engine, example, r, repeat=True) for r in labelled)
        )
        for rollout, verdict in zip(labelled, repeats):
            again = verdict.get("correct")
            if isinstance(again, bool) and rollout.correct is not None:
                rollout.oracle_agreement = again == rollout.correct

    rollout_set = RolloutSet(uid=example.uid, domain=example.domain, rollouts=rollouts)
    rollout_set.meta = {
        "k_requested": k,
        "settings": [s["name"] for s in settings],
        "n_usable": len(rollout_set.usable),
        "n_correct": rollout_set.n_correct,
        "informative": rollout_set.is_informative,
        "salt_namespace": EVAL_ROLLOUT_SALT_NS,
    }
    return rollout_set


async def build_all_rollout_sets(
    engine: LLMEngine,
    examples: Sequence[Example],
    *,
    k: int = 6,
    max_tokens: int = 8192,
    concurrency: int = 16,
    agreement_probe_fraction: float = 0.15,
    top_up: int = 0,
    run_dir: RunDir | None = None,
    reuse: bool = True,
) -> dict[str, RolloutSet]:
    """Build rollout sets for every example, resuming from disk when possible."""
    existing: dict[str, RolloutSet] = {}
    path = Path(run_dir.path) / "rollouts.jsonl" if run_dir is not None else None
    if reuse and path is not None and path.exists():
        existing = load_rollout_sets(path)
        # A set drawn at a smaller k is re-drawn at the larger one. The extra
        # samples are new, but the ones already taken come back from the LLM
        # cache, so widening k costs only the difference.
        existing = {
            uid: rs for uid, rs in existing.items() if rs.meta.get("k_requested", 0) >= k
        }
        if existing:
            logger.info("reusing %d rollout sets from %s", len(existing), path)

    todo = [ex for ex in examples if ex.uid not in existing]
    # A deterministic slice gets the oracle self-consistency probe, so the
    # reliability figure is reproducible rather than dependent on ordering.
    probe_every = max(1, int(1 / agreement_probe_fraction)) if agreement_probe_fraction else 0
    semaphore = asyncio.Semaphore(max(1, concurrency))

    out = dict(existing)
    target = Path(run_dir.path) / "rollouts.jsonl" if run_dir is not None else None
    done = 0
    started = time.monotonic()
    write_lock = asyncio.Lock()

    async def one(index: int, example: Example) -> None:
        nonlocal done
        async with semaphore:
            try:
                rollout_set = await build_rollout_set(
                    engine, example, k=k, max_tokens=max_tokens,
                    agreement_probe=bool(probe_every) and index % probe_every == 0,
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("rollout set failed for %s: %s", example.uid, exc)
                return
        async with write_lock:
            out[example.uid] = rollout_set
            done += 1
            # Checkpoint as we go. Sampling k rollouts for 200 questions takes
            # long enough that losing the lot to a late failure — or to a
            # deliberate restart at higher concurrency — is a real cost, and the
            # LLM cache only avoids re-paying for calls, not for the wait.
            if target is not None and done % CHECKPOINT_EVERY == 0:
                save_rollout_sets(out, target)
            if done % 10 == 0:
                rate = done / max(time.monotonic() - started, 1e-9)
                logger.info(
                    "rollouts %d/%d (%.2f questions/s, eta %.0f min)",
                    done, len(todo), rate, (len(todo) - done) / max(rate, 1e-9) / 60,
                )

    await asyncio.gather(*(one(i, ex) for i, ex in enumerate(todo)))

    if top_up > 0:
        out = await _top_up_uninformative(
            engine, examples, out, k=k, top_up=top_up, max_tokens=max_tokens,
            semaphore=semaphore, target=target,
        )

    if target is not None:
        save_rollout_sets(out, target)
    return out


async def _top_up_uninformative(
    engine: LLMEngine,
    examples: Sequence[Example],
    sets: dict[str, RolloutSet],
    *,
    k: int,
    top_up: int,
    max_tokens: int,
    semaphore: asyncio.Semaphore,
    target: Path | None,
) -> dict[str, RolloutSet]:
    """Draw extra samples only for questions whose labels are not yet mixed.

    A question the policy always gets right (or always wrong) contributes
    nothing to AUC, best-of-n, or separation — there is no pair to order — so it
    is dead weight in every headline metric. Spending the extra samples only on
    those questions converts dead weight into usable questions at a fraction of
    the cost of raising ``k`` everywhere, and leaves the already-informative
    questions untouched so their statistics do not shift underneath the
    comparison.
    """
    by_uid = {ex.uid: ex for ex in examples}
    todo = [
        uid for uid, rs in sets.items()
        if uid in by_uid and not rs.is_informative
    ]
    if not todo:
        return sets
    logger.info(
        "topping up %d questions whose labels are not mixed (k %d -> %d)",
        len(todo), k, k + top_up,
    )
    gained = 0

    async def one(uid: str) -> None:
        nonlocal gained
        async with semaphore:
            try:
                wider = await build_rollout_set(
                    engine, by_uid[uid], k=k + top_up, max_tokens=max_tokens
                )
            except Exception as exc:  # noqa: BLE001
                logger.warning("top-up failed for %s: %s", uid, exc)
                return
        wider.meta["topped_up"] = True
        if wider.is_informative:
            gained += 1
        sets[uid] = wider

    await asyncio.gather(*(one(uid) for uid in todo))
    logger.info("top-up recovered %d/%d questions into the informative set", gained, len(todo))
    if target is not None:
        save_rollout_sets(sets, target)
    return sets


def assert_disjoint_from_generator_rollouts(
    rollout_sets: Mapping[str, RolloutSet], run_dir: "str | Path | RunDir"
) -> dict[str, Any]:
    """Verify no evaluation rollout reappears from the agentic Stage 2 traces.

    The cache-salt namespace already makes a collision impossible, but the point
    of this check is that the guarantee is *verified from the artefacts* rather
    than asserted in prose. If it ever fails, ``agentic`` would be scored on text
    its own drafting stage had seen, which no other source enjoys.
    """
    base = Path(run_dir.path if isinstance(run_dir, RunDir) else run_dir)
    eval_digests = {
        _digest(r.text): (uid, r.rollout_id)
        for uid, rs in rollout_sets.items() for r in rs.rollouts if r.text
    }
    collisions: list[dict[str, str]] = []
    n_generator = 0
    for trace in sorted((base / "traces").glob("agentic*_*.json")):
        try:
            payload = json.loads(trace.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for stage in payload.get("stages") or []:
            if stage.get("stage") != "rollouts":
                continue
            for entry in (stage.get("output") or {}).get("rollouts") or []:
                text = entry.get("text") or ""
                if not text:
                    continue
                n_generator += 1
                hit = eval_digests.get(_digest(text))
                if hit:
                    collisions.append({"uid": hit[0], "rollout_id": hit[1],
                                       "trace": trace.name})
    report = {
        "n_eval_rollouts": len(eval_digests),
        "n_generator_rollouts_checked": n_generator,
        "n_collisions": len(collisions),
        "collisions": collisions[:10],
        "disjoint": not collisions,
    }
    if collisions:
        logger.error("EVAL/GENERATOR ROLLOUT COLLISION: %d overlapping texts", len(collisions))
    else:
        logger.info(
            "verified: %d eval rollouts disjoint from %d generator rollouts",
            len(eval_digests), n_generator,
        )
    return report


def save_rollout_sets(sets: Mapping[str, RolloutSet], path: "str | Path") -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as fh:
        for uid in sorted(sets):
            fh.write(json.dumps(sets[uid].to_dict(), ensure_ascii=False) + "\n")


def load_rollout_sets(path: "str | Path | RunDir") -> dict[str, RolloutSet]:
    base = Path(path.path if isinstance(path, RunDir) else path)
    if base.is_dir():
        base = base / "rollouts.jsonl"
    if not base.exists():
        return {}
    out: dict[str, RolloutSet] = {}
    for line in base.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rollout_set = RolloutSet.from_dict(json.loads(line))
        except (ValueError, KeyError):
            continue
        out[rollout_set.uid] = rollout_set
    return out


def oracle_report(sets: Mapping[str, RolloutSet]) -> dict[str, Any]:
    """Reliability and label-balance summary; read this before any metric."""
    agreements = [
        r.oracle_agreement for rs in sets.values() for r in rs.rollouts
        if r.oracle_agreement is not None
    ]
    usable = [r for rs in sets.values() for r in rs.usable]
    low_conf = [r for r in usable if r.oracle_confidence == "low"]
    no_final = [r for r in usable if not r.final_answer]
    informative = [rs for rs in sets.values() if rs.is_informative]
    all_right = [rs for rs in sets.values() if rs.usable and rs.n_correct == len(rs.usable)]
    all_wrong = [rs for rs in sets.values() if rs.usable and rs.n_correct == 0]
    per_question_rate = [
        rs.n_correct / len(rs.usable) for rs in sets.values() if rs.usable
    ]
    return {
        "n_questions": len(sets),
        "n_rollouts_total": sum(len(rs.rollouts) for rs in sets.values()),
        "n_rollouts_usable": len(usable),
        "rollout_correct_rate": (sum(1 for r in usable if r.correct) / len(usable)) if usable else 0.0,
        "oracle_self_agreement": (sum(1 for a in agreements if a) / len(agreements)) if agreements else None,
        "n_oracle_agreement_probed": len(agreements),
        "low_confidence_rate": (len(low_conf) / len(usable)) if usable else 0.0,
        "missing_final_answer_rate": (len(no_final) / len(usable)) if usable else 0.0,
        "n_informative_questions": len(informative),
        "n_all_correct_questions": len(all_right),
        "n_all_incorrect_questions": len(all_wrong),
        "informative_fraction": (len(informative) / len(sets)) if sets else 0.0,
        "mean_per_question_correct_rate": (
            sum(per_question_rate) / len(per_question_rate) if per_question_rate else 0.0
        ),
    }
