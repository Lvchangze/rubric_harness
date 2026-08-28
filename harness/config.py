"""Typed experiment configuration, loadable from YAML or CLI flags."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

from .llm import DEFAULT_MODEL

logger = logging.getLogger(__name__)

__all__ = ["LLMConfig", "SampleConfig", "AgenticConfig", "EvalConfig", "RunConfig", "load_config"]

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _merge(out[key], value)
        else:
            out[key] = value
    return out


@dataclass
class LLMConfig:
    model: str = DEFAULT_MODEL
    concurrency: int = 32
    reasoning_effort: str = "high"
    max_tokens: int = 16384
    max_attempts: int = 4
    cache_dir: str = "runs/cache"
    request_timeout_s: int = 1200


@dataclass
class SampleConfig:
    domains: list[str] = field(default_factory=lambda: ["rar_science", "rar_medicine"])
    split: str = "val"
    n_per_domain: int = 20
    seed: int = 1234
    # 9.15% of RaR-Science rows repeat a question verbatim; sampling one twice
    # would double-count it in the paired statistics (forensics §10).
    deduplicate_questions: bool = True


@dataclass
class AgenticConfig:
    """Stage switches (for ablations) and per-stage knobs."""

    n_rollouts: int = 3
    enable_decompose: bool = True
    enable_rollouts: bool = True
    enable_reconcile: bool = True
    enable_pitfalls: bool = True
    enable_critic: bool = True          # Stage 6 — the validation loop
    enable_calibration: bool = True
    rollout_max_tokens: int = 12000
    # Stage 6 has two halves with very different leakage properties, and they
    # are switched separately so the confound can be measured rather than
    # argued about. The gold half judges candidates against
    # `example.reference_answer` — the same text the discriminative metric uses
    # as its positive — so any gain it produces is circular by construction.
    # The negative half judges against independently synthesised flawed
    # answers, which the evaluation never sees, so its gain is clean.
    # `use_gold_signal=False` disables *every* gold-derived edit (contradiction
    # drops and the retained-failure demotion included), not just the drop
    # below; gating only `drop_gold_failures` would leave two leak paths open.
    use_gold_signal: bool = True        # master switch for all gold-side edits
    use_negative_signal: bool = True    # master switch for all negative-side edits
    drop_gold_failures: bool = True     # discard criteria the gold answer fails
    demote_indiscriminate: bool = True  # downweight criteria no negative fails
    # The negative half normally mixes wrong rollouts with purpose-built flawed
    # answers. Synthetic negatives are clean but need not be *representative* of
    # how the policy actually fails, which is a live explanation for the
    # negative half underperforming. Setting this keeps only real failed
    # rollouts, so "filter with counterexamples" is tested with counterexamples
    # the model genuinely produced. Questions where every rollout was correct
    # then have no negative at all — recorded, not papered over.
    negatives_from_rollouts_only: bool = False
    max_rollout_negatives: int = 2
    # Stage 2 samples carefully and mostly succeeds, so on this corpus two
    # thirds of questions produce no failed rollout at all. Without mining,
    # `negatives_from_rollouts_only` would therefore be a no-op on most of the
    # set and could not test what it exists to test.
    mine_hard_negatives: bool = True
    target_min_items: int = 6
    target_max_items: int = 16
    # The gold-pass filter is asymmetric by construction: a positive criterion
    # ("must state X") can fail a terse reference answer, whereas a Pitfall
    # ("must not claim Y") passes it by default. Left alone the filter therefore
    # enriches for Pitfalls, which are satisfied by silence and would inflate the
    # score of empty and off-topic responses. Capping the share keeps the
    # category mix inside the range the shipped/baseline rubrics occupy, so the
    # comparison is not confounded by structure. Set before any metric was run.
    max_pitfall_fraction: float = 0.25


@dataclass
class EvalConfig:
    judge_model: str = DEFAULT_MODEL
    judge_max_tokens: int = 12288
    judge_reasoning_effort: str = "high"
    # `None` means "whatever the endpoint defaults to", which is what the pilot
    # ran under. That makes judge verdicts reproducible only while `runs/cache/`
    # survives, and the cache is gitignored — so a fresh clone re-samples the
    # judge and will not reproduce the numbers exactly. Setting this to 0.0
    # fixes that, at the cost of invalidating every cached judgement (the
    # temperature is part of the cache key), so it is left off for continuity
    # with the pilot and should be turned on for the next full run.
    judge_temperature: float | None = None
    shuffle_criteria: bool = True       # position-bias control
    shuffle_seed: int = 7
    # RaR Eq.(1) weighting. `paper_explicit` remaps categories to
    # {Essential 1.0, Important 0.7, Optional 0.3, Pitfall 0.9} and is what the
    # paper actually trains with, so it carries the main result; the other two
    # are robustness checks because the forensics showed the shipped numeric
    # weights are ~a deterministic function of the category (F5).
    aggregation: str = "paper_explicit"          # paper_explicit | raw_weight | uniform
    report_all_aggregations: bool = True
    # Pitfall polarity convention. `detected` is correct; the other modes force a
    # single blanket convention and exist to show the conclusion survives either
    # choice (forensics F6). `favourable` is applied to the comparator rubrics so
    # any agentic win cannot come from mis-scoring them.
    polarity_mode: str = "detected"              # detected | favourable | all_avoidance | all_failure
    polarity_sensitivity_modes: list[str] = field(
        default_factory=lambda: ["favourable", "all_avoidance", "all_failure"]
    )
    comparator_sources: list[str] = field(default_factory=lambda: ["shipped"])
    run_implicit_judge: bool = False             # RaR-IMPLICIT, paper's best variant
    count_controlled: bool = True                # re-run key metrics at equal item counts
    response_variants: list[str] = field(
        default_factory=lambda: [
            "gold",
            "missing_step",
            "numeric_error",
            "right_method_wrong_answer",
            "verbose_empty",
            "terse_correct",
            "off_topic",
        ]
    )
    metrics: list[str] = field(
        default_factory=lambda: [
            # zero-LLM, reuse the forensics detectors — run these first, they are
            # free and the shipped column already has an n=45k reference value
            "grounding",
            "lint",
            "adaptivity",
            # LLM-scored
            "discriminative",
            "coverage",
            "intrinsic",
            "transfer",
            "headtohead",
        ]
    )
    self_agreement_repeats: int = 2
    bootstrap_iters: int = 10000


@dataclass
class RunConfig:
    run_name: str = "pilot"
    runs_dir: str = "runs"
    results_dir: str = "results"
    generators: list[str] = field(default_factory=lambda: ["shipped", "baseline", "agentic"])
    llm: LLMConfig = field(default_factory=LLMConfig)
    sample: SampleConfig = field(default_factory=SampleConfig)
    agentic: AgenticConfig = field(default_factory=AgenticConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "RunConfig":
        raw = dict(raw or {})
        nested = {
            "llm": LLMConfig,
            "sample": SampleConfig,
            "agentic": AgenticConfig,
            "eval": EvalConfig,
        }
        kwargs: dict[str, Any] = {}
        known = {f.name for f in fields(cls)}
        # Unknown keys are still ignored — that is what keeps an older config
        # loadable — but they are announced. Silently dropping them turns a
        # typo'd or renamed flag into a run that looks configured and is not,
        # and the only symptom is a result that quietly used the default.
        for key in sorted(set(raw) - known):
            logger.warning("config: ignoring unknown top-level key %r", key)
        for key, value in raw.items():
            if key not in known:
                continue
            if key in nested and isinstance(value, dict):
                sub_known = {f.name for f in fields(nested[key])}
                for sub in sorted(set(value) - sub_known):
                    logger.warning("config: ignoring unknown key %r under %r", sub, key)
                kwargs[key] = nested[key](**{k: v for k, v in value.items() if k in sub_known})
            else:
                kwargs[key] = value
        return cls(**kwargs)


def load_config(path: str | Path | None, **overrides: Any) -> RunConfig:
    """Load YAML/JSON config, then apply dotted overrides (``llm.concurrency=8``)."""
    raw: dict[str, Any] = {}
    if path:
        text = Path(path).read_text(encoding="utf-8")
        if str(path).endswith((".yaml", ".yml")):
            import yaml  # noqa: PLC0415 - optional dependency, only for YAML configs

            raw = yaml.safe_load(text) or {}
        else:
            raw = json.loads(text)
    for dotted, value in overrides.items():
        if value is None:
            continue
        node = raw
        parts = dotted.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return RunConfig.from_dict(raw)
