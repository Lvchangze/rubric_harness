"""Metric family 2a — query specificity of a rubric, with **zero LLM calls**.

This is the harness-side reimplementation of the anchor-based detector from the
forensics pass, generalised so it can be pointed at *any* rubric source rather
than only the shipped one. Because the detector is identical, the ``shipped``
column produced here is directly comparable to the published n=45,325-question /
n=386,675-criterion numbers in ``docs/01_data_forensics.md`` §4.4 / §4.6 / §6 /
§7.1 — which makes it a free correctness test of this module *and* a
representativeness test of whatever question sample the run drew.

**Import route: we import the forensics detectors, we do not reimplement them.**
``analysis/`` is not a package and its scripts use flat imports (``from common
import ...``), so both the repo root and ``analysis/`` are prepended to
``sys.path`` before importing ``analysis.common`` / ``analysis.rubric_grounding``
/ ``analysis.rubric_similarity`` / ``analysis.rubric_stats``. Every definition
below therefore comes from exactly the code that produced the published numbers:
:func:`analysis.rubric_grounding.instance_anchors`,
:func:`analysis.common.content_tokens`, :func:`analysis.common.ngrams`,
:data:`analysis.rubric_stats.SUBJECTIVE_TERMS`,
:data:`analysis.rubric_stats.STYLE_TERMS` and
:func:`analysis.rubric_similarity.normalise_title`.

Definitions
-----------
**Anchors.** The anchor set of a question is the set of *rare* content tokens
(corpus document frequency <= 1% of the domain's questions) appearing in its
``question`` or ``reference_answer``, plus the numbers appearing there (a number
counts when it has >= 2 characters or a value > 2, which drops the uninformative
"1"/"2"). Note that despite what §4.4's prose says, ``instance_anchors`` never
adds bare symbols; we keep the implementation, not the prose, so the numbers
match.

**Generic criterion.** A criterion is *generic* when its description contains no
anchor of its own question. Lower is better; this is the headline metric.

**Permutation control.** The same test against a *random other* question's
anchors. This is what proves the detector measures query-specificity rather than
generic vocabulary overlap: on the shipped corpus the real rate is 28.87%
(science) / 21.97% (medicine) while the permuted rate is 96.39% / 98.54%. If a
source's real and permuted rates are close, its "grounding" is an artefact.

**Rare-token corpus.** Document frequency is counted over the *full* domain
corpus — all of train+val+test loaded by :func:`analysis.common.load_domain`,
i.e. 22,917 science / 22,408 medicine questions — never over the evaluated
sample. Two reasons: (a) it reproduces the published cutoff exactly (df <= 229
science / 224 medicine), and (b) the vocabulary is then a property of the
*dataset*, so all three rubric sources are scored against one fixed anchor
vocabulary and the comparison between them is fair. The set is cached per domain.

Published shipped-corpus reference points (science / medicine)
-------------------------------------------------------------
=========================================== ================ ================
metric                                      science          medicine
=========================================== ================ ================
``generic_criterion_rate``                  0.2887           0.2197
``generic_rate_permuted``                    0.9639           0.9854
``weak_grounding_rate`` (<2 anchors)         0.5261           0.4353
``mean_anchors_per_criterion``               1.777            2.066
``weight_mass_on_generic_raw``               0.2456           0.1520
``subjective_or_style_rate``                 0.3960           0.2364
``weight_mass_on_subjective``                0.4018           0.2125
``recyclable_rate``                          0.1767           0.1160
``mean_trigram_overlap_with_reference``      0.017            0.1092
``criteria_with_ge50pct_trigram_overlap``    0.0043           0.0741
=========================================== ================ ================

They are also available programmatically in
:data:`SHIPPED_FULL_CORPUS_REFERENCE`, read at import time out of
``analysis/out/*.json`` rather than hard-coded, and
:func:`sample_vs_full_corpus` renders the sample-vs-corpus comparison table.

Caveat on ``recyclable_rate``
-----------------------------
"Recyclable" = generic **and** the normalised title is reused >= 10 times. Title
reuse is necessarily counted *within one rubric source, within one domain,
across the evaluated question set only*. At the pilot's n=80 questions/domain
(~600 criteria) a title has far less opportunity to reach 10 uses than in the
forensics' 22k questions / 172k criteria, so our ``recyclable_rate`` is a
**severe underestimate** and must not be compared to the published 17.67% /
11.60%. ``max_title_reuse``, ``title_reuse_ge2_rate`` and the summary's
``title_reuse_histogram`` are reported instead so boilerplate can still be
compared *between sources at equal n*, which is the comparison we actually need.
"""

from __future__ import annotations

import collections
import json
import logging
import math
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ..config import EvalConfig
from ..schema import Category, Criterion, Example, Rubric
from ..tracing import RunDir
from .stats import mean_ci

# ``analysis/`` holds the scripts that produced docs/01_data_forensics.md. It is
# a plain directory of flat-import scripts, so it needs to be importable both as
# ``analysis.<mod>`` (repo root on the path) and as ``<mod>`` (the directory
# itself on the path, for their internal ``from common import ...``).
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "analysis")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from analysis.common import (  # noqa: E402
    NUMBER_RE,
    content_tokens,
    load_domain,
    ngrams,
)
from analysis.rubric_grounding import instance_anchors  # noqa: E402
from analysis.rubric_similarity import normalise_title  # noqa: E402
from analysis.rubric_stats import STYLE_TERMS, SUBJECTIVE_TERMS  # noqa: E402

logger = logging.getLogger(__name__)

__all__ = [
    "CATEGORY_WEIGHT_REMAP",
    "GroundingResults",
    "MASS_PRODUCED_TITLE_MIN_REUSE",
    "RARE_TOKEN_DOCFREQ_FRACTION",
    "SHIPPED_FULL_CORPUS_REFERENCE",
    "TRIGRAM_COPY_THRESHOLD",
    "WEAK_GROUNDING_MIN_ANCHORS",
    "criterion_tokens",
    "domain_rare_tokens",
    "load_shipped_reference",
    "run_grounding",
    "sample_vs_full_corpus",
]

NAN = float("nan")

#: Document-frequency cutoff for "rare": <= this fraction of the domain's questions.
RARE_TOKEN_DOCFREQ_FRACTION = 0.01
#: A criterion with fewer than this many anchors is "weakly grounded" (§4.4).
WEAK_GROUNDING_MIN_ANCHORS = 2
#: Title reuse count at which a title counts as mass-produced (§4.1 / §4.6).
MASS_PRODUCED_TITLE_MIN_REUSE = 10
#: Trigram-overlap fraction above which a criterion is "copied from reference" (§7.1, F9).
TRIGRAM_COPY_THRESHOLD = 0.5
#: Fixed seed for the permutation control, matching the forensics run.
PERMUTATION_SEED = 0

#: The category -> weight remap RaR-EXPLICIT actually trains with (paper §3.1).
#: Reported alongside the raw numeric weights because the forensics showed the
#: raw weights are close to a deterministic function of the category (F5).
CATEGORY_WEIGHT_REMAP: dict[Category, float] = {
    Category.ESSENTIAL: 1.0,
    Category.IMPORTANT: 0.7,
    Category.OPTIONAL: 0.3,
    Category.PITFALL: 0.9,
}

_ANALYSIS_OUT = _REPO_ROOT / "analysis" / "out"

#: Per-question metrics that get a mean + bootstrap CI in :meth:`GroundingResults.summary`.
_SUMMARY_METRICS: tuple[str, ...] = (
    "generic_criterion_rate",
    "generic_rate_permuted",
    "specificity_lift",
    "weight_mass_on_generic_raw",
    "weight_mass_on_generic_paper",
    "mean_anchors_per_criterion",
    "weak_grounding_rate",
    "recyclable_rate",
    "mass_produced_title_rate",
    "max_title_reuse",
    "title_reuse_ge2_rate",
    "subjective_rate",
    "style_rate",
    "subjective_or_style_rate",
    "generic_and_subjective_rate",
    "weight_mass_on_subjective",
    "weight_mass_on_subjective_paper",
    "mean_trigram_overlap_with_reference",
    "criteria_with_ge50pct_trigram_overlap",
    "criteria_with_ge30pct_trigram_overlap",
    "mean_4gram_overlap_with_question",
    "n_criteria",
)

#: Criterion-level (pooled) metrics. The forensics numbers are pooled over
#: criteria, not averaged over questions, so these are the ones to compare.
_POOLED_METRICS: tuple[str, ...] = (
    "generic_criterion_rate",
    "generic_rate_permuted",
    "weak_grounding_rate",
    "mean_anchors_per_criterion",
    "subjective_rate",
    "style_rate",
    "subjective_or_style_rate",
    "generic_and_subjective_rate",
    "recyclable_rate",
    "mass_produced_title_rate",
    "mean_trigram_overlap_with_reference",
    "criteria_with_ge50pct_trigram_overlap",
    "criteria_with_ge30pct_trigram_overlap",
    "weight_mass_on_generic_raw",
    "weight_mass_on_generic_paper",
    "weight_mass_on_subjective",
    "weight_mass_on_subjective_paper",
)


# ---------------------------------------------------------------------------
# Published shipped-corpus reference values
# ---------------------------------------------------------------------------


def load_shipped_reference() -> dict[str, dict[str, float]]:
    """Forensics full-corpus (n=45,325 question) values, per domain.

    Read out of ``analysis/out/grounding_stats.json``,
    ``analysis/out/composite_stats.json``, ``analysis/out/basic_stats.json`` and
    ``analysis/out/similarity_stats.json`` rather than transcribed, so the
    reference column cannot silently drift from the forensics run. Percentages
    are converted to fractions so they are directly comparable to the metrics
    this module emits. Missing files degrade to an empty dict.

    Includes the ``lint``/``adaptivity`` reference points too (weight-order
    violations, Pitfall mirrors, item-count degeneracy), so the three zero-LLM
    modules share one source of truth.
    """
    blobs: dict[str, Any] = {}
    for name in ("grounding_stats", "composite_stats", "basic_stats", "similarity_stats"):
        path = _ANALYSIS_OUT / f"{name}.json"
        try:
            blobs[name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("cannot read forensics reference %s: %s", path, exc)
            blobs[name] = {}

    def frac(value: Any) -> float:
        return float(value) / 100.0 if isinstance(value, (int, float)) else NAN

    def num(value: Any) -> float:
        return float(value) if isinstance(value, (int, float)) else NAN

    domains = sorted(
        set(blobs.get("grounding_stats", {})) | set(blobs.get("composite_stats", {}))
    )
    out: dict[str, dict[str, float]] = {}
    for domain in domains:
        ground = (blobs["grounding_stats"].get(domain) or {}) if blobs["grounding_stats"] else {}
        anchor = ground.get("anchor_grounding") or {}
        coupling = ground.get("reference_coupling") or {}
        subparts = ground.get("subquestion_coverage") or {}
        comp = (blobs["composite_stats"].get(domain) or {}) if blobs["composite_stats"] else {}
        reward = comp.get("reward_share_of_generic_criteria") or {}
        order = comp.get("weight_order_violations") or {}
        basic = (blobs["basic_stats"].get(domain) or {}) if blobs["basic_stats"] else {}
        verif = basic.get("verifiability") or {}
        degeneracy = basic.get("item_count_degeneracy") or {}
        items = basic.get("items_per_question") or {}
        weights = basic.get("weight_informativeness") or {}
        simulation = weights.get("simulation") or {}
        negative = weights.get("negative_weight_pathology") or {}
        compliance = basic.get("prompt_constraint_compliance") or {}
        sim = (blobs["similarity_stats"].get(domain) or {}) if blobs["similarity_stats"] else {}
        mirrors = ((sim.get("within_question_redundancy") or {}).get(
            "pitfall_mirrors_a_positive_criterion"
        ) or {})

        out[domain] = {
            "n_questions": num(basic.get("n_questions")),
            "n_criteria": num(basic.get("n_criteria")),
            # --- grounding -------------------------------------------------
            "generic_criterion_rate": frac(anchor.get("generic_criterion_rate_pct")),
            "generic_rate_permuted": frac(anchor.get("generic_rate_permutation_control_pct")),
            "weak_grounding_rate": frac(anchor.get("strictly_generic_rate_ge2_anchors_pct")),
            "mean_anchors_per_criterion": num(anchor.get("mean_anchors_per_criterion")),
            "weight_mass_on_generic_raw": frac(
                reward.get("pct_of_total_absolute_weight_on_generic_criteria")
            ),
            "weight_mass_on_subjective": frac(
                reward.get("pct_of_total_absolute_weight_on_subjective_or_style_criteria")
            ),
            "subjective_rate": frac(verif.get("subjective_term_rate_pct")),
            "style_rate": frac(verif.get("style_or_presentation_rate_pct")),
            "subjective_or_style_rate": frac(verif.get("subjective_or_style_rate_pct")),
            "generic_and_subjective_rate": frac(comp.get("generic_and_subjective_rate_pct")),
            "recyclable_rate": frac(comp.get("recyclable_criterion_rate_pct")),
            "mass_produced_title_rate": frac(comp.get("mass_produced_title_rate_pct")),
            "mean_trigram_overlap_with_reference": num(
                coupling.get("mean_trigram_overlap_with_reference")
            ),
            "criteria_with_ge50pct_trigram_overlap": frac(
                coupling.get("criteria_with_ge50pct_trigram_overlap_pct")
            ),
            "criteria_with_ge30pct_trigram_overlap": frac(
                coupling.get("criteria_with_ge30pct_trigram_overlap_pct")
            ),
            "mean_4gram_overlap_with_question": num(
                coupling.get("mean_4gram_overlap_with_question")
            ),
            # --- lint ------------------------------------------------------
            "non_self_contained_rate": frac(
                verif.get("non_self_contained_correctness_rate_pct")
            ),
            "weight_order_violation_rate": frac(
                order.get("questions_with_at_least_one_violation_pct")
            ),
            "pitfall_mirror_rate": frac(mirrors.get("pct_questions_with_a_mirror_ge_0.6")),
            "negative_weight_rate": frac(
                negative.get("criteria_with_negative_weight_pct")
            ),
            "item_count_in_range_rate": frac(compliance.get("items_per_question_in_7_20_pct")),
            "title_2_to_4_words_rate": frac(compliance.get("title_2_to_4_words_pct")),
            "description_single_sentence_rate": frac(
                compliance.get("description_single_sentence_pct")
            ),
            "weight_band_rate": frac(compliance.get("weight_in_prompt_recommended_band_pct")),
            "pitfall_opener_rate": frac(compliance.get("pitfall_opener_compliant_pct")),
            # --- adaptivity ------------------------------------------------
            "n_items": num(items.get("mean")),
            "items_cv": num(degeneracy.get("count_coefficient_of_variation")),
            "modal_count_share": frac(degeneracy.get("modal_count_share_pct")),
            "r_items_vs_question_len": num(degeneracy.get("pearson_count_vs_question_words")),
            "r_items_vs_reference_len": num(degeneracy.get("pearson_count_vs_reference_words")),
            "weight_corr_with_uniform": num(simulation.get("pearson_paperremap_vs_uniform")),
            "orphan_subpart_rate": frac(subparts.get("orphan_subpart_rate_pct")),
            "subpart_question_rate": frac(subparts.get("questions_with_detected_subparts_pct")),
            "mcq_rate": frac(subparts.get("questions_that_are_multiple_choice_pct")),
        }
    return out


#: Forensics full-corpus values keyed by domain, for the n=45k reference column.
SHIPPED_FULL_CORPUS_REFERENCE: dict[str, dict[str, float]] = load_shipped_reference()


# ---------------------------------------------------------------------------
# Corpus vocabulary
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainVocabulary:
    """The rare-token anchor vocabulary of one domain's full corpus."""

    domain: str
    rare_tokens: frozenset[str]
    docfreq_cutoff: int
    n_questions: int
    ok: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "n_rare_tokens": len(self.rare_tokens),
            "docfreq_cutoff": self.docfreq_cutoff,
            "n_corpus_questions": self.n_questions,
            "loaded": self.ok,
        }


@lru_cache(maxsize=8)
def domain_rare_tokens(domain: str) -> DomainVocabulary:
    """Rare-token vocabulary of ``domain``, over train+val+test.

    Cached, because loading the parquet and counting document frequency costs a
    couple of seconds and every rubric source needs the *same* vocabulary. A
    load failure is logged and degrades to an empty vocabulary (anchors then
    reduce to numbers only, which inflates the generic rate — the returned
    ``ok=False`` is surfaced in the summary so this cannot pass unnoticed).
    """
    try:
        frame = load_domain(domain)
    except Exception as exc:  # noqa: BLE001 - a metric must never raise
        logger.warning("cannot load domain corpus %s: %s", domain, exc)
        return DomainVocabulary(domain, frozenset(), 0, 0, ok=False)

    doc_freq: collections.Counter[str] = collections.Counter()
    for question, reference in zip(frame["question"], frame["reference_answer"]):
        doc_freq.update(set(content_tokens(f"{question} {reference}")))
    cutoff = max(2, int(RARE_TOKEN_DOCFREQ_FRACTION * len(frame)))
    rare = frozenset(token for token, count in doc_freq.items() if count <= cutoff)
    logger.info(
        "domain %s: %d questions, rare-token df cutoff %d, %d rare tokens",
        domain, len(frame), cutoff, len(rare),
    )
    return DomainVocabulary(domain, rare, cutoff, int(len(frame)))


def criterion_tokens(text: str) -> set[str]:
    """The token set an anchor test is run against — content tokens plus numbers.

    Identical to the forensics' criterion side of the anchor test.
    """
    lowered = (text or "").lower()
    return set(content_tokens(lowered)) | set(NUMBER_RE.findall(lowered))


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    lowered = (text or "").lower()
    return any(term in lowered for term in terms)


def _paper_weight(criterion: Criterion) -> float:
    return CATEGORY_WEIGHT_REMAP.get(criterion.category, 0.5)


def _overlap(grams: set[tuple[str, ...]], reference: set[tuple[str, ...]]) -> float:
    return len(grams & reference) / len(grams) if grams else 0.0


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class GroundingResults:
    """Per-question and per-criterion grounding rows plus a per-source summary."""

    per_question_rows: list[dict[str, Any]] = field(default_factory=list)
    per_criterion_rows: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """``{rubric_source: {metric: value}}`` plus underscore-prefixed metadata.

        For every metric ``m`` in the per-question rows the entry carries ``m``
        (mean over questions), ``m_ci_low`` / ``m_ci_high`` (bootstrap
        percentile CI) and ``m_n`` (contributing questions). Additionally:

        ``n_questions`` / ``n_criteria_total`` / ``n_errors``
            Rows for this source, criteria across them, rows carrying an error.
            (``n_criteria`` is the per-question mean, with a CI like any other
            metric.)
        ``pooled``
            The same metrics computed **criterion-level** over all criteria of
            the source (weight-mass metrics are pooled over total weight). These
            are the numbers to compare with the forensics, which pools rather
            than averaging per question.
        ``title_reuse_histogram``
            ``{reuse_count: n_criteria}`` — the honest replacement for
            ``recyclable_rate`` at small n (see the module docstring).

        Metadata keys, all underscore-prefixed so a source can never collide
        with them: ``_metric_family``, ``_params``, ``_by_domain``
        (``{domain: {source: {metric: pooled value}}}``) and
        ``_shipped_full_corpus_reference`` (the n=45k forensics column).
        """
        out: dict[str, Any] = {}
        for source in self._sources():
            rows = [r for r in self.per_question_rows if r.get("rubric_source") == source]
            crit = [r for r in self.per_criterion_rows if r.get("rubric_source") == source]
            entry: dict[str, Any] = {
                "n_questions": len(rows),
                # ``n_criteria`` itself is a per-question metric and picks up a
                # mean + CI below; the pooled total gets its own key.
                "n_criteria_total": len(crit),
                "n_errors": sum(1 for r in rows if r.get("error")),
            }
            for metric in _SUMMARY_METRICS:
                ci = mean_ci([r.get(metric) for r in rows], iters=2000, seed=0)
                entry[metric] = ci["mean"]
                entry[f"{metric}_ci_low"] = ci["ci_low"]
                entry[f"{metric}_ci_high"] = ci["ci_high"]
                entry[f"{metric}_n"] = ci["n"]
            entry["pooled"] = _pooled(crit)
            entry["title_reuse_histogram"] = {
                str(k): int(v)
                for k, v in sorted(
                    collections.Counter(
                        int(r.get("title_reuse") or 0) for r in crit
                    ).items()
                )
            }
            out[source] = entry

        by_domain: dict[str, dict[str, Any]] = {}
        for domain in sorted({str(r.get("domain")) for r in self.per_criterion_rows}):
            by_domain[domain] = {
                source: _pooled(
                    [
                        r
                        for r in self.per_criterion_rows
                        if r.get("rubric_source") == source and str(r.get("domain")) == domain
                    ]
                )
                for source in self._sources()
            }

        out["_metric_family"] = "grounding"
        out["_params"] = dict(self.params)
        out["_by_domain"] = by_domain
        out["_shipped_full_corpus_reference"] = SHIPPED_FULL_CORPUS_REFERENCE
        out["_n_errors"] = len(self.errors)
        return out

    def _sources(self) -> list[str]:
        seen: list[str] = []
        for row in self.per_question_rows:
            source = str(row.get("rubric_source"))
            if source not in seen:
                seen.append(source)
        return seen


def _pooled(criterion_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Criterion-level rates over a flat list of per-criterion rows."""
    n = len(criterion_rows)
    out: dict[str, Any] = {"n_criteria": n}
    if n == 0:
        return {**out, **{m: NAN for m in _POOLED_METRICS}}

    def rate(key: str) -> float:
        return float(np.mean([bool(r.get(key)) for r in criterion_rows]))

    def mass(flag: str, weight_key: str) -> float:
        weights = np.asarray([float(r.get(weight_key) or 0.0) for r in criterion_rows])
        total = float(np.abs(weights).sum())
        if total <= 0:
            return NAN
        flagged = np.asarray([bool(r.get(flag)) for r in criterion_rows])
        return float(np.abs(weights[flagged]).sum() / total)

    anchors = np.asarray([float(r.get("n_anchors") or 0.0) for r in criterion_rows])
    out.update(
        {
            "generic_criterion_rate": rate("generic"),
            "generic_rate_permuted": rate("generic_permuted"),
            "weak_grounding_rate": float(
                np.mean(anchors < WEAK_GROUNDING_MIN_ANCHORS)
            ),
            "mean_anchors_per_criterion": float(anchors.mean()),
            "subjective_rate": rate("subjective"),
            "style_rate": rate("style"),
            "subjective_or_style_rate": rate("subjective_or_style"),
            "generic_and_subjective_rate": rate("generic_and_subjective"),
            "recyclable_rate": rate("recyclable"),
            "mass_produced_title_rate": rate("mass_produced_title"),
            "mean_trigram_overlap_with_reference": float(
                np.mean([float(r.get("trigram_overlap_reference") or 0.0) for r in criterion_rows])
            ),
            "criteria_with_ge50pct_trigram_overlap": float(
                np.mean(
                    [
                        float(r.get("trigram_overlap_reference") or 0.0) >= TRIGRAM_COPY_THRESHOLD
                        for r in criterion_rows
                    ]
                )
            ),
            "criteria_with_ge30pct_trigram_overlap": float(
                np.mean(
                    [
                        float(r.get("trigram_overlap_reference") or 0.0) >= 0.3
                        for r in criterion_rows
                    ]
                )
            ),
            "weight_mass_on_generic_raw": mass("generic", "weight"),
            "weight_mass_on_generic_paper": mass("generic", "weight_paper"),
            "weight_mass_on_subjective": mass("subjective_or_style", "weight"),
            "weight_mass_on_subjective_paper": mass("subjective_or_style", "weight_paper"),
        }
    )
    return out


# ---------------------------------------------------------------------------
# The sample-vs-corpus sanity check
# ---------------------------------------------------------------------------


def sample_vs_full_corpus(
    results: GroundingResults, *, source: str = "shipped"
) -> list[dict[str, Any]]:
    """Compare one source's per-domain pooled metrics to the n=45k forensics column.

    The intended use is ``source="shipped"``: the shipped rubrics of the sample
    are the *same objects* the forensics measured, so a large divergence means
    either this detector disagrees with ``analysis/`` or the question sample is
    unrepresentative. ``ratio`` is sample / full corpus; ``1.0`` is perfect.
    ``recyclable_rate`` and ``mass_produced_title_rate`` are expected to be far
    below 1.0 by construction (see the module docstring) and are flagged
    ``comparable=False``.
    """
    small_n_sensitive = {"recyclable_rate", "mass_produced_title_rate"}
    by_domain = results.summary()["_by_domain"]
    rows: list[dict[str, Any]] = []
    for domain, per_source in sorted(by_domain.items()):
        sample = per_source.get(source) or {}
        reference = SHIPPED_FULL_CORPUS_REFERENCE.get(domain) or {}
        for metric in _POOLED_METRICS:
            got, want = sample.get(metric), reference.get(metric)
            if not _finite(got) or not _finite(want):
                continue
            rows.append(
                {
                    "domain": domain,
                    "metric": metric,
                    "sample": float(got),
                    "full_corpus": float(want),
                    "ratio": float(got) / float(want) if want else NAN,
                    "comparable": metric not in small_n_sensitive,
                }
            )
    return rows


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_grounding(
    engine: Any,  # unused: accepted for API uniformity across metric modules
    examples: Sequence[Example],
    rubrics_by_source: Mapping[str, Mapping[str, Rubric]],
    *,
    config: EvalConfig,
    run_dir: RunDir | None = None,
) -> GroundingResults:
    """Score every rubric of every source for query specificity. No LLM calls.

    ``engine`` and ``config`` are accepted only so the CLI can call every metric
    module through one signature; nothing here consumes either.
    """
    del engine, config  # zero-LLM metric; kept in the signature for uniformity

    results = GroundingResults(
        params={
            "sources": list(rubrics_by_source),
            "rare_token_docfreq_fraction": RARE_TOKEN_DOCFREQ_FRACTION,
            "weak_grounding_min_anchors": WEAK_GROUNDING_MIN_ANCHORS,
            "mass_produced_title_min_reuse": MASS_PRODUCED_TITLE_MIN_REUSE,
            "trigram_copy_threshold": TRIGRAM_COPY_THRESHOLD,
            "permutation_seed": PERMUTATION_SEED,
            "detector_source": "imported from analysis/ (not reimplemented)",
            "title_reuse_scope": "within (rubric_source, domain) across the evaluated sample",
        }
    )
    examples = [ex for ex in examples if ex is not None]
    if not examples:
        return results

    vocab = {ex.domain: domain_rare_tokens(ex.domain) for ex in examples}
    results.params["corpus"] = {d: v.to_dict() for d, v in vocab.items()}

    anchors: dict[str, set[str]] = {}
    grams: dict[str, tuple[set, set]] = {}
    for ex in examples:
        rare = vocab[ex.domain].rare_tokens
        try:
            anchors[ex.uid] = instance_anchors(ex.question, ex.reference_answer, rare)
        except Exception as exc:  # noqa: BLE001
            logger.warning("anchor extraction failed for %s: %s", ex.uid, exc)
            anchors[ex.uid] = set()
        ref_tokens = content_tokens(ex.reference_answer or "")
        grams[ex.uid] = (
            ngrams(ref_tokens, 3),
            ngrams(content_tokens(ex.question or ""), 4),
        )

    permuted = _permuted_anchor_map(examples, anchors)
    title_reuse = _title_reuse_counts(examples, rubrics_by_source)

    for ex in examples:
        for source, per_uid in rubrics_by_source.items():
            rubric = per_uid.get(ex.uid)
            if rubric is None:
                continue
            row, criterion_rows = _score_rubric(
                ex,
                source,
                rubric,
                own_anchors=anchors.get(ex.uid, set()),
                other_anchors=permuted.get(ex.uid, set()),
                reference_trigrams=grams[ex.uid][0],
                question_4grams=grams[ex.uid][1],
                title_reuse=title_reuse.get((source, ex.domain), {}),
            )
            results.per_question_rows.append(row)
            results.per_criterion_rows.extend(criterion_rows)
            if row.get("error"):
                results.errors.append(
                    {"uid": ex.uid, "rubric_source": source, "error": row["error"]}
                )

    _persist(results, run_dir)
    return results


def _permuted_anchor_map(
    examples: Sequence[Example], anchors: Mapping[str, set[str]]
) -> dict[str, set[str]]:
    """Map each uid to *another* question's anchor set, within the same domain.

    Staying inside the domain makes the control conservative: a cross-domain
    distractor would share almost no vocabulary and would trivially produce a
    ~100% generic rate. Self is excluded (the forensics, running at n=22k, did
    not bother; at n=80 a self-hit would be a visible 1.25% bias).
    """
    rng = np.random.default_rng(PERMUTATION_SEED)
    by_domain: dict[str, list[str]] = collections.defaultdict(list)
    for ex in examples:
        by_domain[ex.domain].append(ex.uid)

    out: dict[str, set[str]] = {}
    for uids in by_domain.values():
        uids = sorted(uids)
        for i, uid in enumerate(uids):
            if len(uids) < 2:
                out[uid] = set()
                continue
            j = int(rng.integers(len(uids) - 1))
            if j >= i:
                j += 1
            out[uid] = anchors.get(uids[j], set())
    return out


def _title_reuse_counts(
    examples: Sequence[Example], rubrics_by_source: Mapping[str, Mapping[str, Rubric]]
) -> dict[tuple[str, str], dict[str, int]]:
    """``{(source, domain): {normalised_title: n_uses}}`` over the evaluated sample."""
    counts: dict[tuple[str, str], collections.Counter[str]] = collections.defaultdict(
        collections.Counter
    )
    for ex in examples:
        for source, per_uid in rubrics_by_source.items():
            rubric = per_uid.get(ex.uid)
            if rubric is None:
                continue
            counts[(source, ex.domain)].update(
                normalise_title(c.title) for c in rubric.items
            )
    return {key: dict(counter) for key, counter in counts.items()}


def _score_rubric(
    example: Example,
    source: str,
    rubric: Rubric,
    *,
    own_anchors: set[str],
    other_anchors: set[str],
    reference_trigrams: set[tuple[str, ...]],
    question_4grams: set[tuple[str, ...]],
    title_reuse: Mapping[str, int],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row: dict[str, Any] = {
        "uid": example.uid,
        "domain": example.domain,
        "rubric_source": source,
        "n_criteria": len(rubric),
        "n_question_anchors": len(own_anchors),
        "question_words": len((example.question or "").split()),
        "reference_words": len((example.reference_answer or "").split()),
        "error": None,
    }
    for metric in _SUMMARY_METRICS:
        row.setdefault(metric, NAN)
    row["n_criteria"] = len(rubric)

    if len(rubric) == 0:
        row["error"] = "empty rubric"
        return row, []

    criterion_rows: list[dict[str, Any]] = []
    for index, criterion in enumerate(rubric.items):
        try:
            criterion_rows.append(
                _score_criterion(
                    example,
                    source,
                    index,
                    criterion,
                    own_anchors=own_anchors,
                    other_anchors=other_anchors,
                    reference_trigrams=reference_trigrams,
                    question_4grams=question_4grams,
                    title_reuse=title_reuse,
                )
            )
        except Exception as exc:  # noqa: BLE001 - degrade per criterion
            logger.warning(
                "grounding failed for %s/%s criterion %d: %s", source, example.uid, index, exc
            )

    if not criterion_rows:
        row["error"] = "no criterion could be scored"
        return row, []

    pooled = _pooled(criterion_rows)
    for metric in _POOLED_METRICS:
        if metric in pooled:
            row[metric] = pooled[metric]
    anchors_per = np.asarray([r["n_anchors"] for r in criterion_rows], dtype=float)
    row["mean_anchors_per_criterion"] = float(anchors_per.mean())
    row["specificity_lift"] = float(row["generic_rate_permuted"] - row["generic_criterion_rate"])
    row["mean_4gram_overlap_with_question"] = float(
        np.mean([r["fourgram_overlap_question"] for r in criterion_rows])
    )
    reuse = [int(r["title_reuse"]) for r in criterion_rows]
    row["max_title_reuse"] = float(max(reuse))
    row["title_reuse_ge2_rate"] = float(np.mean([v >= 2 for v in reuse]))
    row["n_generic"] = int(sum(1 for r in criterion_rows if r["generic"]))
    row["n_criteria_scored"] = len(criterion_rows)
    return row, criterion_rows


def _score_criterion(
    example: Example,
    source: str,
    index: int,
    criterion: Criterion,
    *,
    own_anchors: set[str],
    other_anchors: set[str],
    reference_trigrams: set[tuple[str, ...]],
    question_4grams: set[tuple[str, ...]],
    title_reuse: Mapping[str, int],
) -> dict[str, Any]:
    # ``Criterion.description`` already has the "Essential Criteria:" prefix
    # stripped, which is exactly the ``body`` column the forensics measured — so
    # criteria never share tokens merely by sharing a category.
    body = criterion.description or ""
    tokens = criterion_tokens(body)
    body_tokens = content_tokens(body.lower())
    n_anchors = len(tokens & own_anchors)
    generic = n_anchors == 0
    subjective = _contains_any(body, SUBJECTIVE_TERMS)
    style = _contains_any(body, STYLE_TERMS)
    norm_title = normalise_title(criterion.title)
    reuse = int(title_reuse.get(norm_title, 0))

    return {
        "uid": example.uid,
        "domain": example.domain,
        "rubric_source": source,
        "criterion_index": index,
        "title": criterion.title,
        "norm_title": norm_title,
        "category": criterion.category.value,
        "weight": int(criterion.weight),
        "weight_paper": _paper_weight(criterion),
        "n_anchors": n_anchors,
        "generic": generic,
        "generic_permuted": len(tokens & other_anchors) == 0,
        "weak_grounding": n_anchors < WEAK_GROUNDING_MIN_ANCHORS,
        "subjective": subjective,
        "style": style,
        "subjective_or_style": subjective or style,
        "generic_and_subjective": generic and (subjective or style),
        "title_reuse": reuse,
        "mass_produced_title": reuse >= MASS_PRODUCED_TITLE_MIN_REUSE,
        "recyclable": generic and reuse >= MASS_PRODUCED_TITLE_MIN_REUSE,
        "trigram_overlap_reference": _overlap(ngrams(body_tokens, 3), reference_trigrams),
        "fourgram_overlap_question": _overlap(ngrams(body_tokens, 4), question_4grams),
    }


def _persist(results: GroundingResults, run_dir: RunDir | None) -> None:
    if run_dir is None:
        return
    try:
        for row in results.per_question_rows:
            run_dir.writer("grounding_per_question").write(row)
        for row in results.per_criterion_rows:
            run_dir.writer("grounding_per_criterion").write(row)
        run_dir.write_json("grounding_summary.json", results.summary())
        run_dir.write_json(
            "grounding_sample_vs_full_corpus.json", sample_vs_full_corpus(results)
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to persist grounding artefacts: %s", exc)
