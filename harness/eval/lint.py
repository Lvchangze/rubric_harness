"""Metric family 5 — paper-constraint compliance and schema hygiene, **zero LLM**.

Two deliberately separated blocks.

**Block A — RaR prompt compliance.** The ten machine-checkable constraints
extracted from the two verbatim generation prompts in ``docs/00_paper_notes.md``
§2.4. This is what makes the ``baseline`` column credible: a faithful one-shot
reproduction of the paper prompt should comply at roughly the rate the shipped
corpus does (``docs/01_data_forensics.md`` §2.4: >95% on every format-level
constraint), and if it does not, the reproduction is broken rather than the
paper being wrong.

**Block B — our own hygiene checks.** Not from the paper; they target the
failure modes the forensics found (F6, F7, F5). They are reported under separate
keys so the two blocks are never conflated in a table.

The ten prompt constraints
--------------------------
=== ======================================= ======================================
 #  check name                              scope / notes
=== ======================================= ======================================
 1  ``item_count_7_20``                     rubric-level
 2  ``title_2_to_4_words``                   words per :func:`analysis.common.words`
 3  ``description_single_sentence``          :data:`analysis.common.SENT_SPLIT_RE`
 4  ``category_legal``                       see the normalisation caveat below
 5  ``weight_in_range``                      see the normalisation caveat below
 6  ``weight_matches_band``                  **medicine only** (Ess=5/Imp 3-4/Opt 1-2)
 7  ``pitfall_opener``                       **medicine only**, Pitfall criteria only
 8  ``rar_keys_only``                        presence check only, see caveat
 9  ``no_verbatim_copy``                     >= 50% of content trigrams shared
10  ``self_contained``                       :data:`analysis.rubric_stats.NON_SELF_CONTAINED_TERMS`
=== ======================================= ======================================

Caveat: three checks are partly vacuous on run artefacts
--------------------------------------------------------
``harness.schema.Criterion`` normalises on construction — it parses and strips
the ``"<Category> Criteria:"`` prefix, coerces the category to one of the four
enum members, and clamps the weight into the legal band via ``_coerce_weight``.
By the time a rubric reaches this module, checks 4, 5 and 8 can therefore only
fail in ways normalisation cannot repair. They are still emitted (so the report
has a complete constraint list, and so they do their job if a generator is ever
linted *before* normalisation), but a 100% pass rate on them is a statement about
the schema layer, not about the generator. Detecting a malformed prefix, an
out-of-range weight or a fourth JSON key genuinely requires the raw model output,
which lives in ``harness/generators/`` — outside this module's ownership. Where a
generator records such damage in ``Criterion.provenance`` under ``schema_errors``
or ``extra_keys``, it is picked up here.

Published shipped-corpus reference points (science / medicine)
-------------------------------------------------------------
=============================================== ========== ==========
metric                                          science    medicine
=============================================== ========== ==========
``check_item_count_7_20_rate``                  0.9997     0.9978
``check_title_2_to_4_words_rate``               0.9867     0.9546
``check_description_single_sentence_rate``       0.9994     0.9965
``weight_band_rate`` (informational for science) 0.7759     0.9999
``check_pitfall_opener_rate``                    0.0354†    0.7856
``check_self_contained_rate``                    0.9796     0.9931
``polarity_single_rate``                         low‡       low‡
``negative_weight_rate``                         0.1305     0.1416
``weight_order_violation_rate``                  0.4130     0.0000
``pitfall_mirror_rate``                          0.0095     0.1621
=============================================== ========== ==========

† The science prompt never mandates the opener, so 3.54% is a domain-style
observation, not a violation; the check is only scored for medicine.
‡ The shipped corpora put a negative weight on 91.8% / 95.6% of *questions*, and
the phrasing conventions are opposite between domains (F6), so essentially no
shipped rubric is single-polarity in the sense we require of ``agentic``.
"""

from __future__ import annotations

import collections
import logging
import math
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from ..config import EvalConfig
from ..schema import Category, Criterion, Example, Polarity, Rubric
from ..tracing import RunDir
from .grounding import SHIPPED_FULL_CORPUS_REFERENCE
from .stats import mean_ci

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_REPO_ROOT), str(_REPO_ROOT / "analysis")):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from analysis.common import SENT_SPLIT_RE, content_tokens, ngrams, words  # noqa: E402
from analysis.rubric_similarity import polarity_stripped  # noqa: E402
from analysis.rubric_stats import (  # noqa: E402
    NON_SELF_CONTAINED_TERMS,
    PROMPT_WEIGHT_RANGE,
    PROMPT_WEIGHT_RULE,
)

logger = logging.getLogger(__name__)

__all__ = [
    "CheckOutcome",
    "COPY_TRIGRAM_THRESHOLD",
    "LintResults",
    "PITFALL_MIRROR_THRESHOLD",
    "PROMPT_CHECKS",
    "pitfall_mirror_scores",
    "run_lint",
    "tfidf_matrix",
    "weight_order_violations",
]

NAN = float("nan")

#: Fraction of a criterion's content trigrams that must be shared with the
#: question or reference before it counts as a "large verbatim block" (check 9).
#: Chosen to match the forensics' F9 measure (§7.1: 0.43% of science / 7.41% of
#: medicine criteria copy >= 50% of their trigrams from the reference answer), so
#: this check's failure rate on ``shipped`` is a published number.
COPY_TRIGRAM_THRESHOLD = 0.5
#: Polarity-stripped TF-IDF cosine at which a Pitfall counts as a mirror of a
#: positive criterion of the same rubric (F7; §5.1 shipped: 0.95% science /
#: 16.21% medicine of questions).
PITFALL_MIRROR_THRESHOLD = 0.6
#: Minimum document frequency for a token to enter the mirror TF-IDF vocabulary,
#: matching the forensics' ``TfidfVectorizer(min_df=2)``.
MIRROR_MIN_DF = 2
#: Rank order the category labels claim; used by the weight-order check (F5).
CATEGORY_RANK: dict[Category, int] = {
    Category.ESSENTIAL: 3,
    Category.IMPORTANT: 2,
    Category.OPTIONAL: 1,
}

_PITFALL_OPENER_RE = re.compile(r"\s*(does not mention|recommends)", re.IGNORECASE)
# sklearn's default ``token_pattern``, replicated so our TF-IDF and the
# forensics' vectoriser tokenise identically.
_SKLEARN_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


@dataclass(frozen=True)
class CheckOutcome:
    """One constraint evaluated against one unit (a criterion or a rubric)."""

    name: str
    applicable: bool
    passed: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "applicable": self.applicable,
            "passed": self.passed,
            "reason": self.reason,
        }


def _ok(name: str) -> CheckOutcome:
    return CheckOutcome(name, True, True)


def _fail(name: str, reason: str) -> CheckOutcome:
    return CheckOutcome(name, True, False, reason)


def _skip(name: str, reason: str) -> CheckOutcome:
    return CheckOutcome(name, False, True, reason)


# ---------------------------------------------------------------------------
# Block A — the ten machine-checkable prompt constraints
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CriterionContext:
    """Everything a per-criterion check needs beyond the criterion itself."""

    domain: str
    is_medicine: bool
    question_trigrams: frozenset
    reference_trigrams: frozenset


def check_title_2_to_4_words(criterion: Criterion, ctx: CriterionContext) -> CheckOutcome:
    """Constraint 2: ``title`` is 2-4 words (shipped: 98.67% / 95.46%)."""
    del ctx
    n = len(words(criterion.title or ""))
    if 2 <= n <= 4:
        return _ok("title_2_to_4_words")
    return _fail("title_2_to_4_words", f"title has {n} words")


def check_description_single_sentence(
    criterion: Criterion, ctx: CriterionContext
) -> CheckOutcome:
    """Constraint 3: the description is one sentence (shipped: 99.94% / 99.65%)."""
    del ctx
    body = (criterion.description or "").strip()
    n = len(SENT_SPLIT_RE.split(body)) if body else 0
    if n <= 1:
        return _ok("description_single_sentence")
    return _fail("description_single_sentence", f"{n} sentences")


def check_category_legal(criterion: Criterion, ctx: CriterionContext) -> CheckOutcome:
    """Constraint 4: the category is one of the four legal values.

    Partly vacuous post-normalisation (see the module docstring); a generator can
    still record what it had to repair under ``provenance["schema_errors"]``.
    """
    del ctx
    if criterion.category not in set(Category):
        return _fail("category_legal", f"category={criterion.category!r}")
    recorded = criterion.provenance.get("schema_errors") if criterion.provenance else None
    if isinstance(recorded, (list, tuple)) and any("categ" in str(e).lower() for e in recorded):
        return _fail("category_legal", f"generator reported {recorded}")
    return _ok("category_legal")


def check_weight_in_range(criterion: Criterion, ctx: CriterionContext) -> CheckOutcome:
    """Constraint 5: Essential/Important/Optional in [1,5], Pitfall in {-1,-2}.

    Partly vacuous post-normalisation (see the module docstring).
    """
    del ctx
    allowed = PROMPT_WEIGHT_RANGE.get(criterion.category.value, set(range(-2, 6)))
    if int(criterion.weight) in allowed:
        return _ok("weight_in_range")
    return _fail(
        "weight_in_range",
        f"{criterion.category.value} weight={criterion.weight} outside {sorted(allowed)}",
    )


def check_weight_matches_band(criterion: Criterion, ctx: CriterionContext) -> CheckOutcome:
    """Constraint 6 (**medicine only**): Essential=5 / Important 3-4 / Optional 1-2.

    Only the medicine prompt states these anchors; the science prompt says just
    "use 1-5". Scoring science against them would report a violation the prompt
    never asked for, so science is skipped here — its 77.59% band rate is emitted
    as the informational ``weight_band_rate`` instead (§2.4 footnote).
    """
    if not ctx.is_medicine:
        return _skip("weight_matches_band", "science prompt gives no numeric anchors")
    allowed = PROMPT_WEIGHT_RULE.get(criterion.category.value, set())
    if int(criterion.weight) in allowed:
        return _ok("weight_matches_band")
    return _fail(
        "weight_matches_band",
        f"{criterion.category.value} weight={criterion.weight} not in {sorted(allowed)}",
    )


def check_pitfall_opener(criterion: Criterion, ctx: CriterionContext) -> CheckOutcome:
    """Constraint 7 (**medicine only**): Pitfall starts "Does not mention"/"Recommends"."""
    if not ctx.is_medicine:
        return _skip("pitfall_opener", "science prompt does not mandate the opener")
    if criterion.category is not Category.PITFALL:
        return _skip("pitfall_opener", "not a Pitfall")
    if _PITFALL_OPENER_RE.match(criterion.description or ""):
        return _ok("pitfall_opener")
    return _fail("pitfall_opener", f"opens with {' '.join((criterion.description or '').split()[:4])!r}")


def check_rar_keys_only(criterion: Criterion, ctx: CriterionContext) -> CheckOutcome:
    """Constraint 8: exactly ``title`` / ``description`` / ``weight``, no extras.

    Presence check only after normalisation (see the module docstring): an extra
    key in the model's JSON is dropped by ``Criterion.from_dict`` long before we
    see it, so the only recoverable signals are a missing/blank required field
    and an ``extra_keys`` note left by the generator.
    """
    del ctx
    missing = [
        name
        for name, value in (
            ("title", (criterion.title or "").strip()),
            ("description", (criterion.description or "").strip()),
        )
        if not value
    ]
    if missing:
        return _fail("rar_keys_only", f"empty {'/'.join(missing)}")
    extra = criterion.provenance.get("extra_keys") if criterion.provenance else None
    if extra:
        return _fail("rar_keys_only", f"generator reported extra keys {extra}")
    return _ok("rar_keys_only")


def check_no_verbatim_copy(criterion: Criterion, ctx: CriterionContext) -> CheckOutcome:
    """Constraint 9: no large verbatim block lifted from question/reference.

    Operationalised as the forensics' F9 measure: the fraction of the criterion's
    *content-token trigrams* (stopwords dropped, so formulaic rubric phrasing
    does not count) that also occur in the reference answer or the question. A
    criterion fails at >= :data:`COPY_TRIGRAM_THRESHOLD` = 50%, the same cut
    §7.1 reports (0.43% of science / 7.41% of medicine criteria).
    """
    trigrams = ngrams(content_tokens((criterion.description or "").lower()), 3)
    if not trigrams:
        return _skip("no_verbatim_copy", "fewer than 3 content tokens")
    ref = len(trigrams & ctx.reference_trigrams) / len(trigrams)
    question = len(trigrams & ctx.question_trigrams) / len(trigrams)
    worst = max(ref, question)
    if worst < COPY_TRIGRAM_THRESHOLD:
        return _ok("no_verbatim_copy")
    origin = "reference" if ref >= question else "question"
    return _fail("no_verbatim_copy", f"{worst:.0%} of trigrams shared with {origin}")


def check_self_contained(criterion: Criterion, ctx: CriterionContext) -> CheckOutcome:
    """Constraint 10: judgeable without external information.

    Approximated with the forensics' heuristic
    (:data:`analysis.rubric_stats.NON_SELF_CONTAINED_TERMS`): a criterion that
    asserts the answer is "correct" / matches "the reference" without saying what
    correct *is* cannot be settled from the response alone. Shipped: 2.04% of
    science / 0.69% of medicine criteria fail (§6). This is a lower bound — it
    catches the explicit phrasings only.
    """
    del ctx
    lowered = (criterion.description or "").lower()
    hits = [term for term in NON_SELF_CONTAINED_TERMS if term in lowered]
    if not hits:
        return _ok("self_contained")
    return _fail("self_contained", f"asserts correctness via {hits[0]!r}")


#: Per-criterion prompt constraints, in the order of ``docs/00_paper_notes.md`` §2.4.
PROMPT_CHECKS: tuple[Callable[[Criterion, CriterionContext], CheckOutcome], ...] = (
    check_title_2_to_4_words,
    check_description_single_sentence,
    check_category_legal,
    check_weight_in_range,
    check_weight_matches_band,
    check_pitfall_opener,
    check_rar_keys_only,
    check_no_verbatim_copy,
    check_self_contained,
)

#: Constraint 1 is rubric-level, so it lives outside :data:`PROMPT_CHECKS`.
ITEM_COUNT_CHECK = "item_count_7_20"
#: All ten constraint names, constraint 1 first. Order matches §2.4.
_CHECK_NAMES: tuple[str, ...] = (
    ITEM_COUNT_CHECK,
    "title_2_to_4_words",
    "description_single_sentence",
    "category_legal",
    "weight_in_range",
    "weight_matches_band",
    "pitfall_opener",
    "rar_keys_only",
    "no_verbatim_copy",
    "self_contained",
)


def check_item_count(rubric: Rubric) -> CheckOutcome:
    """Constraint 1: 7-20 items (shipped: 99.97% / 99.78%)."""
    n = len(rubric)
    if 7 <= n <= 20:
        return _ok(ITEM_COUNT_CHECK)
    return _fail(ITEM_COUNT_CHECK, f"{n} items")


# ---------------------------------------------------------------------------
# Block B — our hygiene checks (F5, F6, F7)
# ---------------------------------------------------------------------------


def tfidf_matrix(
    documents: Sequence[str], *, min_df: int = MIRROR_MIN_DF
) -> tuple[np.ndarray, dict[str, int]]:
    """L2-normalised sublinear-TF / smoothed-IDF matrix, pure numpy.

    Replicates ``TfidfVectorizer(sublinear_tf=True, min_df=min_df)``: sklearn's
    default ``\\b\\w\\w+\\b`` tokenisation and lowercasing, ``tf = 1 + log(tf)``,
    ``idf = ln((1+n)/(1+df)) + 1``, then L2 row normalisation. Written out here
    rather than imported so this module has no sklearn dependency and so the
    ``min_df`` corpus is explicit — it is the pool of documents passed in, which
    for the Pitfall-mirror check is every criterion of one ``(source, domain)``
    group, matching how the forensics fitted its vectoriser.
    """
    tokenised = [_SKLEARN_TOKEN_RE.findall((doc or "").lower()) for doc in documents]
    doc_freq: collections.Counter[str] = collections.Counter()
    for tokens in tokenised:
        doc_freq.update(set(tokens))
    vocab = {
        token: i
        for i, token in enumerate(
            sorted(t for t, c in doc_freq.items() if c >= max(1, int(min_df)))
        )
    }
    matrix = np.zeros((len(tokenised), len(vocab)), dtype=np.float32)
    if not vocab:
        return matrix, vocab

    for row, tokens in enumerate(tokenised):
        for token, count in collections.Counter(tokens).items():
            column = vocab.get(token)
            if column is not None:
                matrix[row, column] = 1.0 + math.log(count)
    n_docs = len(tokenised)
    df = np.asarray([doc_freq[token] for token in vocab], dtype=np.float64)
    matrix *= (np.log((1.0 + n_docs) / (1.0 + df)) + 1.0).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.where(norms > 0, norms, 1.0), vocab


def pitfall_mirror_scores(rubrics: Mapping[str, Rubric]) -> dict[str, dict[str, Any]]:
    """F7: does a Pitfall merely restate a positive criterion of the same rubric?

    The polarity wrapper ("Does not mention ..." / "The response must state
    that ...") is stripped with
    :func:`analysis.rubric_similarity.polarity_stripped` first, so a Pitfall and
    its positive twin become lexically identical; what remains is compared by
    TF-IDF cosine. A question is flagged at cosine >=
    :data:`PITFALL_MIRROR_THRESHOLD`. Shipped reference (§5.1): 0.95% of science
    and 16.21% of medicine questions.

    ``rubrics`` should be one ``(source, domain)`` group, because the IDF is
    fitted over whatever is passed in.
    """
    keys: list[tuple[str, int]] = []
    documents: list[str] = []
    is_pitfall: list[bool] = []
    for uid, rubric in rubrics.items():
        for index, criterion in enumerate(rubric.items):
            keys.append((uid, index))
            documents.append(polarity_stripped(criterion.description or ""))
            is_pitfall.append(criterion.category is Category.PITFALL)

    out: dict[str, dict[str, Any]] = {
        uid: {"max_similarity": 0.0, "n_mirrors": 0, "mirror_pairs": []}
        for uid in rubrics
    }
    if not documents:
        return out
    matrix, _vocab = tfidf_matrix(documents)
    flags = np.asarray(is_pitfall)
    rows_by_uid: dict[str, list[int]] = collections.defaultdict(list)
    for row, (uid, _index) in enumerate(keys):
        rows_by_uid[uid].append(row)

    for uid, rows in rows_by_uid.items():
        rows_arr = np.asarray(rows)
        pitfalls = rows_arr[flags[rows_arr]]
        positives = rows_arr[~flags[rows_arr]]
        if pitfalls.size == 0 or positives.size == 0:
            continue
        block = matrix[pitfalls] @ matrix[positives].T
        best = float(block.max())
        hits = np.argwhere(block >= PITFALL_MIRROR_THRESHOLD)
        out[uid] = {
            "max_similarity": best,
            "n_mirrors": int(len(hits)),
            "mirror_pairs": [
                {
                    "pitfall_index": keys[int(pitfalls[a])][1],
                    "positive_index": keys[int(positives[b])][1],
                    "similarity": float(block[a, b]),
                }
                for a, b in hits[:5]
            ],
        }
    return out


def weight_order_violations(rubric: Rubric) -> dict[str, Any]:
    """F5: inside one rubric, does a lower-priority item weigh >= a higher one?

    ``Essential > Important > Optional`` is the semantics the category labels
    claim, so an ordered pair that contradicts it makes the weighted sum of
    Eq. 1 disagree with the labels. Pitfalls are excluded (their weight is
    negative by convention, not by priority). Shipped (§9.1): 41.30% of science
    questions contain at least one violation, 0.00% of medicine.
    """
    ranked = [(CATEGORY_RANK[c.category], float(c.weight)) for c in rubric.items if c.category in CATEGORY_RANK]
    if len(ranked) < 2:
        return {"n_pairs": 0, "n_violations": 0, "pair_rate": NAN, "any": False}
    rank = np.asarray([r for r, _w in ranked], dtype=float)
    weight = np.asarray([w for _r, w in ranked], dtype=float)
    higher = rank[:, None] > rank[None, :]
    bad = higher & (weight[:, None] <= weight[None, :])
    n_pairs = int(higher.sum())
    n_bad = int(bad.sum())
    return {
        "n_pairs": n_pairs,
        "n_violations": n_bad,
        "pair_rate": n_bad / n_pairs if n_pairs else NAN,
        "any": bool(n_bad),
    }


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

#: Per-question metrics that get a mean + bootstrap CI in the summary.
_SUMMARY_METRICS: tuple[str, ...] = (
    "compliance_rate",
    "fully_compliant_rubric",
    "item_count_ok",
    "n_items",
    "polarity_single_rate",
    "negative_weight_rate",
    "weight_order_violation_rate",
    "weight_order_violation_pair_rate",
    "pitfall_mirror_rate",
    "n_pitfall_mirrors",
    "max_pitfall_mirror_similarity",
    "pitfall_fraction",
    "weight_band_rate",
    "unclassified_polarity_rate",
) + tuple(f"check_{name}_rate" for name in _CHECK_NAMES)

_HYGIENE_METRICS: tuple[str, ...] = (
    "polarity_single_rate",
    "negative_weight_rate",
    "weight_order_violation_rate",
    "weight_order_violation_pair_rate",
    "pitfall_mirror_rate",
    "pitfall_fraction",
    "unclassified_polarity_rate",
)


@dataclass
class LintResults:
    """Per-question and per-criterion lint rows plus a per-source summary."""

    per_question_rows: list[dict[str, Any]] = field(default_factory=list)
    per_criterion_rows: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    params: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> dict[str, Any]:
        """``{rubric_source: {metric: value}}`` plus underscore-prefixed metadata.

        Per source, every metric ``m`` comes with ``m``, ``m_ci_low``,
        ``m_ci_high`` and ``m_n``. The headline is ``compliance_rate``: the
        fraction of criteria passing *every applicable* per-criterion prompt
        constraint (checks 2-10). Constraint 1 is rubric-level and reported as
        ``item_count_ok``; ``fully_compliant_rubric`` is the conjunction of the
        two. Per-check pass rates are ``check_<name>_rate``.

        ``check_breakdown``
            Per-constraint breakdown pooled over criteria:
            ``{unit, n_applicable, n_failed, rate, top_reasons}``. This is the
            block-A table. (The per-question and per-criterion rows carry the raw
            outcomes for their own unit under ``checks``.)
        ``hygiene``
            Block B only (F5/F6/F7), kept separate so prompt compliance and our
            own hygiene targets are never summed together.

        Metadata: ``_metric_family``, ``_params``, ``_by_domain`` and
        ``_shipped_full_corpus_reference``.
        """
        out: dict[str, Any] = {}
        for source in self._sources():
            rows = [r for r in self.per_question_rows if r.get("rubric_source") == source]
            crit = [r for r in self.per_criterion_rows if r.get("rubric_source") == source]
            entry: dict[str, Any] = {
                "n_questions": len(rows),
                "n_criteria": len(crit),
                "n_errors": sum(1 for r in rows if r.get("error")),
            }
            for metric in _SUMMARY_METRICS:
                ci = mean_ci([r.get(metric) for r in rows], iters=2000, seed=0)
                entry[metric] = ci["mean"]
                entry[f"{metric}_ci_low"] = ci["ci_low"]
                entry[f"{metric}_ci_high"] = ci["ci_high"]
                entry[f"{metric}_n"] = ci["n"]
            entry["check_breakdown"] = _check_breakdown(rows, crit)
            entry["hygiene"] = {m: entry[m] for m in _HYGIENE_METRICS}
            out[source] = entry

        by_domain: dict[str, dict[str, Any]] = {}
        for domain in sorted({str(r.get("domain")) for r in self.per_question_rows}):
            by_domain[domain] = {}
            for source in self._sources():
                rows = [
                    r
                    for r in self.per_question_rows
                    if r.get("rubric_source") == source and str(r.get("domain")) == domain
                ]
                crit = [
                    r
                    for r in self.per_criterion_rows
                    if r.get("rubric_source") == source and str(r.get("domain")) == domain
                ]
                if not rows:
                    continue
                by_domain[domain][source] = {
                    "n_questions": len(rows),
                    "n_criteria": len(crit),
                    **{m: _safe_mean([r.get(m) for r in rows]) for m in _SUMMARY_METRICS},
                    "check_breakdown": _check_breakdown(rows, crit),
                }

        out["_metric_family"] = "lint"
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


def _check_breakdown(
    question_rows: Sequence[Mapping[str, Any]], criterion_rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """Pooled pass rate and the commonest failure reasons, per constraint."""
    out: dict[str, Any] = {}
    for name in _CHECK_NAMES:
        # Constraint 1 has the rubric as its unit; the rest have the criterion.
        units: Sequence[Mapping[str, Any]] = (
            question_rows if name == ITEM_COUNT_CHECK else criterion_rows
        )
        applicable = [u for u in units if bool((u.get("checks") or {}).get(name, {}).get("applicable"))]
        failed = [u for u in applicable if not (u.get("checks") or {})[name].get("passed")]
        reasons = collections.Counter(
            str((u.get("checks") or {})[name].get("reason", "")) for u in failed
        )
        out[name] = {
            "unit": "rubric" if name == ITEM_COUNT_CHECK else "criterion",
            "n_applicable": len(applicable),
            "n_failed": len(failed),
            "rate": (len(applicable) - len(failed)) / len(applicable) if applicable else NAN,
            "top_reasons": [
                {"reason": reason, "n": n} for reason, n in reasons.most_common(5)
            ],
        }
    return out


def _safe_mean(values: Sequence[Any]) -> float:
    finite = [float(v) for v in values if _finite(v)]
    return float(np.mean(finite)) if finite else NAN


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


async def run_lint(
    engine: Any,  # unused: accepted for API uniformity across metric modules
    examples: Sequence[Example],
    rubrics_by_source: Mapping[str, Mapping[str, Rubric]],
    *,
    config: EvalConfig,
    run_dir: RunDir | None = None,
) -> LintResults:
    """Lint every rubric of every source against §2.4 plus our hygiene checks."""
    del engine, config  # zero-LLM metric; kept in the signature for uniformity

    results = LintResults(
        params={
            "sources": list(rubrics_by_source),
            "copy_trigram_threshold": COPY_TRIGRAM_THRESHOLD,
            "pitfall_mirror_threshold": PITFALL_MIRROR_THRESHOLD,
            "mirror_min_df": MIRROR_MIN_DF,
            "mirror_idf_scope": "all criteria of one (rubric_source, domain) group",
            "checks": list(_CHECK_NAMES),
            "medicine_only_checks": ["weight_matches_band", "pitfall_opener"],
            "vacuous_after_schema_normalisation": [
                "category_legal",
                "weight_in_range",
                "rar_keys_only",
            ],
        }
    )
    examples = [ex for ex in examples if ex is not None]
    if not examples:
        return results

    by_uid = {ex.uid: ex for ex in examples}
    contexts = {ex.uid: _context(ex) for ex in examples}

    # The mirror check needs a corpus to fit IDF on: one per (source, domain).
    mirror: dict[tuple[str, str], dict[str, dict[str, Any]]] = {}
    for source, per_uid in rubrics_by_source.items():
        grouped: dict[str, dict[str, Rubric]] = collections.defaultdict(dict)
        for uid, rubric in per_uid.items():
            example = by_uid.get(uid)
            if example is not None:
                grouped[example.domain][uid] = rubric
        for domain, group in grouped.items():
            try:
                mirror[(source, domain)] = pitfall_mirror_scores(group)
            except Exception as exc:  # noqa: BLE001 - degrade, never raise
                logger.warning("pitfall mirror failed for %s/%s: %s", source, domain, exc)
                mirror[(source, domain)] = {}

    for ex in examples:
        for source, per_uid in rubrics_by_source.items():
            rubric = per_uid.get(ex.uid)
            if rubric is None:
                continue
            try:
                row, criterion_rows = _lint_rubric(
                    ex,
                    source,
                    rubric,
                    contexts[ex.uid],
                    mirror.get((source, ex.domain), {}).get(ex.uid),
                )
            except Exception as exc:  # noqa: BLE001 - degrade per (question, source)
                logger.warning("lint failed for %s/%s: %s", source, ex.uid, exc)
                row, criterion_rows = _blank_row(ex, source, rubric), []
                row["error"] = f"exception: {exc}"[:300]
            results.per_question_rows.append(row)
            results.per_criterion_rows.extend(criterion_rows)
            if row.get("error"):
                results.errors.append(
                    {"uid": ex.uid, "rubric_source": source, "error": row["error"]}
                )

    _persist(results, run_dir)
    return results


def _context(example: Example) -> CriterionContext:
    return CriterionContext(
        domain=example.domain,
        is_medicine="medicine" in (example.domain or "").lower(),
        question_trigrams=frozenset(ngrams(content_tokens((example.question or "").lower()), 3)),
        reference_trigrams=frozenset(
            ngrams(content_tokens((example.reference_answer or "").lower()), 3)
        ),
    )


def _blank_row(example: Example, source: str, rubric: Rubric) -> dict[str, Any]:
    row: dict[str, Any] = {
        "uid": example.uid,
        "domain": example.domain,
        "rubric_source": source,
        "n_items": len(rubric),
        "error": None,
        "checks": {},
    }
    for metric in _SUMMARY_METRICS:
        row.setdefault(metric, NAN)
    row["n_items"] = len(rubric)
    return row


def _lint_rubric(
    example: Example,
    source: str,
    rubric: Rubric,
    ctx: CriterionContext,
    mirror: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    row = _blank_row(example, source, rubric)
    item_count = check_item_count(rubric)
    row["checks"] = {ITEM_COUNT_CHECK: item_count.to_dict()}
    row["item_count_ok"] = float(item_count.passed)
    row[f"check_{ITEM_COUNT_CHECK}_rate"] = float(item_count.passed)

    if len(rubric) == 0:
        row["error"] = "empty rubric"
        row["compliance_rate"] = NAN
        row["fully_compliant_rubric"] = 0.0
        return row, []

    criterion_rows: list[dict[str, Any]] = []
    per_check_pass: dict[str, list[bool]] = collections.defaultdict(list)
    fully_compliant = 0
    for index, criterion in enumerate(rubric.items):
        outcomes = {}
        for check in PROMPT_CHECKS:
            try:
                outcome = check(criterion, ctx)
            except Exception as exc:  # noqa: BLE001 - one bad check must not kill the row
                outcome = CheckOutcome(getattr(check, "__name__", "check"), True, False, f"error: {exc}"[:120])
            outcomes[outcome.name] = outcome
        applicable = [o for o in outcomes.values() if o.applicable]
        passed_all = all(o.passed for o in applicable)
        fully_compliant += int(passed_all)
        for name, outcome in outcomes.items():
            if outcome.applicable:
                per_check_pass[name].append(outcome.passed)
        criterion_rows.append(
            {
                "uid": example.uid,
                "domain": example.domain,
                "rubric_source": source,
                "criterion_index": index,
                "title": criterion.title,
                "category": criterion.category.value,
                "weight": int(criterion.weight),
                "polarity": (criterion.polarity or Polarity.POSITIVE).value,
                "polarity_method": criterion.polarity_method,
                "n_checks_applicable": len(applicable),
                "n_checks_failed": sum(1 for o in applicable if not o.passed),
                "compliant": passed_all,
                "failed_checks": [o.name for o in applicable if not o.passed],
                "checks": {name: o.to_dict() for name, o in outcomes.items()},
            }
        )

    n = len(criterion_rows)
    row["compliance_rate"] = fully_compliant / n
    row["fully_compliant_rubric"] = float(bool(item_count.passed) and fully_compliant == n)
    for name in _CHECK_NAMES:
        if name == ITEM_COUNT_CHECK:
            continue
        hits = per_check_pass.get(name, [])
        row[f"check_{name}_rate"] = float(np.mean(hits)) if hits else NAN

    # --- block B: hygiene ---------------------------------------------------
    polarity = rubric.polarity_stats()
    row["polarity_single_rate"] = float(bool(polarity["single_polarity"]))
    row["unclassified_polarity_rate"] = polarity["n_unclassified_polarity"] / n
    row["negative_weight_rate"] = float(
        np.mean([int(c.weight) < 0 for c in rubric.items])
    )
    row["pitfall_fraction"] = float(
        np.mean([c.category is Category.PITFALL for c in rubric.items])
    )
    order = weight_order_violations(rubric)
    row["weight_order_violation_rate"] = float(order["any"])
    row["weight_order_violation_pair_rate"] = order["pair_rate"]
    row["n_weight_order_pairs"] = order["n_pairs"]
    band = [
        int(c.weight) in PROMPT_WEIGHT_RULE.get(c.category.value, set()) for c in rubric.items
    ]
    row["weight_band_rate"] = float(np.mean(band)) if band else NAN

    mirror = mirror or {}
    similarity = float(mirror.get("max_similarity", 0.0) or 0.0)
    row["max_pitfall_mirror_similarity"] = similarity
    row["n_pitfall_mirrors"] = int(mirror.get("n_mirrors", 0) or 0)
    row["pitfall_mirror_rate"] = float(similarity >= PITFALL_MIRROR_THRESHOLD)
    row["pitfall_mirror_pairs"] = list(mirror.get("mirror_pairs") or [])
    return row, criterion_rows


def _persist(results: LintResults, run_dir: RunDir | None) -> None:
    if run_dir is None:
        return
    try:
        for row in results.per_question_rows:
            run_dir.writer("lint_per_question").write(row)
        for row in results.per_criterion_rows:
            run_dir.writer("lint_per_criterion").write(row)
        run_dir.write_json("lint_summary.json", results.summary())
    except Exception as exc:  # noqa: BLE001
        logger.warning("failed to persist lint artefacts: %s", exc)
