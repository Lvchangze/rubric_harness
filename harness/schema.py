"""Canonical data structures shared by every generator and every metric.

Every rubric — whether shipped with the RaR dataset, re-synthesised by our
single-pass baseline, or produced by the agentic pipeline — is normalised into
:class:`Rubric` so that evaluation can treat ``rubric_source`` as the *only*
free variable.

Serialisation contract: ``Rubric.to_dict()`` / ``Rubric.from_dict()`` round-trip
losslessly through JSON, which is what the run artefacts under ``runs/`` store.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Iterator, Sequence

__all__ = [
    "Category",
    "Polarity",
    "Criterion",
    "Rubric",
    "Example",
    "CATEGORY_PREFIXES",
    "parse_category_prefix",
    "strip_category_prefix",
    "detect_polarity",
]


class Category(str, Enum):
    """RaR criterion categories.

    ``PITFALL`` criteria are *negative*: they carry a negative weight and are
    satisfied when the response avoids the named mistake.
    """

    ESSENTIAL = "Essential"
    IMPORTANT = "Important"
    OPTIONAL = "Optional"
    PITFALL = "Pitfall"

    @classmethod
    def coerce(cls, value: Any, default: "Category" = None) -> "Category":  # type: ignore[assignment]
        if isinstance(value, cls):
            return value
        if value is None:
            return default or cls.IMPORTANT
        text = str(value).strip().rstrip(":").strip()
        text = re.sub(r"\s*criteria$", "", text, flags=re.IGNORECASE).strip()
        for member in cls:
            if member.value.lower() == text.lower():
                return member
        return default or cls.IMPORTANT


class Polarity(str, Enum):
    """Which way round a criterion's literal truth maps onto response quality.

    The RaR corpora need this because their Pitfall convention is inverted
    between domains: RaR-Science phrases 80.3% of Pitfalls as *avoidance*
    ("Avoids rounding intermediate values early" — true means the response is
    good), while RaR-Medicine phrases 88.1% as *failure* ("Does not mention the
    overcorrection risk" — true means the response is bad). Both carry weight
    -1/-2, so any single sign convention scores one domain backwards. Storing
    polarity explicitly, rather than inferring it from the weight's sign or the
    category, is what makes scoring correct for both.

    See ``docs/01_data_forensics.md`` F6 for the measurement.
    """

    #: Literal satisfaction of the sentence means the response is GOOD.
    POSITIVE = "positive"
    #: Literal satisfaction of the sentence means the response is BAD.
    NEGATIVE = "negative"

    @classmethod
    def coerce(cls, value: Any, default: "Polarity" = None) -> "Polarity":  # type: ignore[assignment]
        if isinstance(value, cls):
            return value
        if value is None:
            return default or cls.POSITIVE
        text = str(value).strip().lower()
        if text in {"positive", "pos", "+", "avoidance", "avoid", "good", "1", "true"}:
            return cls.POSITIVE
        if text in {"negative", "neg", "-", "failure", "fail", "bad", "0", "false"}:
            return cls.NEGATIVE
        return default or cls.POSITIVE


# Phrasing patterns transcribed verbatim from ``analysis/collect_examples.py`` so
# the harness classifies polarity exactly as the forensics pass measured it.
# One deliberate difference: the forensics resolves overlaps in favour of the
# failure reading, we resolve them in favour of avoidance, because the only
# genuinely overlapping form ("does not incorrectly assume X") is an avoidance.
_POLARITY_SUBJECT = r"(?:the\s+(?:response|answer|explanation)\s+)?"
_FAILURE_PHRASING_RE = re.compile(
    rf"^\s*{_POLARITY_SUBJECT}("
    r"does\s+not\s+(?:mention|state|recommend|include|identify|address|note|specify|suggest"
    r"|list|provide|discuss|explain)"
    r"|fails?\s+to|omits?|neglects?\s+to|overlooks?|ignores?|misses"
    r"|recommends?|suggests?|claims?|misidentifies|confuses|mistakes)",
    re.IGNORECASE,
)
_AVOIDANCE_PHRASING_RE = re.compile(
    rf"^\s*{_POLARITY_SUBJECT}("
    r"avoid(?:s|ing|ed)?"
    r"|does\s+not\s+(?:incorrectly|wrongly|falsely|erroneously|mistakenly)"
    r"|(?:must|should)\s+(?:not|avoid|never|refrain|steer)"
    r"|do\s+not|refrains?|never|warns?\s+against|cautions?\s+against|ensures?|prevents?)",
    re.IGNORECASE,
)


def detect_polarity(description: str, category: Category | None = None) -> tuple[Polarity, str]:
    """Classify a criterion's polarity from its phrasing.

    Returns ``(polarity, method)`` where ``method`` is ``"avoidance"``,
    ``"failure"``, ``"category_default"`` or ``"unclassified"`` — callers use it
    to decide whether an LLM tie-break is warranted and to report how many
    criteria fell back to a default.

    Non-Pitfall criteria are stated as requirements ("States that ..."), so they
    are positive by construction; only Pitfalls are genuinely ambiguous.
    """
    text = strip_category_prefix(description or "")
    if category is not None and category is not Category.PITFALL:
        return Polarity.POSITIVE, "category_default"
    # Avoidance is checked first: "does not incorrectly assume X" matches both
    # patterns, and the avoidance reading is the correct one.
    if _AVOIDANCE_PHRASING_RE.match(text):
        return Polarity.POSITIVE, "avoidance"
    if _FAILURE_PHRASING_RE.match(text):
        return Polarity.NEGATIVE, "failure"
    return Polarity.POSITIVE, "unclassified"


CATEGORY_PREFIXES: dict[Category, str] = {
    Category.ESSENTIAL: "Essential Criteria:",
    Category.IMPORTANT: "Important Criteria:",
    Category.OPTIONAL: "Optional Criteria:",
    Category.PITFALL: "Pitfall Criteria:",
}

_PREFIX_RE = re.compile(
    r"^\s*(essential|important|optional|pitfall)\s*criteri(?:a|on)\s*:\s*",
    flags=re.IGNORECASE,
)


def parse_category_prefix(description: str) -> Category | None:
    """Return the ``Category`` encoded in a RaR-style description prefix."""
    match = _PREFIX_RE.match(description or "")
    if match is None:
        return None
    return Category.coerce(match.group(1))


def strip_category_prefix(description: str) -> str:
    """Description text with the ``<Category> Criteria:`` prefix removed."""
    return _PREFIX_RE.sub("", description or "").strip()


@dataclass
class Criterion:
    """A single checkable rubric item.

    Attributes
    ----------
    title:
        2-4 word handle, as in the RaR schema.
    description:
        Full criterion text *without* the category prefix. Use
        :meth:`prefixed_description` to render the RaR-formatted string.
    weight:
        Positive magnitude 1-5 for Essential/Important/Optional; for ``PITFALL``
        the stored value is the RaR negative weight (-1 or -2).
    category:
        See :class:`Category`.
    provenance:
        Free-form dict recording *how* the criterion came to exist (which agent
        stage, which evidence). Only the agentic generator populates this; it is
        what makes traces auditable.
    validation:
        Populated by the agentic critic stage: gold pass/fail, negative
        discrimination counts, etc. ``None`` for unvalidated criteria.
    """

    title: str
    description: str
    weight: int
    category: Category = Category.IMPORTANT
    provenance: dict[str, Any] = field(default_factory=dict)
    validation: dict[str, Any] | None = None
    polarity: Polarity | None = None
    polarity_method: str = ""

    def __post_init__(self) -> None:
        self.category = Category.coerce(self.category)
        self.title = (self.title or "").strip()
        desc = (self.description or "").strip()
        # If a generator emitted the RaR prefix inline, believe the prefix over
        # any separately-supplied category and normalise it out of the text.
        inline = parse_category_prefix(desc)
        if inline is not None:
            self.category = inline
            desc = strip_category_prefix(desc)
        self.description = desc
        self.weight = _coerce_weight(self.weight, self.category)
        if self.polarity is None:
            self.polarity, self.polarity_method = detect_polarity(desc, self.category)
        else:
            self.polarity = Polarity.coerce(self.polarity)
            self.polarity_method = self.polarity_method or "declared"

    @property
    def magnitude(self) -> int:
        """Absolute importance, 1-5 scale, sign-independent."""
        return abs(int(self.weight))

    @property
    def is_negative(self) -> bool:
        return self.category is Category.PITFALL

    def satisfied_when(self, literally_true: bool) -> bool:
        """Map the judge's literal yes/no onto "is this good for the response?".

        The judge is deliberately asked only whether the sentence is literally
        true of the response; the good/bad direction is applied here, from the
        stored polarity, so a mixed-convention rubric scores correctly.
        """
        return literally_true if self.polarity is Polarity.POSITIVE else not literally_true

    def prefixed_description(self) -> str:
        return f"{CATEGORY_PREFIXES[self.category]} {self.description}".strip()

    def to_dict(self, *, include_meta: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {
            "title": self.title,
            "description": self.description,
            "weight": int(self.weight),
            "category": self.category.value,
            "polarity": (self.polarity or Polarity.POSITIVE).value,
        }
        if include_meta and self.polarity_method:
            out["polarity_method"] = self.polarity_method
        if include_meta:
            if self.provenance:
                out["provenance"] = self.provenance
            if self.validation is not None:
                out["validation"] = self.validation
        return out

    def to_rar_dict(self) -> dict[str, Any]:
        """Exactly the three-key shape used by the shipped dataset."""
        return {
            "title": self.title,
            "description": self.prefixed_description(),
            "weight": int(self.weight),
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Criterion":
        desc = str(raw.get("description", "") or "")
        category = raw.get("category")
        if category is None:
            category = parse_category_prefix(desc) or Category.IMPORTANT
        return cls(
            title=str(raw.get("title", "") or ""),
            description=desc,
            weight=raw.get("weight", 3),
            category=Category.coerce(category),
            provenance=dict(raw.get("provenance") or {}),
            validation=raw.get("validation"),
            polarity=Polarity.coerce(raw["polarity"]) if raw.get("polarity") else None,
            polarity_method=str(raw.get("polarity_method", "") or ""),
        )


def _coerce_weight(value: Any, category: Category) -> int:
    try:
        weight = int(round(float(value)))
    except (TypeError, ValueError):
        weight = -1 if category is Category.PITFALL else 3
    if category is Category.PITFALL:
        weight = -abs(weight) if weight != 0 else -1
        return max(-2, min(-1, weight))
    weight = abs(weight)
    if weight == 0:
        weight = 1
    return max(1, min(5, weight))


@dataclass
class Rubric:
    """An ordered list of criteria plus bookkeeping metadata."""

    items: list[Criterion] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)

    def __iter__(self) -> Iterator[Criterion]:
        return iter(self.items)

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Criterion:
        return self.items[idx]

    @property
    def source(self) -> str:
        return str(self.meta.get("source", "unknown"))

    def by_category(self, category: Category) -> list[Criterion]:
        return [c for c in self.items if c.category is category]

    def total_positive_weight(self) -> int:
        return sum(c.weight for c in self.items if not c.is_negative)

    def polarity_stats(self) -> dict[str, Any]:
        """Polarity mix, used to report F6 compliance per rubric source."""
        pitfalls = self.by_category(Category.PITFALL)
        methods: dict[str, int] = {}
        for c in self.items:
            methods[c.polarity_method or "unknown"] = methods.get(c.polarity_method or "unknown", 0) + 1
        n_negative = sum(1 for c in self.items if c.polarity is Polarity.NEGATIVE)
        return {
            "n_items": len(self.items),
            "n_pitfall": len(pitfalls),
            "n_negative_polarity": n_negative,
            "negative_polarity_fraction": n_negative / len(self.items) if self.items else 0.0,
            "single_polarity": n_negative == 0,
            "n_unclassified_polarity": methods.get("unclassified", 0),
            "polarity_methods": methods,
        }

    def to_dict(self, *, include_meta: bool = True) -> dict[str, Any]:
        return {
            "items": [c.to_dict(include_meta=include_meta) for c in self.items],
            "meta": dict(self.meta) if include_meta else {},
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, **kwargs)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Rubric":
        items_raw: Iterable[dict[str, Any]]
        if isinstance(raw, list):  # tolerate a bare criterion list
            items_raw, meta = raw, {}
        else:
            items_raw = raw.get("items") or raw.get("rubric") or []
            meta = dict(raw.get("meta") or {})
        return cls(items=[Criterion.from_dict(r) for r in items_raw], meta=meta)

    @classmethod
    def from_criteria(
        cls, criteria: Sequence[Criterion], **meta: Any
    ) -> "Rubric":
        return cls(items=list(criteria), meta=dict(meta))


@dataclass
class Example:
    """One dataset row, domain-tagged and stably identified.

    ``uid`` is a content hash so that the same question keeps the same id across
    reruns, sample sizes and machines — every artefact under ``runs/`` and
    ``results/`` is keyed by it.
    """

    uid: str
    domain: str
    split: str
    row_index: int
    question: str
    reference_answer: str
    question_source: str
    shipped_rubric: Rubric

    def to_dict(self, *, include_rubric: bool = True) -> dict[str, Any]:
        out = {
            "uid": self.uid,
            "domain": self.domain,
            "split": self.split,
            "row_index": self.row_index,
            "question": self.question,
            "reference_answer": self.reference_answer,
            "question_source": self.question_source,
        }
        if include_rubric:
            out["shipped_rubric"] = self.shipped_rubric.to_dict()
        return out

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "Example":
        return cls(
            uid=raw["uid"],
            domain=raw["domain"],
            split=raw.get("split", "val"),
            row_index=int(raw.get("row_index", -1)),
            question=raw["question"],
            reference_answer=raw["reference_answer"],
            question_source=raw.get("question_source", ""),
            shipped_rubric=Rubric.from_dict(raw.get("shipped_rubric") or {}),
        )
