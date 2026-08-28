"""Dataset loading and sampling for the RaR parquet corpora.

Exposes the two things the rest of the harness needs: a reproducible sample of
:class:`~harness.schema.Example` objects, and the shipped RaR rubric parsed into
the canonical :class:`~harness.schema.Rubric` structure.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any, Iterable, Sequence

import pandas as pd

from .schema import Criterion, Example, Rubric, parse_category_prefix

logger = logging.getLogger(__name__)

__all__ = [
    "DOMAINS",
    "data_root",
    "load_split",
    "parse_shipped_rubric",
    "sample_examples",
    "load_examples_jsonl",
    "write_examples_jsonl",
]

DOMAINS: tuple[str, ...] = ("rar_science", "rar_medicine")

_DEFAULT_ROOT = Path(__file__).resolve().parent.parent / "data"


def data_root(root: str | Path | None = None) -> Path:
    return Path(root) if root else _DEFAULT_ROOT


def _uid(domain: str, question: str, reference_answer: str) -> str:
    digest = hashlib.sha1(
        f"{domain}\x00{question}\x00{reference_answer}".encode("utf-8")
    ).hexdigest()
    return f"{domain.replace('rar_', '')[:3]}-{digest[:12]}"


def parse_shipped_rubric(raw: Any, *, source: str = "shipped") -> Rubric:
    """Convert the parquet ``rubric`` column into a :class:`Rubric`.

    The shipped rows carry ``{title, description, weight}`` with the category
    encoded as a text prefix on ``description``; :class:`Criterion` strips and
    normalises that automatically.
    """
    items: list[Criterion] = []
    if raw is None:
        return Rubric(items=[], meta={"source": source})
    for entry in raw:
        if hasattr(entry, "item") and not isinstance(entry, (dict, str)):
            entry = entry.item()
        if isinstance(entry, str):
            try:
                entry = json.loads(entry)
            except Exception:  # noqa: BLE001 - fall back to prefix-only parsing
                entry = {"title": "", "description": entry, "weight": 3}
        entry = dict(entry)
        description = str(entry.get("description", ""))
        category = parse_category_prefix(description)
        items.append(
            Criterion(
                title=str(entry.get("title", "")),
                description=description,
                weight=entry.get("weight", 3),
                category=category or "Important",
                provenance={"stage": "shipped"},
            )
        )
    return Rubric(items=items, meta={"source": source, "n_items": len(items)})


def load_split(
    domain: str, split: str = "val", *, root: str | Path | None = None
) -> pd.DataFrame:
    path = data_root(root) / domain / f"{split}-00000-of-00001.parquet"
    if not path.exists():
        raise FileNotFoundError(f"missing parquet: {path}")
    return pd.read_parquet(path)


def _row_to_example(domain: str, split: str, idx: int, row: pd.Series) -> Example:
    question = str(row["question"])
    reference = str(row["reference_answer"])
    return Example(
        uid=_uid(domain, question, reference),
        domain=domain,
        split=split,
        row_index=int(idx),
        question=question,
        reference_answer=reference,
        question_source=str(row.get("question_source", "")),
        shipped_rubric=parse_shipped_rubric(row.get("rubric")),
    )


def sample_examples(
    domain: str,
    *,
    split: str = "val",
    n: int = 20,
    seed: int = 1234,
    root: str | Path | None = None,
    min_question_chars: int = 80,
    max_question_chars: int = 6000,
    max_reference_chars: int = 12000,
    min_rubric_items: int = 4,
    question_sources: Sequence[str] | None = None,
    deduplicate_questions: bool = True,
) -> list[Example]:
    """Deterministically draw ``n`` usable examples from one domain/split.

    Filtering keeps the pilot honest rather than cherry-picking: it only removes
    rows that are degenerate for *every* method (empty/stub questions, reference
    answers too long to fit judging prompts comfortably, or shipped rubrics with
    too few items to compare against).

    ``deduplicate_questions`` matters more than it looks: 9.15% of RaR-Science
    rows are duplicate questions and 12.3% of its test split also appears in
    train (``docs/01_data_forensics.md`` §10). Sampling a question twice would
    make paired statistics treat one question as two independent observations.
    """
    df = load_split(domain, split, root=root)
    if question_sources:
        df = df[df["question_source"].isin(list(question_sources))]
    if deduplicate_questions:
        df = df[~df["question"].astype(str).str.strip().duplicated(keep="first")]

    q_len = df["question"].astype(str).str.len()
    r_len = df["reference_answer"].astype(str).str.len()
    mask = (
        q_len.between(min_question_chars, max_question_chars)
        & r_len.between(40, max_reference_chars)
        & df["rubric_count"].fillna(0).astype(int).ge(min_rubric_items)
    )
    pool = df[mask]
    if len(pool) < n:
        logger.warning(
            "%s/%s: only %d rows pass filters, requested %d", domain, split, len(pool), n
        )
    rng = random.Random(f"{seed}:{domain}:{split}")
    indices = sorted(pool.index.tolist())
    rng.shuffle(indices)
    chosen = indices[:n]
    return [_row_to_example(domain, split, i, df.loc[i]) for i in chosen]


def write_examples_jsonl(examples: Iterable[Example], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex.to_dict(), ensure_ascii=False) + "\n")
    return path


def load_examples_jsonl(path: str | Path) -> list[Example]:
    out: list[Example] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(Example.from_dict(json.loads(line)))
    return out
