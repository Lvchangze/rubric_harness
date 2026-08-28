"""Shared loading / tokenisation helpers for the RaR rubric forensics scripts."""

from __future__ import annotations

import pathlib
import re
from typing import Iterable

import pandas as pd

REPO = pathlib.Path(__file__).resolve().parent.parent
DATA = REPO / "data"
OUT = REPO / "analysis" / "out"
DOMAINS = ("rar_science", "rar_medicine")
SPLITS = ("train", "val", "test")

CATEGORY_RE = re.compile(r"^\s*([A-Za-z]+)\s+Criteria\s*:\s*", re.IGNORECASE)
CANONICAL_CATEGORIES = ("Essential", "Important", "Optional", "Pitfall")

WORD_RE = re.compile(r"[A-Za-z][A-Za-z\-']+")
TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-'\.]*")
NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")
SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z(])")

# Very small English stoplist; enough to keep n-gram / anchor statistics honest
# without pulling in an extra dependency.
STOPWORDS = frozenset(
    """a an the and or but if then than that this these those of to in on at by for with
    from as is are was were be been being it its it's they them their there here which who
    whom whose what when where why how all any both each few more most other some such no
    nor not only own same so too very can will just should now must may might could would
    does do did doing done have has had having i you he she we us our your his her also
    into about over under between within without across during before after above below
    response answer criteria criterion mention mentions mentioned state states stated
    explain explains explained include includes included provide provides provided must
    should may correctly correct clearly clear
    """.split()
)


def load_domain(domain: str, splits: Iterable[str] = SPLITS) -> pd.DataFrame:
    """Concatenate all splits of one domain into a single frame with a `split` column."""
    frames = []
    for split in splits:
        path = DATA / domain / f"{split}-00000-of-00001.parquet"
        frame = pd.read_parquet(path)
        frame["split"] = split
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    df["qid"] = [f"{domain}:{i}" for i in range(len(df))]
    return df


def explode_criteria(df: pd.DataFrame, domain: str) -> pd.DataFrame:
    """Flatten the nested `rubric` column into one row per criterion."""
    rows = []
    for row in df.itertuples(index=False):
        for position, item in enumerate(row.rubric):
            description = item.get("description") or ""
            title = item.get("title") or ""
            weight = item.get("weight")
            match = CATEGORY_RE.match(description)
            raw_category = match.group(1) if match else None
            body = description[match.end():] if match else description
            rows.append(
                {
                    "domain": domain,
                    "qid": row.qid,
                    "split": row.split,
                    "pos": position,
                    "n_items": len(row.rubric),
                    "title": title,
                    "description": description,
                    "body": body,
                    "weight": weight,
                    "raw_category": raw_category,
                    "category": _canonical_category(raw_category),
                    "has_prefix": match is not None,
                }
            )
    return pd.DataFrame(rows)


def _canonical_category(raw: str | None) -> str:
    if raw is None:
        return "MISSING"
    lowered = raw.lower()
    for canonical in CANONICAL_CATEGORIES:
        if lowered == canonical.lower():
            return canonical
    return "MALFORMED"


def words(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def content_tokens(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.lower()) if t not in STOPWORDS and len(t) > 2]


def ngrams(tokens: list[str], n: int) -> set[tuple[str, ...]]:
    if len(tokens) < n:
        return set()
    return {tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


def pct(numerator: float, denominator: float) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0
