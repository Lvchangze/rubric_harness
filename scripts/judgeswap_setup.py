#!/usr/bin/env python
"""Build the judge-swap run directory: same inputs, balanced paired subset.

The swap experiment's only free variable is the judge model, so every other
input is *reused verbatim* rather than regenerated:

* ``rubrics_{source}.jsonl`` — hard-linked from the pilot, all six sources.
* ``responses.jsonl`` — hard-linked; the degraded ladder is shared across
  sources by construction and is byte-identical for every one of them.
* ``examples.jsonl`` — the only file that differs: it is *narrowed* to the
  chosen subset, which is how ``harness.pipeline.load_all_rubrics`` scopes the
  evaluation without touching any rubric.

The subset is drawn from the *original run's paired queue* — questions where the
incumbent judge produced a complete response ladder under all six sources — so
every question in the swap can be compared against the original judge on the
same question, which is far stronger than two independent samples. Within that
queue the two domains are balanced to equal size, because the headline domain
contrast (a science-only z_separation effect) is the thing being rechecked and
unequal domain sizes would confound it with precision.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

TIER_ORDER = [
    "gold", "missing_step", "terse_correct", "right_method_wrong_answer",
    "numeric_error", "verbose_empty", "off_topic",
]

SOURCES = ["shipped", "baseline", "agentic", "agentic-noval",
           "agentic-goldonly", "agentic-negonly"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    out = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def paired_queue(rows: list[dict[str, Any]], sources: list[str]) -> tuple[list[str], dict[str, str]]:
    """uids whose ladder is complete for every source under the original judge."""
    have: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    domains: dict[str, str] = {}
    for row in rows:
        if row.get("error") or not row.get("usable", True):
            continue
        if not isinstance(row.get("score"), (int, float)):
            continue
        uid, src, variant = row.get("uid"), row.get("rubric_source"), row.get("variant")
        if not (uid and src and variant):
            continue
        have[uid][src].add(variant)
        if row.get("domain"):
            domains[uid] = str(row["domain"])
    keep = [
        uid for uid, by_src in have.items()
        if all(set(TIER_ORDER) <= by_src.get(s, set()) for s in sources)
    ]
    return sorted(keep), domains


def link_or_copy(src: Path, dst: Path) -> str:
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-run", default="runs/pilot_v2")
    ap.add_argument("--dst-run", default="runs/judgeswap")
    ap.add_argument("--sources", nargs="*", default=SOURCES)
    ap.add_argument("--per-domain", type=int, default=0,
                    help="questions per domain; 0 = the largest balanced subset")
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()

    src = Path(args.src_run)
    dst = Path(args.dst_run)
    dst.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(src / "metrics" / "discriminative_per_response_rows.jsonl")
    queue, domains = paired_queue(rows, args.sources)
    by_domain: dict[str, list[str]] = defaultdict(list)
    for uid in queue:
        by_domain[domains.get(uid, "?")].append(uid)

    print(f"original paired queue: n={len(queue)}")
    for dom, uids in sorted(by_domain.items()):
        print(f"  {dom}: {len(uids)}")

    per_domain = args.per_domain or min(len(v) for v in by_domain.values())
    rng = random.Random(args.seed)
    chosen: list[str] = []
    for dom in sorted(by_domain):
        uids = sorted(by_domain[dom])
        if len(uids) <= per_domain:
            pick = uids
        else:
            pick = sorted(rng.sample(uids, per_domain))
        chosen.extend(pick)
        print(f"  -> {dom}: taking {len(pick)}")
    chosen_set = set(chosen)

    # examples.jsonl: narrowed to the subset, order preserved from the pilot.
    all_examples = read_jsonl(src / "examples.jsonl")
    subset = [e for e in all_examples if e.get("uid") in chosen_set]
    missing = chosen_set - {e.get("uid") for e in subset}
    if missing:
        raise SystemExit(f"{len(missing)} chosen uids absent from examples.jsonl")
    with (dst / "examples.jsonl").open("w", encoding="utf-8") as fh:
        for e in subset:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")
    print(f"wrote {dst / 'examples.jsonl'} with {len(subset)} examples")

    # Everything else is reused byte-for-byte.
    modes = {}
    for source in args.sources:
        name = f"rubrics_{source}.jsonl"
        modes[name] = link_or_copy(src / name, dst / name)
    modes["responses.jsonl"] = link_or_copy(src / "responses.jsonl", dst / "responses.jsonl")
    print("reused inputs:", json.dumps(modes))

    # Fingerprint the reused inputs so the report can assert they were untouched.
    import hashlib

    fp = {}
    for name in list(modes):
        h = hashlib.sha256((dst / name).read_bytes()).hexdigest()[:16]
        fp[name] = {"sha256_16": h, "bytes": (dst / name).stat().st_size, "mode": modes[name]}

    manifest = {
        "src_run": str(src),
        "dst_run": str(dst),
        "sources": args.sources,
        "original_paired_queue_n": len(queue),
        "original_paired_queue_by_domain": {k: len(v) for k, v in sorted(by_domain.items())},
        "per_domain_selected": per_domain,
        "n_selected": len(chosen),
        "selected_by_domain": {
            dom: sum(1 for u in chosen if domains.get(u) == dom)
            for dom in sorted(by_domain)
        },
        "seed": args.seed,
        "uids": chosen,
        "input_fingerprints": fp,
    }
    (dst / "judgeswap_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"wrote {dst / 'judgeswap_manifest.json'}")
    print(json.dumps({k: v for k, v in manifest.items() if k != "uids"}, indent=2))


if __name__ == "__main__":
    main()
