#!/usr/bin/env python3
"""Cut RubricBench into a development half and a held-out half, once.

Why this exists
---------------
The best generator in this study (``framed``) was written after reading
``baseline``'s failures on all 1,147 cases. That is a legitimate way to get an
idea and an illegitimate way to measure one: every case it was tuned on is a
case its score is optimistic about. Continuing to iterate against the full
benchmark would compound that until none of the numbers meant anything.

So the benchmark is cut in two before any further work. Every subsequent
iteration reads only ``dev``; ``holdout`` is spent a bounded number of times and
each spend is logged. The split is written to disk and never regenerated -- a
split that can be recomputed with a different seed is not a split.

Stratification is by the five official domain groups, because the headline score
is a group-weighted average and an unstratified cut would change the mixture
between the two halves. Group sizes are allocated by largest remainder so the
two halves keep the benchmark's composition to within one case per group.

Usage::

    python scripts/rubricbench_split.py --dev-size 600 --seed 20260831
    python scripts/rubricbench_split.py --verify          # re-derive and diff
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.rubricbench import group_of, load_cases  # noqa: E402

SPLIT_PATH = Path("results/rubricbench/split.json")


def allocate(sizes: dict[str, int], total: int, target: int) -> dict[str, int]:
    """Largest-remainder allocation of ``target`` slots across groups.

    Proportional rounding one group at a time can drift by several cases; the
    largest-remainder method keeps every group within one of its exact share and
    makes the totals add up without a fudge factor at the end.
    """
    exact = {g: n * target / total for g, n in sizes.items()}
    alloc = {g: int(v) for g, v in exact.items()}
    short = target - sum(alloc.values())
    order = sorted(exact, key=lambda g: (-(exact[g] - alloc[g]), g))
    for g in order[:short]:
        alloc[g] += 1
    return alloc


def build(dev_size: int, seed: int) -> dict:
    cases = load_cases()
    by_group: dict[str, list[str]] = collections.defaultdict(list)
    domain_of: dict[str, str] = {}
    for case in cases:
        by_group[group_of(case.domain) or "other"].append(case.case_id)
        domain_of[case.case_id] = case.domain

    sizes = {g: len(v) for g, v in by_group.items()}
    alloc = allocate(sizes, len(cases), dev_size)

    rng = random.Random(seed)
    dev: list[str] = []
    holdout: list[str] = []
    for group in sorted(by_group):
        # Sort before sampling: dict order must not be able to influence the cut.
        members = sorted(by_group[group])
        picked = set(rng.sample(members, alloc[group]))
        dev.extend(m for m in members if m in picked)
        holdout.extend(m for m in members if m not in picked)

    return {
        "created": "2026-08-31",
        "seed": seed,
        "dev_size": len(dev),
        "holdout_size": len(holdout),
        "stratified_by": "official domain group (chat/code/if/safety/stem)",
        "allocation": {g: {"total": sizes[g], "dev": alloc[g], "holdout": sizes[g] - alloc[g]}
                       for g in sorted(sizes)},
        "note": ("Frozen. Every iteration is developed on dev; holdout is spent at most "
                 "three times and each spend is recorded in opt/HOLDOUT_LOG.md."),
        "dev": sorted(dev),
        "holdout": sorted(holdout),
    }


def load_case_ids(spec: str) -> list[str]:
    """Resolve ``--case-ids`` for the runner and the comparison script.

    Accepts ``path/to/split.json:dev`` (a named key inside a split file), a JSON
    file holding a bare list of ids, or a text file with one id per line.
    """
    path_part, _, key = spec.partition(":")
    path = Path(path_part)
    if not path.exists():
        raise SystemExit(f"--case-ids: no such file {path}")
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        raw = json.loads(text)
        if isinstance(raw, dict):
            if not key:
                raise SystemExit(
                    f"--case-ids: {path} holds an object; name the key, e.g. '{path}:dev'")
            if key not in raw:
                raise SystemExit(f"--case-ids: {path} has no key {key!r}")
            raw = raw[key]
        if not isinstance(raw, list):
            raise SystemExit(f"--case-ids: {path}:{key} is not a list")
        return [str(x) for x in raw]
    return [line.strip() for line in text.splitlines() if line.strip()]


def describe(split: dict) -> str:
    cases = {c.case_id: c for c in load_cases()}
    out = [f"dev={split['dev_size']}  holdout={split['holdout_size']}  seed={split['seed']}", ""]
    out.append(f"{'group':<8} {'total':>6} {'dev':>6} {'hold':>6}")
    for g, a in split["allocation"].items():
        out.append(f"{g:<8} {a['total']:>6} {a['dev']:>6} {a['holdout']:>6}")
    out.append("")
    # Balance on axes the split did not control for; if these drifted badly the
    # two halves would not be comparable even though the group mixture matched.
    for axis, fn in (("domain", lambda c: c.domain),
                     ("source", lambda c: c.source),
                     ("label", lambda c: str(c.label))):
        out.append(f"{axis:<18} {'dev%':>7} {'hold%':>7}")
        dev_v = collections.Counter(fn(cases[i]) for i in split["dev"])
        hold_v = collections.Counter(fn(cases[i]) for i in split["holdout"])
        for key in sorted(set(dev_v) | set(hold_v), key=lambda k: -(dev_v[k] + hold_v[k])):
            out.append(f"  {key:<16} {dev_v[key] / split['dev_size']:>6.1%} "
                       f"{hold_v[key] / split['holdout_size']:>6.1%}")
        out.append("")
    return "\n".join(out)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dev-size", type=int, default=600)
    p.add_argument("--seed", type=int, default=20260831)
    p.add_argument("--out", default=str(SPLIT_PATH))
    p.add_argument("--verify", action="store_true",
                   help="re-derive the split and confirm the file on disk matches")
    args = p.parse_args()

    split = build(args.dev_size, args.seed)
    out = Path(args.out)

    if args.verify:
        if not out.exists():
            raise SystemExit(f"{out} does not exist yet")
        on_disk = json.loads(out.read_text(encoding="utf-8"))
        same = (on_disk["dev"] == split["dev"] and on_disk["holdout"] == split["holdout"])
        print(f"{out}: {'reproduces exactly' if same else 'DOES NOT MATCH the seed'}")
        print(describe(on_disk))
        return 0 if same else 1

    if out.exists():
        raise SystemExit(f"{out} already exists -- the split is frozen; delete it deliberately "
                         f"if you really mean to recut, and say so in the log")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(split, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(describe(split))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
