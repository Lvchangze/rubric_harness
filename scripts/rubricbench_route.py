#!/usr/bin/env python3
"""Bound what per-domain routing between existing rubric sources could ever buy.

Why this runs before any router is written
------------------------------------------
`contrastive` recovered the right criteria on CHAT-style cases and lost 16
points on STEM (OPTIMIZATION_LOG §4), which suggests routing: pick the
generator per task type. A router has two parts, though — inferring the type
and choosing well given it — and only the second can be bounded for free. The
per-case verdicts for every candidate are already on disk, so the *oracle*
router (perfect type inference, using the dataset's own domain label, plus the
per-group argmax) can be evaluated with no model call at all.

That number is an upper bound on the whole family: a real router must infer the
type, so it can only do worse. If the oracle does not clear the reference by
dev's minimum detectable difference, there is nothing to implement.

The bound is deliberately loose in two ways, both of which make it *larger*
than any deployable router:

1. It is handed the ground-truth domain label.
2. Each group's winner is chosen on the same 600 cases the result is read on,
   so it collects the winner's curse — with 5 groups and k sources, picking the
   per-group maximum of noisy estimates is biased upward. ``--cross-fit``
   quantifies that bias by choosing on one half and reading on the other.

Also reported: the per-case oracle, which picks the right source case by case.
That is not a routing bound (no router sees per-case outcomes) but it is the
absolute limit of source *selection*, so it says whether the candidates disagree
enough for any selection scheme to matter.

Usage::

    python scripts/rubricbench_route.py --sources framed contrastive balanced
    python scripts/rubricbench_route.py --sources framed contrastive balanced \
        --cross-fit --out results/rubricbench/opt/route_bound.md
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import sys
from pathlib import Path
from typing import Sequence

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness.rubricbench import group_of  # noqa: E402
from rubricbench_compare import GROUPS, LETTER, hit, load, mcnemar, mde  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--sources", nargs="+", required=True,
                   help="candidates the oracle router may choose between")
    p.add_argument("--reference", default="framed")
    p.add_argument("--case-ids", default="results/rubricbench/split.json:dev")
    p.add_argument("--results-dir", nargs="+",
                   default=["results/rubricbench/opt/runs", "results/rubricbench"])
    p.add_argument("--mde", type=float, default=0.038,
                   help="the decision threshold; dev's minimum detectable difference")
    p.add_argument("--cross-fit", action="store_true",
                   help="also choose per-group winners out-of-sample, by halving dev")
    p.add_argument("--out", default=None)
    return p.parse_args()


def acc(hits: Sequence[int | None]) -> float:
    """ACC with the official convention: an unusable verdict counts as wrong."""
    return sum(0 if h is None else h for h in hits) / max(1, len(hits))


def load_any(source: str, roots: Sequence[Path]) -> dict[str, dict]:
    """Resolve a source name, accepting the ``dev_`` tag the sweep writes.

    Candidate runs land as ``dev_<name>_verdicts.jsonl`` under ``opt/runs`` while
    the shared controls are untagged in the results root. Trying both keeps the
    source names in the tables the names used in the log.
    """
    try:
        return load(source, roots)
    except SystemExit:
        return load(f"dev_{source}", roots)


def main() -> int:
    args = parse_args()
    roots = [Path(r) for r in args.results_dir]
    sources = list(dict.fromkeys(args.sources))
    if args.reference not in sources:
        sources.insert(0, args.reference)
    data = {s: load_any(s, roots) for s in sources}

    from rubricbench_split import load_case_ids  # noqa: PLC0415

    wanted = set(load_case_ids(args.case_ids))
    common = sorted(set.intersection(*(set(d) for d in data.values())) & wanted)
    if not common:
        raise SystemExit("the requested sources share no cases in this subset")

    group = {cid: group_of(data[sources[0]][cid]["domain"]) or "other" for cid in common}
    hits = {s: {cid: hit(data[s][cid]) for cid in common} for s in sources}
    by_group: dict[str, list[str]] = {}
    for cid in common:
        by_group.setdefault(group[cid], []).append(cid)
    nonsafety = [cid for cid in common if group[cid] != "safety"]

    lines: list[str] = []

    def emit(text: str = "") -> None:
        print(text)
        lines.append(text)

    emit(f"# Oracle routing bound on {len(common)} cases ({args.case_ids})")
    emit()
    emit(f"Sources: {', '.join('`' + s + '`' for s in sources)}. "
         f"Reference `{args.reference}`. No model calls: every number below is a "
         "recombination of verdicts already on disk.")
    emit()

    # -- per-group accuracy, and who wins each group ----------------------
    emit("## Per-group ACC, and the oracle's pick")
    emit()
    emit("| group | n | " + " | ".join(f"`{s}`" for s in sources) + " | oracle picks |")
    emit("|---|--:|" + "--:|" * len(sources) + "---|")
    pick: dict[str, str] = {}
    for g in GROUPS + ["other"]:
        ids = by_group.get(g)
        if not ids:
            continue
        scores = {s: acc([hits[s][cid] for cid in ids]) for s in sources}
        # Ties go to the reference: a router that switches away from the
        # incumbent on a tie is adding a moving part for nothing.
        best = max(sources, key=lambda s: (scores[s], s == args.reference))
        pick[g] = best
        cells = " | ".join(f"{scores[s]:.4f}" + ("*" if s == best else "") for s in sources)
        emit(f"| {g.upper()} | {len(ids)} | {cells} | `{best}` |")
    emit()
    emit("`*` marks the group winner. Ties resolve to the reference.")
    emit()

    routed = {cid: hits[pick[group[cid]]][cid] for cid in common}
    # Invariant: choosing each group's best can never score below the best fixed
    # source. If this trips, the pick or the regrouping is wrong and the bound
    # would be understated -- which is the direction that silently closes a
    # family that should have stayed open.
    best_fixed = max(acc([hits[s][cid] for cid in common]) for s in sources)
    assert acc([routed[cid] for cid in common]) >= best_fixed - 1e-9, \
        "oracle routing scored below a fixed source; the per-group pick is broken"

    # -- the bound, in both readings --------------------------------------
    emit("## The bound")
    emit()
    emit("| reading | n | `" + args.reference + "` | oracle-routed | Δ | McNemar p | clears +"
         f"{args.mde:.3f}? |")
    emit("|---|--:|--:|--:|--:|--:|---|")
    verdicts: dict[str, tuple[float, bool]] = {}
    for label, ids in (("forward, all", common), ("forward, ex-SAFETY", nonsafety)):
        pairs = [(hits[args.reference][cid], routed[cid]) for cid in ids]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        a_h, b_h = [x[0] for x in pairs], [x[1] for x in pairs]
        delta = (sum(b_h) - sum(a_h)) / len(pairs)
        _, _, p = mcnemar(a_h, b_h)
        clears = delta > args.mde
        verdicts[label] = (delta, clears)
        emit(f"| {label} | {len(pairs)} | {acc(a_h):.4f} | {acc(b_h):.4f} | {delta:+.4f} | "
             f"{p:.3g} | {'**yes**' if clears else 'no'} |")
    emit()

    # -- how much of that Δ is the winner's curse -------------------------
    if args.cross_fit:
        emit("## The same selection, made out of sample")
        emit()
        emit("Dev is halved by a hash of the case id (stable, no seed to pick). "
             "Each half's per-group winner is chosen on the *other* half, so the "
             "reported score no longer contains the selection it came from. The "
             "difference between this and the row above is the winner's curse.")
        emit()
        fold = {cid: int(hashlib.sha1(cid.encode()).hexdigest(), 16) % 2 for cid in common}
        cf: dict[str, int | None] = {}
        for f in (0, 1):
            train = [cid for cid in common if fold[cid] != f]
            train_groups: dict[str, list[str]] = {}
            for cid in train:
                train_groups.setdefault(group[cid], []).append(cid)
            for cid in (c for c in common if fold[c] == f):
                ids = train_groups.get(group[cid], [])
                if not ids:
                    cf[cid] = hits[args.reference][cid]
                    continue
                scores = {s: acc([hits[s][t] for t in ids]) for s in sources}
                best = max(sources, key=lambda s: (scores[s], s == args.reference))
                cf[cid] = hits[best][cid]
        emit("| reading | n | `" + args.reference + "` | cross-fitted routing | Δ | McNemar p |")
        emit("|---|--:|--:|--:|--:|--:|")
        for label, ids in (("forward, all", common), ("forward, ex-SAFETY", nonsafety)):
            pairs = [(hits[args.reference][cid], cf[cid]) for cid in ids]
            pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
            a_h, b_h = [x[0] for x in pairs], [x[1] for x in pairs]
            _, _, p = mcnemar(a_h, b_h)
            emit(f"| {label} | {len(pairs)} | {acc(a_h):.4f} | {acc(b_h):.4f} | "
                 f"{(sum(b_h) - sum(a_h)) / len(pairs):+.4f} | {p:.3g} |")
        emit()

    # -- what any selection scheme could reach ----------------------------
    emit("## For scale: the per-case oracle")
    emit()
    emit("Not a routing bound — no router sees per-case outcomes — but it says "
         "whether the candidates disagree enough for selection to be worth "
         "anything at all.")
    emit()
    emit("| reading | n | union of correct | Δ vs reference |")
    emit("|---|--:|--:|--:|")
    for label, ids in (("forward, all", common), ("forward, ex-SAFETY", nonsafety)):
        best_case = [max((hits[s][cid] or 0) for s in sources) for cid in ids]
        ref_h = [hits[args.reference][cid] for cid in ids]
        emit(f"| {label} | {len(ids)} | {acc(best_case):.4f} | "
             f"{acc(best_case) - acc(ref_h):+.4f} |")
    emit()

    # -- is that complementarity reachable without labels? ----------------
    # The per-case oracle is large, which raises the obvious question of whether
    # any label-free rule gets at it. Majority vote over the sources' *verdicts*
    # is the cheapest such rule and, unlike the oracle, it is implementable: it
    # costs one judge pass per source and no labels. It is not a rubric-writing
    # method and is not a candidate here -- it is reported because it says
    # whether the spread between sources is signal or noise.
    emit("## Diagnostic: is the per-case spread reachable without labels?")
    emit()
    emit("Majority vote over the sources' forward verdicts. Not a candidate "
         "(it judges k times per case, and it is not a rubric-generation "
         "method) — reported only because it distinguishes usable "
         "complementarity from noise.")
    emit()
    emit("| reading | n | majority vote | Δ vs reference | McNemar p |")
    emit("|---|--:|--:|--:|--:|")
    votes: dict[str, int | None] = {}
    for cid in common:
        picks = [data[s][cid].get("forward") for s in sources]
        picks = [p for p in picks if p is not None]
        if not picks:
            votes[cid] = None
            continue
        top = max(set(picks), key=picks.count)
        # A tie is no information, so it falls back to the incumbent rather than
        # to whichever letter sorts first.
        if picks.count(top) * 2 == len(picks):
            votes[cid] = hits[args.reference][cid]
        else:
            votes[cid] = int(LETTER[top] == data[sources[0]][cid]["label"])
    for label, ids in (("forward, all", common), ("forward, ex-SAFETY", nonsafety)):
        pairs = [(hits[args.reference][cid], votes[cid]) for cid in ids]
        pairs = [(a, b) for a, b in pairs if a is not None and b is not None]
        a_h, b_h = [x[0] for x in pairs], [x[1] for x in pairs]
        _, _, p = mcnemar(a_h, b_h)
        emit(f"| {label} | {len(pairs)} | {acc(b_h):.4f} | "
             f"{(sum(b_h) - sum(a_h)) / len(pairs):+.4f} | {p:.3g} |")
    emit()

    # -- and what this sample size can see at all -------------------------
    pairs = [(hits[args.reference][cid], routed[cid]) for cid in nonsafety]
    nd = sum(1 for a, b in pairs if a != b)
    emit(f"Oracle routing and `{args.reference}` disagree on {nd}/{len(pairs)} "
         f"ex-SAFETY cases; the smallest |Δ| an exact McNemar could call "
         f"significant there is {mde(nd, len(pairs)):.4f}.")
    emit()

    # -- best fixed pair, as a sanity check on the group structure --------
    # If some *single* source beats the routed combination, the per-group
    # differences were noise and the routing story was never there.
    emit("## Is the group structure real?")
    emit()
    emit("| source (fixed, no routing) | ex-SAFETY ACC |")
    emit("|---|--:|")
    for s in sources:
        emit(f"| `{s}` | {acc([hits[s][cid] for cid in nonsafety]):.4f} |")
    emit(f"| **oracle-routed** | **{acc([routed[cid] for cid in nonsafety]):.4f}** |")
    emit()
    if len(sources) > 2:
        group_acc = {g: {s: acc([hits[s][t] for t in ids]) for s in sources}
                     for g, ids in by_group.items()}
        combos = []
        for r in range(2, len(sources) + 1):
            for combo in itertools.combinations(sources, r):
                winner = {g: max(combo, key=lambda s: (group_acc[g][s], s == args.reference))
                          for g in by_group}
                sub = acc([hits[winner[group[cid]]][cid] for cid in nonsafety])
                combos.append((sub, combo))
        best_acc, best_combo = max(combos)
        emit(f"Best subset to route between (ex-SAFETY): "
             f"{', '.join('`' + s + '`' for s in best_combo)} at {best_acc:.4f}.")
        emit()

    delta_ns, clears_ns = verdicts["forward, ex-SAFETY"]
    emit("## Verdict")
    emit()
    emit(f"Oracle routing with ground-truth labels moves ex-SAFETY ACC by "
         f"{delta_ns:+.4f}. The pre-registered rule was to open the routing "
         f"family only if this exceeded +{args.mde:.3f}. It "
         f"{'does' if clears_ns else 'does not'}, so the family is "
         f"{'open' if clears_ns else '**closed**'}.")
    emit()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"wrote {args.out}")
    print(json.dumps({"delta_ex_safety": delta_ns, "clears": clears_ns}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
