#!/usr/bin/env python3
"""Rehearse the interrupted rollout run without touching the network.

The run was cut in half by a serving outage and will be finished by somebody who
was not here when it broke. Before handing that over, the resume path has to be
shown to work -- specifically the two failure modes this project has already
paid for:

1. **Overwriting writes.** Three separate artefacts have been destroyed by a
   partial run persisting its own view of a shared file: 8,000 judge verdicts,
   the zero-LLM metric rows, then a screening checkpoint. ``rollouts.jsonl`` is
   the next candidate, because it holds all 780 screened questions while the
   resumed run only handles the 128 that were selected.
2. **Silent cohort drift.** ``compute_table`` intersects uids across whatever
   sources it is given, so a half-finished source shrinks the paired queue
   without raising anything.

Every check runs on a copy under a temporary directory, so a failing rehearsal
cannot damage the real artefacts. Nothing here calls an LLM: the generator and
the judge are replaced by stubs, which is enough because the question being
asked is about orchestration and file handling, not about model output.
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harness.eval.rollouts import (  # noqa: E402
    RolloutSet,
    load_rollout_sets,
    merge_rollout_sets,
)

def _load(name: str):
    """Import a sibling CLI script by path.

    ``scripts/`` is deliberately not a package -- every script there is meant to
    be run as ``python3 scripts/foo.py`` -- so this loads by file path rather
    than adding an ``__init__.py`` just to let the rehearsal import two helpers.
    """
    import importlib.util

    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_dryrun_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


REPO = Path(__file__).resolve().parent.parent
RUN = REPO / "runs" / "rollout_v2"
SOURCES = ["shipped", "baseline", "agentic", "agentic-goldonly", "agentic-negonly"]

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'OK  ' if ok else 'FAIL'} {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


def check_rollout_merge(tmp: Path) -> None:
    """Widening k for 128 questions must not drop the other 652."""
    print("\n== rollouts.jsonl survives a k=4 -> k=8 resume ==")
    src = RUN / "rollouts.jsonl"
    target = tmp / "rollouts.jsonl"
    shutil.copy(src, target)

    before = load_rollout_sets(target)
    selected = {
        json.loads(line)["uid"]
        for line in (RUN / "examples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    check("checkpoint holds the whole screen", len(before) == 780, f"{len(before)} sets")
    check("selected questions are a subset", selected <= set(before), f"{len(selected)} selected")

    # Exactly what build_all_rollout_sets does on resume: it keeps only the sets
    # already drawn at k>=8, re-derives the rest, and persists its own view.
    kept = {u: rs for u, rs in before.items() if rs.meta.get("k_requested", 0) >= 8}
    redrawn = {}
    for uid in selected:
        rollout_set = RolloutSet.from_dict(before[uid].to_dict())
        rollout_set.meta["k_requested"] = 8
        redrawn[uid] = rollout_set
    run_view = {**kept, **redrawn}
    check("a run's own view is much smaller than the file",
          len(run_view) < len(before), f"{len(run_view)} vs {len(before)}")

    merge_rollout_sets(run_view, target)
    after = load_rollout_sets(target)
    check("no screened question is lost", len(after) == len(before),
          f"{len(before)} -> {len(after)}")
    check("selected questions were updated to k=8",
          all(after[u].meta.get("k_requested") == 8 for u in selected))
    check("untouched questions keep k=4",
          all(after[u].meta.get("k_requested") == 4
              for u in set(after) - selected - set(kept)))

    # And the regression this replaces: the plain writer would have truncated.
    naive = tmp / "naive.jsonl"
    shutil.copy(src, naive)
    from harness.eval.rollouts import save_rollout_sets
    save_rollout_sets(run_view, naive)
    check("the non-merging writer really would have truncated",
          len(load_rollout_sets(naive)) < len(before),
          f"{len(load_rollout_sets(naive))} sets would remain")


def check_row_merge(tmp: Path) -> None:
    """Re-running one source must not disturb rows belonging to the others."""
    print("\n== metric rows survive a per-source re-run ==")
    _write_rows_preserving_other_sources = _load("eval_rubrics")._write_rows_preserving_other_sources

    path = tmp / "rows.jsonl"
    original = (
        [{"uid": f"q{i}", "rubric_source": "baseline", "v": 1} for i in range(5)]
        + [{"uid": f"q{i}", "rubric_source": "agentic", "v": 1} for i in range(5)]
        + [{"uid": f"q{i}", "pair": "a_vs_b", "v": 1} for i in range(3)]  # no source
    )
    path.write_text(
        "\n".join(json.dumps(r) for r in original) + "\n", encoding="utf-8"
    )

    incoming = [{"uid": f"q{i}", "rubric_source": "agentic", "v": 2} for i in range(5)]
    _write_rows_preserving_other_sources(path, incoming, ["agentic"])
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    by_source = {}
    for row in rows:
        by_source.setdefault(row.get("rubric_source", "<none>"), []).append(row)

    check("other source untouched", len(by_source.get("baseline", [])) == 5,
          f"{len(by_source.get('baseline', []))} rows")
    check("re-run source replaced, not appended",
          len(by_source.get("agentic", [])) == 5,
          f"{len(by_source.get('agentic', []))} rows")
    check("re-run source actually updated",
          all(r["v"] == 2 for r in by_source.get("agentic", [])))
    check("source-less rows (head-to-head) not duplicated",
          len(by_source.get("<none>", [])) == 3,
          f"{len(by_source.get('<none>', []))} rows")


def check_pairing_guard(tmp: Path) -> None:
    """A half-finished source must fail loudly, not shrink the cohort quietly."""
    print("\n== a partially generated source cannot silently shrink the cohort ==")
    import pandas as pd

    compute_table = _load("aggregate_results").compute_table

    rows = []
    for source, n in [("baseline", 20), ("agentic", 20), ("agentic-negonly", 12)]:
        for i in range(n):
            rows.append({"uid": f"q{i}", "rubric_source": source, "domain": "d",
                         "_family": "grounding",
                         "generic_criterion_rate": 0.1 * (i % 3)})
    df = pd.DataFrame(rows)

    full = compute_table(df, sources=["baseline", "agentic"],
                         reference="baseline", iters=200, seed=1)
    partial = compute_table(df, sources=["baseline", "agentic", "agentic-negonly"],
                            reference="baseline", iters=200, seed=1)
    n_full = full[0]["n_paired"] if full else None
    n_partial = partial[0]["n_paired"] if partial else None
    check("complete sources pair on all questions", n_full == 20, f"n={n_full}")
    check("adding a half-finished source does shrink the queue",
          n_partial == 12, f"n={n_partial}")
    check("--expect-paired would catch it", n_partial != n_full,
          "so the resume script must pass --expect-paired")


def check_stage_coverage() -> None:
    """The resume script must actually cover every remaining stage."""
    print("\n== run_rollout_v2.sh covers the remaining stages ==")
    script = (REPO / "scripts" / "run_rollout_v2.sh").read_text(encoding="utf-8")
    for stage, needle in [
        ("rubric generation", "gen_rubrics.py"),
        ("rollout widening to k=8", "build_rollouts.py"),
        ("judging", "eval_rubrics.py"),
        ("aggregation", "aggregate_results.py"),
        ("rollout tables", "rollout_report.py"),
        ("reward hacking probe", "reward_hacking_probe.py"),
        ("cohort assertion", "--expect-paired"),
        ("concurrent-run lock", "mkdir \"$LOCK\""),
    ]:
        check(f"stage present: {stage}", needle in script)

    watchdog = (REPO / "scripts" / "resume_when_up.sh").read_text(encoding="utf-8")
    check("watchdog calls the resume script", "run_rollout_v2.sh" in watchdog)
    check("watchdog needs consecutive healthy probes", "NEEDED_OK" in watchdog)


def check_generation_resume() -> None:
    """Generation must skip the 62 questions that already have rubrics."""
    print("\n== generation resumes rather than regenerates ==")
    from harness.tracing import RunDir

    run_dir = RunDir("runs", "rollout_v2")
    selected = {
        json.loads(line)["uid"]
        for line in (RUN / "examples.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    for source, expect_done in [("shipped", 128), ("baseline", 128), ("agentic", 62),
                                ("agentic-goldonly", 62), ("agentic-negonly", 62)]:
        done = run_dir.existing_uids(f"rubrics_{source}")
        todo = selected - done
        check(f"{source}: {len(done)} done, {len(todo)} to do",
              len(done & selected) == expect_done)


def main() -> int:
    print("Dry run of the rollout_v2 resume path. No LLM calls are made.")
    have_run = (RUN / "examples.jsonl").exists() and (RUN / "rollouts.jsonl").exists()
    with tempfile.TemporaryDirectory() as raw:
        tmp = Path(raw)
        if have_run:
            check_rollout_merge(tmp)
        else:
            print(f"\n== skipped: {RUN} not present, artefact checks need it ==")
        check_row_merge(tmp)
        check_pairing_guard(tmp)
    check_stage_coverage()
    if have_run:
        check_generation_resume()

    print()
    if failures:
        print(f"{len(failures)} CHECK(S) FAILED:")
        for name in failures:
            print(f"  - {name}")
        return 1
    print("ALL RESUME CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
