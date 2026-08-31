#!/usr/bin/env bash
# Cheap end-to-end health check: no LLM calls, a few seconds.
# Verifies every module imports, every CLI parses its arguments, the numeric
# helpers pass their own tests, and the baseline prompts still match the paper.
set -uo pipefail
cd "$(dirname "$0")/.." || exit 1

fail=0

echo "== byte-compile =="
python -m compileall -q harness scripts || fail=1

echo
echo "== module imports =="
python - <<'PY' || fail=1
import importlib
mods = [
    "harness.schema", "harness.llm", "harness.data", "harness.config",
    "harness.tracing", "harness.pipeline", "harness.generators",
    "harness.generators.agentic", "harness.generators.single_pass",
    "harness.generators.shipped", "harness.prompts.rar_original",
    "harness.prompts.agentic", "harness.prompts.eval_prompts",
    "harness.prompts.responses", "harness.eval.judge", "harness.eval.stats",
    "harness.eval.responses", "harness.eval.discriminative",
    "harness.eval.transfer", "harness.eval.coverage", "harness.eval.intrinsic",
    "harness.eval.headtohead", "harness.eval.grounding", "harness.eval.lint",
    "harness.eval.adaptivity",
    "harness.prompts.tools", "harness.tools", "harness.tools.base",
    "harness.tools.sandbox", "harness.tools.compute", "harness.tools.inspection",
    "harness.tools.verification", "harness.tools.negatives",
    "harness.rubricbench",
]
bad = 0
for m in mods:
    try:
        importlib.import_module(m)
    except Exception as exc:
        bad += 1
        print(f"  FAIL {m}: {type(exc).__name__}: {exc}")
print(f"  {len(mods) - bad}/{len(mods)} modules import")
raise SystemExit(1 if bad else 0)
PY

echo
echo "== CLI argument parsing =="
for s in gen_rubrics build_responses eval_rubrics aggregate_results \
         main_table report_facts case_study check_prompt_fidelity _repair_jsonl \
         confound_audit rubricbench_run rubricbench_compare rubricbench_verify \
         rubricbench_split rubricbench_gen rubricbench_failures rubricbench_route \
         rubricbench_rubric_stats rubricbench_tilt_audit; do
  if python "scripts/$s.py" --help >/dev/null 2>&1; then
    echo "  OK   $s"
  else
    echo "  FAIL $s"
    fail=1
  fi
done

echo
echo "== metric dispatcher binds every signature =="
python - <<'PY' || fail=1
import importlib.util, inspect
spec = importlib.util.spec_from_file_location("er", "scripts/eval_rubrics.py")
er = importlib.util.module_from_spec(spec)
spec.loader.exec_module(er)
available = {"engine": 1, "examples": 2, "rubrics_by_source": 3,
             "response_sets": 4, "config": 5, "run_dir": 6, "rollout_sets": 7}
bad = 0
for metric in sorted(er.METRIC_MODULES):
    fn = er._load(metric)
    try:
        inspect.signature(fn).bind(**er._call_kwargs(fn, available))
    except TypeError as exc:
        bad += 1
        print(f"  FAIL {metric}: {exc}")
print(f"  {len(er.METRIC_MODULES) - bad}/{len(er.METRIC_MODULES)} metrics bind")
raise SystemExit(1 if bad else 0)
PY

echo
echo "== numeric self-tests =="
python -m harness.eval.stats >/dev/null 2>&1 \
  && echo "  OK   harness.eval.stats" || { echo "  FAIL harness.eval.stats"; fail=1; }

echo
echo "== row-merge preserves and never duplicates =="
python - <<'PY' || fail=1
import json, sys, tempfile
from pathlib import Path
sys.path.insert(0, "scripts")
import eval_rubrics as er

ok = True
with tempfile.TemporaryDirectory() as tmp:
    # Source-keyed rows: re-running one source must leave the others alone.
    path = Path(tmp) / "m_per_question_rows.jsonl"
    er._write_rows_preserving_other_sources(
        path, [{"uid": "a", "rubric_source": "x"}, {"uid": "a", "rubric_source": "y"}], ["x", "y"]
    )
    er._write_rows_preserving_other_sources(path, [{"uid": "a", "rubric_source": "y"}], ["y"])
    rows = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    if len(rows) != 2 or {r["rubric_source"] for r in rows} != {"x", "y"}:
        print(f"  FAIL source-keyed merge: {rows}"); ok = False
    else:
        print("  OK   source-keyed rows preserved across a scoped re-run")

    # Rows with no source (head-to-head) must be replaced, not accumulated.
    pair = Path(tmp) / "h_per_pair_rows.jsonl"
    batch = [{"uid": "a", "source_a": "x", "source_b": "y"}]
    er._write_rows_preserving_other_sources(pair, batch, ["x", "y"])
    er._write_rows_preserving_other_sources(pair, batch, ["x", "y"])
    rows = [json.loads(l) for l in pair.read_text().splitlines() if l.strip()]
    if len(rows) != 1:
        print(f"  FAIL unattributed rows duplicated on re-run: {len(rows)} rows"); ok = False
    else:
        print("  OK   unattributed rows replaced rather than appended")
raise SystemExit(0 if ok else 1)
PY

echo
echo "== FDR families are declared, not inferred from the table =="
python - <<'PY' || fail=1
import sys
sys.path.insert(0, "scripts")
import aggregate_results as ag, rollout_report as rr

ok = True
# The rollout family is a fixed list; a metric may only join it deliberately.
family = {m for m, _, _ in rr.FDR_FAMILY}
expected = {"auc", "best_of_n_accuracy", "best_of_n_lift",
            "spearman", "separation", "z_separation"}
if family != expected:
    print(f"  FAIL rollout FDR family changed: {sorted(family ^ expected)}"); ok = False
else:
    print(f"  OK   rollout FDR family pinned to {len(family)} metrics")
if {m for m, _ in rr.DESCRIPTIVE} & family:
    print("  FAIL descriptive metrics leaked into the corrected family"); ok = False
else:
    print("  OK   descriptive metrics excluded from correction")
# The main table's family is METRIC_SPEC, so it cannot shrink with the run's scope.
if len(ag.METRIC_SPEC) != len({(f, c) for f, c, _, _ in ag.METRIC_SPEC}):
    print("  FAIL METRIC_SPEC has duplicate (family, column) entries"); ok = False
else:
    print(f"  OK   main-table FDR family pinned to {len(ag.METRIC_SPEC)} metrics")
raise SystemExit(0 if ok else 1)
PY

echo
echo "== polarity: FAVOURABLE is documented as an alias, not a concession =="
python - <<'PY' || fail=1
import sys
sys.path.insert(0, ".")
from harness.eval.judge import PolarityMode, effective_polarity
from harness.schema import Category, Criterion

ok = True
probes = [
    Criterion(title="t", description="Pitfall Criteria: Fails to state X", weight=3, category=Category.PITFALL),
    Criterion(title="t", description="Pitfall Criteria: Avoids stating X", weight=3, category=Category.PITFALL),
    Criterion(title="t", description="Pitfall Criteria: Something unparseable", weight=3, category=Category.PITFALL),
    Criterion(title="t", description="Essential Criteria: States X", weight=5, category=Category.ESSENTIAL),
]
for c in probes:
    if effective_polarity(c, PolarityMode.FAVOURABLE) is not effective_polarity(c, PolarityMode.DETECTED):
        print(f"  FAIL FAVOURABLE now differs from DETECTED for {c.description!r}"); ok = False
if ok:
    print("  OK   FAVOURABLE == DETECTED on every branch (as documented)")
raise SystemExit(0 if ok else 1)
PY

echo
echo "== tools: schemas, sandbox isolation, and the no-tools arm stays no-tools =="
python - <<'PY' || fail=1
import asyncio, sys
sys.path.insert(0, ".")
from harness.generators import REGISTRY
from harness.schema import Example, Rubric
from harness.tools import DEFAULT_TOOLS, ToolContext, build_registry

ok = True
reg = build_registry()
if sorted(reg.names) != sorted(DEFAULT_TOOLS):
    print(f"  FAIL registry {sorted(reg.names)} != {sorted(DEFAULT_TOOLS)}"); ok = False
else:
    print(f"  OK   {len(reg)} tools registered")

for schema in reg.schemas():
    fn = schema["function"]
    if not fn.get("name") or not fn.get("description") or "properties" not in fn.get("parameters", {}):
        print(f"  FAIL malformed schema: {fn.get('name')}"); ok = False
else:
    print("  OK   every tool exports a well-formed function schema")

# The ablation only means something if `agentic` really has no toolbelt.
if REGISTRY["agentic"](engine=None).tools_active:
    print("  FAIL `agentic` has tools active; the no-tools arm is contaminated"); ok = False
elif not REGISTRY["agentic-tools"](engine=None).tools_active:
    print("  FAIL `agentic-tools` has no toolbelt"); ok = False
else:
    print("  OK   agentic=no tools, agentic-tools=tools")

ex = Example(uid="t", domain="rar_science", split="val", row_index=0, question="q",
             reference_answer="r", question_source="", shipped_rubric=Rubric())
ctx = ToolContext(example=ex, engine=None)

async def probe():
    checks = [
        ("arithmetic", {"code": "2+2"}, True),
        ("network blocked", {"code": "import socket"}, False),
        ("process blocked", {"code": "import os; os.system('ls')"}, False),
        ("sympy available", {"code": "print(sympy is not None and np is not None)"}, True),
    ]
    good = True
    for label, args, want in checks:
        inv = await reg.dispatch("python_eval", args, ctx)
        if inv.result["ok"] != want:
            print(f"  FAIL sandbox {label}: ok={inv.result['ok']}, expected {want}")
            good = False
    return good

if asyncio.run(probe()):
    print("  OK   sandbox runs maths and refuses network/process access")
else:
    ok = False
raise SystemExit(0 if ok else 1)
PY

echo
echo "== RubricBench: numeric helpers, and the report still matches its artefacts =="
# Checks the exact McNemar / Clopper-Pearson / Wilson / Fisher implementations
# against frozen reference values, and re-derives every headline number in
# results/rubricbench/REPORT.md from the verdict files -- including §9's ceiling
# validity result (under a third, independent tokenisation) and §10's round-two
# candidate deltas and offline bounds. Only the failures are printed; there are
# ~72 assertions.
if rb_out=$(python scripts/rubricbench_verify.py --section selftest 2>&1); then
  echo "  OK   $(printf '%s\n' "$rb_out" | grep -c '^  OK') assertions (stats helpers + REPORT.md facts)"
else
  printf '%s\n' "$rb_out" | grep -E '^  FAIL|FAILED' | sed 's/^/  /'
  echo "  FAIL rubricbench verify"
  fail=1
fi

echo
echo "== baseline prompt fidelity vs docs/00_paper_notes.md =="
python scripts/check_prompt_fidelity.py >/dev/null 2>&1 \
  && echo "  OK   prompts verbatim" || { echo "  FAIL prompt drift"; fail=1; }

echo
if [ "$fail" -eq 0 ]; then
  echo "ALL SELFTESTS PASSED"
else
  echo "SELFTESTS FAILED"
fi
exit "$fail"
