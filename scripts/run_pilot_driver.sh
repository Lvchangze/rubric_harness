#!/usr/bin/env bash
# Drives one phase of the pilot end to end.
#
#   bash scripts/run_pilot_driver.sh generate   # rubrics + response ladder
#   bash scripts/run_pilot_driver.sh evaluate   # metrics + aggregation
#
# Guarded by an atomic mkdir lock. The surrounding execution environment has
# been observed launching the same command more than once concurrently, and a
# second generator appending to the same artefacts corrupts them. Only the
# first invocation proceeds; the rest exit 0. Never rmdir the lock from the
# launching command -- that reintroduces the race it exists to prevent.
set -uo pipefail

cd "$(dirname "$0")/.." || exit 1

PHASE="${1:-generate}"
LOCK="/tmp/rubric_pilot_${PHASE}.lock"
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "[driver] another invocation holds $LOCK; exiting"
  exit 0
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT

export LLM_REQUEST_TIMEOUT_S=1200
CFG=configs/pilot.yaml
RUN=runs/pilot_v2
CC=48

step() { echo; echo "[driver] ===== $* ===== $(date '+%H:%M:%S')"; }

if [ "$PHASE" = "generate" ]; then
  # Rubric generation and the response ladder are independent: responses depend
  # only on the questions, not on any rubric. Run them side by side.
  step "1/4 agentic rubrics + response ladder (parallel)"
  python scripts/gen_rubrics.py --config $CFG --sources agentic --concurrency $CC \
    > $RUN/gen_agentic.log 2>&1 &
  PID_GEN=$!
  python scripts/build_responses.py --config $CFG --concurrency $CC \
    > $RUN/responses.log 2>&1 &
  PID_RESP=$!
  wait $PID_GEN; echo "[driver] agentic gen rc=$?"
  wait $PID_RESP; echo "[driver] responses rc=$?"

  step "2/4 agentic-noval ablation rubrics"
  python scripts/gen_rubrics.py --config $CFG --sources agentic-noval --concurrency 64 \
    > $RUN/gen_noval.log 2>&1
  echo "[driver] noval gen rc=$?"

  step "3/4 deduplicate append-only artefacts"
  python scripts/_repair_jsonl.py $RUN/rubrics_*.jsonl 2>&1 | sed 's/^/[driver] /'

  step "4/4 zero-LLM structural metrics (all sources)"
  python scripts/eval_rubrics.py --config $CFG --metrics grounding lint adaptivity \
    > $RUN/eval_structural.log 2>&1
  echo "[driver] structural rc=$?"

  step "GENERATE_PHASE_DONE"

elif [ "$PHASE" = "evaluate" ]; then
  step "1/3 LLM-judged metrics (all sources)"
  python scripts/eval_rubrics.py --config $CFG \
    --metrics discriminative transfer coverage intrinsic headtohead \
    --concurrency 64 > $RUN/eval_llm.log 2>&1
  echo "[driver] llm metrics rc=$?"

  step "2/3 count-controlled re-evaluation"
  python scripts/eval_rubrics.py --config $CFG --count-controlled \
    --metrics discriminative coverage grounding \
    --concurrency 64 > $RUN/eval_cc.log 2>&1
  echo "[driver] count-controlled rc=$?"

  step "3/3 aggregate results"
  python scripts/aggregate_results.py --run-name pilot_v2 > $RUN/aggregate.log 2>&1
  echo "[driver] aggregate rc=$?"

  step "EVALUATE_PHASE_DONE"

elif [ "$PHASE" = "leakage" ]; then
  # Stage 6 filters criteria against `reference_answer`, which the
  # discriminative metric also uses as its `gold` positive, so its gain is
  # confounded. These two ablations split the loop by leakage: `goldonly` keeps
  # only the leaking half, `negonly` only the half judged against independently
  # synthesised negatives the evaluation never sees. Whichever carries the gain
  # decides whether the headline result is circular.
  step "1/3 split-critic ablation rubrics"
  python scripts/gen_rubrics.py --config $CFG --sources agentic-goldonly \
    --concurrency 64 > $RUN/gen_goldonly.log 2>&1
  echo "[driver] goldonly gen rc=$?"
  python scripts/gen_rubrics.py --config $CFG --sources agentic-negonly \
    --concurrency 64 > $RUN/gen_negonly.log 2>&1
  echo "[driver] negonly gen rc=$?"
  python scripts/_repair_jsonl.py $RUN/rubrics_agentic-goldonly.jsonl \
    $RUN/rubrics_agentic-negonly.jsonl 2>&1 | sed 's/^/[driver] /'

  step "2/3 judge the ablations on the same shared response ladder"
  python scripts/eval_rubrics.py --config $CFG \
    --sources agentic-goldonly agentic-negonly \
    --metrics discriminative grounding lint adaptivity \
    --concurrency 64 > $RUN/eval_leakage.log 2>&1
  echo "[driver] leakage eval rc=$?"

  step "3/3 confound audit over all sources"
  python scripts/confound_audit.py --run-name pilot_v2 --reference baseline --by-domain \
    > $RUN/confound_audit.log 2>&1
  echo "[driver] audit rc=$?"

  step "LEAKAGE_PHASE_DONE"
else
  echo "[driver] unknown phase: $PHASE (expected 'generate', 'evaluate' or 'leakage')"
  exit 2
fi
