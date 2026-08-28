#!/usr/bin/env bash
# Remaining stages of the powered rollout evaluation, in order, resumable.
#
# The run was cut in half by a serving-endpoint outage: screening and question
# selection completed, generation did not. Every stage resumes from disk by uid
# and every finished LLM call is cached, so re-running this after an
# interruption costs only the work that had not finished.
#
#   bash scripts/run_rollout_v2.sh          # run it now
#   bash scripts/resume_when_up.sh          # or wait for the endpoint first
#
# Stage costs, measured or estimated from the pilot. "calls" are LLM requests;
# cached ones are free.
#
#   1 rubrics      66 questions x 3 agentic variants   ~3,000 calls   ~40 min
#   2 structural   0 calls (upgrades the 62-question preview to 128)  ~2 min
#   3 aggregate    0 calls                                            ~1 min
#   4 rollouts     128 questions, k=4 -> k=8           ~1,100 calls   ~25 min
#   5 judging      128 questions x 8 rollouts x 5 src  ~5,100 calls   ~90 min
#   6 tables       0 calls                                            ~2 min
#
# Concurrency 48 is deliberate. The screen held 192 for an hour, but the outage
# began under that load; 48 was the pilot's setting for generation and never
# produced an error. Raise it only after the endpoint has been stable for a
# while.
set -euo pipefail

cd "$(dirname "$0")/.."
export LLM_REQUEST_TIMEOUT_S=1200

RUN=rollout_v2
CFG=configs/rollout_v2.yaml
SOURCES="shipped baseline agentic agentic-goldonly agentic-negonly"
N_SELECTED=128
CONC=${CONC:-48}
LOCK=".lock_${RUN}"

# The shell in this environment has been observed to execute a command more than
# once; two generators writing one jsonl produced a 598-line file for a
# 200-question run. An mkdir lock is atomic on every filesystem we care about.
if ! mkdir "$LOCK" 2>/dev/null; then
  echo "another run holds $LOCK; exiting" >&2
  exit 1
fi
trap 'rmdir "$LOCK" 2>/dev/null || true' EXIT

mkdir -p logs
strip() { grep -v "HTTP Request" || true; }

echo "== 1/6 rubrics for the selected questions (66 x 3 agentic variants) =="
python3 scripts/gen_rubrics.py --config "$CFG" --run-name "$RUN" \
  --sources $SOURCES --concurrency "$CONC" 2>&1 | tee logs/gen_v2.log | strip

echo "== 2/6 structural metrics, zero LLM calls =="
# Run before judging so that if the endpoint dies again the structural result on
# the full 128 is already banked. This also replaces the 62-question preview in
# the report with the complete cohort.
python3 scripts/eval_rubrics.py --config "$CFG" --run-name "$RUN" \
  --metrics grounding lint adaptivity --sources $SOURCES 2>&1 \
  | tee logs/struct_v2.log | strip

echo "== 3/6 aggregate structural metrics, with a cohort assertion =="
# --expect-paired is load-bearing: summarize_metric intersects uids across every
# source it is given, so if any source is still short the paired queue shrinks
# silently and the table is computed on a different question set than it claims.
python3 scripts/aggregate_results.py --run-name "$RUN" --results-dir results \
  --sources $SOURCES --expect-paired "$N_SELECTED" --expect-metrics 0

echo "== 4/6 widen the screened rollouts from k=4 to k=8 =="
# The first four draws come back from the cache; only the extra ones are paid
# for. The write merges, so the 652 screened-but-not-selected questions stay in
# rollouts.jsonl rather than being replaced by this run's 128.
python3 scripts/build_rollouts.py --config "$CFG" --run-name "$RUN" \
  --k 8 --concurrency "$CONC" --agreement-probe-fraction 0.05 2>&1 \
  | tee logs/rollouts_v2.log | strip

echo "== 5/6 score every rollout under every rubric =="
python3 scripts/eval_rubrics.py --config "$CFG" --run-name "$RUN" \
  --metrics rollout --sources $SOURCES --concurrency "$CONC" 2>&1 \
  | tee logs/eval_v2.log | strip

echo "== 6/6 tables, zero LLM calls =="
python3 scripts/rollout_report.py --run-name "$RUN" --results-dir results --sources $SOURCES
python3 scripts/reward_hacking_probe.py --run-name "$RUN" --results-dir results
python3 scripts/main_table.py --run-name "$RUN" --runs-dir runs

cat <<EOF

done. Read, in this order:
  results/$RUN/rollout_report.md        AUC / best-of-n / Spearman, per domain, BH-FDR
  results/$RUN/reward_hacking_probe.md  score-vs-length on oracle-incorrect rollouts
  results/$RUN/summary_table.md         structural metrics on all $N_SELECTED questions

Then apply the pre-registered reading rules in HANDOFF.md section 5 BEFORE
looking at which way the numbers went.
EOF
