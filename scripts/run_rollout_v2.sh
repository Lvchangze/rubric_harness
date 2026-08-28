#!/usr/bin/env bash
# Remaining stages of the powered rollout evaluation, in order, resumable.
#
# Every stage resumes from disk and every LLM call is cached, so re-running this
# after an interruption costs only the work that had not finished. It exists
# because the run was cut in half by a serving-endpoint outage: the screen and
# the question selection completed, generation did not.
#
#   bash scripts/run_rollout_v2.sh
set -euo pipefail

cd "$(dirname "$0")/.."
export LLM_REQUEST_TIMEOUT_S=1200

RUN=rollout_v2
CFG=configs/rollout_v2.yaml
SOURCES="shipped baseline agentic agentic-goldonly agentic-negonly"
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

echo "== 1/4 rubrics for the selected questions =="
python3 scripts/gen_rubrics.py --config "$CFG" --run-name "$RUN" \
  --sources $SOURCES --concurrency 48 2>&1 | tee logs/gen_v2.log | grep -v "HTTP Request" || true

echo "== 2/4 widen the screened rollouts from k=4 to k=8 =="
# The first four come back from the cache; only the extra draws are paid for.
python3 scripts/build_rollouts.py --config "$CFG" --run-name "$RUN" \
  --k 8 --concurrency 48 --agreement-probe-fraction 0.05 2>&1 \
  | tee logs/rollouts_v2.log | grep -v "HTTP Request" || true

echo "== 3/4 score every rollout under every rubric =="
python3 scripts/eval_rubrics.py --config "$CFG" --run-name "$RUN" \
  --metrics rollout --sources $SOURCES --concurrency 48 2>&1 \
  | tee logs/eval_v2.log | grep -v "HTTP Request" || true

echo "== 4/4 tables (no LLM calls) =="
python3 scripts/rollout_report.py --run-name "$RUN" --results-dir results --sources $SOURCES
python3 scripts/reward_hacking_probe.py --run-name "$RUN" --results-dir results

echo "done: results/$RUN/"
