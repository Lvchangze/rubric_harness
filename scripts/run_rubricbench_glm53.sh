#!/usr/bin/env bash
# Re-measure the RubricBench comparison with GLM-5.3 as both generator and judge.
#
# The published table used GLM-5.2 for both roles. GLM-5.2 is retired, so every
# source is re-judged here — `none` and `expert` included, since they depend on
# the judge even though nothing is generated for them. Mixing judges across rows
# would make the comparison meaningless.
#
# Dev half only (600 cases). The holdout half may be scored at most three times
# and each spend has to be logged before the number is read
# (results/rubricbench/opt/HOLDOUT_LOG.md). A model swap is worth a spend only if
# dev shows something to confirm.
#
# Each source is skipped once its score file exists, and every finished LLM call
# is cached, so restarting after an interruption only pays for what was lost.
#
#   tmux new-session -d -s rbench53 "bash scripts/run_rubricbench_glm53.sh >> logs/rubricbench_glm53.log 2>&1"
set -uo pipefail
cd "$(dirname "$0")/.."

MODEL=${MODEL:-GLM-5.3-H20-t2-copy}
OUT=${OUT:-results/rubricbench_glm53}
CASES=${CASES:-results/rubricbench/split.json:dev}
CONC=${CONC:-128}
SOURCES=${SOURCES:-"none expert baseline framed agentic-tools agentic"}

echo "$(date '+%F %T') model=$MODEL cases=$CASES out=$OUT concurrency=$CONC"
for src in $SOURCES; do
  for attempt in 1 2 3; do
    [ -f "$OUT/${src}_score.json" ] && break
    echo "$(date '+%F %T') === $src (attempt $attempt) ==="
    python3 scripts/rubricbench_run.py --source "$src" --case-ids "$CASES" \
      --model "$MODEL" --concurrency "$CONC" --out-dir "$OUT"
    echo "$(date '+%F %T') === $src exit $? ==="
  done
  [ -f "$OUT/${src}_score.json" ] || echo "$(date '+%F %T') !!! $src failed three times"
done
echo "$(date '+%F %T') === all done ==="
