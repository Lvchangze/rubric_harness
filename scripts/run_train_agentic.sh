#!/usr/bin/env bash
# agentic-tools over the RaR train split, supervised.
#
# The first attempt ran at concurrency 128 and finished 3 questions in 13 hours.
# It was not slow, it was thrashing: 905 requests hit the 1200s hard cap and
# 5,884 came back empty, all in the rollout stage, and the pipeline never
# reached stage 3 on any question. The endpoint returned to a 1.5s single-call
# latency the moment the load came off. 64 is the only concurrency this
# workload has been measured clean at (runs/tools_v1: 128 questions in 3,093s).
#
# What that cost was 13 hours of nothing, discovered by hand. So this script
# supervises: it samples the output every STALL_CHECK_S and aborts if the
# completed count has not moved across two consecutive checks. A run that is
# making no progress should stop by itself rather than hold the endpoint.
#
#   nohup bash scripts/run_train_agentic.sh > logs/train_agentic_tools.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."
export LLM_REQUEST_TIMEOUT_S=1200

CFG=configs/train_full.yaml
RUN=train_full
CONC="${CONC:-64}"
OUT="runs/$RUN/rubrics_agentic-tools.jsonl"
STALL_CHECK_S="${STALL_CHECK_S:-1800}"      # 30 min
STALL_STRIKES="${STALL_STRIKES:-2}"         # abort after this many flat checks

count() { [ -f "$OUT" ] && wc -l < "$OUT" || echo 0; }

echo "$(date '+%F %T') start: concurrency=$CONC, already done $(count)/28720"

python3 scripts/gen_rubrics.py --config "$CFG" --run-name "$RUN" \
  --sources agentic-tools --concurrency "$CONC" 2>&1 | grep -v "HTTP Request" &
GEN_PID=$!

strikes=0
last=$(count)
while kill -0 "$GEN_PID" 2>/dev/null; do
  sleep "$STALL_CHECK_S"
  kill -0 "$GEN_PID" 2>/dev/null || break
  now=$(count)
  rate=$(( (now - last) * 3600 / STALL_CHECK_S ))
  echo "$(date '+%F %T') [supervisor] $now/28720 done (+$((now-last)) in last $((STALL_CHECK_S/60))min, ~${rate}/h)"
  if [ "$now" -le "$last" ]; then
    strikes=$((strikes + 1))
    echo "$(date '+%F %T') [supervisor] NO PROGRESS (strike $strikes/$STALL_STRIKES)"
    if [ "$strikes" -ge "$STALL_STRIKES" ]; then
      echo "$(date '+%F %T') [supervisor] ABORTING: stalled. Lower CONC and restart; progress is kept."
      kill "$GEN_PID" 2>/dev/null
      pkill -f "gen_rubrics.py.*agentic-tools" 2>/dev/null
      exit 2
    fi
  else
    strikes=0
  fi
  last=$now
done

wait "$GEN_PID"; rc=$?
echo "$(date '+%F %T') finished: $(count)/28720 done, exit $rc"
exit $rc
