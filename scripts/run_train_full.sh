#!/usr/bin/env bash
# Chain the two train-split generation stages.
#
# They are sequential on purpose. Run together they would put 64+128=192
# concurrent requests on the endpoint, and a 192-concurrency screening load is
# what preceded the 2026-08-28 outage that cost most of a day. Baseline is ~3h
# of a ~5-day total, so serialising costs little and removes the one failure
# mode we have already hit.
#
# Both stages resume by uid, so killing and restarting this script is safe and
# only pays for questions that had not finished.
set -uo pipefail
cd "$(dirname "$0")/.."
export LLM_REQUEST_TIMEOUT_S=1200
CFG=configs/train_full.yaml
RUN=train_full

wait_for() {           # $1 = pid to wait on, may already be gone
  while kill -0 "$1" 2>/dev/null; do sleep 60; done
}

if [ "${1:-}" = "--after" ]; then
  echo "$(date '+%F %T') waiting for pid $2 (baseline) to finish"
  wait_for "$2"
  echo "$(date '+%F %T') baseline finished"
fi

echo "$(date '+%F %T') === agentic-tools, concurrency 128 ==="
python3 scripts/gen_rubrics.py --config "$CFG" --run-name "$RUN" \
  --sources agentic-tools --concurrency 128 2>&1 | grep -v "HTTP Request"
echo "$(date '+%F %T') === done, exit $? ==="
