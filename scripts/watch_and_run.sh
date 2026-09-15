#!/usr/bin/env bash
# Wait for the serving endpoint, then run a command; retry if it dies mid-way.
#
# Generalised from `resume_when_up.sh`, which is welded to the rollout_v2 run.
# That one stays as-is because HANDOFF.md documents it; this one takes the
# command as arguments.
#
#   nohup bash scripts/watch_and_run.sh -- python3 scripts/gen_rubrics.py ... &
#
# Environment:
#   PROBE_TIMEOUT_S  per-probe cap                      (default 60)
#   INTERVAL_S       gap between probes                 (default 120)
#   NEEDED_OK        consecutive healthy probes to start (default 2)
#   MAX_RUNS         attempts of the command itself     (default 5)
#   MODEL            model to probe
#
# Two consecutive probes, not one: a freshly-restarted deployment can answer a
# single request and still not have all instances routable, and starting a
# high-concurrency job into that state fails slowly against a 1200s timeout
# instead of failing fast.
#
# Retrying the command is safe because every generator resumes by uid and every
# completed LLM call is served from the content-addressed cache, so a retry pays
# only for what did not finish.
set -uo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

PROBE_TIMEOUT_S=${PROBE_TIMEOUT_S:-60}
INTERVAL_S=${INTERVAL_S:-120}
NEEDED_OK=${NEEDED_OK:-2}
MAX_RUNS=${MAX_RUNS:-5}
MODEL=${MODEL:-hy-t2t-glm-5.2-384k-fp8-L20A-t1-v2}

[ "${1:-}" = "--" ] && shift
if [ "$#" -eq 0 ]; then
  echo "usage: $0 -- <command...>" >&2
  exit 2
fi

probe() {
  MODEL="$MODEL" LLM_REQUEST_TIMEOUT_S="$PROBE_TIMEOUT_S" \
  timeout $((PROBE_TIMEOUT_S + 30)) python3 - <<'PY'
import asyncio, os, sys
sys.path.insert(0, '/apdcephfs_zwfy6/share_302970870/hunyuan/changzelv/dev/hyctx_data')
from contextagent.llm import LLMClient

async def main():
    client = LLMClient(os.environ["MODEL"], concurrency=1, reasoning_effort='high')
    out = await client.chat('What is 2+2? Reply with just the number.', max_tokens=1024)
    text = out.get('response', '') if isinstance(out, dict) else str(out)
    sys.exit(0 if text.strip() else 1)

asyncio.run(main())
PY
}

wait_until_up() {
  local ok=0
  while true; do
    if probe >/dev/null 2>&1; then
      ok=$((ok + 1))
      echo "$(date '+%F %T') healthy probe $ok/$NEEDED_OK"
      [ "$ok" -ge "$NEEDED_OK" ] && return 0
      sleep 10          # short gap between confirmations, not a full interval
    else
      [ "$ok" -gt 0 ] && echo "$(date '+%F %T') health lost, resetting"
      ok=0
      # Logged every time rather than suppressed: a watchdog that prints nothing
      # while waiting is indistinguishable from one that has died.
      echo "$(date '+%F %T') endpoint down, waiting ${INTERVAL_S}s"
      sleep "$INTERVAL_S"
    fi
  done
}

echo "$(date '+%F %T') watching $MODEL; will run: $*"
for attempt in $(seq 1 "$MAX_RUNS"); do
  wait_until_up
  echo "$(date '+%F %T') === run attempt $attempt/$MAX_RUNS ==="
  if "$@"; then
    echo "$(date '+%F %T') === COMMAND SUCCEEDED ==="
    exit 0
  fi
  echo "$(date '+%F %T') attempt $attempt failed; re-checking the endpoint before retrying"
done
echo "$(date '+%F %T') === GAVE UP after $MAX_RUNS attempts ==="
exit 1
