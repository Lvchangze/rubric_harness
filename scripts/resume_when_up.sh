#!/usr/bin/env bash
# Wait for the serving endpoint to come back, then finish the rollout run.
#
# The powered rollout evaluation was cut off mid-generation by an endpoint
# outage: a two-token request timed out at concurrency 1, and the fallback
# vendors returned 401. Everything up to that point is on disk and every
# finished call is cached, so the run only needs someone to notice the endpoint
# is healthy again. This does the noticing.
#
#   nohup bash scripts/resume_when_up.sh > logs/resume.log 2>&1 &
set -uo pipefail

cd "$(dirname "$0")/.."
mkdir -p logs

PROBE_TIMEOUT_S=${PROBE_TIMEOUT_S:-60}
INTERVAL_S=${INTERVAL_S:-120}
# Two consecutive healthy probes before committing the budget: a single lucky
# response during a partial recovery would start a run that then fails slowly
# against a 1200s per-request timeout.
NEEDED_OK=${NEEDED_OK:-2}

probe() {
  LLM_REQUEST_TIMEOUT_S="$PROBE_TIMEOUT_S" timeout $((PROBE_TIMEOUT_S + 30)) python3 - <<'PY'
import asyncio, sys
sys.path.insert(0, '/apdcephfs_zwfy6/share_302970870/hunyuan/changzelv/dev/hyctx_data')
from contextagent.llm import LLMClient

async def main():
    client = LLMClient('hy-t2t-glm-5.2-384k-fp8-L20A-t1-v2', concurrency=1, reasoning_effort='high')
    out = await client.chat('What is 2+2? Reply with just the number.', max_tokens=1024)
    text = out.get('response', '') if isinstance(out, dict) else str(out)
    sys.exit(0 if text.strip() else 1)

asyncio.run(main())
PY
}

ok=0
while true; do
  if probe >/dev/null 2>&1; then
    ok=$((ok + 1))
    echo "$(date '+%F %T') healthy probe $ok/$NEEDED_OK"
  else
    [ "$ok" -gt 0 ] && echo "$(date '+%F %T') health lost, resetting"
    ok=0
    # Logged every time rather than suppressed: a watchdog that prints nothing
    # while waiting is indistinguishable from one that has died.
    echo "$(date '+%F %T') endpoint still down, waiting ${INTERVAL_S}s"
  fi

  if [ "$ok" -ge "$NEEDED_OK" ]; then
    echo "$(date '+%F %T') endpoint healthy — resuming rollout_v2"
    bash scripts/run_rollout_v2.sh
    echo "$(date '+%F %T') run_rollout_v2.sh exited with $?"
    exit 0
  fi
  sleep "$INTERVAL_S"
done
