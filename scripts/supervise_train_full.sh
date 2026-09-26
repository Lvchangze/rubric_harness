#!/usr/bin/env bash
# Keep the train-split generation running until it finishes.
#
# The in-process guards in harness/pipeline.py catch an endpoint that dies or
# stalls while the driver itself is healthy. They cannot catch the driver
# stopping: on 2026-09-25 the generator went silent at 08:01, never logged again,
# and was SIGKILLed at 15:57 during a machine-wide OOM caused by other workloads
# on the node. The stall watchdog never fired because it runs inside the event
# loop that had stopped, and twenty hours passed before anyone noticed.
#
# This runs outside the process and watches the one signal that cannot lie:
# rows landing in the output file. It restarts the generator when it exits
# abnormally or when no row has landed for STALL_S, and stops when the
# generator exits cleanly (every question attempted). Resume is by uid, so a
# restart costs only the questions that were in flight.
#
#   nohup setsid bash scripts/supervise_train_full.sh > logs/supervise.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/.."

CFG=${CFG:-configs/train_full.yaml}
RUN=${RUN:-train_full_glm53}
SOURCE=${SOURCE:-agentic-tools}
# Concurrency comes from $CFG. In-flight is the measured pairing for 128 slots.
INFLIGHT=${INFLIGHT:-192}
OUT=${OUT:-runs/$RUN/rubrics_$SOURCE.jsonl}
GEN_LOG=${GEN_LOG:-logs/train_${RUN#train_full_}_supervised.log}
GEN_CMD=${GEN_CMD:-"python3 scripts/gen_rubrics.py --config $CFG --run-name $RUN --sources $SOURCE --max-in-flight $INFLIGHT"}

POLL_S=${POLL_S:-10}     # how quickly a dead generator is noticed
CHECK_S=${CHECK_S:-300}  # how often progress is measured
# Longer than the in-process stall watchdog (3600s), so that one gets the first
# chance to exit with a logged reason; this is the backstop for when the process
# cannot act at all. Also well past restart warm-up, which takes ~20 minutes
# before the first question lands.
STALL_S=${STALL_S:-5400}
# A run that dies this fast is not a transient: it is a bug or a dead endpoint
# that the health probe missed. Stop rather than restart-loop against it.
FAST_S=${FAST_S:-900}
MAX_FAST_FAILS=${MAX_FAST_FAILS:-5}
BACKOFF_S=${BACKOFF_S:-60}
PROBE=${PROBE:-1}
PROBE_INTERVAL_S=${PROBE_INTERVAL_S:-120}

log() { echo "$(date '+%F %T') [supervise] $*"; }

rows() { if [ -f "$OUT" ]; then wc -l < "$OUT"; else echo 0; fi; }

# A row with no criteria is a failure record, and on resume its uid counts as
# done — so an endpoint outage that the consecutive-empty guard stopped after 50
# questions would otherwise leave those 50 permanently skipped. Only ever called
# while no generator is running.
purge_empty() {
  [ -f "$OUT" ] || return 0
  python3 - "$OUT" <<'PY'
import json, sys, pathlib
p = pathlib.Path(sys.argv[1])
rows = [json.loads(l) for l in p.open(encoding="utf-8") if l.strip()]
keep = [r for r in rows if r.get("rubric", {}).get("items")]
if len(keep) == len(rows):
    sys.exit(0)
tmp = p.with_suffix(p.suffix + ".tmp")
with tmp.open("w", encoding="utf-8") as fh:
    for r in keep:
        fh.write(json.dumps(r, ensure_ascii=False) + "\n")
assert sum(1 for l in tmp.open(encoding="utf-8") if l.strip()) == len(keep)
tmp.replace(p)
print(f"purged {len(rows) - len(keep)} empty rows, {len(keep)} remain")
PY
}

probe_once() {
  timeout 180 python3 - "$CFG" >/dev/null 2>&1 <<'PY'
import asyncio, sys
sys.path.insert(0, ".")
from harness.config import load_config
from harness.llm import LLMEngine

model = load_config(sys.argv[1]).llm.model

async def main() -> bool:
    engine = LLMEngine(model, concurrency=1, cache_dir=None, max_attempts=1, request_timeout_s=150)
    text = await engine.chat("What is 2+2? Reply with just the number.", max_tokens=1024, tag="supervise:probe")
    return bool(text and text.strip())

sys.exit(0 if asyncio.run(main()) else 1)
PY
}

# Two consecutive healthy probes: one lucky reply during a partial recovery
# would start a run that then fails slowly.
wait_healthy() {
  local ok=0
  while [ "$ok" -lt 2 ]; do
    if probe_once; then
      ok=$((ok + 1))
    else
      [ "$ok" -gt 0 ] && log "endpoint health lost, resetting"
      ok=0
      log "endpoint unhealthy, retrying in ${PROBE_INTERVAL_S}s"
      sleep "$PROBE_INTERVAL_S"
    fi
  done
}

fast_fails=0
while true; do
  [ "$PROBE" = 1 ] && wait_healthy
  purge_empty | sed "s/^/$(date '+%F %T') [supervise] /"

  started=$(date +%s)
  last=$(rows)
  last_change=$started
  log "starting generator (rows=$last)"
  # exec so that $! is the generator's own pid, not a wrapper shell's.
  bash -c "exec $GEN_CMD" >> "$GEN_LOG" 2>&1 &
  pid=$!

  next_check=$(( started + CHECK_S ))
  while kill -0 "$pid" 2>/dev/null; do
    sleep "$POLL_S"
    [ "$(date +%s)" -lt "$next_check" ] && continue
    next_check=$(( $(date +%s) + CHECK_S ))
    kill -0 "$pid" 2>/dev/null || break
    now=$(rows)
    if [ "$now" != "$last" ]; then
      last=$now
      last_change=$(date +%s)
    elif [ $(( $(date +%s) - last_change )) -ge "$STALL_S" ]; then
      log "no new rows for $(( ($(date +%s) - last_change) / 60 )) min (rows=$last); killing pid $pid"
      kill -KILL "$pid" 2>/dev/null
      break
    fi
  done

  # Blocks until the old generator is really gone, even one stuck in the kernel,
  # so two writers never append to the same file.
  wait "$pid"
  rc=$?
  ran=$(( $(date +%s) - started ))
  log "generator exited rc=$rc after $(( ran / 60 )) min (rows=$(rows))"

  if [ "$rc" -eq 0 ]; then
    log "clean exit: every question attempted"
    purge_empty | sed "s/^/$(date '+%F %T') [supervise] /"
    exit 0
  fi

  if [ "$ran" -lt "$FAST_S" ]; then
    fast_fails=$((fast_fails + 1))
  else
    fast_fails=0
  fi
  if [ "$fast_fails" -ge "$MAX_FAST_FAILS" ]; then
    log "giving up: $fast_fails consecutive runs died within $(( FAST_S / 60 )) min"
    exit 1
  fi
  sleep $(( fast_fails * BACKOFF_S ))
done
