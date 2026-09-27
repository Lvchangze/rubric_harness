#!/usr/bin/env bash
# Restart supervise_train_full.sh if it is killed; stop when it exits on purpose.
#
# On 2026-09-27 the supervisor and its generator were killed together with no
# trace — no OOM record, no reboot, other long-running processes untouched —
# and seven hours passed before anyone looked. The supervisor guards the
# generator; this guards the supervisor.
#
# Exit codes are the supervisor's own contract: 0 means every question was
# attempted, 1 means it gave up after repeated fast deaths. Both are decisions
# to respect, not failures to retry. Anything else (137 for SIGKILL, 143 for
# SIGTERM) means something outside killed it.
#
#   tmux new-session -d -s rubric-train "bash scripts/keep_supervisor.sh"
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs
LOG=${SUPERVISE_LOG:-logs/supervise.log}

while true; do
  bash scripts/supervise_train_full.sh >> "$LOG" 2>&1
  rc=$?
  case "$rc" in
    0|1) echo "$(date '+%F %T') [keep] supervisor exited rc=$rc on purpose; stopping" >> "$LOG"; exit "$rc" ;;
    *)   echo "$(date '+%F %T') [keep] supervisor died rc=$rc; restarting in 60s" >> "$LOG"; sleep 60 ;;
  esac
done
