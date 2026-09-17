#!/usr/bin/env bash
# Chain the two train-split generation stages on Kimi-K3.
#
# They are sequential on purpose, and on this endpoint the reason is stronger
# than it was on glm5.2: it saturates, and it saturates early. Measured on the
# real workload (`scripts/gen_rubrics.py` baseline, long structured output):
#
#     conc  64 -> 0.71 samples/s        conc 256 -> 0.88 samples/s
#
# 4x the concurrency for 24% more work, and 256 also produced ~6,300 stream
# errors in ten minutes. Running both stages at once would not finish sooner;
# it would just move the same capacity around and add failures. 128 sits at the
# knee.
#
# Do not size this from `scripts/probe_endpoint.py`, which reported 4.2 req/s
# at conc 64 for the same endpoint. That probe asked for three sentences; this
# endpoint is token-bound, so short-output probing overstates it by ~6x. The
# workload-shaped numbers above are the ones that held up.
#
# Kimi is roughly 5x slower here than the retired glm5.2 deployment (which did
# 3.67 samples/s on baseline at conc 64), so expect agentic-tools to take weeks
# rather than the ~4 days it was tracking before. It resumes by uid and the
# output is usable at any point, so it is meant to be left running and drawn
# from as needed rather than waited on.
#
# Both stages resume by uid, so killing and restarting this script is safe and
# only pays for questions that had not finished.
set -uo pipefail
cd "$(dirname "$0")/.."
export LLM_REQUEST_TIMEOUT_S=1200
CFG=configs/train_full_kimi.yaml
RUN=train_full_kimi
mkdir -p logs

# `--after PID` attaches to a baseline already running rather than starting a
# second one, so recovering from an interrupted launch does not put two jobs on
# a saturated endpoint at once.
if [ "${1:-}" = "--after" ]; then
  echo "$(date '+%F %T') waiting for pid $2 (baseline) to finish"
  while kill -0 "$2" 2>/dev/null; do sleep 60; done
  echo "$(date '+%F %T') baseline finished"
else
  echo "$(date '+%F %T') === baseline, concurrency 128 (~9-11h) ==="
  python3 scripts/gen_rubrics.py --config "$CFG" --run-name "$RUN" \
    --sources baseline --concurrency 128 --max-in-flight 128 2>&1 | grep -v "HTTP Request"
fi

echo "$(date '+%F %T') === agentic-tools, concurrency 128 ==="
python3 scripts/gen_rubrics.py --config "$CFG" --run-name "$RUN" \
  --sources agentic-tools --concurrency 128 --max-in-flight 96 2>&1 | grep -v "HTTP Request"

echo "$(date '+%F %T') === done, exit $? ==="
