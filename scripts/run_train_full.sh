#!/usr/bin/env bash
# Chain the two train-split generation stages on GLM-5.3.
#
# They are sequential on purpose. Run together they would put both stages'
# concurrency on the endpoint at once, and a 192-concurrency screening load is
# what preceded the 2026-08-28 outage that cost most of a day. Baseline is ~1h
# of a ~2-day total, so serialising costs little and removes a failure mode we
# have already hit.
#
# Both stages resume by uid, so killing and restarting this script is safe and
# only pays for questions that had not finished.
set -uo pipefail
cd "$(dirname "$0")/.."
export LLM_REQUEST_TIMEOUT_S=1200
CFG=configs/train_full.yaml
RUN=train_full_glm53
CONC=128

# --max-in-flight bounds live *questions*, which is the setting that actually
# governs throughput here and is separate from --concurrency (live requests).
# Omitting it is not a neutral default, it is unbounded: the glm5.2 attempt ran
# without it and opened all 28,717 pipelines at once, which is why its rate
# decayed from 0.53 to 0.07 samples/s over 23h rather than holding steady.
#
# Sizing it: a question is mostly sequential (staged pipeline), and the smoke
# measured ~73 calls over ~1155s, i.e. ~0.063 req/s of demand per live question.
# Filling 128 slots at this endpoint's measured 11.65 rps therefore needs
# ~185 questions in flight, so 192. Baseline is one call per question, so there
# demand and slots are the same thing and in-flight tracks concurrency.
BASELINE_INFLIGHT=$CONC
AGENTIC_INFLIGHT=192

echo "$(date '+%F %T') === baseline, concurrency $CONC, in-flight $BASELINE_INFLIGHT ==="
python3 scripts/gen_rubrics.py --config "$CFG" --run-name "$RUN" \
  --sources baseline --concurrency "$CONC" --max-in-flight "$BASELINE_INFLIGHT" 2>&1 \
  | grep -v "HTTP Request"
echo "$(date '+%F %T') === baseline done ==="

echo "$(date '+%F %T') === agentic-tools, concurrency $CONC, in-flight $AGENTIC_INFLIGHT ==="
python3 scripts/gen_rubrics.py --config "$CFG" --run-name "$RUN" \
  --sources agentic-tools --concurrency "$CONC" --max-in-flight "$AGENTIC_INFLIGHT" 2>&1 \
  | grep -v "HTTP Request"
echo "$(date '+%F %T') === agentic-tools done, exit $? ==="
