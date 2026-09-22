#!/usr/bin/env bash
# Chain the two train-split generation stages on GLM-5.3.
#
# baseline runs first and alone; agentic-tools then runs as two shards, one per
# GLM-5.3 deployment. They are separate capacity pools (t1 on 29.81.228.x, t2 on
# 29.127/191/232.x, identical output at temperature 0), so the shards genuinely
# run in parallel rather than splitting one queue.
#
# Why shard at all: the endpoint serves these calls at ~1.65 rps with 128 slots
# full, i.e. ~77s per call — reasoning-heavy generation is far slower than the
# short-prompt probe suggested, and one endpoint puts this stage at ~9 days.
# Nothing here is CPU-bound (the driver sits at 7%), so a second endpoint is
# close to a straight doubling.
#
# Every stage resumes by uid from its own file, so killing and restarting this
# script is safe and only pays for questions that had not finished.
set -uo pipefail
cd "$(dirname "$0")/.."
CFG=configs/train_full.yaml
RUN=train_full_glm53
CONC=128

# --max-in-flight bounds live *questions*, separate from --concurrency (live
# requests), and it is what actually governs throughput. Omitting it is not a
# neutral default, it is unbounded: the glm5.2 attempt ran without it, opened
# all 28,717 pipelines at once and decayed from 0.53 to 0.07 samples/s over 23h.
# 192 is measured, not guessed — at that setting the driver holds all 128
# connections open, so the slots stay full without over-spreading.
INFLIGHT=192

echo "$(date '+%F %T') === baseline, concurrency $CONC (no-op if already complete) ==="
python3 scripts/gen_rubrics.py --config "$CFG" --run-name "$RUN" \
  --sources baseline --concurrency "$CONC" --max-in-flight "$CONC" 2>&1 \
  | grep -v "HTTP Request"

# Hand the questions already finished by the unsharded run to whichever shard
# now owns them. Without this the t2 shard regenerates its half from scratch:
# the cache key includes the model name, so t1's cached calls are unreachable
# from t2 even though the two serve the same weights.
python3 scripts/seed_shards.py --run-dir "runs/$RUN" --source agentic-tools --shards 2

echo "$(date '+%F %T') === agentic-tools, 2 shards x (conc $CONC, in-flight $INFLIGHT) ==="
python3 scripts/gen_rubrics.py --config "$CFG" --run-name "$RUN" \
  --sources agentic-tools --model GLM-5.3-H20-t1 --shard 0/2 \
  --concurrency "$CONC" --max-in-flight "$INFLIGHT" 2>&1 \
  | grep -v "HTTP Request" | sed 's/^/[t1] /' &
PID_T1=$!
python3 scripts/gen_rubrics.py --config "$CFG" --run-name "$RUN" \
  --sources agentic-tools --model GLM-5.3-H20-t2 --shard 1/2 \
  --concurrency "$CONC" --max-in-flight "$INFLIGHT" 2>&1 \
  | grep -v "HTTP Request" | sed 's/^/[t2] /' &
PID_T2=$!

wait $PID_T1; echo "$(date '+%F %T') === shard 0 (t1) done ==="
wait $PID_T2; echo "$(date '+%F %T') === shard 1 (t2) done ==="

python3 scripts/merge_shards.py --run-dir "runs/$RUN" --source agentic-tools --force
echo "$(date '+%F %T') === all done ==="
