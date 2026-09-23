#!/usr/bin/env bash
# Chain the two train-split generation stages on GLM-5.3.
#
# Single endpoint again. The run was briefly sharded across GLM-5.3 t1 and t2,
# which were separate capacity pools; both were retired on 2026-09-18 and
# t2-copy is the only GLM-5.3 deployment left. The sharding machinery stays in
# the tree (--shard, seed_shards.py, merge_shards.py) because a second endpoint
# may reappear, but splitting one pool across two processes buys nothing.
#
# Every stage resumes by uid, so killing and restarting this script is safe and
# only pays for questions that had not finished. Partial shard output from the
# earlier attempt was merged back into the unsharded file before this ran, so
# resume sees all of it.
set -uo pipefail
cd "$(dirname "$0")/.."
CFG=configs/train_full.yaml
RUN=train_full_glm53
CONC=128

# --max-in-flight bounds live *questions*, separate from --concurrency (live
# requests), and it is what actually governs throughput. Omitting it is not a
# neutral default, it is unbounded: the glm5.2 attempt ran without it, opened
# all 28,717 pipelines at once and decayed from 0.53 to 0.07 samples/s over 23h.
#
# Held at 1.5x concurrency, the ratio measured to keep the slots full: at 192
# in-flight the driver held all 128 connections open. A question is mostly
# sequential, which is why in-flight has to exceed concurrency at all.
INFLIGHT=192

echo "$(date '+%F %T') === baseline, concurrency $CONC (no-op if already complete) ==="
python3 scripts/gen_rubrics.py --config "$CFG" --run-name "$RUN" \
  --sources baseline --concurrency "$CONC" --max-in-flight "$CONC" 2>&1 \
  | grep -v "HTTP Request"

echo "$(date '+%F %T') === agentic-tools, concurrency $CONC, in-flight $INFLIGHT ==="
python3 scripts/gen_rubrics.py --config "$CFG" --run-name "$RUN" \
  --sources agentic-tools --concurrency "$CONC" --max-in-flight "$INFLIGHT" 2>&1 \
  | grep -v "HTTP Request"
echo "$(date '+%F %T') === agentic-tools done, exit $? ==="
