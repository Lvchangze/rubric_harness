#!/usr/bin/env bash
# Generate and score a list of candidate variants on dev, one at a time.
#
# Sequential on purpose. The endpoint has been knocked over at 192 concurrent
# requests, so the budget is one process at --concurrency 64 rather than several
# smaller ones whose sum drifts past it.
#
# Dev scoring runs BOTH presentation orders, because the protocol reports the
# position-controlled reading (a case counts only if the verdict survives the
# swap) alongside the forward one. Set ORDER=single to halve the judge cost when
# only forward is wanted: the forward call is byte-identical either way, verified
# at 40/40 against `framed`'s original both-order run.
#
# Specs:
#   <variant>                        generate that variant under its own name
#   gen:<name>:<variant>:<flags>     ... with extra rubricbench_gen.py flags
#   mech:<name>:<transform>[:<src>]  mechanical rewrite, no model call
#
# Usage:  scripts/rubricbench_sweep.sh qform balanced contrastive typed
#         scripts/rubricbench_sweep.sh mech:framed_noweight:strip_weights
#         scripts/rubricbench_sweep.sh 'gen:cf3:contrastive:--fewshot 3'
set -uo pipefail
cd "$(dirname "$0")/.."

export LLM_REQUEST_TIMEOUT_S=1200
SPLIT="${SPLIT:-results/rubricbench/split.json:dev}"
CONC="${CONC:-64}"
ORDER_FLAG=""
[[ "${ORDER:-both}" == "single" ]] && ORDER_FLAG="--single-order"
RUBRICS=results/rubricbench/opt/rubrics

for spec in "$@"; do
  if [[ "$spec" == mech:* ]]; then
    IFS=: read -r _ name kind src <<<"$spec"
    src="${src:-results/rubricbench/framed_rubrics.json}"
    echo "=== [$(date +%H:%M:%S)] transform $name ($kind from $src) ==="
    python3 scripts/rubricbench_gen.py --from "$src" --transform "$kind" \
      --name "$name" --case-ids "$SPLIT" || { echo "FAILED transform $name"; continue; }
  elif [[ "$spec" == gen:* ]]; then
    IFS=: read -r _ name variant extra <<<"$spec"
    echo "=== [$(date +%H:%M:%S)] generate $name (variant=$variant ${extra:-}) ==="
    # shellcheck disable=SC2086 -- extra flags are meant to word-split
    python3 scripts/rubricbench_gen.py --variant "$variant" --name "$name" \
      --case-ids "$SPLIT" --concurrency "$CONC" ${extra:-} \
      || { echo "FAILED gen $name"; continue; }
  else
    name="$spec"
    echo "=== [$(date +%H:%M:%S)] generate $name ==="
    python3 scripts/rubricbench_gen.py --variant "$name" --case-ids "$SPLIT" \
      --concurrency "$CONC" || { echo "FAILED gen $name"; continue; }
  fi

  echo "=== [$(date +%H:%M:%S)] judge $name ==="
  python3 scripts/rubricbench_run.py --source "file:$RUBRICS/$name.json" \
    --case-ids "$SPLIT" $ORDER_FLAG --concurrency "$CONC" --tag "dev_$name" --out-dir results/rubricbench/opt/runs \
    || { echo "FAILED judge $name"; continue; }
  echo "=== [$(date +%H:%M:%S)] done $name ==="
done

echo "=== [$(date +%H:%M:%S)] sweep complete ==="
