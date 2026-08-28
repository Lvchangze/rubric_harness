#!/usr/bin/env bash
# End-to-end pilot driver. Every stage is resumable, so re-running after an
# interruption picks up where it stopped instead of redoing work.
#
#   bash scripts/run_pilot.sh [config] [run_name]
set -euo pipefail

CONFIG="${1:-configs/pilot.yaml}"
RUN_NAME="${2:-pilot_v2}"
cd "$(dirname "$0")/.."

export LLM_REQUEST_TIMEOUT_S=1200
LOG_DIR="runs/${RUN_NAME}"
mkdir -p "$LOG_DIR"

step() { echo; echo "=== $* ==="; date '+%F %T'; }

# The baseline column is only meaningful if it reproduces the paper's one-shot
# generator exactly, so refuse to spend budget on a run whose prompts drifted.
step "0/7 verify prompt fidelity against docs/00_paper_notes.md"
python scripts/check_prompt_fidelity.py 2>&1 | tee -a "$LOG_DIR/fidelity.log"

step "1/7 generate rubrics (shipped, baseline, agentic)"
python scripts/gen_rubrics.py --config "$CONFIG" --run-name "$RUN_NAME" \
    --sources shipped baseline agentic 2>&1 | tee -a "$LOG_DIR/gen.log"

step "2/7 generate ablation rubrics (agentic-noval)"
python scripts/gen_rubrics.py --config "$CONFIG" --run-name "$RUN_NAME" \
    --sources agentic-noval 2>&1 | tee -a "$LOG_DIR/gen_ablation.log"

# Zero-LLM metrics run before anything expensive: they reuse the forensics
# detectors, so if the shipped column here disagrees with the published n=45k
# numbers, something is wrong with the sample and the pilot should not proceed.
step "3/7 zero-LLM structural metrics (grounding, lint, adaptivity)"
python scripts/eval_rubrics.py --config "$CONFIG" --run-name "$RUN_NAME" \
    --sources shipped baseline agentic-noval agentic \
    --metrics grounding lint adaptivity 2>&1 | tee -a "$LOG_DIR/eval_structural.log"

step "4/7 build the shared response ladder"
python scripts/build_responses.py --config "$CONFIG" --run-name "$RUN_NAME" \
    2>&1 | tee -a "$LOG_DIR/responses.log"

step "5/7 evaluate every rubric source (LLM-judged)"
python scripts/eval_rubrics.py --config "$CONFIG" --run-name "$RUN_NAME" \
    --sources shipped baseline agentic-noval agentic --metrics all \
    2>&1 | tee -a "$LOG_DIR/eval.log"

# Confound guard: if agentic only wins because it writes more criteria, that is
# not a win. Re-scored with every source cut to the same per-question count.
step "6/7 count-controlled re-evaluation"
python scripts/eval_rubrics.py --config "$CONFIG" --run-name "$RUN_NAME" \
    --sources shipped baseline agentic-noval agentic --count-controlled \
    --metrics grounding coverage discriminative 2>&1 | tee -a "$LOG_DIR/eval_count_controlled.log"

step "7/7 aggregate into results/"
python scripts/aggregate_results.py --run-name "$RUN_NAME" --reference baseline --by-domain \
    2>&1 | tee -a "$LOG_DIR/aggregate.log"
python scripts/case_study.py --run-name "$RUN_NAME" --top 2 \
    --out "results/${RUN_NAME}/cases.md"

step "done"
echo "tables  -> results/${RUN_NAME}/summary_table.md"
echo "cases   -> results/${RUN_NAME}/cases.md"
