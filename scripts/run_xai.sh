#!/bin/bash

# scripts/run_xai.sh

# Change to project root directory
cd "$(dirname "$0")/.." || exit 1

# Set device
export CUDA_VISIBLE_DEVICES=0

CONFIG="configs/roberta_glue.yaml"
TASKS=("sst2" "mrpc")
METHODS=("fft" "static_tada" "lora" "flex_tada")
SEED=42

echo "================================================================"
echo "🧠 Starting XAI Deep Analysis (Faithfulness & Sufficiency) on FULL Data"
echo "Tasks: ${TASKS[*]} | Methods: ${METHODS[*]}"
echo "================================================================"

mkdir -p "outputs/xai_results"

for TASK in "${TASKS[@]}"
do
    echo "------------------------------------------------"
    echo "🔬 Analyzing Task: ${TASK^^}"
    echo "------------------------------------------------"

    for METHOD in "${METHODS[@]}"
    do
        echo "⏳ Running XAI Evaluator for: ${METHOD^^}..."

        RUN_DIR="outputs/${TASK}_${METHOD}_${SEED}"
        RESULT_JSON="${RUN_DIR}/results_${TASK}_${METHOD}.json"

        if [ ! -d "$RUN_DIR" ]; then
            echo "❌ ERROR: Run directory does not exist: $RUN_DIR"
            continue
        fi

        if [ ! -f "$RESULT_JSON" ]; then
            echo "❌ ERROR: Result JSON not found: $RESULT_JSON"
            continue
        fi

        MODEL_PATH=$(python - <<PY
import json
import os
import glob
import math

run_dir = "$RUN_DIR"
result_json = "$RESULT_JSON"

with open(result_json, "r") as f:
    results = json.load(f)

best_epoch = float(results["epoch"])

checkpoints = sorted(
    glob.glob(os.path.join(run_dir, "checkpoint-*")),
    key=lambda p: int(os.path.basename(p).split("-")[-1])
)

selected = None

# Preferred: match using trainer_state.json epoch
for ckpt in checkpoints:
    state_path = os.path.join(ckpt, "trainer_state.json")
    if not os.path.exists(state_path):
        continue

    try:
        with open(state_path, "r") as f:
            state = json.load(f)

        ckpt_epoch = state.get("epoch", None)

        if ckpt_epoch is not None and math.isclose(float(ckpt_epoch), best_epoch, rel_tol=1e-4, abs_tol=1e-4):
            selected = ckpt
            break

    except Exception:
        pass

# Fallback: assume checkpoints are saved once per epoch
if selected is None:
    epoch_index = int(round(best_epoch)) - 1
    if 0 <= epoch_index < len(checkpoints):
        selected = checkpoints[epoch_index]

if selected is None:
    raise SystemExit(f"Could not find checkpoint for epoch {best_epoch} in {run_dir}")

print(selected)
PY
)

        if [ -z "$MODEL_PATH" ]; then
            echo "❌ ERROR: Could not determine checkpoint path for $METHOD on $TASK"
            continue
        fi

        echo "✅ Selected checkpoint: $MODEL_PATH"

        python run_xai_analysis.py \
            --config "$CONFIG" \
            --task "$TASK" \
            --method "$METHOD" \
            --model_path "$MODEL_PATH" \
            --output_dir "outputs/xai_results" \
            --max_samples -1

        if [ $? -eq 0 ]; then
            echo "✅ XAI Analysis complete for $METHOD on $TASK."
        else
            echo "❌ ERROR: XAI Analysis failed for $METHOD on $TASK using $MODEL_PATH."
        fi
    done
done

echo "================================================================"
echo "🎉 XAI Analysis Completed!"
echo "Check 'outputs/xai_results/' for the JSON files."
echo "================================================================"