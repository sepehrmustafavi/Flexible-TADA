#!/bin/bash

# scripts/run_validity_analysis.sh
# Architectural Boundaries Experiment: Model Depth vs. Causal Masking

TASKS=("sst2" "qnli")

CONFIGS=(
    "configs/Qwen2-0.5B_glue.yaml"
    "configs/deberta-v2-xxlarge_glue.yaml"
    "configs/Qwen3.5-2B_glue.yaml"
)

METHODS=("lora" "static_tada" "flex_tada")

SEED=42

echo "----------------------------------------------------------------"
echo "🚀 Starting Validity Analysis (Depth vs. Causal Masking)"
echo "----------------------------------------------------------------"

for CONFIG in "${CONFIGS[@]}"
do
    MODEL_NAME=$(basename "$CONFIG" .yaml)

    echo "================================================================"
    echo "🤖 Processing Model: $MODEL_NAME"
    echo "================================================================"

    mkdir -p "outputs/${MODEL_NAME}"

    for TASK in "${TASKS[@]}"
    do
        echo "📚 Task: ${TASK^^}"

        for METHOD in "${METHODS[@]}"
        do
            echo "   ⏳ Running Method: ${METHOD^^}"

            python main.py \
                --config "$CONFIG" \
                --task "$TASK" \
                --method "$METHOD" \
                --seed "$SEED"

            if [ $? -eq 0 ]; then
                echo "   ✅ Finished: $METHOD on $TASK"

                OUTPUT_DIR="outputs/${TASK}_${METHOD}_${SEED}"

                if [ -d "$OUTPUT_DIR" ]; then

                    # Remove Trainer checkpoints
                    echo "   🧹 Removing checkpoints..."
                    rm -rf "$OUTPUT_DIR"/checkpoint-*

                    # Move final output into model folder
                    DEST_DIR="outputs/${MODEL_NAME}/$(basename "$OUTPUT_DIR")"

                    echo "   📦 Moving output to $DEST_DIR"
                    mv "$OUTPUT_DIR" "$DEST_DIR"
                fi

            else
                echo "   ❌ Error: $METHOD failed on $TASK"
            fi

            echo "------------------------------------------------"
        done
    done
done

echo "🎉 Validity Analysis experiments are fully completed!"