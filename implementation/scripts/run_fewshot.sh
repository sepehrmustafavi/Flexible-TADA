#!/bin/bash

# scripts/run_fewshot.sh

# This script evaluates the "Data Drought Resistance" (Scenario 4).
# It runs experiments with limited training samples (10, 50, 100)
# across multiple random seeds to ensure statistical significance.

# 1. Configuration
CONFIG="configs/roberta_glue.yaml"
METHODS=("fft" "static_tada" "lora" "flex_tada")TASKS=("mnli" "rte")         # MNLI for high-resource/low-shot, RTE for low-resource
SHOTS=(10 50 100)            # Number of training samples
SEEDS=(42 123 2026)          # Multiple seeds for mean/std-dev calculation

echo "================================================================"
echo "🌵 Starting Few-Shot Robustness Study (Data Drought)"
echo "Config: $CONFIG | Methods: ${METHODS[*]}"
echo "================================================================"

for TASK in "${TASKS[@]}"
do
    for SHOT in "${SHOTS[@]}"
    do
        for SEED in "${SEEDS[@]}"
        do
            for METHOD in "${METHODS[@]}"
            do
                echo "------------------------------------------------"
                echo "📊 Task: ${TASK^^} | Shot: $SHOT | Seed: $SEED | Method: ${METHOD^^}"
                echo "------------------------------------------------"

                # Define a unique output name for this specific trial
                TRIAL_NAME="${TASK}_${METHOD}_shot${SHOT}_seed${SEED}"
                
                # Execute the main pipeline with the --few_shot flag
                python main.py \
                    --config "$CONFIG" \
                    --method "$METHOD" \
                    --task "$TASK" \
                    --seed "$SEED" \
                    --few_shot "$SHOT"

                # Check if successful
                if [ $? -eq 0 ]; then
                    echo "✅ Trial Success: $TRIAL_NAME"
                    # Move results to a dedicated few-shot folder for organization
                    mkdir -p "outputs/few_shot_results"
                    mv "outputs/${TASK}_${METHOD}_${SEED}" "outputs/few_shot_results/$TRIAL_NAME"
                else
                    echo "❌ Trial Failed: $TRIAL_NAME"
                fi

            done
        done
    done
done

echo "================================================================"
echo "🎉 Few-Shot Experiments Completed!"
echo "All results are gathered in 'outputs/few_shot_results/'"
echo "You can now calculate the mean and variance for Table 3."
echo "================================================================"