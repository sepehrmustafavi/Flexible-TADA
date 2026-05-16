#!/bin/bash

# scripts/run_xai_table5.sh

# This script automates the Explainable AI (XAI) analysis for Table 5.
# It evaluates Faithfulness and Sufficiency on the BEST saved checkpoints
# to prove that Flexible TADA relies on deep semantics rather than surface heuristics.

CONFIG="configs/roberta_glue.yaml"
# We choose SST-2 and MRPC as they are great for word-level semantic importance
TASKS=("sst2" "mrpc") 
METHODS=("fft" "static_tada" "lora" "flex_tada")
SEED=42

echo "================================================================"
echo "🧠 Starting XAI Deep Analysis (Faithfulness & Sufficiency)"
echo "Tasks: ${TASKS[*]} | Methods: ${METHODS[*]}"
echo "================================================================"

# Create a directory specifically for XAI results
mkdir -p "outputs/xai_results"

for TASK in "${TASKS[@]}"
do
    echo "------------------------------------------------"
    echo "🔬 Analyzing Task: ${TASK^^}"
    echo "------------------------------------------------"
    
    for METHOD in "${METHODS[@]}"
    do
        echo "⏳ Running XAI Evaluator for: ${METHOD^^}..."
        
        # The path where the best model checkpoint was saved during training
        MODEL_PATH="outputs/${TASK}_${METHOD}_${SEED}"
        
        # We assume you will have a python script named 'run_xai_analysis.py'
        # in the root directory to handle the loading and XAI metric calculation.
        python run_xai_analysis.py \
            --config "$CONFIG" \
            --task "$TASK" \
            --method "$METHOD" \
            --model_path "$MODEL_PATH" \
            --output_dir "outputs/xai_results"
            
        if [ $? -eq 0 ]; then
            echo "✅ XAI Analysis complete for $METHOD on $TASK."
        else
            echo "❌ ERROR: XAI Analysis failed for $METHOD. Ensure the model exists at $MODEL_PATH."
        fi
    done
done

echo "================================================================"
echo "🎉 XAI Analysis Completed!"
echo "Check 'outputs/xai_results/' for the JSON files containing"
echo "the Faithfulness and Sufficiency scores for Table 5."
echo "================================================================"