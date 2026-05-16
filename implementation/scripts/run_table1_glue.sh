#!/bin/bash

# scripts/run_table1_glue.sh
# Automated generation for Table 1 across 4 Encoder Models

# 1. Configuration Array (All 4 models)
CONFIGS=(
    "configs/roberta_glue.yaml"
    "configs/bert_glue.yaml"
    "configs/electra_glue.yaml"
    "configs/deberta_glue.yaml"
    "configs/qwen_glue.yaml" 
)
SEED=42
METHODS=("fft" "static_tada" "lora" "flex_tada")
TASKS=("sst2" "mnli" "qnli" "qqp" "mrpc" "cola" "rte" "stsb")

echo "----------------------------------------------------------------"
echo "🚀 Starting Massive Table 1 Experiments (4 Encoder Models)"
echo "----------------------------------------------------------------"

# 2. Outer Loop: Iterate through each Model Config
for CONFIG in "${CONFIGS[@]}"
do
    echo "================================================================"
    echo "🤖 Loading Model Configuration: $CONFIG"
    echo "================================================================"

    # 3. Middle Loop: Iterate through each GLUE task
    for TASK in "${TASKS[@]}"
    do
        echo "Processing Task: ${TASK^^}"
        
        # 4. Inner Loop: Iterate through each tuning method
        for METHOD in "${METHODS[@]}"
        do
            echo "Running Method: ${METHOD^^} on $TASK..."
            
            python main.py \
                --config "$CONFIG" \
                --method "$METHOD" \
                --task "$TASK" \
                --seed "$SEED"
                
            if [ $? -eq 0 ]; then
                echo "✅ Finished: $METHOD on $TASK ($CONFIG)"
            else
                echo "❌ Error: $METHOD failed on $TASK ($CONFIG)."
            fi
            
        done
    done
done

echo "🎉 ALL experiments for Table 1 (All Models) are successfully completed!"