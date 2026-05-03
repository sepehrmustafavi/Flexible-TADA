#!/bin/bash

# scripts/run_ablation.sh

# This script automates the Deep Layer Ablation study.
# It dynamically modifies the YAML config to unfreeze different transformer layers
# and tracks how the representation depth affects the task performance.

CONFIG="configs/roberta_glue.yaml"
TASK="sst2" # We use SST-2 because it's fast and sensitive to semantic changes
METHOD="flex_tada"
SEED=42

# The layers we want to test: Layer 0 (Bottom), 6 (Middle), 10 (Penultimate), 11 (Top)
LAYERS=(0 6 10 11)

echo "================================================================"
echo "🔬 Starting Deep Layer Ablation Study"
echo "Task: ${TASK^^} | Method: ${METHOD^^}"
echo "================================================================"

for LAYER in "${LAYERS[@]}"
do
    echo "------------------------------------------------"
    echo "⚙️ Modifying config to unfreeze Layer: $LAYER"
    
    # We use 'sed' to dynamically update the layer number in the YAML file.
    # This searches for "encoder.layer.X" and replaces X with the current loop layer.
    sed -i "s/encoder\.layer\.[0-9]\+/encoder.layer.$LAYER/g" "$CONFIG"
    
    echo "🚀 Running Training for Layer $LAYER..."
    
    # Execute the pipeline
    python main.py \
        --config "$CONFIG" \
        --method "$METHOD" \
        --task "$TASK" \
        --seed "$SEED"
        
    # Check if the execution was successful
    if [ $? -eq 0 ]; then
        echo "✅ Successfully Finished Layer $LAYER"
        
        # CRITICAL: Rename the output folder so the next loop doesn't overwrite it!
        # E.g., renames "outputs/sst2_flex_tada_42" to "outputs/sst2_flex_tada_42_layer11"
        mv "outputs/${TASK}_${METHOD}_${SEED}" "outputs/${TASK}_${METHOD}_${SEED}_layer${LAYER}"
    else
        echo "❌ CRITICAL ERROR: Failed on Layer $LAYER."
        # Restore the config before exiting to avoid leaving it in a broken state
        sed -i "s/encoder\.layer\.[0-9]\+/encoder.layer.11/g" "$CONFIG"
        exit 1
    fi
done

# Restore the YAML file to its default state (Layer 11) for future experiments
sed -i "s/encoder\.layer\.[0-9]\+/encoder.layer.11/g" "$CONFIG"

echo "================================================================"
echo "🎉 Ablation Study Completed!"
echo "Check the 'outputs/' folder for the JSON results of each layer."
echo "You can now plot the accuracy curve for the paper."
echo "================================================================"