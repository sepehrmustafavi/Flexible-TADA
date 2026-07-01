#!/bin/bash

# scripts/run_cka_analysis.sh
# Bash wrapper for Representation Collapse Analysis (CKA)

# Change to project root directory
cd "$(dirname "$0")/.." || exit 1

# Set device
export CUDA_VISIBLE_DEVICES=0

echo "================================================================"
echo "🧠 Starting Representation Analysis (CKA) via Python Engine..."
echo "================================================================"

# Execute the core Python script located in the root directory
python run_representation_analysis.py

if [ $? -eq 0 ]; then
    echo "================================================================"
    echo "🎉 CKA Analysis Execution Completed Successfully!"
    echo "================================================================"
else
    echo "❌ ERROR: CKA Analysis failed. Please check the logs."
    exit 1
fi
