#!/bin/bash

# scripts/run_stacked_heatmap.sh
# Bash wrapper for generating stacked XAI heatmaps for the journal paper


echo "================================================================"
echo "🎨 Starting Stacked Heatmap Generation via Python Engine..."
echo "================================================================"

# Execute the core Python script located in the root directory
python generate_stacked_heatmap.py

if [ $? -eq 0 ]; then
    echo "================================================================"
    echo "🎉 Stacked Heatmap Successfully Generated and Saved!"
    echo "================================================================"
else
    echo "❌ ERROR: Heatmap generation failed. Please check the logs."
    exit 1
fi