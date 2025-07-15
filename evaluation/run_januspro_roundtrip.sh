#!/bin/bash

# Janus Pro Roundtrip Generation Script
# This script demonstrates how to run Janus Pro roundtrip generation

# Configuration - Update these paths as needed
MODEL_PATH="/path/to/your/januspro/model"  # Update this to your Janus Pro model path
PROMPTS_FILE="prompts.txt"  # Path to your prompts file
OUTPUT_DIR="januspro_roundtrip_results"
DEVICE=0
SEED=42

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path $MODEL_PATH does not exist!"
    echo "Please update the MODEL_PATH variable in this script."
    echo "You can download Janus Pro models from:"
    echo "- Janus-Pro-1B: https://huggingface.co/deepseek-ai/Janus-Pro-1B"
    echo "- Janus-Pro-7B: https://huggingface.co/deepseek-ai/Janus-Pro-7B"
    exit 1
fi

# Check if prompts file exists
if [ ! -f "$PROMPTS_FILE" ]; then
    echo "Error: Prompts file $PROMPTS_FILE not found!"
    echo "Please create a prompts.txt file or update the PROMPTS_FILE variable."
    exit 1
fi

echo "Janus Pro Roundtrip Generation"
echo "=============================="
echo "Model path: $MODEL_PATH"
echo "Prompts file: $PROMPTS_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Seed: $SEED"
echo ""

# Run the roundtrip generation
python roundtrip_generation.py \
    "$MODEL_PATH" \
    --model_type januspro \
    --prompts_file "$PROMPTS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --seed "$SEED"

echo ""
echo "Janus Pro roundtrip generation completed!"
echo "Results saved to: $OUTPUT_DIR" 