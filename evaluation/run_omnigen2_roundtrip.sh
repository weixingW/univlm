#!/bin/bash

# OmniGen2 Roundtrip Generation Script
# This script demonstrates how to run OmniGen2 roundtrip generation

# Configuration - Update these paths as needed
MODEL_PATH="/path/to/your/omnigen2/model"  # Update this to your OmniGen2 model path
PROMPTS_FILE="prompts.txt"  # Path to your prompts file
OUTPUT_DIR="omnigen2_roundtrip_results"
DEVICE=0
SEED=42

# Check if model path exists
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model path $MODEL_PATH does not exist!"
    echo "Please update the MODEL_PATH variable in this script."
    exit 1
fi

# Check if prompts file exists
if [ ! -f "$PROMPTS_FILE" ]; then
    echo "Error: Prompts file $PROMPTS_FILE not found!"
    echo "Please create a prompts.txt file or update the PROMPTS_FILE variable."
    exit 1
fi

echo "OmniGen2 Roundtrip Generation"
echo "============================="
echo "Model path: $MODEL_PATH"
echo "Prompts file: $PROMPTS_FILE"
echo "Output directory: $OUTPUT_DIR"
echo "Device: $DEVICE"
echo "Seed: $SEED"
echo ""

# Run the roundtrip generation
python roundtrip_generation.py \
    "$MODEL_PATH" \
    --model_type omnigen2 \
    --prompts_file "$PROMPTS_FILE" \
    --output_dir "$OUTPUT_DIR" \
    --device "$DEVICE" \
    --seed "$SEED"

echo ""
echo "OmniGen2 roundtrip generation completed!"
echo "Results saved to: $OUTPUT_DIR" 