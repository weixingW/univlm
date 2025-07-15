#!/bin/bash

# Show-o Roundtrip Generation Script
# This script provides various options for running Show-o roundtrip generation

set -e  # Exit on any error

# Default configuration
DEFAULT_MODEL_PATH="/path/to/showo/model"
DEFAULT_CONFIG_PATH="/path/to/showo/config.yaml"
DEFAULT_DEVICE=0
DEFAULT_SEED=42
DEFAULT_PROMPTS_FILE="test_prompts.txt"
DEFAULT_OUTPUT_DIR="showo_roundtrip_results"
DEFAULT_START_IDX=0
DEFAULT_END_IDX=""
DEFAULT_BATCH_SIZE=1

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to show usage
show_usage() {
    cat << EOF
Show-o Roundtrip Generation Script

Usage: $0 [OPTIONS] COMMAND

Commands:
    test           Run a simple test with a single prompt
    single         Run single roundtrip generation
    batch          Run batch roundtrip generation
    file           Run roundtrip generation from prompts file
    example        Run example usage script

Options:
    -m, --model-path PATH     Path to Show-o model (default: $DEFAULT_MODEL_PATH)
    -c, --config-path PATH    Path to Show-o config file (default: $DEFAULT_CONFIG_PATH)
    -d, --device DEVICE       CUDA device ID (default: $DEFAULT_DEVICE)
    -s, --seed SEED           Random seed (default: $DEFAULT_SEED)
    -p, --prompts-file PATH   Path to prompts file (default: $DEFAULT_PROMPTS_FILE)
    -o, --output-dir DIR      Output directory (default: $DEFAULT_OUTPUT_DIR)
    -b, --start-idx IDX       Starting prompt index (default: $DEFAULT_START_IDX)
    -e, --end-idx IDX         Ending prompt index (default: all)
    -h, --help                Show this help message

Examples:
    # Test the setup
    $0 test -m /path/to/model -c /path/to/config.yaml

    # Run single roundtrip generation
    $0 single -m /path/to/model -c /path/to/config.yaml

    # Run batch generation from file
    $0 file -m /path/to/model -c /path/to/config.yaml -p prompts.txt -o results

    # Run with specific device and seed
    $0 file -m /path/to/model -c /path/to/config.yaml -d 1 -s 123

EOF
}

# Function to validate required files
validate_files() {
    local model_path="$1"
    local config_path="$2"
    
    if [[ ! -f "$model_path" ]]; then
        print_error "Model path does not exist: $model_path"
        exit 1
    fi
    
    if [[ ! -f "$config_path" ]]; then
        print_error "Config path does not exist: $config_path"
        exit 1
    fi
}

# Function to run test
run_test() {
    local model_path="$1"
    local config_path="$2"
    local device="$3"
    local seed="$4"
    
    print_info "Running Show-o roundtrip test..."
    
    python test_showo.py \
        --model_path "$model_path" \
        --config_path "$config_path" \
        --device "$device" \
        --test_prompt "A beautiful sunset over the ocean with palm trees"
    
    print_success "Test completed successfully!"
}

# Function to run single roundtrip
run_single() {
    local model_path="$1"
    local config_path="$2"
    local device="$3"
    local seed="$4"
    
    print_info "Running single Show-o roundtrip generation..."
    
    python example_showo_usage.py \
        --model_path "$model_path" \
        --config_path "$config_path" \
        --device "$device" \
        --example single
    
    print_success "Single roundtrip completed successfully!"
}

# Function to run batch roundtrip
run_batch() {
    local model_path="$1"
    local config_path="$2"
    local device="$3"
    local seed="$4"
    
    print_info "Running batch Show-o roundtrip generation..."
    
    python example_showo_usage.py \
        --model_path "$model_path" \
        --config_path "$config_path" \
        --device "$device" \
        --example batch
    
    print_success "Batch roundtrip completed successfully!"
}

# Function to run file-based roundtrip
run_file() {
    local model_path="$1"
    local config_path="$2"
    local device="$3"
    local seed="$4"
    local prompts_file="$5"
    local output_dir="$6"
    local start_idx="$7"
    local end_idx="$8"
    
    print_info "Running file-based Show-o roundtrip generation..."
    print_info "Prompts file: $prompts_file"
    print_info "Output directory: $output_dir"
    print_info "Start index: $start_idx"
    if [[ -n "$end_idx" ]]; then
        print_info "End index: $end_idx"
    else
        print_info "End index: all prompts"
    fi
    
    # Check if prompts file exists
    if [[ ! -f "$prompts_file" ]]; then
        print_error "Prompts file does not exist: $prompts_file"
        exit 1
    fi
    
    # Create output directory
    mkdir -p "$output_dir"
    
    # Run roundtrip generation
    python roundtrip_generation.py \
        "$model_path" \
        --model_type showo \
        --config_path "$config_path" \
        --prompts_file "$prompts_file" \
        --output_dir "$output_dir" \
        --start_idx "$start_idx" \
        --end_idx "$end_idx" \
        --device "$device" \
        --seed "$seed"
    
    print_success "File-based roundtrip completed successfully!"
}

# Function to run example
run_example() {
    local model_path="$1"
    local config_path="$2"
    local device="$3"
    local prompts_file="$4"
    
    print_info "Running Show-o example usage..."
    
    python example_showo_usage.py \
        --model_path "$model_path" \
        --config_path "$config_path" \
        --device "$device" \
        --example file \
        --prompts_file "$prompts_file"
    
    print_success "Example completed successfully!"
}

# Parse command line arguments
COMMAND=""
MODEL_PATH="$DEFAULT_MODEL_PATH"
CONFIG_PATH="$DEFAULT_CONFIG_PATH"
DEVICE="$DEFAULT_DEVICE"
SEED="$DEFAULT_SEED"
PROMPTS_FILE="$DEFAULT_PROMPTS_FILE"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
START_IDX="$DEFAULT_START_IDX"
END_IDX="$DEFAULT_END_IDX"

while [[ $# -gt 0 ]]; do
    case $1 in
        test|single|batch|file|example)
            COMMAND="$1"
            shift
            ;;
        -m|--model-path)
            MODEL_PATH="$2"
            shift 2
            ;;
        -c|--config-path)
            CONFIG_PATH="$2"
            shift 2
            ;;
        -d|--device)
            DEVICE="$2"
            shift 2
            ;;
        -s|--seed)
            SEED="$2"
            shift 2
            ;;
        -p|--prompts-file)
            PROMPTS_FILE="$2"
            shift 2
            ;;
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -b|--start-idx)
            START_IDX="$2"
            shift 2
            ;;
        -e|--end-idx)
            END_IDX="$2"
            shift 2
            ;;
        -h|--help)
            show_usage
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if command is provided
if [[ -z "$COMMAND" ]]; then
    print_error "No command specified"
    show_usage
    exit 1
fi

# Print configuration
print_info "Configuration:"
print_info "  Model path: $MODEL_PATH"
print_info "  Config path: $CONFIG_PATH"
print_info "  Device: $DEVICE"
print_info "  Seed: $SEED"
print_info "  Command: $COMMAND"

# Validate files for all commands except test
if [[ "$COMMAND" != "test" ]]; then
    validate_files "$MODEL_PATH" "$CONFIG_PATH"
fi

# Run the specified command
case "$COMMAND" in
    test)
        run_test "$MODEL_PATH" "$CONFIG_PATH" "$DEVICE" "$SEED"
        ;;
    single)
        run_single "$MODEL_PATH" "$CONFIG_PATH" "$DEVICE" "$SEED"
        ;;
    batch)
        run_batch "$MODEL_PATH" "$CONFIG_PATH" "$DEVICE" "$SEED"
        ;;
    file)
        run_file "$MODEL_PATH" "$CONFIG_PATH" "$DEVICE" "$SEED" "$PROMPTS_FILE" "$OUTPUT_DIR" "$START_IDX" "$END_IDX"
        ;;
    example)
        run_example "$MODEL_PATH" "$CONFIG_PATH" "$DEVICE" "$PROMPTS_FILE"
        ;;
    *)
        print_error "Unknown command: $COMMAND"
        show_usage
        exit 1
        ;;
esac

print_success "All operations completed successfully!" 