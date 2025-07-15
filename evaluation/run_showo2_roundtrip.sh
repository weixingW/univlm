#!/bin/bash

# Show-o2 Roundtrip Generation Script
# This script provides easy access to Show-o2 roundtrip generation functionality

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
MODEL_PATH="showlab/show-o2-7b"
CONFIG_PATH="../configs/showo2_config.yaml"
DEVICE=0
SEED=42
PROMPTS_FILE="prompts.txt"
OUTPUT_DIR="showo2_roundtrip_results"
START_IDX=0
END_IDX=""
GENERATE_FROM_CAPTIONS=false
CAPTIONS_DIR=""

# Function to print usage
print_usage() {
    echo -e "${BLUE}Show-o2 Roundtrip Generation Script${NC}"
    echo ""
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -m, --model-path PATH     Path to Show-o2 model (default: $MODEL_PATH)"
    echo "  -c, --config-path PATH    Path to configuration file (default: $CONFIG_PATH)"
    echo "  -d, --device DEVICE       CUDA device ID (default: $DEVICE)"
    echo "  -s, --seed SEED           Random seed (default: $SEED)"
    echo "  -p, --prompts-file FILE   Path to prompts file (default: $PROMPTS_FILE)"
    echo "  -o, --output-dir DIR      Output directory (default: $OUTPUT_DIR)"
    echo "  --start-idx IDX           Starting prompt index (default: $START_IDX)"
    echo "  --end-idx IDX             Ending prompt index (default: all)"
    echo "  --from-captions           Generate images from existing captions"
    echo "  --captions-dir DIR        Directory containing caption files"
    echo "  -h, --help                Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 -m showlab/show-o2-7b -c ../configs/showo2_config.yaml -p my_prompts.txt -o results"
    echo "  $0 --from-captions --captions-dir results/captions -o caption_images"
    echo "  $0 -m showlab/show-o2-1.5b --start-idx 10 --end-idx 20"
}

# Function to check dependencies
check_dependencies() {
    echo -e "${BLUE}Checking dependencies...${NC}"
    
    # Check if Python is available
    if ! command -v python3 &> /dev/null; then
        echo -e "${RED}Error: python3 is not installed${NC}"
        exit 1
    fi
    
    # Check if required Python packages are available
    python3 -c "import torch" 2>/dev/null || {
        echo -e "${RED}Error: PyTorch is not installed${NC}"
        exit 1
    }
    
    python3 -c "import PIL" 2>/dev/null || {
        echo -e "${RED}Error: Pillow is not installed${NC}"
        exit 1
    }
    
    # Check if Show-o2 directory exists
    if [ ! -d "../Show-o/show-o2" ]; then
        echo -e "${RED}Error: Show-o2 directory not found at ../Show-o/show-o2${NC}"
        echo -e "${YELLOW}Make sure the Show-o folder is in the univlm directory${NC}"
        exit 1
    fi
    
    echo -e "${GREEN}✓ Dependencies check passed${NC}"
}

# Function to check CUDA availability
check_cuda() {
    echo -e "${BLUE}Checking CUDA availability...${NC}"
    
    if python3 -c "import torch; print(torch.cuda.is_available())" 2>/dev/null | grep -q "True"; then
        echo -e "${GREEN}✓ CUDA is available${NC}"
        python3 -c "import torch; print(f'CUDA device: {torch.cuda.get_device_name($DEVICE)}')"
    else
        echo -e "${YELLOW}⚠ CUDA is not available, will use CPU${NC}"
    fi
}

# Function to run roundtrip generation
run_roundtrip() {
    echo -e "${BLUE}Running Show-o2 roundtrip generation...${NC}"
    
    # Build command
    cmd="python3 roundtrip_main.py"
    cmd="$cmd $MODEL_PATH"
    cmd="$cmd --model_type showo2"
    if [ -n "$CONFIG_PATH" ]; then
        cmd="$cmd --config_path $CONFIG_PATH"
    fi
    cmd="$cmd --device $DEVICE"
    cmd="$cmd --seed $SEED"
    cmd="$cmd --prompts_file $PROMPTS_FILE"
    cmd="$cmd --output_dir $OUTPUT_DIR"
    cmd="$cmd --start_idx $START_IDX"
    
    if [ -n "$END_IDX" ]; then
        cmd="$cmd --end_idx $END_IDX"
    fi
    
    if [ "$GENERATE_FROM_CAPTIONS" = true ]; then
        cmd="$cmd --generate_from_captions"
        if [ -n "$CAPTIONS_DIR" ]; then
            cmd="$cmd --captions_dir $CAPTIONS_DIR"
        fi
    fi
    
    echo -e "${YELLOW}Executing: $cmd${NC}"
    echo ""
    
    # Execute command
    eval $cmd
}

# Function to run test
run_test() {
    echo -e "${BLUE}Running Show-o2 roundtrip test...${NC}"
    
    if [ -f "test_showo2_roundtrip.py" ]; then
        python3 test_showo2_roundtrip.py
    else
        echo -e "${RED}Error: test_showo2_roundtrip.py not found${NC}"
        exit 1
    fi
}

# Function to run example
run_example() {
    echo -e "${BLUE}Running Show-o2 roundtrip example...${NC}"
    
    if [ -f "example_showo2_usage.py" ]; then
        python3 example_showo2_usage.py
    else
        echo -e "${RED}Error: example_showo2_usage.py not found${NC}"
        exit 1
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
        --start-idx)
            START_IDX="$2"
            shift 2
            ;;
        --end-idx)
            END_IDX="$2"
            shift 2
            ;;
        --from-captions)
            GENERATE_FROM_CAPTIONS=true
            shift
            ;;
        --captions-dir)
            CAPTIONS_DIR="$2"
            shift 2
            ;;
        --test)
            check_dependencies
            check_cuda
            run_test
            exit 0
            ;;
        --example)
            check_dependencies
            check_cuda
            run_example
            exit 0
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            print_usage
            exit 1
            ;;
    esac
done

# Main execution
echo -e "${BLUE}Show-o2 Roundtrip Generation${NC}"
echo "=================================="

# Check dependencies
check_dependencies

# Check CUDA
check_cuda

    # Validate inputs
    if [ "$GENERATE_FROM_CAPTIONS" = false ] && [ ! -f "$PROMPTS_FILE" ]; then
        echo -e "${RED}Error: Prompts file '$PROMPTS_FILE' not found${NC}"
        exit 1
    fi

    if [ "$GENERATE_FROM_CAPTIONS" = true ] && [ -n "$CAPTIONS_DIR" ] && [ ! -d "$CAPTIONS_DIR" ]; then
        echo -e "${RED}Error: Captions directory '$CAPTIONS_DIR' not found${NC}"
        exit 1
    fi

    if [ -n "$CONFIG_PATH" ] && [ ! -f "$CONFIG_PATH" ]; then
        echo -e "${RED}Error: Config file '$CONFIG_PATH' not found${NC}"
        exit 1
    fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run roundtrip generation
run_roundtrip

echo -e "${GREEN}Show-o2 roundtrip generation completed!${NC}"
echo -e "Results saved to: $OUTPUT_DIR" 