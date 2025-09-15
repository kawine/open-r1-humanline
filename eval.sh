#!/bin/bash

# Configuration
MODEL_NAME="Qwen2.5-1.5B-Instruct_128_010_100"
BASE_DIR="/scratch/kawin/open-r1/data"
MODEL_DIR="${BASE_DIR}/${MODEL_NAME}"
NUM_GPUS=1
TASK="math_500"


# CRITICAL: Fix CUDA fork issue
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export TOKENIZERS_PARALLELISM=false
# Optional: Force single worker
export VLLM_NUM_WORKER=0  # This forces vLLM to use no additional worker processes

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting evaluation pipeline for model: ${MODEL_NAME}${NC}"
echo -e "${GREEN}Model directory: ${MODEL_DIR}${NC}"

# Check if model directory exists
if [ ! -d "${MODEL_DIR}" ]; then
    echo -e "${RED}Error: Model directory ${MODEL_DIR} does not exist${NC}"
    exit 1
fi

# Find all checkpoint directories
CHECKPOINTS=$(find "${MODEL_DIR}" -type d -name "checkpoint-*" | sort -V)

if [ -z "$CHECKPOINTS" ]; then
    echo -e "${RED}No checkpoint directories found in ${MODEL_DIR}${NC}"
    exit 1
fi

echo -e "${GREEN}Found checkpoints:${NC}"
echo "$CHECKPOINTS"
echo ""

# Process each checkpoint
for CHECKPOINT_DIR in $CHECKPOINTS; do
    CHECKPOINT_NAME=$(basename "$CHECKPOINT_DIR")
    echo -e "${YELLOW}========================================${NC}"
    echo -e "${YELLOW}Processing ${CHECKPOINT_NAME}${NC}"
    echo -e "${YELLOW}========================================${NC}"
    
    # Check if already converted (pytorch_model.bin exists)
    if [ -f "${CHECKPOINT_DIR}/pytorch_model.bin" ]; then
        echo -e "${GREEN}Checkpoint already converted (pytorch_model.bin exists)${NC}"
    else
        # Check if zero_to_fp32.py exists
        if [ ! -f "${CHECKPOINT_DIR}/zero_to_fp32.py" ]; then
            echo -e "${RED}Warning: zero_to_fp32.py not found in ${CHECKPOINT_DIR}, skipping...${NC}"
            continue
        fi
        
        echo -e "${GREEN}Converting Zero-3 checkpoint to consolidated format...${NC}"
        
        # Run the conversion
        cd "${CHECKPOINT_DIR}"
        python zero_to_fp32.py . pytorch_model.bin
        
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Successfully converted ${CHECKPOINT_NAME}${NC}"
            
            # Optional: Clean up sharded files to save space after conversion
            # Uncomment the following lines if you want to delete the sharded files
            # echo "Cleaning up sharded checkpoint files..."
            # rm -f zero_pp_rank_*.pt
            # rm -f bf16_zero_pp_rank_*.pt
            # rm -f fp32_zero_pp_rank_*.pt
        else
            echo -e "${RED}Failed to convert ${CHECKPOINT_NAME}, skipping evaluation...${NC}"
            continue
        fi
        
        cd - > /dev/null
    fi
    
    # Run evaluation on the converted checkpoint
    echo -e "${GREEN}Running evaluation on ${CHECKPOINT_NAME}...${NC}"
    
    # Set up evaluation parameters
    MODEL_PATH="${CHECKPOINT_DIR}"
    OUTPUT_DIR="${BASE_DIR}/evals/${MODEL_NAME}/${CHECKPOINT_NAME}/${TASK}"
    
    # Create output directory
    mkdir -p "${OUTPUT_DIR}"
    
    # Set up model arguments for lighteval
    MODEL_ARGS="model_name=${MODEL_PATH},dtype=bfloat16,max_model_length=4096,gpu_memory_utilization=0.8,generation_parameters={max_new_tokens:2048,temperature:0.6,top_p:0.95}"
    
    # Run lighteval
    echo "Command: lighteval vllm ${MODEL_ARGS} \"lighteval|${TASK}|0|0\" --use-chat-template --output-dir ${OUTPUT_DIR}"
    
    lighteval vllm $MODEL_ARGS "lighteval|${TASK}|0|0" \
        --use-chat-template \
        --output-dir "${OUTPUT_DIR}"
    
    if [ $? -eq 0 ]; then
        echo -e "${GREEN}Evaluation completed for ${CHECKPOINT_NAME}${NC}"
    else
        echo -e "${RED}Evaluation failed for ${CHECKPOINT_NAME}${NC}"
    fi
    
    echo ""
done

echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All evaluations completed!${NC}"
echo -e "${GREEN}Results saved in: ${BASE_DIR}/evals/${MODEL_NAME}/${NC}"
echo -e "${GREEN}========================================${NC}"

# Optional: Create a summary of all results
SUMMARY_FILE="${BASE_DIR}/evals/${MODEL_NAME}/evaluation_summary.txt"
echo "Evaluation Summary for ${MODEL_NAME}" > "${SUMMARY_FILE}"
echo "Generated on: $(date)" >> "${SUMMARY_FILE}"
echo "" >> "${SUMMARY_FILE}"

for CHECKPOINT_DIR in $CHECKPOINTS; do
    CHECKPOINT_NAME=$(basename "$CHECKPOINT_DIR")
    RESULT_DIR="${BASE_DIR}/evals/${MODEL_NAME}/${CHECKPOINT_NAME}/${TASK}"
    
    echo "Checkpoint: ${CHECKPOINT_NAME}" >> "${SUMMARY_FILE}"
    
    # Look for result files (adjust based on lighteval output format)
    if [ -d "${RESULT_DIR}" ]; then
        # Find and append key metrics (adjust based on actual output files)
        find "${RESULT_DIR}" -name "*.json" -o -name "*.txt" | while read -r result_file; do
            echo "  - $(basename "$result_file")" >> "${SUMMARY_FILE}"
        done
    else
        echo "  - No results found" >> "${SUMMARY_FILE}"
    fi
    echo "" >> "${SUMMARY_FILE}"
done

echo -e "${GREEN}Summary saved to: ${SUMMARY_FILE}${NC}"
