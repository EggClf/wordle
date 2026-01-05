#!/bin/bash

# ============================================
# Single GPU Training Script for Wordle
# ============================================

set -euo pipefail  # Exit on error, undefined vars, pipe failures

echo "=========================================="
echo "Wordle Training - Single GPU Mode"
echo "=========================================="

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Is CUDA installed?"
    exit 1
fi

# Display GPU info
echo ""
echo "GPU Information:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

# Check if CUDA is available in PyTorch
python3 << 'PYCHECK'
import torch
if not torch.cuda.is_available():
    print("Error: CUDA not available in PyTorch")
    exit(1)
print(f"PyTorch CUDA version: {torch.version.cuda}")
print(f"Available GPUs: {torch.cuda.device_count()}")
print(f"Current GPU: {torch.cuda.get_device_name(0)}")
PYCHECK

# Set environment variables
export CUDA_VISIBLE_DEVICES=0  # Use first GPU
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false

# Training configuration
SCRIPT_PATH="/workspace/scripts/train_wordle_offline.py"
OUTPUT_DIR="/workspace/outputs"
LOG_DIR="/workspace/logs"

# Verify required files exist
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Training script not found at $SCRIPT_PATH"
    exit 1
fi

if [ ! -d "/workspace/data/wordle-grpo" ]; then
    echo "Error: Dataset not found at /workspace/data/wordle-grpo"
    exit 1
fi

if [ ! -f "/workspace/data/five_letter_words.csv" ]; then
    echo "Error: Word list not found at /workspace/data/five_letter_words.csv"
    exit 1
fi

if [ ! -d "/workspace/models/Qwen2.5-0.5B" ]; then
    echo "Error: Base model not found at /workspace/models/Qwen2.5-0.5B"
    exit 1
fi

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# Log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/train_single_gpu_${TIMESTAMP}.log"

echo "All required files validated successfully"
echo "Starting training..."
echo "Log file: $LOG_FILE"
echo ""

# Run training with logging
python3 $SCRIPT_PATH 2>&1 | tee $LOG_FILE

# Check training status
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "Training completed successfully!"
    echo "=========================================="
    echo ""
    echo "Outputs saved to:"
    echo "  - SFT model: $OUTPUT_DIR/wordle-sft"
    echo "  - GRPO model: $OUTPUT_DIR/wordle-grpo"
    echo "  - Training log: $LOG_FILE"
    echo ""
    echo "Next steps:"
    echo "  1. View logs: cat $LOG_FILE"
    echo "  2. TensorBoard: tensorboard --logdir $OUTPUT_DIR"
    echo "  3. Inference: python /workspace/scripts/inference_wordle.py"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "Training failed! Check log file:"
    echo "  $LOG_FILE"
    echo "=========================================="
    exit 1
fi
