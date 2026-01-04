#!/bin/bash

# ============================================
# Multi-GPU Training Script for Wordle
# Uses DeepSpeed for distributed training
# ============================================

set -e

echo "=========================================="
echo "Wordle Training - Multi-GPU Mode"
echo "=========================================="

# Check CUDA availability
if ! command -v nvidia-smi &> /dev/null; then
    echo "Error: nvidia-smi not found. Is CUDA installed?"
    exit 1
fi

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)

if [ $NUM_GPUS -eq 0 ]; then
    echo "Error: No GPUs detected!"
    exit 1
fi

echo ""
echo "Detected $NUM_GPUS GPU(s):"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv,noheader
echo ""

# Ask user which GPUs to use
read -p "Use all GPUs? (y/n, default=y): " use_all_gpus
use_all_gpus=${use_all_gpus:-y}

if [[ $use_all_gpus == "n" || $use_all_gpus == "N" ]]; then
    read -p "Enter GPU IDs (comma-separated, e.g., 0,1,2): " gpu_ids
    export CUDA_VISIBLE_DEVICES=$gpu_ids
    NUM_GPUS=$(echo $gpu_ids | tr ',' '\n' | wc -l)
    echo "Using GPUs: $gpu_ids"
else
    echo "Using all $NUM_GPUS GPUs"
fi

# Check DeepSpeed installation
python3 << 'PYCHECK'
import sys
try:
    import deepspeed
    print(f"DeepSpeed version: {deepspeed.__version__}")
except ImportError:
    print("Error: DeepSpeed not installed")
    sys.exit(1)

import torch
if not torch.cuda.is_available():
    print("Error: CUDA not available in PyTorch")
    sys.exit(1)
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")
PYCHECK

if [ $? -ne 0 ]; then
    exit 1
fi

# Set environment variables
export OMP_NUM_THREADS=8
export TOKENIZERS_PARALLELISM=false
export NCCL_DEBUG=INFO  # Enable NCCL debug for troubleshooting

# Training configuration
SCRIPT_PATH="/workspace/scripts/train_wordle_offline.py"
DS_CONFIG="/workspace/configs/ds_config.json"
OUTPUT_DIR="/workspace/outputs"
LOG_DIR="/workspace/logs"

# Create directories
mkdir -p $OUTPUT_DIR
mkdir -p $LOG_DIR

# Verify DeepSpeed config exists
if [ ! -f "$DS_CONFIG" ]; then
    echo "Error: DeepSpeed config not found at $DS_CONFIG"
    exit 1
fi

echo ""
echo "DeepSpeed Configuration:"
cat $DS_CONFIG
echo ""

# Log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/train_multi_gpu_${TIMESTAMP}.log"

echo "=========================================="
echo "Training Configuration:"
echo "  - GPUs: $NUM_GPUS"
echo "  - Script: $SCRIPT_PATH"
echo "  - DeepSpeed Config: $DS_CONFIG"
echo "  - Output: $OUTPUT_DIR"
echo "  - Log: $LOG_FILE"
echo "=========================================="
echo ""

read -p "Start training? (y/n, default=y): " start_training
start_training=${start_training:-y}

if [[ $start_training != "y" && $start_training != "Y" ]]; then
    echo "Training cancelled."
    exit 0
fi

echo ""
echo "Starting training..."
echo ""

# Run DeepSpeed training
deepspeed --num_gpus=$NUM_GPUS \
    --master_port=29500 \
    $SCRIPT_PATH \
    --deepspeed $DS_CONFIG \
    2>&1 | tee $LOG_FILE

# Check training status
TRAIN_EXIT_CODE=${PIPESTATUS[0]}

if [ $TRAIN_EXIT_CODE -eq 0 ]; then
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
    echo "  2. TensorBoard: tensorboard --logdir $OUTPUT_DIR --bind_all"
    echo "  3. Inference: python /workspace/scripts/inference_wordle.py"
    echo ""
    echo "Performance Summary:"
    echo "  - GPUs used: $NUM_GPUS"
    grep -i "samples/s" $LOG_FILE | tail -5 || echo "  (Check log for details)"
    echo ""
else
    echo ""
    echo "=========================================="
    echo "Training failed with exit code: $TRAIN_EXIT_CODE"
    echo "=========================================="
    echo ""
    echo "Check the log file for details:"
    echo "  $LOG_FILE"
    echo ""
    echo "Common issues:"
    echo "  1. Out of memory -> Reduce batch size in train_wordle_offline.py"
    echo "  2. NCCL errors -> Check GPU connectivity"
    echo "  3. DeepSpeed errors -> Verify ds_config.json"
    echo ""
    exit 1
fi

# Optional: Show GPU memory usage
echo "Final GPU Memory Usage:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv,noheader
echo ""
