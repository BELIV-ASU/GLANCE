#!/usr/bin/env bash
# =============================================================
#  train_teacher.sh  –  Launch SafetyVLM Teacher Training
#  Fine-tunes Qwen3-VL with QLoRA + CoT + Grounding
#  on driving_handbook_data (2× A100-80GB)
# =============================================================
set -euo pipefail

# ---- Environment setup ----
module load ffmpeg-6.0-gcc-12.1.0
# NOTE: Do NOT load cuda-12.9.0 – PyTorch 2.10 ships its own CUDA 12.8 runtime.
# Loading cuda-12.9 causes CUBLAS_STATUS_INVALID_VALUE in all bf16 matmuls.
# module load cuda-12.9.0-gcc-12.1.0
module load cmake/3.30.2
module load mamba/latest
source activate new_beginning
export PATH="${HOME}/.local/bin:${PATH}"
export HF_HOME="/scratch/rbaskar5/.hf_cache"
export HF_TOKEN="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"
# Restrict to single GPU – QLoRA 4-bit model (~10GB) fits on one A100-80GB.
# Prevents Trainer from wrapping in DataParallel which causes CUBLAS errors.
export CUDA_VISIBLE_DEVICES=0
# Force cuBLASLt backend – more tolerant of CUDA version mismatches
export TORCH_BLAS_PREFER_CUBLASLT=1


# ---- Paths ----
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DATA_ROOT="/scratch/rbaskar5/Dataset/DriveLM"
DATA_JSON="${DATA_ROOT}/v1_1_train_nus.json"
DATA_DIR="${SCRIPT_DIR}/data_drivelm"
OUTPUT_DIR="${SCRIPT_DIR}/checkpoints"
mkdir -p "${OUTPUT_DIR}"
MODEL_NAME="Qwen/Qwen2.5-VL-32B-Instruct"

# ---- Step 1: Build the instruction-tuning dataset (skip if already built) ----
if [ ! -f "${DATA_DIR}/train.jsonl" ]; then
    echo "============================================================"
    echo "  Step 1: Building dataset from DriveLM..."
    echo "============================================================"
    python3 "${SCRIPT_DIR}/src/data.py" \
        --data_json  "${DATA_JSON}" \
        --data_root  "${DATA_ROOT}" \
        --output_dir "${DATA_DIR}" \
        --val_ratio  0.05 \
        --seed       42 \
        --with_depth
else
    echo "============================================================"
    echo "  Step 1: Dataset already exists, skipping build."
    echo "============================================================"
fi

echo ""
echo "  Dataset files:"
wc -l "${DATA_DIR}/train.jsonl" "${DATA_DIR}/val.jsonl"
echo ""

# ---- Step 2: Train the 32B teacher with QLoRA ----
echo "============================================================"
echo "  Step 2: Fine-tuning ${MODEL_NAME} with QLoRA"
echo "  GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ', ')"
echo "============================================================"

# FP8 model (~30GB) fits across 2× A100-80GB via HF device_map="auto".
# We use a single process so device_map handles GPU sharding internally
# (accelerate multi-GPU conflicts with device_map="auto").
LAUNCH_CMD="python3"

${LAUNCH_CMD} "${SCRIPT_DIR}/src/train_teacher.py" \
    --model_name            "${MODEL_NAME}" \
    --load_in_4bit \
    --attn_implementation   eager \
    --data_dir              "${DATA_DIR}" \
    --output_dir            "${OUTPUT_DIR}" \
    --num_train_epochs      3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size  1 \
    --gradient_accumulation_steps 8 \
    --learning_rate         2e-4 \
    --lr_scheduler_type     cosine \
    --warmup_ratio          0.05 \
    --weight_decay          0.01 \
    --max_grad_norm         1.0 \
    --max_seq_length        2048 \
    --lora_r                64 \
    --lora_alpha            128 \
    --lora_dropout          0.05 \
    --logging_steps         5 \
    --save_steps            200 \
    --eval_steps            200 \
    --save_total_limit      3 \
    --seed                  42 \
    --gradient_checkpointing \
    --bf16 \
    --tf32 \
    2>&1 | tee "${OUTPUT_DIR}/train.log"

echo ""
echo "============================================================"
echo "  Training complete!"
echo "  Checkpoints: ${OUTPUT_DIR}"
echo "  LoRA adapter: ${OUTPUT_DIR}/final_lora"
echo "============================================================"