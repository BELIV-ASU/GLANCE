#!/usr/bin/env bash
# =============================================================
#  train_teacher.sh  –  Launch SafetyVLM Teacher Training
#  Fine-tunes Qwen3-VL with QLoRA + CoT + Grounding
#  on driving_handbook_data (2× A100-80GB)
# =============================================================
set -euo pipefail

# ---- Environment setup ----
module load ffmpeg-6.0-gcc-12.1.0
module load cuda-12.9.0-gcc-12.1.0
module load cmake/3.30.2


# ---- Paths ----
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DATA_ROOT="/scratch/jnolas77/SafetyVLM/driving_handbook_data"
DATA_JSON="${DATA_ROOT}/data.json"
DATA_DIR="${SCRIPT_DIR}/data"
OUTPUT_DIR="${SCRIPT_DIR}/checkpoints"
MODEL_NAME="Qwen/Qwen3-VL-30B-A3B-Instruct"

# ---- Step 1: Build the instruction-tuning dataset ----
echo "============================================================"
echo "  Step 1: Building dataset from driving handbook..."
echo "============================================================"
python3 "${SCRIPT_DIR}/safety_data.py" \
    --data_json  "${DATA_JSON}" \
    --data_root  "${DATA_ROOT}" \
    --output_dir "${DATA_DIR}" \
    --val_ratio  0.05 \
    --seed       42

echo ""
echo "  Dataset files:"
wc -l "${DATA_DIR}/train.jsonl" "${DATA_DIR}/val.jsonl"
echo ""

# ---- Step 2: Train the 32B teacher with QLoRA ----
echo "============================================================"
echo "  Step 2: Fine-tuning ${MODEL_NAME} with QLoRA"
echo "  GPUs: $(nvidia-smi --query-gpu=name --format=csv,noheader | paste -sd ', ')"
echo "============================================================"

# Use accelerate for multi-GPU (2× A100-80GB)
# If accelerate config doesn't exist, run with defaults
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)

if [ "${NUM_GPUS}" -gt 1 ]; then
    LAUNCH_CMD="accelerate launch --num_processes ${NUM_GPUS} --mixed_precision bf16"
else
    LAUNCH_CMD="python3"
fi

${LAUNCH_CMD} "${SCRIPT_DIR}/train_teacher.py" \
    --model_name            "${MODEL_NAME}" \
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
    --max_seq_length        4096 \
    --lora_r                64 \
    --lora_alpha            128 \
    --lora_dropout          0.05 \
    --logging_steps         5 \
    --save_steps            200 \
    --eval_steps            200 \
    --save_total_limit      3 \
    --seed                  42 \
    --load_in_4bit \
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
