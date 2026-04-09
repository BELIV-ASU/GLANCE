#!/usr/bin/env bash
# =============================================================
#  train_waymo.sh  –  Fine-tune 7B VLMs on Waymo images
#
#  Supports Qwen2.5-VL-7B, Cosmos R1-7B, MiMo-VL-7B,
#  OpenVLA-7B, VILA-7B via vlm_factory.py
#
#  Hardware: 2× A100-80GB (bf16, LoRA, gradient checkpointing)

#cd /scratch/rbaskar5/GLANCE && source /scratch/rbaskar5/set.bash >/dev/null 2>&1 && source activate new_beginning >/dev/null 2>&1 && export CUDA_VISIBLE_DEVICES=0,1,2,3 TRAJ_SPEED_PROFILE=fast TRAJ_MAX_SAMPLES=0 TRAJ_PARALLEL_WORKERS=4 TRAJ_RESUME=1 FORCE_REGENERATE=0 NUM_EPOCHS=5 PYTHONPATH=/scratch/rbaskar5/GLANCE/src && bash scripts/run_waymo_qwen32b_trajectory.sh
# =============================================================
set -euo pipefail

# ── Environment setup ────────────────────────────────────────────────────
module load ffmpeg-6.0-gcc-12.1.0
# NOTE: Do NOT load cuda-12.9.0 – PyTorch 2.10 ships its own CUDA 12.8 runtime.
# Loading cuda-12.9 causes CUBLAS_STATUS_INVALID_VALUE in all bf16 matmuls.
module load cmake/3.30.2
module load mamba/latest
source activate new_beginning
export PATH="${HOME}/.local/bin:${PATH}"
export HF_HOME="/scratch/rbaskar5/.hf_cache"
export HF_TOKEN="${HF_TOKEN:-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}"
export CUDA_VISIBLE_DEVICES=0,1
# Force cuBLASLt backend – avoids CUBLAS_STATUS_INVALID_VALUE
export TORCH_BLAS_PREFER_CUBLASLT=1
# Triton / misc caches
export TRITON_CACHE_DIR="/scratch/rbaskar5/.triton_cache"
export TMPDIR="/scratch/rbaskar5/.tmp"
mkdir -p "${TRITON_CACHE_DIR}" "${TMPDIR}"

# ── Paths ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${SCRIPT_DIR}/configs/waymo_finetune_config.yaml"
OUTPUT_DIR="${SCRIPT_DIR}/checkpoints/waymo"
mkdir -p "${OUTPUT_DIR}"

# ── Model selection (override via CLI: ./train_waymo.sh mimo-vl-7b) ─────
MODEL_NAME="${1:-qwen2.5-vl-7b}"

echo "============================================================"
echo "  Waymo VLM Fine-tuning"
echo "  Model:  ${MODEL_NAME}"
echo "  Config: ${CONFIG}"
echo "  GPUs:   $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | paste -sd ', ' || echo 'N/A')"
echo "============================================================"

# ── Launch ───────────────────────────────────────────────────────────────
# 7B models in bf16 (~14GB) fit on each A100-80GB.
# Use Accelerate multi-GPU for proper DDP across 2 GPUs.
ACCELERATE_CONFIG="${SCRIPT_DIR}/configs/accelerate_2gpu.yaml"

if [ -f "${ACCELERATE_CONFIG}" ]; then
    echo "Using accelerate config: ${ACCELERATE_CONFIG}"
    LAUNCH_CMD="accelerate launch --config_file ${ACCELERATE_CONFIG}"
else
    echo "Accelerate config not found, falling back to single-GPU"
    LAUNCH_CMD="python3"
fi

${LAUNCH_CMD} "${SCRIPT_DIR}/src/train_waymo.py" \
    --config  "${CONFIG}" \
    --model   "${MODEL_NAME}" \
    --output_dir "${OUTPUT_DIR}/${MODEL_NAME}" \
    2>&1 | tee "${OUTPUT_DIR}/${MODEL_NAME}_train.log"

echo ""
echo "============================================================"
echo "  Training complete!"
echo "  Model:       ${MODEL_NAME}"
echo "  Checkpoints: ${OUTPUT_DIR}/${MODEL_NAME}"
echo "  LoRA adapter: ${OUTPUT_DIR}/${MODEL_NAME}/final_lora"
echo "============================================================"
