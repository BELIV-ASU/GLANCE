#!/usr/bin/env bash
# =============================================================
#  infer_checkpoint400.sh  –  Inference with checkpoint-400
#  on val data from driving_road_data
# =============================================================
set -euo pipefail

# ---- Environment setup (same as train_teacher.sh) ----
module load ffmpeg-6.0-gcc-12.1.0
module load cmake/3.30.2
module load mamba/latest
source activate new_beginning
export PATH="${HOME}/.local/bin:${PATH}"
export HF_HOME="/scratch/jnolas77/.hf_cache"
export HF_TOKEN=""
export CUDA_VISIBLE_DEVICES=0
export TORCH_BLAS_PREFER_CUBLASLT=1

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "============================================================"
echo "  Inference: checkpoint-400 on val data"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "============================================================"

python3 "${SCRIPT_DIR}/src/infer_checkpoint400.py" \
    --base_model    "Qwen/Qwen2.5-VL-32B-Instruct" \
    --adapter_path  "${SCRIPT_DIR}/checkpoints/checkpoint-400" \
    --val_dir       "/scratch/jnolas77/driving_road_data/val" \
    --img_root      "/scratch/jnolas77/driving_road_data" \
    --handbook_dir  "/scratch/jnolas77/driving_handbook_data" \
    --data_type     both \
    --num_samples   5 \
    --num_handbook  5 \
    --max_new_tokens 512 \
    --temperature   0.7 \
    --seed          42 \
    2>&1 | tee "${SCRIPT_DIR}/logs/inference_ckpt400.log"

echo ""
echo "  Done! Results saved to ${SCRIPT_DIR}/results/inference_results_ckpt400.json"
