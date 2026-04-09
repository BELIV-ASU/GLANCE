#!/usr/bin/env bash
# run_dual_distill.sh
# ──────────────────────────────────────────────────────────────────────────
# Launch dual-teacher distillation:
#   Vision teacher : Qwen2.5-VL-32B + Waymo LoRA adapter
#   Language teacher: QwQ-32B + reasoning QLoRA adapter
#   Student        : Qwen2.5-VL-7B-Instruct with new LoRA
#
# Both teachers are loaded 4-bit (QLoRA) and frozen.
# The student is trained in bf16 with LoRA (r=64).
#
# Hardware requirement: 2× A100-80GB (or equivalent)
# ──────────────────────────────────────────────────────────────────────────
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ── Environment setup (matches set.bash) ─────────────────────────────────
module load ffmpeg-6.0-gcc-12.1.0
# Do NOT load cuda-12.9.0 — PyTorch 2.10 ships its own CUDA 12.8 runtime.
# Loading cuda-12.9 causes CUBLAS_STATUS_INVALID_VALUE in bf16 matmuls.
module load cuda-12.8.1-gcc-12.1.0
module load cmake/3.30.2
module load glibc-2.38-gcc-12.1.0
module load mamba/latest
source activate new_beginning

# Pull in all cache/path exports from the central set.bash
source /scratch/rbaskar5/set.bash >/dev/null 2>&1 || true

cd "${ROOT_DIR}"

# ── Teacher checkpoints ─────────────────────────────────────────────────
VISION_BASE="${VISION_BASE:-Qwen/Qwen2.5-VL-32B-Instruct}"
VISION_LORA="${VISION_LORA:-${ROOT_DIR}/checkpoints/waymo/qwen2.5-vl-32b-trajectory/final_lora}"

LANG_BASE="${LANG_BASE:-Qwen/QwQ-32B}"
LANG_LORA="${LANG_LORA:-${ROOT_DIR}/reasoning_distillation/checkpoints/teacher_qwq32b_qlora/teacher_lora_adapter}"

# ── Student ─────────────────────────────────────────────────────────────
STUDENT="${STUDENT:-Qwen/Qwen2.5-VL-7B-Instruct}"

# ── Data ────────────────────────────────────────────────────────────────
WAYMO_JSONL="${WAYMO_JSONL:-${ROOT_DIR}/data/waymo_trajectory/train.jsonl}"
REASON_JSONL="${REASON_JSONL:-${ROOT_DIR}/reasoning_distillation/data/distill_train_qwen25vl7b.jsonl}"
IMAGE_ROOT="${IMAGE_ROOT:-/scratch/rbaskar5/Dataset/waymo_front}"

# ── Output ───────────────────────────────────────────────────────────────
OUT_DIR="${OUT_DIR:-${ROOT_DIR}/checkpoints/dual_distilled_7b_v2}"

# ── Hyperparams ──────────────────────────────────────────────────────────
EPOCHS="${EPOCHS:-5}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-8}"
MAX_STEPS="${MAX_STEPS:--1}"
LR="${LR:-8e-6}"
ALPHA_VISION="${ALPHA_VISION:-0.4}"
ALPHA_LANG="${ALPHA_LANG:-0.2}"
ALPHA_CE="${ALPHA_CE:-0.4}"
TEMPERATURE="${TEMPERATURE:-3.0}"
SAVE_EVERY="${SAVE_EVERY:-100}"

echo "========================================================"
echo " Dual-teacher distillation"
echo "  Vision teacher  : ${VISION_BASE}"
echo "  Vision LoRA     : ${VISION_LORA}"
echo "  Language teacher: ${LANG_BASE}"
echo "  Language LoRA   : ${LANG_LORA}"
echo "  Student         : ${STUDENT}"
echo "  Waymo data      : ${WAYMO_JSONL}"
echo "  Reasoning data  : ${REASON_JSONL}"
echo "  Output          : ${OUT_DIR}"
echo "  α_vision=${ALPHA_VISION}  α_lang=${ALPHA_LANG}  α_ce=${ALPHA_CE}  T=${TEMPERATURE}"
echo "========================================================"

mkdir -p "${OUT_DIR}"

# Run on a single process — device_map="auto" spreads both teachers and the
# student across available GPUs.  If you need DDP, switch to accelerate launch
# and add appropriate DistributedSampler handling.
python src/distill_dual_teachers.py \
    --vision_base  "${VISION_BASE}" \
    --vision_lora  "${VISION_LORA}" \
    --lang_base    "${LANG_BASE}" \
    --lang_lora    "${LANG_LORA}" \
    --student      "${STUDENT}" \
    --waymo_jsonl  "${WAYMO_JSONL}" \
    --reason_jsonl "${REASON_JSONL}" \
    --image_root   "${IMAGE_ROOT}" \
    --out_dir      "${OUT_DIR}" \
    --epochs       "${EPOCHS}" \
    --batch_size   "${BATCH_SIZE}" \
    --grad_accum   "${GRAD_ACCUM}" \
    --max_steps    "${MAX_STEPS}" \
    --lr           "${LR}" \
    --alpha_vision "${ALPHA_VISION}" \
    --alpha_lang   "${ALPHA_LANG}" \
    --alpha_ce     "${ALPHA_CE}" \
    --temperature  "${TEMPERATURE}" \
    --save_every   "${SAVE_EVERY}" \
    --log_every    10 \
    2>&1 | tee "${OUT_DIR}/distill.log"

echo "Distillation complete. Adapter saved to: ${OUT_DIR}/final"
