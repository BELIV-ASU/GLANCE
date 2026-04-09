#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
CONFIG="${ROOT_DIR}/configs/waymo_finetune_qwen32b_trajectory.yaml"
ACCEL_CFG="${ROOT_DIR}/configs/accelerate_2gpu.yaml"

IMAGE_ROOT="${IMAGE_ROOT:-/scratch/rbaskar5/Dataset/waymo_front}"
TRAJ_DIR="${TRAJ_DIR:-${ROOT_DIR}/data/waymo_trajectory}"
RAW_JSONL="${RAW_JSONL:-${TRAJ_DIR}/trajectory_full.jsonl}"
TRAIN_JSONL="${TRAIN_JSONL:-${TRAJ_DIR}/train.jsonl}"
VAL_JSONL="${VAL_JSONL:-${TRAJ_DIR}/val.jsonl}"

# Keep this modest by default so fine-tuning can start quickly.
TRAJ_MAX_SAMPLES="${TRAJ_MAX_SAMPLES:-2000}"
VAL_RATIO="${VAL_RATIO:-0.05}"
SEED="${SEED:-42}"
NUM_EPOCHS="${NUM_EPOCHS:-5}"
FORCE_REGENERATE="${FORCE_REGENERATE:-0}"
TRAJ_PARALLEL_WORKERS="${TRAJ_PARALLEL_WORKERS:-2}"
TRAJ_BATCH_SIZE="${TRAJ_BATCH_SIZE:-2}"
TRAJ_MAX_NEW_TOKENS="${TRAJ_MAX_NEW_TOKENS:-128}"
TRAJ_RESUME="${TRAJ_RESUME:-1}"
TRAJ_MODEL_NAME="${TRAJ_MODEL_NAME:-qwen2.5-vl-32b}"
TRAJ_IMAGE_MAX_SIDE="${TRAJ_IMAGE_MAX_SIDE:-0}"
TRAJ_SPEED_PROFILE="${TRAJ_SPEED_PROFILE:-quality}"

if [[ "${TRAJ_SPEED_PROFILE}" == "fast" ]]; then
  # Fast profile targets throughput for large-scale annotation generation.
  if [[ "${TRAJ_MODEL_NAME}" == "qwen2.5-vl-32b" ]]; then
    TRAJ_MODEL_NAME="qwen2.5-vl-7b"
  fi
  if [[ "${TRAJ_MAX_NEW_TOKENS}" == "128" ]]; then
    TRAJ_MAX_NEW_TOKENS="64"
  fi
  if [[ "${TRAJ_BATCH_SIZE}" == "2" ]]; then
    TRAJ_BATCH_SIZE="4"
  fi
  if [[ "${TRAJ_IMAGE_MAX_SIDE}" == "0" ]]; then
    TRAJ_IMAGE_MAX_SIDE="896"
  fi
fi

mkdir -p "${TRAJ_DIR}"

# Preserve user-requested GPU visibility; env setup may overwrite this.
ORIG_CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

source /scratch/rbaskar5/set.bash >/dev/null 2>&1
source activate new_beginning >/dev/null 2>&1

if [[ -n "${ORIG_CUDA_VISIBLE_DEVICES}" ]]; then
  export CUDA_VISIBLE_DEVICES="${ORIG_CUDA_VISIBLE_DEVICES}"
fi

cd "${ROOT_DIR}"

echo "[1/4] Verifying Waymo dataset cache..."
python src/verify_waymo_dataset.py

echo "[2/4] Building trajectory annotations with Qwen2.5-VL-32B..."
echo "  Trajectory profile=${TRAJ_SPEED_PROFILE} model=${TRAJ_MODEL_NAME} batch=${TRAJ_BATCH_SIZE} max_new_tokens=${TRAJ_MAX_NEW_TOKENS} image_max_side=${TRAJ_IMAGE_MAX_SIDE}"
if [[ "${FORCE_REGENERATE}" == "1" && -f "${RAW_JSONL}" ]]; then
    echo "  FORCE_REGENERATE=1, removing existing ${RAW_JSONL}"
    rm -f "${RAW_JSONL}"
fi
if [[ "${FORCE_REGENERATE}" == "1" ]]; then
  rm -f "${RAW_JSONL}.part"*
fi

if [[ ! -f "${RAW_JSONL}" ]]; then
  resume_flag=""
  if [[ "${TRAJ_RESUME}" == "1" ]]; then
    resume_flag="--resume"
  fi

  if [[ "${TRAJ_PARALLEL_WORKERS}" -gt 1 ]]; then
    echo "  Running sharded trajectory generation with ${TRAJ_PARALLEL_WORKERS} workers..."
    gpu_ids=()
    if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      IFS=',' read -r -a gpu_ids <<< "${CUDA_VISIBLE_DEVICES}"
    elif command -v nvidia-smi >/dev/null 2>&1; then
      while IFS= read -r idx; do
        idx="${idx//[[:space:]]/}"
        [[ -n "${idx}" ]] && gpu_ids+=("${idx}")
      done < <(nvidia-smi --query-gpu=index --format=csv,noheader)
    fi
    if [[ "${#gpu_ids[@]}" -eq 0 ]]; then
      gpu_ids=("0")
    fi
    echo "  Using GPU IDs: ${gpu_ids[*]}"

    pids=()
    parts=()
    for ((shard=0; shard<TRAJ_PARALLEL_WORKERS; shard++)); do
      gpu_id="${gpu_ids[$((shard % ${#gpu_ids[@]}))]}"
      part="${RAW_JSONL}.part${shard}"
      parts+=("${part}")
      CUDA_VISIBLE_DEVICES="${gpu_id}" python src/trajectory_analysis.py \
        --config "${CONFIG}" \
        --model_name "${TRAJ_MODEL_NAME}" \
        --image_root "${IMAGE_ROOT}" \
        --output "${part}" \
        --max_samples "${TRAJ_MAX_SAMPLES}" \
        --num_shards "${TRAJ_PARALLEL_WORKERS}" \
        --shard_id "${shard}" \
        --batch_size "${TRAJ_BATCH_SIZE}" \
        --image_max_side "${TRAJ_IMAGE_MAX_SIDE}" \
        --max_new_tokens "${TRAJ_MAX_NEW_TOKENS}" \
        ${resume_flag} \
        --temperature 0.0 &
      pids+=("$!")
      echo "    shard=${shard} gpu=${gpu_id} pid=${pids[-1]} -> ${part}"
    done

    for pid in "${pids[@]}"; do
      wait "${pid}"
    done

    cat "${parts[@]}" > "${RAW_JSONL}"
    rm -f "${parts[@]}"
    echo "  Merged shard outputs into ${RAW_JSONL}"
  else
    python src/trajectory_analysis.py \
      --config "${CONFIG}" \
      --model_name "${TRAJ_MODEL_NAME}" \
      --image_root "${IMAGE_ROOT}" \
      --output "${RAW_JSONL}" \
      --max_samples "${TRAJ_MAX_SAMPLES}" \
      --batch_size "${TRAJ_BATCH_SIZE}" \
      --image_max_side "${TRAJ_IMAGE_MAX_SIDE}" \
      --max_new_tokens "${TRAJ_MAX_NEW_TOKENS}" \
      ${resume_flag} \
      --temperature 0.0
  fi
else
  echo "  Found existing ${RAW_JSONL}, skipping generation."
fi

echo "[3/4] Splitting train/val JSONL..."
RAW_JSONL="${RAW_JSONL}" TRAIN_JSONL="${TRAIN_JSONL}" VAL_JSONL="${VAL_JSONL}" VAL_RATIO="${VAL_RATIO}" SEED="${SEED}" python - <<'PY'
import json
import os
import random
from pathlib import Path

raw = Path(os.environ["RAW_JSONL"])
train = Path(os.environ["TRAIN_JSONL"])
val = Path(os.environ["VAL_JSONL"])
val_ratio = float(os.environ["VAL_RATIO"])
seed = int(os.environ["SEED"])

rows = []
with raw.open("r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            rows.append(json.loads(line))

if not rows:
    raise SystemExit("No rows found in trajectory JSONL")

rnd = random.Random(seed)
rnd.shuffle(rows)

val_n = max(1, int(len(rows) * val_ratio))
val_rows = rows[:val_n]
train_rows = rows[val_n:]
if not train_rows:
    train_rows, val_rows = rows, rows[:1]

train.parent.mkdir(parents=True, exist_ok=True)
with train.open("w", encoding="utf-8") as f:
    for r in train_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

with val.open("w", encoding="utf-8") as f:
    for r in val_rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")

print(f"Wrote train={len(train_rows):,}  val={len(val_rows):,}")
PY

echo "[4/4] Starting fine-tuning with accelerate..."
accelerate launch --config_file "${ACCEL_CFG}" \
  src/train_waymo.py \
  --config "${CONFIG}" \
  --model qwen2.5-vl-32b \
    --num_epochs "${NUM_EPOCHS}" \
  --output_dir "${ROOT_DIR}/checkpoints/waymo/qwen2.5-vl-32b-trajectory"
