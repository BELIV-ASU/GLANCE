#!/usr/bin/env bash
# run_distill.sh  –  2×A100-80GB distillation launch
# Usage: bash /scratch/rbaskar5/GLANCE/scripts/run_distill.sh

set -euo pipefail

# ---- Environment ----
source /scratch/rbaskar5/set.bash

# Reduce allocator fragmentation – avoids OOM spikes during KL log_softmax
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Hard-coded binary paths for this HPC setup:
#   mamba env new_beginning uses system Python at /packages/apps/mamba/2.0.8
#   packages are installed via pip --user into /home/rbaskar5/.local
#   accelerate binary lives at /home/rbaskar5/.local/bin/accelerate
PYTHON_BIN="/packages/apps/mamba/2.0.8/bin/python"

# Verify we have the right python
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "ERROR: python not found at ${PYTHON_BIN}" >&2
    exit 1
fi

# Verify accelerate is importable (we use 'python -m accelerate', not the binary)
if ! "${PYTHON_BIN}" -c "import accelerate" 2>/dev/null; then
    echo "ERROR: accelerate not importable. Run: pip install accelerate" >&2
    exit 1
fi

# Confirm the active conda env is new_beginning
if [[ "${CONDA_DEFAULT_ENV:-}" != "new_beginning" ]]; then
    echo "ERROR: expected conda env 'new_beginning', got '${CONDA_DEFAULT_ENV:-none}'" >&2
    echo "       Run: source activate new_beginning" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="${SCRIPT_DIR}/distilled_student"
mkdir -p "${OUT_DIR}"

echo "============================================================"
echo " Distillation: 2×A100-80GB  |  $(date)"
echo " Python:    ${PYTHON_BIN}"
echo " Conda env: ${CONDA_DEFAULT_ENV}"
echo " Output →  ${OUT_DIR}"
echo "============================================================"
echo ""

# Run the accelerate binary as a Python script using the explicit interpreter.
# This bypasses the binary's shebang (which would pick up the wrong Python)
# while still respecting PYTHONPATH set by set.bash.
"${PYTHON_BIN}" /home/rbaskar5/.local/bin/accelerate launch \
    --config_file "${SCRIPT_DIR}/configs/accelerate_2gpu.yaml" \
    "${SCRIPT_DIR}/src/distill_4b.py" \
        --teacher       Qwen/Qwen2.5-VL-32B-Instruct \
        --teacher_adapter "${SCRIPT_DIR}/checkpoints/checkpoint-800" \
        --student       Qwen/Qwen2.5-VL-7B-Instruct \
        --out_dir       "${OUT_DIR}" \
        --data_dir      "${SCRIPT_DIR}/data_drivelm" \
        --teacher_4bit \
        --student_4bit \
        --use_qlora_student \
        --epochs        1 \
        --batch_size    1 \
        --grad_accum    8 \
        --lr            2e-5 \
        --max_length    2048 \
        --temperature   2.0 \
        --alpha_ce      0.2 \
        --beta_kl       0.8 \
        --lora_r        16 \
        --lora_alpha    32 \
        --log_every     10 \
        --save_steps    200 \
        --save_total_limit 3 \
        --resume_from   "${SCRIPT_DIR}/distilled_student/checkpoints/step-200" \
        --infer_after    50 \
        --infer_samples 40 \
        --infer_max_new_tokens 256

echo ""
echo "============================================================"
echo " Done: $(date)"
echo " Results → ${OUT_DIR}/results/distilled_inference_results.json"
echo "============================================================"
