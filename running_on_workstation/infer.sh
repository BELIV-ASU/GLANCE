#!/usr/bin/env bash
# =============================================================
#  infer.sh — Run tinyvlm C++ inference (Qwen2.5-VL-7B Q4_K_M)
#
#  Usage:
#    ./infer.sh                              # interactive mode
#    ./infer.sh --batch                      # batch all images/
#    ./infer.sh --batch --low-vram           # batch, <5GB VRAM
#    ./infer.sh --image images/01.png "prompt"  # single image
#    ./infer.sh --low-vram                   # interactive, <5GB VRAM
# =============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "${SCRIPT_DIR}")"
BUILD_DIR="${SCRIPT_DIR}/build"
BINARY="${BUILD_DIR}/tinyvlm"

MODEL="${PROJECT_ROOT}/models/gguf/qwen2.5-vl-7b-Q4_K_M.gguf"
MMPROJ="${PROJECT_ROOT}/models/gguf/qwen2.5-vl-7b-mmproj-f16.gguf"

# Parse --low-vram from any position
LOW_VRAM=""
ARGS=()
for arg in "$@"; do
    if [[ "$arg" == "--low-vram" ]]; then
        LOW_VRAM="--low-vram"
    else
        ARGS+=("$arg")
    fi
done
set -- "${ARGS[@]+"${ARGS[@]}"}"

# ── Build if needed ──
if [[ ! -x "${BINARY}" ]]; then
    echo "[*] Building tinyvlm..."
    cmake -B "${BUILD_DIR}" -S "${SCRIPT_DIR}" \
        -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release 2>&1 | tail -3
    cmake --build "${BUILD_DIR}" --target tinyvlm -j"$(nproc)" 2>&1 | tail -5
fi

echo "============================================================"
echo "  tinyvlm — Qwen2.5-VL-7B Q4_K_M + CUDA"
echo "  GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
[[ -n "${LOW_VRAM}" ]] && echo "  VRAM mode: low (<5 GB)"
echo "============================================================"

if [[ "${1:-}" == "--batch" ]]; then
    echo "  Mode: Batch (all images in ${PROJECT_ROOT}/images/)"
    echo "============================================================"
    exec "${BINARY}" \
        --model "${MODEL}" \
        --mmproj "${MMPROJ}" \
        --batch "${PROJECT_ROOT}/images" \
        --json "${PROJECT_ROOT}/outputs/batch_results.json" \
        --max-tokens 512 \
        --temp 0.7 \
        ${LOW_VRAM}
elif [[ "${1:-}" == "--image" ]]; then
    shift
    IMAGE_PATH="${1:?Missing image path}"
    shift
    PROMPT="${*:-Describe this traffic scene.}"
    echo "  Mode: Single image — ${IMAGE_PATH}"
    echo "============================================================"
    exec "${BINARY}" \
        --model "${MODEL}" \
        --mmproj "${MMPROJ}" \
        --image "${IMAGE_PATH}" \
        --max-tokens 512 \
        --temp 0.7 \
        ${LOW_VRAM} \
        "${PROMPT}"
else
    echo "  Mode: Interactive"
    echo "============================================================"
    exec "${BINARY}" \
        --model "${MODEL}" \
        --mmproj "${MMPROJ}" \
        --interactive \
        --max-tokens 512 \
        --temp 0.7 \
        ${LOW_VRAM}
fi
