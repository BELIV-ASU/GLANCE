#!/bin/bash
# ================================================================
#  setup_glance_waymo_qwen.sh  –  Complete setup for GLANCE training
#
#  Prepares Waymo front camera dataset and launches Qwen VLM training
#  on 2× A100s using LoRA fine-tuning
#
#  Usage:
#    ./setup_glance_waymo_qwen.sh [prepare|train|both]
#
#  Default: setup complete pipeline (prepare → train)
# ================================================================

set -euo pipefail

# ── Colors for output ────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# ── Logging functions ────────────────────────────────────────────────
log_info()    { echo -e "${BLUE}ℹ${NC} $*"; }
log_success() { echo -e "${GREEN}✓${NC} $*"; }
log_warn()    { echo -e "${YELLOW}⚠${NC} $*"; }
log_error()   { echo -e "${RED}✗${NC} $*"; }

# ── Setup environment ────────────────────────────────────────────────
PROJECT_ROOT="/scratch/rbaskar5/GLANCE"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PREPARE_SCRIPT="${PROJECT_ROOT}/scripts/prepare_waymo_front_camera.py"
TRAIN_SCRIPT="${PROJECT_ROOT}/scripts/train_waymo.sh"

log_info "Project root: ${PROJECT_ROOT}"
log_info "Script directory: ${SCRIPT_DIR}"

# ── Check prerequisites ──────────────────────────────────────────────
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check if Waymo dataset exists
    if [ ! -d "/scratch/rbaskar5/.hf_cache/datasets--AnnaZhang--waymo_open_dataset_v_1_4_3/blobs" ]; then
        log_error "Waymo dataset not found at expected location"
        log_error "Make sure you've downloaded it: ~/waymo_loader.py"
        exit 1
    fi
    log_success "Waymo dataset found"
    
    # Check GLANCE project structure
    if [ ! -f "${PREPARE_SCRIPT}" ]; then
        log_error "Prepare script not found: ${PREPARE_SCRIPT}"
        exit 1
    fi
    log_success "Prepare script found"
    
    if [ ! -f "${TRAIN_SCRIPT}" ]; then
        log_error "Training script not found: ${TRAIN_SCRIPT}"
        exit 1
    fi
    log_success "Training script found"
    
    # Check mamba environment
    if ! command -v mamba &> /dev/null; then
        log_warn "mamba not in PATH, trying to load modules..."
        module load mamba/latest 2>/dev/null || true
    fi
    
    log_success "All prerequisites met"
}

# ── Prepare data ─────────────────────────────────────────────────────
prepare_data() {
    log_info ""
    log_info "════════════════════════════════════════════════════════════"
    log_info "PHASE 1: Preparing Waymo Front Camera Data"
    log_info "════════════════════════════════════════════════════════════"
    log_info ""
    
    cd "${PROJECT_ROOT}"
    
    # Source environment
    log_info "Setting up environment..."
    source /scratch/rbaskar5/set.bash
    
    # Check if data is already prepared
    if [ -f "/scratch/rbaskar5/Dataset/waymo_front/annotations/train.json" ]; then
        log_warn "Front camera data already prepared"
        log_warn "Found: /scratch/rbaskar5/Dataset/waymo_front/annotations/train.json"
        read -p "Continue with existing data? (y/n) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Re-preparing data..."
        else
            log_info "Using existing prepared data"
            return 0
        fi
    fi
    
    # Run preparation script
    log_info "Extracting FRONT camera images from TFRecords..."
    log_info "(This may take 10-30 minutes for full dataset)"
    python "${PREPARE_SCRIPT}"
    
    if [ $? -eq 0 ]; then
        log_success "Data preparation complete"
    else
        log_error "Data preparation failed"
        exit 1
    fi
}

# ── Launch training ─────────────────────────────────────────────────
launch_training() {
    log_info ""
    log_info "════════════════════════════════════════════════════════════"
    log_info "PHASE 2: Launching Qwen2.5-VL-7B Training"
    log_info "════════════════════════════════════════════════════════════"
    log_info ""
    
    cd "${PROJECT_ROOT}"
    
    # Source environment
    source /scratch/rbaskar5/set.bash
    
    log_info "Model: Qwen2.5-VL-7B"
    log_info "Hardware: 2× A100-80GB"
    log_info "Training: LoRA fine-tuning with bf16 + gradient checkpointing"
    log_info ""
    
    # Make script executable
    chmod +x "${TRAIN_SCRIPT}"
    
    # Launch training with Qwen2.5-VL-7B
    log_info "Starting training..."
    bash "${TRAIN_SCRIPT}" qwen2.5-vl-7b
    
    if [ $? -eq 0 ]; then
        log_success "Training launched successfully"
    else
        log_error "Training launch failed"
        exit 1
    fi
}

# ── Verify data and config ───────────────────────────────────────────
verify_setup() {
    log_info ""
    log_info "════════════════════════════════════════════════════════════"
    log_info "Verifying Setup"
    log_info "════════════════════════════════════════════════════════════"
    log_info ""
    
    # Check extracted data
    TRAIN_JSON="/scratch/rbaskar5/Dataset/waymo_front/annotations/train.json"
    VAL_JSON="/scratch/rbaskar5/Dataset/waymo_front/annotations/val.json"
    
    if [ -f "${TRAIN_JSON}" ]; then
        TRAIN_COUNT=$(jq 'length' "${TRAIN_JSON}" 2>/dev/null || echo "?")
        log_success "Train annotations: ${TRAIN_COUNT} samples"
    else
        log_warn "Train annotations not found"
    fi
    
    if [ -f "${VAL_JSON}" ]; then
        VAL_COUNT=$(jq 'length' "${VAL_JSON}" 2>/dev/null || echo "?")
        log_success "Validation annotations: ${VAL_COUNT} samples"
    else
        log_warn "Validation annotations not found"
    fi
    
    # Check config
    CONFIG="${PROJECT_ROOT}/configs/waymo_finetune_config.yaml"
    if grep -q 'qwen2.5-vl-7b' "${CONFIG}"; then
        log_success "Config uses Qwen2.5-VL-7B ✓"
    else
        log_warn "Config may not have Qwen2.5-VL-7B set"
    fi
}

# ── Print usage info ─────────────────────────────────────────────────
print_usage() {
    echo ""
    echo "════════════════════════════════════════════════════════════"
    echo "GLANCE + Waymo + Qwen Setup Complete!"
    echo "════════════════════════════════════════════════════════════"
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Inspect trained checkpoints:"
    echo "   ls -lh ${PROJECT_ROOT}/checkpoints/waymo_finetune/qwen2.5-vl-7b/"
    echo ""
    echo "2. Monitor training (in another terminal):"
    echo "   tail -f ${PROJECT_ROOT}/checkpoints/waymo/qwen2.5-vl-7b_train.log"
    echo ""
    echo "3. Run inference with trained LoRA adapter:"
    echo "   python ${PROJECT_ROOT}/src/infer_teacher.py --lora /path/to/lora_adapter"
    echo ""
    echo "4. Evaluate on test set:"
    echo "   python ${PROJECT_ROOT}/src/infer_distilled_student.py \\\"
    echo "     --model qwen2.5-vl-7b \\\"
    echo "     --lora ${PROJECT_ROOT}/checkpoints/waymo/qwen2.5-vl-7b/final_lora"
    echo ""
}

# ── Main ─────────────────────────────────────────────────────────────
main() {
    local mode="${1:-both}"
    
    echo ""
    echo "╔═══════════════════════════════════════════════════════════╗"
    echo "║  GLANCE + Waymo + Qwen VLM Setup                          ║"
    echo "║  ─────────────────────────────────────────────────────── ║"
    echo "║  Mode: $mode                                              ║"
    echo "║  Dataset: Waymo (Front Camera Only)                       ║"
    echo "║  Model: Qwen2.5-VL-7B-Instruct (LoRA)                     ║"
    echo "║  Hardware: 2× A100-80GB                                   ║"
    echo "╚═══════════════════════════════════════════════════════════╝"
    echo ""
    
    check_prerequisites
    
    case "${mode}" in
        prepare)
            prepare_data
            verify_setup
            print_usage
            ;;
        train)
            # Assume data is already prepared
            launch_training
            ;;
        both|full)
            prepare_data
            verify_setup
            launch_training
            print_usage
            ;;
        *)
            log_error "Unknown mode: ${mode}"
            log_info "Usage: $0 [prepare|train|both]"
            exit 1
            ;;
    esac
    
    log_success "Setup complete!"
}

main "$@"
