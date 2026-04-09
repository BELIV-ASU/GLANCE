# GLANCE Project – Progress Log

## Phase 1: Dataset Acquisition & Preparation

- Downloaded **DriveLM** dataset (v1.1) from HuggingFace, including `v1_1_train_nus.json`.
- Cloned DriveLM repo, pulled LFS files, unzipped all archives.
- Dataset lives at `/scratch/rbaskar5/Dataset/DriveLM/`.
- nuScenes camera images at `/scratch/rbaskar5/Dataset/nuscenes/samples/` (34,149 CAM_FRONT images).

## Phase 2: Data Pipeline (`data.py`)

- Built `data.py` to convert DriveLM scene-level JSON into chat-format JSONL.
- Flattens scene → frame → task → QA into individual samples with system/user/assistant messages.
- Each user message includes scene metadata + camera image as `file://` URI.
- Generated **359,058 train** + **18,897 val** samples in `data_drivelm/`.

## Phase 3: Training Setup

- **Model**: `Qwen/Qwen2.5-VL-32B-Instruct` (non-MoE, `Qwen2_5_VLForConditionalGeneration`).
  - Initially tried Qwen3-VL-30B-A3B (MoE) but hit `torch._grouped_mm` crashes → switched.
- **Quantisation**: QLoRA 4-bit NF4 + double quantisation + bfloat16 compute (BitsAndBytes 0.49.2).
- **LoRA**: r=64, alpha=128, dropout=0.05, targets `q_proj,k_proj,v_proj,o_proj` → ~134M trainable params.
- **Environment**: ASU Sol cluster, conda env `new_beginning`, A100-SXM4-80GB, CUDA 12.8, PyTorch 2.10.

### CUBLAS Bug Workarounds (PyTorch 2.10 + cu128)

Three distinct CUBLAS crashes were resolved:

1. **`cublasSgemmStridedBatched` in rotary embeddings** – Monkey-patched `Qwen2_5_VLRotaryEmbedding.forward` to use `torch.einsum` instead of `@` operator.
2. **`cublasGemmEx` in bitsandbytes 4-bit matmul** – Set `torch.backends.cuda.preferred_blas_library("cublaslt")` globally + `TORCH_BLAS_PREFER_CUBLASLT=1` env var.
3. **`torch._grouped_mm` contraction dimension mismatch** – MoE-specific; resolved by switching to Qwen2.5-VL-32B (non-MoE).

## Phase 4: First Training Run → checkpoint-200

- Launched via `train_teacher.sh` on single A100-80GB (`CUDA_VISIBLE_DEVICES=0`).
- Training config: batch_size=1, grad_accum=8, lr=2e-4 cosine, max_seq_length=2048.
- Produced `checkpoints/checkpoint-200` (200 optimiser steps).
- Loss dropped from ~16.3 → ~7.2 over 200 steps.

## Phase 5: Inference & Visualisation

- Rewrote `infer_checkpoint200.py` for DriveLM val.jsonl format (chat messages with `file://` image URIs).
- Ran inference on 20 val samples → saved to `inference_results_ckpt200.json`.
- Created `visualize_results.py` → rendered 20 PNGs to `viz_ckpt200/` (camera frame + prompt + response + ground truth).

## Phase 6: Depth Map Integration (Depth Anything V2)

**Goal**: Add pre-computed monocular depth maps as a second visual input alongside RGB, giving the model explicit spatial/distance awareness for autonomous driving VQA.

### 6a. Depth Pre-Computation (`compute_depth.py`)

- Uses `depth-anything/Depth-Anything-V2-Small-hf` (~25M params) from HuggingFace transformers.
- Processes all 34,149 CAM_FRONT images in batches of 8 on GPU.
- Saves 16-bit grayscale PNGs (0–65535 range) to `/scratch/rbaskar5/Dataset/nuscenes/depth/CAM_FRONT/`.
- Skips already-computed files for restartability.
- **Status**: Running in background — ~5,700/34,149 computed so far.

### 6b. Pipeline Updates for Dual-Image Training

**`data.py`** — Updated:
- Added `_derive_depth_path()`: maps RGB path (`samples/CAM_FRONT/xxx.jpg` → `depth/CAM_FRONT/xxx.png`).
- Added `--with_depth` CLI flag. When set, each sample gets a second `{"type": "image"}` entry for the depth map.
- Graceful fallback: if depth PNG doesn't exist for a sample, only the RGB image is included.

**`train_teacher.py`** — Updated:
- `QwenVLCollator._load_images_from_messages()` now handles 16-bit depth PNGs:
  - Detects PIL mode `I;16` or `I`.
  - Normalises to 8-bit, converts to 3-channel grayscale RGB.
  - VL processor receives uniform `(H, W, 3)` uint8 tensors for both RGB and depth.
- Fallback data-build also passes `--with_depth`.

**`infer_checkpoint200.py`** — Updated:
- Added `_load_image_robust()` with the same 16-bit depth handling.
- `generate()` uses robust loader for all images (RGB + depth).

**`train_teacher.sh`** — Updated:
- Data build step passes `--with_depth` to `data.py`.

## Phase 7: Second Training Run (current)

- Deleted old JSONL and relaunched `train_teacher.sh`.
- JSONL was rebuilt by `data.py` (triggered by `train_teacher.py` since files were removed).
- **Note**: Since depth computation is still in progress (~17% done), the rebuilt JSONL currently has depth paths only for samples whose depth maps already exist. The collator gracefully skips missing depth files.
- Training is running: 134,649 total steps (3 epochs × 359,058 samples ÷ batch 1 ÷ grad_accum 8).
- Based on the previous run's throughput, full training is expected to take **several days** on a single A100.

## Phase 7 Update: Depth Computation Complete ✓

- `compute_depth.py` finished processing all **34,149 CAM_FRONT images** in **190.3 minutes** (~3 img/s on GPU).
- All 16-bit grayscale PNGs saved to `/scratch/rbaskar5/Dataset/nuscenes/depth/CAM_FRONT/`.
- Depth integration in `data.py` / `train_teacher.py` / `infer_checkpoint200.py` is fully ready for reuse.

---

## Phase 8: Knowledge Distillation (32B → 7B)

**Goal**: Distil the fine-tuned 32B teacher into a leaner 7B student for deployment.

### Setup

| | Teacher | Student |
|---|---|---|
| Base model | `Qwen/Qwen2.5-VL-32B-Instruct` | `Qwen/Qwen2.5-VL-7B-Instruct` |
| LoRA adapter | `checkpoints/checkpoint-800` | Freshly initialised |
| Quantisation | 4-bit NF4 (frozen) | 4-bit NF4 + QLoRA |
| Trainable params | 0 (inference only) | ~47.6M / 8.34B (0.57%) |
| LoRA config | — | r=16, alpha=32 |

### Training Config

- **Script**: `distill_4b.py` launched via `run_distill.sh` (2×A100-80GB, `accelerate launch`)
- **Data**: 359,058 train samples, 18,897 val samples (`data_drivelm/`)
- **Epochs**: 1 → ~22,442 update steps (179,529 samples/rank ÷ grad_accum 8)
- **Batch / grad_accum**: 1 / 8
- **LR**: 2e-5 cosine
- **Max length**: 2048 tokens
- **Loss**: α=0.2 × CE  +  β=0.8 × KL (temperature=2.0)
- **Checkpoints**: every 200 steps, keep 3 → `distilled_student/checkpoints/step-{N}/`
- **Mid-training inference**: every 50 steps, 40 val samples → `distilled_student/distilled_inference_results.json`
- **GPU memory**: ~68 GB / 80 GB per GPU

### Debugging Resolved

1. `python -m accelerate` — accelerate cannot be run as a module; fixed to use binary path via `${PYTHON_BIN} /home/rbaskar5/.local/bin/accelerate launch`.
2. Stray `\` passed as positional argument to `distill_4b.py` — argparse rejected it; fixed shell line continuations.
3. `--infer_after 10\` (no space before backslash) passed `10` as an unrecognised argument — fixed formatting.
4. `--infer_after` was `store_true` boolean; changed to `type=int` to support interval-based mid-training inference.

### Status

- First successful run completed at **step ≥ 10** (loss: 16.59 → 16.30 at step 30).
- `distilled_inference_results.json` produced (40 samples).
- Early inference quality is poor as expected (e.g., predicted "straight path" vs GT "Turn right.") — distillation is still in early steps.
- Distillation run ongoing (current `--infer_after 50`).

---

## File Inventory

| File | Purpose |
|------|---------|
| `train_teacher.py` | Core training script (Qwen2.5-VL + QLoRA + dual-image collator) |
| `train_teacher.sh` | Training launcher (env setup, data build, train) |
| `data.py` | DriveLM JSON → train/val JSONL converter (with `--with_depth`) |
| `compute_depth.py` | Pre-compute depth maps with Depth Anything V2 (complete) |
| `infer_checkpoint200.py` | Inference on val data with LoRA checkpoint |
| `visualize_results.py` | Render inference results as images |
| `distill_4b.py` | Knowledge distillation script (32B teacher → 7B student, QLoRA) |
| `run_distill.sh` | Distillation launcher (2×A100, accelerate, mid-training inference) |
| `accelerate_2gpu.yaml` | Accelerate config for 2-GPU DDP |
| `data_drivelm/train.jsonl` | 359,058 training samples |
| `data_drivelm/val.jsonl` | 18,897 validation samples |
| `distilled_student/` | Distilled student weights + inference results |
| `checkpoints/checkpoint-800` | Teacher LoRA adapter (used to initialise teacher for distillation) |
| `checkpoints/checkpoint-200` | First training checkpoint (200 steps, no depth) |

## Environment

- **Cluster**: ASU Sol (SLURM)
- **GPU**: NVIDIA A100-SXM4-80GB
- **CUDA**: 12.8 (bundled with PyTorch 2.10)
- **Python**: 3.12, conda env `new_beginning`
- **Key packages**: transformers, peft 0.18.1, bitsandbytes 0.49.2, datasets 4.6.0, trl 0.29.0, accelerate

---

## Phase 9: Validation Visual Script (Image + Output)

- Added `src/validate_with_visuals.py` for validation-time inference with visual inspection output.
- Script reads a validation JSONL, runs model generation, and writes:
  - `validation_results.jsonl` with prompt, ground truth, and model prediction.
  - `visualizations/sample_XXXXX.png` with side-by-side image and text panel.
- Visualization panel includes:
  - Prompt text
  - Ground truth text
  - Model output text
- Increased text readability for outputs by adding configurable `--font_size` (default 34), with larger section headers.
- Supports base model + optional LoRA adapter via:
  - `--base_model`
  - `--adapter_path`
- Added configurable display width for text panel via `--panel_width`.
