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

---

## File Inventory

| File | Purpose |
|------|---------|
| `train_teacher.py` | Core training script (Qwen2.5-VL + QLoRA + dual-image collator) |
| `train_teacher.sh` | Training launcher (env setup, data build, train) |
| `data.py` | DriveLM JSON → train/val JSONL converter (with `--with_depth`) |
| `compute_depth.py` | Pre-compute depth maps with Depth Anything V2 |
| `infer_checkpoint200.py` | Inference on val data with LoRA checkpoint |
| `visualize_results.py` | Render inference results as images |
| `data_drivelm/train.jsonl` | 359,058 training samples |
| `data_drivelm/val.jsonl` | 18,897 validation samples |
| `checkpoints/checkpoint-200` | First training checkpoint (200 steps, no depth) |

## Environment

- **Cluster**: ASU Sol (SLURM)
- **GPU**: NVIDIA A100-SXM4-80GB
- **CUDA**: 12.8 (bundled with PyTorch 2.10)
- **Python**: 3.12, conda env `new_beginning`
- **Key packages**: transformers, peft 0.18.1, bitsandbytes 0.49.2, datasets 4.6.0, trl 0.29.0, accelerate
