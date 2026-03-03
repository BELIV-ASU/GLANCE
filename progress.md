# GLANCE / SafetyVLM — Progress Log

**Date:** March 2, 2026  
**Hardware:** NVIDIA RTX 5000 Ada Generation (30 GB VRAM), Dell Precision 3660  
**Model:** Qwen2.5-VL-7B-Instruct + LoRA adapter (r=16, α=32, dropout=0.05)

---

## Phase 1 — Workstation Adaptation

Migrated the project from the university HPC cluster (SOL, A100 GPUs, 32B model) to the local workstation (RTX 5000 Ada, 7B model).

- Rewrote `infer_teacher.py` — loads `Qwen/Qwen2.5-VL-7B-Instruct` with bitsandbytes NF4 4-bit quantization, Flash Attention 2, and LoRA `merge_and_unload()`.
- Rewrote `infer.sh` — activates `.venv`, sets CUDA env vars, supports `--images` and val.jsonl modes.
- Created `.venv` with Python 3.10, installed all dependencies (transformers 4.51.3, torch 2.6.0+cu124, peft 0.18.1, flash-attn 2.7.4.post1, bitsandbytes 0.49.2, qwen-vl-utils 0.0.14).
- Renamed `running_on_jetson/` → `running_on_workstation/`, updated C++ code for RTX 5000 Ada specs.

## Phase 2 — Video Frame Extraction & Inference

- Downloaded a YouTube dashcam video → `video/source.mp4`.
- Extracted 20 evenly-spaced frames → `images/01_frame_117s.png` through `images/20_frame_497s.png`.
- Ran inference on all 20 frames, producing driving safety analyses.
- Created `save_results.py` — generates visual result cards (per-image) and a stitched `report.png`.
- Results saved to `results/` with individual cards + report.png + results.json.

## Phase 3 — GPU Optimization (bitsandbytes NF4)

Applied optimizations to the NF4 pipeline:

- Flash Attention 2 (`attn_implementation="flash_attention_2"`)
- LoRA `merge_and_unload()` — frees adapter overhead
- TF32 math on Ampere+ (`torch.backends.cuda.matmul.allow_tf32 = True`)
- `torch.inference_mode()` context
- Image pixel cap (min 200K, max 800K pixels)
- `torch.cuda.empty_cache()` after merge

**NF4 Baseline Performance:**
| Metric | Value |
|---|---|
| Peak VRAM | 10,072 MiB (32.8% of 30 GB) |
| Time per image | ~8.8s |
| Avg GPU utilization | 66% |

## Phase 4 — Jetson AGX Orin Deployment (Prepared)

Created deployment scripts for NVIDIA Jetson AGX Orin edge device:

- `running_on_jetson/export_gguf.py` — exports model to GGUF format for llama.cpp
- `running_on_jetson/infer_jetson.py` — dual-mode inference (HuggingFace fp16 or llama-cpp GGUF)
- `running_on_jetson/CMakeLists.txt` — C++ build for the Jetson

## Phase 5 — AWQ 4-bit Quantization (Merge & Quant Strategy)

### The Problem

The initial attempt to use `Qwen/Qwen2.5-VL-7B-Instruct-AWQ` (pre-quantized AWQ model from HuggingFace) **failed completely** because the LoRA adapter in `checkpoints/` was trained on the non-AWQ base model (`Qwen/Qwen2.5-VL-7B-Instruct`). Loading a LoRA onto a pre-quantized model causes:

1. **Layer structure mismatch** — AWQ replaces `nn.Linear` with `WQLinear`. LoRA expects standard linear layers, so all ~200 vision encoder LoRA keys (`visual.blocks.*`) were missing.
2. **lm_head dtype crash** — AWQ wrapped `lm_head` expecting INT weights (`qweight`), but the checkpoint has fp16 weights → `RuntimeError: expected scalar type Int but found Half`.
3. **merge_and_unload impossible** — AWQ quantized layers don't support merging LoRA back in.

### The Solution: Merge First, Then Quantize

Implemented the industry-standard "Merge & Quant" pipeline:

**Step 1 — Merge LoRA into base model:**
- Loaded `Qwen/Qwen2.5-VL-7B-Instruct` in fp16 on GPU
- Applied LoRA adapter via `PeftModel.from_pretrained()`
- Called `merge_and_unload()` — permanently bakes all LoRA weights (including vision encoder keys) into the base model
- Saved merged model (15.4 GB, safetensors)
- **Verified:** Merged model produces correct driving safety analysis without any adapter

**Step 2 — AWQ quantization:**
- Quantized merged fp16 model with AWQ 4-bit (GEMM kernel, group_size=128, zero_point=True)
- Calibration: 32 samples, 256 seq_len
- Fixed `lm_head` crash by adding `modules_to_not_convert: ["visual", "lm_head"]` to config
- Saved AWQ model (6.4 GB)

### AWQ Performance Results

| Metric | NF4 (before) | AWQ 4-bit (after) | Improvement |
|---|---|---|---|
| Model VRAM | ~8.7 GB | **6.5 GB** | -25% |
| Peak VRAM | 10.1 GB | **7.5 GB** | -26% |
| Time per image | 8.8s | **5.4s** | -39% faster |
| Model on disk | 16 GB (HF cache) | **6.4 GB** | -60% |

## Phase 6 — Workspace Reorganization

Restructured the project into a clean layout:

- Moved all model files under `models/` (checkpoints, merged model, AWQ model)
- Moved scripts into `scripts/`
- Moved inference outputs into `outputs/`
- Updated all paths and verified inference still works

## Phase 7 — GGUF Conversion & C++ Native Pipeline

Converted from the Python-based inference pipeline to a pure **C++ pipeline using llama.cpp**. This eliminates the Python runtime entirely.

### GGUF Model Conversion

1. **Re-merged LoRA** into fp16 base model (16 GB)
2. **Converted** fp16 HF model → GGUF fp16 (15.2 GB) using `convert_hf_to_gguf.py`
3. **Extracted** vision projector → `qwen2.5-vl-7b-mmproj-f16.gguf` (1.3 GB)
4. **Quantized** to Q4_K_M → `qwen2.5-vl-7b-Q4_K_M.gguf` (4.4 GB)
5. Deleted intermediate fp16 merged model and AWQ model to save disk space

### C++ Application (`tinyvlm`)

Built a complete C++ inference application using llama.cpp's native multimodal (`mtmd`) API:

- **vlm.h / vlm.cpp** — `VLM` class wrapping llama.cpp model loading, vision encoding, and autoregressive generation
- **main.cpp** — CLI with batch, single-image, and interactive modes; JSON output support
- **CMakeLists.txt** — FetchContent of llama.cpp master, CUDA enabled, links llama/ggml/common/mtmd

**llama.cpp API fixes applied:**
- `flash_attn` (bool) → `flash_attn_type` (enum `LLAMA_FLASH_ATTN_TYPE_ENABLED`)
- `llama_kv_cache_clear()` → `llama_memory_clear(llama_get_memory(ctx_), true)`
- `mtmd_bitmap_init_from_file()` → `mtmd_helper_bitmap_init_from_file(ctx, fname)`
- `mtmd_tokenize()` now takes `mtmd_input_text` struct

### Performance (Full GPU Mode)

All 20 dashcam images processed — live GPU monitoring with nvidia-smi during inference:

| Metric | Value |
|---|---|
| **Token throughput** | 77.1 tok/s (range 75.8–80.3) |
| **Avg prefill** | 1.2s per image |
| **Avg time/image** | 4.7s |
| **Total compute** | 94s for 20 images |
| **Peak VRAM** | 8,701 MiB (28.3% of 30 GB) |
| **Avg GPU utilization** | 72% (peak 100%) |
| **Power draw** | avg 184W, peak 250W |
| **Temperature** | avg 71°C, peak 80°C |
| **GPU clocks** | avg 2108 MHz, peak 2790 MHz |

#### Per-Image Results (Full GPU)

| Image | Tokens | tok/s | Prefill | Generate | Total |
|---|---|---|---|---|---|
| 01_frame_117s.png | 267 | 80.3 | 1383 ms | 3327 ms | 4.7s |
| 02_frame_137s.png | 333 | 78.0 | 1191 ms | 4267 ms | 5.5s |
| 03_frame_157s.png | 247 | 77.4 | 1200 ms | 3192 ms | 4.4s |
| 04_frame_177s.png | 243 | 76.6 | 1195 ms | 3172 ms | 4.4s |
| 05_frame_197s.png | 262 | 77.1 | 1197 ms | 3397 ms | 4.6s |
| 06_frame_217s.png | 253 | 75.8 | 1196 ms | 3336 ms | 4.5s |
| 07_frame_237s.png | 268 | 77.2 | 1205 ms | 3471 ms | 4.7s |
| 08_frame_257s.png | 236 | 76.2 | 1206 ms | 3096 ms | 4.3s |
| 09_frame_277s.png | 281 | 77.6 | 1211 ms | 3622 ms | 4.8s |
| 10_frame_297s.png | 278 | 76.1 | 1216 ms | 3653 ms | 4.9s |
| 11_frame_317s.png | 196 | 76.4 | 1216 ms | 2565 ms | 3.8s |
| 12_frame_337s.png | 268 | 76.2 | 1216 ms | 3519 ms | 4.7s |
| 13_frame_357s.png | 303 | 76.6 | 1215 ms | 3956 ms | 5.2s |
| 14_frame_377s.png | 249 | 76.6 | 1216 ms | 3252 ms | 4.5s |
| 15_frame_397s.png | 376 | 76.5 | 1223 ms | 4918 ms | 6.1s |
| 16_frame_417s.png | 264 | 76.2 | 1218 ms | 3465 ms | 4.7s |
| 17_frame_437s.png | 319 | 78.3 | 1230 ms | 4072 ms | 5.3s |
| 18_frame_457s.png | 188 | 77.6 | 1224 ms | 2423 ms | 3.6s |
| 19_frame_477s.png | 267 | 77.7 | 1230 ms | 3438 ms | 4.7s |
| 20_frame_497s.png | 266 | 78.4 | 1229 ms | 3393 ms | 4.6s |
| **Totals** | **5,364** | **77.1 avg** | **1,221 avg** | **3,477 avg** | **94.0s** |

#### GPU Timeline (sampled during inference)

| Time | GPU % | VRAM (MiB) | Power | Temp | GPU MHz |
|---|---|---|---|---|---|
| 21:12:35 | 20% | 724 | 29W | 54°C | 240 |
| 21:12:51 | 94% | 8633 | 206W | 64°C | 2445 |
| 21:12:59 | 94% | 8626 | 241W | 69°C | 2775 |
| 21:13:07 | 94% | 8627 | 241W | 72°C | 2760 |
| 21:13:15 | 100% | 8676 | 212W | 74°C | 2610 |
| 21:13:23 | 94% | 8693 | 249W | 77°C | 2745 |
| 21:13:31 | 95% | 8687 | 249W | 78°C | 2745 |
| 21:13:39 | 95% | 8684 | 244W | 78°C | 2745 |
| 21:13:47 | 94% | 8676 | 230W | 74°C | 2760 |
| 21:13:55 | 94% | 8659 | 249W | 79°C | 2745 |
| 21:14:03 | 34% | 8673 | 215W | 79°C | 2340 |
| 21:14:11 | 94% | 8416 | 249W | 80°C | 2715 |
| 21:14:19 | 94% | 8407 | 250W | 80°C | 2730 |
| 21:14:27 | 0% | 440 | 28W | 66°C | 210 |

## Phase 8 — Low-VRAM Mode (< 5 GB Target)

Added `--low-vram` and `--vision-cpu` flags to keep peak VRAM under 5 GB:

**Strategy:** Run the vision encoder on CPU (saves ~2 GB VRAM), offload only 20/28 text decoder layers to GPU, reduce context to 2048.

**Implementation:**
- Added `vision_cpu` config field to `VLMConfig`
- `--low-vram` sets: vision on CPU, 20 GPU layers, ctx 2048, batch 1024
- `--vision-cpu` moves only the vision encoder to CPU without other changes
- Updated `infer.sh` to accept `--low-vram` flag in any position

### Performance Comparison

| Metric | Full GPU | Low VRAM (`--low-vram`) |
|---|---|---|
| **Peak VRAM** | 8,701 MiB | **4,628 MiB (4.5 GB)** |
| **Token generation** | 77.1 tok/s | **80.4 tok/s** |
| **Prefill/image** | 1.2s | ~41s (vision on CPU) |
| **Total (20 images)** | 94s | 879s |

Token generation speed is comparable (even slightly faster with more GPU headroom). The trade-off is slower vision encoding per image (~41s on CPU vs 1.2s on GPU).

#### Per-Image Results (Low VRAM)

| Image | Tokens | tok/s | Prefill | Generate | Total |
|---|---|---|---|---|---|
| 01_frame_117s.png | 256 | 78.8 | 39184 ms | 3249 ms | 42.4s |
| 02_frame_137s.png | 256 | 79.4 | 41297 ms | 3226 ms | 44.5s |
| 03_frame_157s.png | 209 | 78.9 | 41417 ms | 2648 ms | 44.1s |
| 04_frame_177s.png | 256 | 79.6 | 41559 ms | 3215 ms | 44.8s |
| 05_frame_197s.png | 242 | 78.9 | 41359 ms | 3068 ms | 44.4s |
| 06_frame_217s.png | 241 | 79.7 | 41213 ms | 3026 ms | 44.2s |
| 07_frame_237s.png | 222 | 77.5 | 41247 ms | 2866 ms | 44.1s |
| 08_frame_257s.png | 255 | 78.1 | 41089 ms | 3266 ms | 44.4s |
| 09_frame_277s.png | 256 | 79.6 | 41346 ms | 3216 ms | 44.6s |
| 10_frame_297s.png | 238 | 78.5 | 41148 ms | 3032 ms | 44.2s |
| 11_frame_317s.png | 256 | 78.1 | 41149 ms | 3277 ms | 44.4s |
| 12_frame_337s.png | 217 | 78.7 | 41166 ms | 2757 ms | 43.9s |
| 13_frame_357s.png | 256 | 78.3 | 41457 ms | 3271 ms | 44.7s |
| 14_frame_377s.png | 173 | 82.5 | 40879 ms | 2097 ms | 43.0s |
| 15_frame_397s.png | 256 | 84.4 | 40887 ms | 3032 ms | 43.9s |
| 16_frame_417s.png | 242 | 83.9 | 40881 ms | 2884 ms | 43.8s |
| 17_frame_437s.png | 249 | 83.3 | 40658 ms | 2990 ms | 43.6s |
| 18_frame_457s.png | 189 | 83.3 | 40835 ms | 2270 ms | 43.1s |
| 19_frame_477s.png | 202 | 82.9 | 41103 ms | 2437 ms | 43.5s |
| 20_frame_497s.png | 256 | 84.1 | 40120 ms | 3045 ms | 43.2s |
| **Totals** | **4,727** | **80.4 avg** | **41,000 avg** | **2,944 avg** | **878.9s** |

### Usage

```bash
# Low VRAM batch (<5 GB)
./running_on_workstation/infer.sh --batch --low-vram

# Full GPU batch (~8.7 GB, fast)
./running_on_workstation/infer.sh --batch

# Single image, low VRAM
./running_on_workstation/build/tinyvlm \
  --model models/gguf/qwen2.5-vl-7b-Q4_K_M.gguf \
  --mmproj models/gguf/qwen2.5-vl-7b-mmproj-f16.gguf \
  --low-vram --image photo.jpg "Describe this traffic scene"

# Interactive mode
./running_on_workstation/infer.sh
./running_on_workstation/infer.sh --low-vram
```

---

## Current File Layout

```
GLANCE/
├── progress.md               # This file
├── ReadME.md
├── pip_freeze.txt
├── data/
│   ├── train.jsonl
│   └── val.jsonl
├── images/                   # 20 dashcam frames
├── video/                    # Source dashcam video
├── models/
│   ├── checkpoints/          # LoRA adapter (r=16, α=32)
│   │   ├── adapter_config.json
│   │   └── adapter_model.safetensors
│   └── gguf/
│       ├── qwen2.5-vl-7b-Q4_K_M.gguf      # Text model (4.4 GB)
│       └── qwen2.5-vl-7b-mmproj-f16.gguf   # Vision projector (1.3 GB)
├── outputs/
│   ├── batch_results.json         # Full GPU inference results
│   ├── batch_results_lowvram.json # Low VRAM inference results
│   ├── gpu_stats.csv              # GPU monitoring data
│   ├── gpu_monitor.csv
│   ├── inference.log
│   └── results/                   # Visual report cards
├── running_on_workstation/   # C++ inference (RTX 5000 Ada)
│   ├── CMakeLists.txt
│   ├── infer.sh              # Convenience wrapper
│   ├── build/                # Build output (tinyvlm binary)
│   └── src/
│       ├── vlm.h             # VLM class header
│       ├── vlm.cpp           # VLM implementation
│       └── main.cpp          # CLI entry point
└── running_on_jetson/        # Jetson AGX Orin deployment
    ├── CMakeLists.txt
    ├── export_gguf.py
    ├── infer_jetson.py
    └── src/
```

## Key Technical Details

| Component | Details |
|---|---|
| Base model | Qwen/Qwen2.5-VL-7B-Instruct |
| LoRA | r=16, α=32, dropout=0.05, targets: q/k/v/o/gate/up/down_proj |
| Quantization | GGUF Q4_K_M (4.4 GB text + 1.3 GB vision) |
| C++ runtime | llama.cpp (master), CUDA, cmake FetchContent |
| GPU arch | Ada Lovelace (sm_89) |
| Full GPU VRAM | ~8.7 GB peak, 77 tok/s |
| Low VRAM | ~4.5 GB peak, 80 tok/s generation (slower prefill) |
