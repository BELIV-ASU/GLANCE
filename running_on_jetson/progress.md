# SafetyVLM — Jetson AGX Orin Progress

## Platform

| Component | Version |
|-----------|---------|
| Board | NVIDIA Jetson AGX Orin (sm_87 Ampere) |
| CUDA | 12.2 |
| TensorRT | 8.6.2 |
| NPP | 12.2 |
| OpenCV | 4.13.0 (CPU; no CUDA modules) |
| ROS2 | Humble (with cv_bridge, sensor_msgs, image_transport) |
| Model | Qwen2.5-VL-7B-Instruct GGUF Q4_K_M (4.4 GB) + mmproj-f16 (1.3 GB) |
| Backend | llama.cpp (master, FetchContent) with GGML CUDA + mtmd multimodal API |
| Build | ament_cmake / colcon |

---

## File Structure

```
running_on_jetson/
├── CMakeLists.txt          # ament_cmake build with CUDA, NPP, llama.cpp, ROS2
├── package.xml             # ROS2 package manifest (tinyvlm_jetson)
├── export_gguf.py          # GGUF export helper (pre-existing)
├── infer_jetson.py         # Python inference script (pre-existing)
├── progress.md             # This file
└── src/
    ├── vlm.h               # VLM engine header — VLMConfig, VLM class, chat_with_image_data()
    ├── vlm.cpp             # VLM engine — llama.cpp + mtmd Qwen2-VL pipeline
    ├── cuda_preprocess.cuh # CUDA GPU image preprocessing API (GpuImage, preprocess_bgr_to_rgb_resize)
    ├── cuda_preprocess.cu  # NPP BGR→RGB channel swap + bilinear resize on GPU
    └── ros2_node.cpp       # ROS2 real-time inference node (3-thread, recording, overlay)
```

---

## What Was Done

### 1. VLM Engine (`vlm.h` / `vlm.cpp`)

- Created a C++ VLM inference engine wrapping llama.cpp and the mtmd (multimodal) API.
- `VLMConfig::for_jetson_orin()` preset: n_ctx=4096, n_batch=2048, max_tokens=256, 99 GPU layers offloaded to CUDA.
- `chat_with_image_data(prompt, width, height, rgb_data)` accepts raw RGB pixel data directly from the preprocessing pipeline — no filesystem round-trip.
- Internally builds a Qwen2.5-VL `<|im_start|>` chat-format prompt with `mtmd_default_marker()` for the vision encoder.
- Returns `GenerationResult` with text, token count, tok/s, prefill time, and generation time.

### 2. CUDA Preprocessing (`cuda_preprocess.cuh` / `cuda_preprocess.cu`)

- GPU-accelerated image preprocessing using NVIDIA NPP (Performance Primitives).
- 4-step pipeline:
  1. `cudaMemcpy2DAsync` — upload BGR8 frame to GPU pitched memory
  2. `nppiSwapChannels_8u_C3R` — BGR → RGB channel swap
  3. `nppiResize_8u_C3R` — bilinear resize to target resolution
  4. `cudaMemcpy2DAsync` — download RGB result to host
- Persistent static GPU buffers with lazy allocation (avoids per-frame malloc).
- Supports arbitrary source resolution up to 1920×1080 (configurable).
- Falls back to CPU (OpenCV `cv::resize` + `cv::cvtColor`) if CUDA init fails.

### 3. ROS2 Inference Node (`ros2_node.cpp`)

#### Architecture — 3-Thread Real-Time Pipeline

| Thread | Role |
|--------|------|
| **ROS callback** (main) | Receives every frame from `/sensing/camera/camera0/image_rect_color`, updates display buffer at full framerate, passes every N-th frame to preprocess thread |
| **Preprocess thread** | Waits on condition_variable, runs CUDA NPP BGR→RGB+resize, double-buffers output, signals inference thread |
| **Inference thread** | Waits on condition_variable, runs VLM `chat_with_image_data()` as fast as possible, updates overlay results |

#### Low-Latency Design

- **Frame skipping**: only every N-th frame (default: 3) is sent for analysis — configurable via `skip_frames` parameter.
- **Double-buffered preprocessing**: buffer A is written by preprocess while buffer B is read by inference, then swapped — zero-copy handoff.
- **Condition variables**: zero-polling wake-up between threads (no `sleep_for` loops).
- **No fixed interval**: inference runs back-to-back as fast as the model allows.
- **Smaller defaults**: 480×320 target resolution, 128 max_tokens, concise prompt — tuned for speed.

#### On-Screen Overlay (OpenCV HUD)

- Semi-transparent bottom bar with word-wrapped model response.
- Status indicator: green "READY" / orange "ANALYZING…".
- Live stats:
  - Display FPS (rendering framerate)
  - Camera FPS (incoming ROS topic rate)
  - Frame counter
  - Skip ratio (e.g. "Skip: 1/3")
  - Inference count
  - Preprocess latency (ms)
  - Inference stats: tokens generated, tok/s, prefill time, generation time, total time

#### MP4 Video Recording

- Records the full display output (camera + overlay + inference text) to MP4.
- H.264 (`avc1`) with XVID/AVI automatic fallback.
- Timestamped filenames: `recordings/safetyvlm_YYYYMMDD_HHMMSS.mp4`.
- Blinking red **REC** indicator in the top-right corner.
- Configurable: `record` (bool), `record_path`, `record_fps` (default: 20).

### 4. Build System (`CMakeLists.txt` / `package.xml`)

- `ament_cmake` project: `tinyvlm_jetson`, languages CXX, C, CUDA.
- `CMAKE_CUDA_ARCHITECTURES="87"` for Jetson AGX Orin.
- `FetchContent` pulls llama.cpp master with `GGML_CUDA=ON`.
- Links NPP libraries: `nppc`, `nppicc`, `nppig`, `nppidei`.
- Depends on: `rclcpp`, `sensor_msgs`, `cv_bridge`, `OpenCV`.
- `target_link_libraries` uses plain (non-keyword) signature for `ament_target_dependencies` compatibility.
- Build time: ~25–30 seconds (incremental), ~6–7 minutes (clean with llama.cpp).

### 5. Model Files

- Downloaded from Google Drive via `gdown`:
  - `qwen2.5-vl-7b-Q4_K_M.gguf` — 4.4 GB (quantized model weights)
  - `qwen2.5-vl-7b-mmproj-f16.gguf` — 1.3 GB (vision projector, FP16)
- Stored at `/home/beliv/GLANCE/gguf/`.

---

## ROS2 Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_path` | *(required)* | Path to GGUF model file |
| `mmproj_path` | *(required)* | Path to mmproj GGUF file |
| `prompt` | Traffic sign identification prompt | VLM instruction prompt |
| `max_tokens` | 128 | Maximum tokens to generate |
| `target_width` | 480 | Preprocessing target width |
| `target_height` | 320 | Preprocessing target height |
| `skip_frames` | 3 | Analyze every N-th frame |
| `topic` | `/sensing/camera/camera0/image_rect_color` | ROS2 image topic |
| `use_cuda_preprocess` | true | Use CUDA NPP or CPU fallback |
| `record` | true | Enable MP4 recording |
| `record_path` | `/home/beliv/GLANCE/recordings` | Recording output directory |
| `record_fps` | 20.0 | Recording framerate |

---

## Usage

```bash
cd /home/beliv/GLANCE
source install/setup.bash

ros2 run tinyvlm_jetson ros2_inference_node --ros-args \
    -p model_path:=/home/beliv/GLANCE/gguf/qwen2.5-vl-7b-Q4_K_M.gguf \
    -p mmproj_path:=/home/beliv/GLANCE/gguf/qwen2.5-vl-7b-mmproj-f16.gguf \
    -p skip_frames:=3 \
    -p record:=true
```

Press **ESC** or **q** to quit. Recording is saved automatically on exit.

---

## Known Notes

- OpenCV linker warning (`libopencv_*.so.4.5d` vs `4.5`) from cv_bridge is harmless — ABI compatible.
- Model paths must be absolute (ROS2 does not expand `~`).
- `target_link_libraries` must use plain (non-keyword) signature to avoid conflict with `ament_target_dependencies`.
