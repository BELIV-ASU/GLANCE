# Empirical Study of VLMs (7B) Performance on Edge in Real-World Traffic Violations

A hybrid C++/Python research project for deploying distilled 7B Vision-Language Models on edge hardware for real-time traffic violation detection.

## Architecture Overview

```
┌──────────────────────────────────────────────────────────────────────┐
│                        OFFLINE (Python)                              │
│                                                                      │
│  ┌─────────────────┐    ┌──────────────────┐    ┌────────────────┐  │
│  │ Reasoning LLM   │    │  Vision LLM      │    │ Ground Truth   │  │
│  │ (Teacher 1)     │    │  (Teacher 2)     │    │ Annotations    │  │
│  └────────┬────────┘    └────────┬─────────┘    └───────┬────────┘  │
│           │                      │                      │           │
│           └──────────┬───────────┘                      │           │
│                      ▼                                  │           │
│           ┌──────────────────────┐                      │           │
│           │   Knowledge          │◄─────────────────────┘           │
│           │   Distillation       │                                  │
│           │   (distill_teachers  │                                  │
│           │    _to_student.py)   │                                  │
│           └──────────┬───────────┘                                  │
│                      ▼                                              │
│           ┌──────────────────────┐                                  │
│           │   Export to ONNX /   │                                  │
│           │   TensorRT           │                                  │
│           │   (export_to_onnx.py)│                                  │
│           └──────────┬───────────┘                                  │
└──────────────────────┼──────────────────────────────────────────────┘
                       │  .onnx / .engine files
┌──────────────────────┼──────────────────────────────────────────────┐
│                      ▼         EDGE (C++)                           │
│                                                                      │
│  ┌─────────────────────────────────────────────────────────────┐    │
│  │                   ROS 2 Runtime                              │    │
│  │                                                              │    │
│  │  ┌───────────────────┐    ┌────────────────────────────┐    │    │
│  │  │ DataIngestionNode │───►│ VLMInferenceEngine         │    │    │
│  │  │ (Camera, Depth,   │    │ (TensorRT / ONNX Runtime)  │    │    │
│  │  │  Semantic, RosBag)│    └──────────┬─────────────────┘    │    │
│  │  └───────────────────┘               │                      │    │
│  │                                      ▼                      │    │
│  │                           ┌──────────────────────┐          │    │
│  │                           │ SNNRuntimeInterface  │          │    │
│  │                           │ (Spike Encoding/     │          │    │
│  │                           │  Decoding, LIF sim)  │          │    │
│  │                           └──────────┬───────────┘          │    │
│  │                                      ▼                      │    │
│  │                           ┌──────────────────────┐          │    │
│  │                           │ EdgeEvaluator        │          │    │
│  │                           │ (Latency, GPU, mIoU, │          │    │
│  │                           │  SNN metrics)        │          │    │
│  │                           └──────────────────────┘          │    │
│  └─────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────┘
```

## Distilled Student Models

| Model | Parameters | Architecture |
|-------|-----------|--------------|
| Qwen2.5-7B | 7B | Transformer (instruction-tuned) |
| Cosmos R1-7B | 7B | World-model reasoning |
| MiMo-7B | 7B | Multi-modal reasoning |
| Open VLA-7B | 7B | Vision-Language-Action |
| VILA-7B | 7B | Visual language pre-training |

## Directory Structure

```
edge_vlm_study/
├── CMakeLists.txt                  # CMake build (ROS 2 / ament_cmake)
├── package.xml                     # ROS 2 package manifest
├── README.md                       # This file
│
├── include/edge_vlm_study/         # C++ public headers
│   ├── DataIngestionNode.hpp       # ROS 2 data ingestion node
│   ├── VLMInferenceEngine.hpp      # TensorRT / ONNX Runtime inference
│   ├── SNNRuntimeInterface.hpp     # SNN spike encode / simulate / decode
│   └── EdgeEvaluator.hpp           # Benchmarking & evaluation harness
│
├── src/                            # C++ source files
│   ├── main.cpp                    # Entry point (ROS 2 node)
│   ├── DataIngestionNode.cpp       # Camera, depth, semantic ingestion
│   ├── VLMInferenceEngine.cpp      # Multi-backend VLM inference
│   ├── SNNRuntimeInterface.cpp     # LIF neuron simulation stubs
│   └── EdgeEvaluator.cpp           # Latency/accuracy/GPU benchmarks
│
├── launch/                         # ROS 2 launch files
│   └── edge_vlm_launch.py
│
├── config/                         # YAML configuration
│   └── edge_vlm_config.yaml
│
├── offline_distillation/           # Python training / export scripts
│   ├── __init__.py
│   ├── distill_teachers_to_student.py
│   └── export_to_onnx.py
│
├── models/                         # Serialized model files (gitignored)
├── data/                           # Evaluation data (gitignored)
├── scripts/                        # Utility scripts
└── tests/                          # Unit tests
```

## Prerequisites

### Edge (C++)
- ROS 2 Humble (or later)
- CMake ≥ 3.22
- CUDA Toolkit 12.x
- LibTorch (C++ PyTorch)
- TensorRT 8.x / 9.x
- ONNX Runtime 1.16+
- C++20 compiler (GCC 11+ / Clang 14+)

### Offline (Python)
- Python 3.10+
- PyTorch 2.x
- Transformers (HuggingFace)
- PEFT (LoRA)
- Accelerate
- ONNX + ONNX Runtime

## Building

```bash
# Source ROS 2
source /opt/ros/humble/setup.bash

# Build
cd edge_vlm_study
mkdir build && cd build
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DTorch_DIR=/path/to/libtorch/share/cmake/Torch \
    -DTENSORRT_ROOT=/usr/local/TensorRT \
    -DONNXRUNTIME_ROOT=/usr/local/onnxruntime
make -j$(nproc)
```

## Running

```bash
# Live inference
ros2 launch edge_vlm_study edge_vlm_launch.py

# Benchmark mode
ros2 run edge_vlm_study edge_vlm_node --benchmark

# Offline distillation
cd offline_distillation
python distill_teachers_to_student.py \
    --student Qwen/Qwen2.5-7B-Instruct \
    --dataset ../data/traffic_violations \
    --output ../checkpoints/distilled_qwen2_5

# Export to ONNX
python export_to_onnx.py \
    --checkpoint ../checkpoints/distilled_qwen2_5 \
    --output ../models/qwen2_5_7b_distilled.onnx
```

## TODO

- [ ] Implement RosBag2 replay in `DataIngestionNode`
- [ ] Implement TensorRT engine loading in `VLMInferenceEngine`
- [ ] Implement ONNX Runtime session in `VLMInferenceEngine`
- [ ] Implement image preprocessing (resize, normalize, HWC→CHW)
- [ ] Implement time-synchronized message assembly
- [ ] Implement NVML GPU profiling in `EdgeEvaluator`
- [ ] Implement confusion-matrix-based precision/recall/F1
- [ ] Implement neuromorphic hardware backend for SNNRuntime
- [ ] Implement full distillation training loop
- [ ] Implement ONNX/TensorRT export pipeline
- [ ] Add unit tests (Google Test / Catch2)
- [ ] Add CI/CD pipeline
