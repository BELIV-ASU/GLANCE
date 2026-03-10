////////////////////////////////////////////////////////////////////////////////
/// @file   VLMInferenceEngine.hpp
/// @brief  Inference engine for distilled 7B Vision-Language Models on edge.
///
/// Project: Empirical Study of VLMs (7B) Performance on Edge
///          in Real-World Traffic Violations
///
/// Supports TensorRT and ONNX Runtime backends to load and execute distilled
/// models (Qwen2.5, Cosmos R1, MiMo, Open VLA, VILA) converted to
/// ONNX / TensorRT engine formats.
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <optional>

namespace edge_vlm_study {

/// ──────────────────────────────────────────────────────────────────────────
/// Supported inference backends
/// ──────────────────────────────────────────────────────────────────────────
enum class InferenceBackend {
    TENSORRT,       ///< NVIDIA TensorRT (FP16/INT8, max throughput)
    ONNX_RUNTIME,   ///< ONNX Runtime  (cross-platform, CUDA EP)
    LIBTORCH        ///< LibTorch C++ (for debugging / fallback)
};

/// ──────────────────────────────────────────────────────────────────────────
/// Distilled model identifiers
/// ──────────────────────────────────────────────────────────────────────────
enum class ModelID {
    QWEN2_5_7B,
    COSMOS_R1_7B,
    MIMO_7B,
    OPEN_VLA_7B,
    VILA_7B
};

/// ──────────────────────────────────────────────────────────────────────────
/// Engine configuration
/// ──────────────────────────────────────────────────────────────────────────
struct InferenceEngineConfig {
    /// Which backend to use
    InferenceBackend backend{InferenceBackend::TENSORRT};

    /// Path to the serialized model file (.onnx, .engine, .pt)
    std::string model_path{};

    /// Which distilled model this engine wraps
    ModelID model_id{ModelID::QWEN2_5_7B};

    /// GPU device index
    int device_id{0};

    /// Maximum batch size
    int max_batch_size{1};

    /// Use FP16 precision (TensorRT)
    bool fp16{true};

    /// Use INT8 quantization (TensorRT – requires calibration data)
    bool int8{false};

    /// Maximum sequence length for text generation
    int max_seq_length{512};

    /// Workspace size in bytes for TensorRT builder
    size_t trt_workspace_bytes{1ULL << 30};  // 1 GiB
};

/// ──────────────────────────────────────────────────────────────────────────
/// Inference result
/// ──────────────────────────────────────────────────────────────────────────
struct InferenceResult {
    /// Detected traffic violation class (e.g., "red_light_running")
    std::string violation_class;

    /// Confidence score [0.0, 1.0]
    float confidence{0.0f};

    /// Natural-language reasoning / description from the VLM
    std::string reasoning_text;

    /// Bounding box [x_min, y_min, x_max, y_max] normalized to [0,1]
    std::vector<float> bbox;

    /// Wall-clock inference latency
    std::chrono::microseconds latency{0};

    /// Additional key-value metadata
    std::unordered_map<std::string, std::string> metadata;
};

/// ──────────────────────────────────────────────────────────────────────────
/// @class VLMInferenceEngine
/// @brief Loads a distilled 7B VLM and runs inference on edge hardware.
///
/// Lifecycle:
///   1. Construct with config.
///   2. Call initialize() to load model and allocate GPU resources.
///   3. Call infer() per frame.
///   4. Destructor releases all GPU memory.
/// ──────────────────────────────────────────────────────────────────────────
class VLMInferenceEngine {
public:
    /// @brief Construct engine with configuration.
    explicit VLMInferenceEngine(const InferenceEngineConfig& config);

    /// @brief Destructor – releases GPU resources.
    ~VLMInferenceEngine();

    // Non-copyable, movable
    VLMInferenceEngine(const VLMInferenceEngine&) = delete;
    VLMInferenceEngine& operator=(const VLMInferenceEngine&) = delete;
    VLMInferenceEngine(VLMInferenceEngine&&) noexcept;
    VLMInferenceEngine& operator=(VLMInferenceEngine&&) noexcept;

    /// @brief Initialize the engine (load model, build TRT engine, etc.).
    /// @return true on success.
    bool initialize();

    /// @brief Run inference on a single image + optional text prompt.
    /// @param image_data  Raw pixel data (H x W x C, uint8).
    /// @param width       Image width.
    /// @param height      Image height.
    /// @param channels    Number of channels (3 = RGB).
    /// @param text_prompt Optional text prompt for the VLM.
    /// @return InferenceResult with violation class, confidence, reasoning.
    [[nodiscard]] InferenceResult infer(
        const std::vector<uint8_t>& image_data,
        int width, int height, int channels = 3,
        const std::optional<std::string>& text_prompt = std::nullopt);

    /// @brief Run batched inference.
    [[nodiscard]] std::vector<InferenceResult> infer_batch(
        const std::vector<std::vector<uint8_t>>& images,
        int width, int height, int channels = 3,
        const std::optional<std::string>& text_prompt = std::nullopt);

    /// @brief Warm up the engine with dummy data (for stable latency measurement).
    /// @param warmup_iters Number of warm-up iterations.
    void warmup(int warmup_iters = 10);

    /// @brief Query whether the engine is ready for inference.
    [[nodiscard]] bool is_initialized() const noexcept;

    /// @brief Get the model identifier.
    [[nodiscard]] ModelID model_id() const noexcept;

    /// @brief Get current backend.
    [[nodiscard]] InferenceBackend backend() const noexcept;

private:
    // ── Backend-specific initialization ─────────────────────────────────
    bool init_tensorrt();
    bool init_onnxruntime();
    bool init_libtorch();

    // ── Preprocessing ───────────────────────────────────────────────────
    // TODO: Implement image normalization, tokenization, etc.
    std::vector<float> preprocess_image(
        const std::vector<uint8_t>& raw, int w, int h, int c);

    // ── State ───────────────────────────────────────────────────────────
    InferenceEngineConfig config_;
    bool initialized_{false};

    // Opaque pointers to backend-specific handles
    // (PIMPL pattern to avoid leaking TRT/ORT headers)
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace edge_vlm_study
