/*
 * vlm.h — Qwen2.5-VL-7B-Instruct inference via llama.cpp
 *
 * Jetson AGX Orin edition (CUDA 12.2, sm_87, shared memory).
 * Uses llama.cpp's native multimodal (mtmd) support.
 *
 * Model: Qwen/Qwen2.5-VL-7B-Instruct (GGUF Q4_K_M)
 * Target: Jetson AGX Orin (32/64 GB unified memory)
 */

#pragma once

#include <string>
#include <vector>
#include <functional>
#include <cstdint>

// llama.cpp forward declarations
struct llama_model;
struct llama_context;
struct llama_sampler;

// mtmd (multimodal) forward declarations
struct mtmd_context;
struct mtmd_input_chunks;
struct mtmd_bitmap;

namespace tinyvlm {

// ─── Configuration ──────────────────────────────────────────
struct VLMConfig {
    std::string model_path;           // Path to GGUF (text model)
    std::string mmproj_path;          // Path to mmproj GGUF (vision projector)

    int   n_gpu_layers  = -1;         // Layers on GPU (-1 = auto)
    int   n_ctx         = 4096;       // Context length (conservative for Orin)
    int   n_batch       = 2048;       // Prompt batch size
    int   n_ubatch      = 512;        // Micro-batch for decode
    bool  use_mmap      = true;       // Memory-map model
    bool  flash_attn    = true;       // Flash attention
    bool  vision_cpu    = false;      // Run vision encoder on CPU
    int   n_threads     = 0;          // CPU threads (0 = auto)

    // Sampling
    float temperature    = 0.7f;
    float top_p          = 0.9f;
    int   top_k          = 40;
    float repeat_penalty = 1.1f;
    int   max_tokens     = 256;       // Shorter for real-time

    // System prompt
    std::string system_prompt =
        "You are SafetyVLM, an expert driving instructor and traffic-rule analyst. "
        "When shown a traffic scene, identify signs, markings, and signals. "
        "Explain the rules that apply and state correct driver behaviour. "
        "Be concise.";

    // Auto-configure for Jetson AGX Orin (32 GB unified memory)
    static VLMConfig for_jetson_orin(const std::string& model_path,
                                     const std::string& mmproj_path);
};

// ─── Generation result ──────────────────────────────────────
struct GenerationResult {
    std::string text;
    int    tokens_generated = 0;
    int    tokens_prompt    = 0;
    double time_prefill_ms  = 0;
    double time_generate_ms = 0;
    double tokens_per_sec   = 0;
    bool   stopped_eos      = false;
};

// ─── Streaming callback ─────────────────────────────────────
using TokenCallback = std::function<bool(const std::string& piece)>;

// ─── VLM Engine ─────────────────────────────────────────────
class VLM {
public:
    explicit VLM(const VLMConfig& cfg);
    ~VLM();

    VLM(const VLM&) = delete;
    VLM& operator=(const VLM&) = delete;

    bool load();
    bool is_loaded() const { return model_ != nullptr && ctx_mtmd_ != nullptr; }

    // Text-only chat
    GenerationResult chat(const std::string& user_msg,
                          const TokenCallback& cb = nullptr);

    // Chat with image loaded from file
    GenerationResult chat_with_image(const std::string& user_msg,
                                      const std::string& image_path,
                                      const TokenCallback& cb = nullptr);

    // Chat with raw RGB pixel data (w * h * 3 bytes, RGBRGBRGB...)
    // Used by the ROS node to pass preprocessed camera frames directly
    GenerationResult chat_with_image_data(const std::string& user_msg,
                                          uint32_t width, uint32_t height,
                                          const unsigned char* rgb_data,
                                          const TokenCallback& cb = nullptr);

    void reset();
    void print_stats() const;

private:
    VLMConfig cfg_;

    llama_model*   model_    = nullptr;
    llama_context* ctx_      = nullptr;
    llama_sampler* smpl_     = nullptr;
    mtmd_context*  ctx_mtmd_ = nullptr;

    int gpu_layers_ = 0;
    int n_past_     = 0;

    bool init_sampler();
    int  auto_gpu_layers() const;

    GenerationResult generate_from_chunks(mtmd_input_chunks* chunks,
                                           const TokenCallback& cb);

    std::string build_prompt(const std::string& user_msg,
                              bool has_image) const;
};

}  // namespace tinyvlm
