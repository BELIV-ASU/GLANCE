/*
 * vlm.h — Qwen2.5-VL-7B-Instruct inference via llama.cpp
 *
 * Uses llama.cpp's native multimodal (mtmd) support.
 * No separate CLIP encoder needed — llama.cpp handles vision
 * internally for Qwen2-VL architecture.
 *
 * Model: Qwen/Qwen2.5-VL-7B-Instruct (GGUF Q4_K_M)
 * Target: GTX 1650 (4GB VRAM, 896 CUDA cores)
 *
 * Layer offloading:
 *   - mmap the full GGUF from disk (zero-copy cold storage)
 *   - Offload as many layers to GPU as VRAM allows
 *   - Remaining layers execute on CPU
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
    std::string model_path;           // Path to GGUF (text + vision combined)
    std::string mmproj_path;          // Path to mmproj GGUF (vision projector)

    int   n_gpu_layers  = -1;         // Layers on GPU (-1 = auto for 4GB)
    int   n_ctx         = 4096;       // Context length
    int   n_batch       = 2048;       // Prompt batch size
    int   n_ubatch      = 512;        // Micro-batch for decode
    bool  use_mmap      = true;       // Memory-map model (disk tier)
    bool  flash_attn    = true;       // Flash attention
    int   n_threads     = 0;          // CPU threads (0 = auto)

    // Sampling
    float temperature    = 0.7f;
    float top_p          = 0.9f;
    int   top_k          = 40;
    float repeat_penalty = 1.1f;
    int   max_tokens     = 512;

    // Auto-configure for GTX 1650 (4GB VRAM)
    static VLMConfig for_gtx1650(const std::string& model_path,
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
// Return false to stop generation
using TokenCallback = std::function<bool(const std::string& piece)>;

// ─── VLM Engine ─────────────────────────────────────────────
class VLM {
public:
    explicit VLM(const VLMConfig& cfg);
    ~VLM();

    VLM(const VLM&) = delete;
    VLM& operator=(const VLM&) = delete;

    /// Load both text model and vision projector
    bool load();
    bool is_loaded() const { return model_ != nullptr && ctx_mtmd_ != nullptr; }

    /// Text-only generation
    GenerationResult chat(const std::string& user_msg,
                          const TokenCallback& cb = nullptr);

    /// Vision + text generation (provide image file path)
    GenerationResult chat_with_image(const std::string& user_msg,
                                      const std::string& image_path,
                                      const TokenCallback& cb = nullptr);

    /// Vision + text with raw image bytes
    GenerationResult chat_with_image_bytes(const std::string& user_msg,
                                            const uint8_t* rgba, int w, int h,
                                            const TokenCallback& cb = nullptr);

    /// Reset conversation (clears KV cache)
    void reset();

    void print_stats() const;

private:
    VLMConfig cfg_;

    // llama.cpp core
    llama_model*   model_   = nullptr;
    llama_context* ctx_     = nullptr;
    llama_sampler* smpl_    = nullptr;

    // Multimodal context (handles vision encoding for Qwen2-VL)
    mtmd_context*  ctx_mtmd_ = nullptr;

    int gpu_layers_ = 0;
    int n_past_     = 0;   // Tokens already in KV cache

    // Internal
    bool init_sampler();
    int  auto_gpu_layers() const;

    // Core generation from mtmd input chunks
    GenerationResult generate_from_chunks(mtmd_input_chunks* chunks,
                                           const TokenCallback& cb);

    // Build chat prompt string
    std::string build_prompt(const std::string& user_msg,
                              bool has_image) const;
};

}  // namespace tinyvlm
