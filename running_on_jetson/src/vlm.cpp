/*
 * vlm.cpp — Qwen2.5-VL-7B-Instruct inference via llama.cpp + mtmd
 *
 * Jetson AGX Orin edition (CUDA 12.2, sm_87).
 *
 * llama.cpp's mtmd (multimodal) API handles:
 *   - Qwen2-VL vision encoder (ViT)
 *   - Vision-language projector (mmproj)
 *   - Image tokenization + embedding injection
 *   - All quantization (Q4_K_M dequant, CUDA kernels)
 *   - KV cache management
 *   - Layer offloading (GPU/CPU split via n_gpu_layers + mmap)
 *
 * We drive the high-level flow:
 *   1. Build chat prompt with image markers
 *   2. Let mtmd tokenize + encode images
 *   3. Feed chunks through llama_decode
 *   4. Sample tokens in a loop
 */

#include "vlm.h"

#include "llama.h"
#include "common.h"
#include "mtmd.h"
#include "mtmd-helper.h"

#include <iostream>
#include <chrono>
#include <algorithm>
#include <thread>
#include <cstring>

namespace tinyvlm {

// ═══════════════════════════════════════════════════════════
//  VLMConfig preset for Jetson AGX Orin
// ═══════════════════════════════════════════════════════════

VLMConfig VLMConfig::for_jetson_orin(const std::string& model_path,
                                      const std::string& mmproj_path) {
    VLMConfig c;
    c.model_path  = model_path;
    c.mmproj_path = mmproj_path;

    // Jetson AGX Orin: unified memory (32/64 GB shared CPU+GPU)
    // Qwen2.5-VL-7B Q4_K_M: ~4.4 GB text + 1.3 GB vision
    // Conservative settings for real-time inference
    c.n_ctx        = 4096;
    c.n_batch      = 2048;
    c.n_ubatch     = 512;
    c.use_mmap     = true;
    c.flash_attn   = true;
    c.n_gpu_layers = 99;     // offload all layers (unified memory)
    c.n_threads    = std::max(1, (int)std::thread::hardware_concurrency() / 2);

    c.temperature    = 0.7f;
    c.top_p          = 0.9f;
    c.top_k          = 40;
    c.repeat_penalty = 1.1f;
    c.max_tokens     = 256;  // shorter for real-time use

    return c;
}

// ═══════════════════════════════════════════════════════════
//  Auto GPU layers for Jetson AGX Orin
// ═══════════════════════════════════════════════════════════

int VLM::auto_gpu_layers() const {
    // Jetson AGX Orin has unified memory — no separate VRAM budget.
    // Offload everything to GPU for maximum throughput.
    // Q4_K_M 7B: ~4.4 GB + vision ~1.3 GB + KV ~0.3 GB ≈ 6 GB
    // Leaves plenty for OS + ROS on 32 GB Orin.
    return 99;
}

// ═══════════════════════════════════════════════════════════
//  Constructor / Destructor
// ═══════════════════════════════════════════════════════════

VLM::VLM(const VLMConfig& cfg) : cfg_(cfg) {
    llama_backend_init();
}

VLM::~VLM() {
    if (ctx_mtmd_) mtmd_free(ctx_mtmd_);
    if (smpl_)     llama_sampler_free(smpl_);
    if (ctx_)      llama_free(ctx_);
    if (model_)    llama_model_free(model_);
    llama_backend_free();
}

// ═══════════════════════════════════════════════════════════
//  Load model
// ═══════════════════════════════════════════════════════════

bool VLM::load() {
    if (cfg_.model_path.empty()) {
        std::cerr << "[tinyvlm] error: model_path is empty\n";
        return false;
    }
    if (cfg_.mmproj_path.empty()) {
        std::cerr << "[tinyvlm] error: mmproj_path is empty\n";
        return false;
    }

    // ── Model ──
    llama_model_params mp = llama_model_default_params();
    gpu_layers_ = cfg_.n_gpu_layers < 0 ? auto_gpu_layers() : cfg_.n_gpu_layers;
    mp.n_gpu_layers = gpu_layers_;
    mp.use_mmap     = cfg_.use_mmap;

    std::cout << "[tinyvlm-jetson] Loading " << cfg_.model_path << "\n";
    std::cout << "[tinyvlm-jetson] GPU layers: " << gpu_layers_
              << " | mmap: " << (cfg_.use_mmap ? "on" : "off") << "\n";

    model_ = llama_model_load_from_file(cfg_.model_path.c_str(), mp);
    if (!model_) {
        std::cerr << "[tinyvlm-jetson] Failed to load model\n";
        return false;
    }

    // ── Context ──
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx       = cfg_.n_ctx;
    cp.n_batch     = cfg_.n_batch;
    cp.n_ubatch    = cfg_.n_ubatch;
    cp.flash_attn_type = cfg_.flash_attn
                             ? LLAMA_FLASH_ATTN_TYPE_ENABLED
                             : LLAMA_FLASH_ATTN_TYPE_DISABLED;
    int threads    = cfg_.n_threads > 0
                      ? cfg_.n_threads
                      : std::max(1, (int)std::thread::hardware_concurrency() / 2);
    cp.n_threads       = threads;
    cp.n_threads_batch = threads;

    ctx_ = llama_init_from_model(model_, cp);
    if (!ctx_) {
        std::cerr << "[tinyvlm-jetson] Failed to create llama context\n";
        return false;
    }

    // ── Multimodal context (vision encoder + projector) ──
    mtmd_context_params mcp = mtmd_context_params_default();
    mcp.use_gpu        = (gpu_layers_ > 0) && !cfg_.vision_cpu;
    mcp.n_threads      = threads;
    mcp.print_timings  = false;  // quiet for real-time

    std::cout << "[tinyvlm-jetson] Vision encoder: "
              << (mcp.use_gpu ? "GPU" : "CPU") << "\n";

    ctx_mtmd_ = mtmd_init_from_file(cfg_.mmproj_path.c_str(), model_, mcp);
    if (!ctx_mtmd_) {
        std::cerr << "[tinyvlm-jetson] Failed to load mmproj from "
                  << cfg_.mmproj_path << "\n";
        return false;
    }

    // ── Sampler ──
    if (!init_sampler()) {
        std::cerr << "[tinyvlm-jetson] Failed to init sampler\n";
        return false;
    }

    std::cout << "[tinyvlm-jetson] Ready.\n";
    print_stats();
    return true;
}

// ═══════════════════════════════════════════════════════════
//  Sampler setup
// ═══════════════════════════════════════════════════════════

bool VLM::init_sampler() {
    auto sp = llama_sampler_chain_default_params();
    smpl_ = llama_sampler_chain_init(sp);
    if (!smpl_) return false;

    llama_sampler_chain_add(smpl_, llama_sampler_init_penalties(
        64, cfg_.repeat_penalty, 0.0f, 0.0f));
    llama_sampler_chain_add(smpl_, llama_sampler_init_top_k(cfg_.top_k));
    llama_sampler_chain_add(smpl_, llama_sampler_init_top_p(cfg_.top_p, 1));
    llama_sampler_chain_add(smpl_, llama_sampler_init_temp(cfg_.temperature));
    llama_sampler_chain_add(smpl_, llama_sampler_init_dist(42));

    return true;
}

// ═══════════════════════════════════════════════════════════
//  Chat prompt building (Qwen2.5-VL format)
// ═══════════════════════════════════════════════════════════

std::string VLM::build_prompt(const std::string& user_msg,
                               bool has_image) const {
    std::string marker = mtmd_default_marker();

    std::string p;
    p += "<|im_start|>system\n";
    p += cfg_.system_prompt;
    p += "<|im_end|>\n";
    p += "<|im_start|>user\n";
    if (has_image) {
        p += marker;
        p += "\n";
    }
    p += user_msg;
    p += "<|im_end|>\n";
    p += "<|im_start|>assistant\n";
    return p;
}

// ═══════════════════════════════════════════════════════════
//  Text-only chat
// ═══════════════════════════════════════════════════════════

GenerationResult VLM::chat(const std::string& user_msg,
                            const TokenCallback& cb) {
    std::string prompt = build_prompt(user_msg, false);

    mtmd_input_text text;
    text.text          = prompt.c_str();
    text.add_special   = true;
    text.parse_special = true;

    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    int32_t rc = mtmd_tokenize(ctx_mtmd_, chunks, &text, nullptr, 0);
    if (rc != 0) {
        std::cerr << "[tinyvlm-jetson] Tokenization failed (rc=" << rc << ")\n";
        mtmd_input_chunks_free(chunks);
        return {};
    }

    auto result = generate_from_chunks(chunks, cb);
    mtmd_input_chunks_free(chunks);
    return result;
}

// ═══════════════════════════════════════════════════════════
//  Vision + text chat (image from file)
// ═══════════════════════════════════════════════════════════

GenerationResult VLM::chat_with_image(const std::string& user_msg,
                                        const std::string& image_path,
                                        const TokenCallback& cb) {
    mtmd_bitmap* bmp = mtmd_helper_bitmap_init_from_file(ctx_mtmd_, image_path.c_str());
    if (!bmp) {
        std::cerr << "[tinyvlm-jetson] Failed to load image: " << image_path << "\n";
        return {};
    }

    std::string prompt = build_prompt(user_msg, true);

    mtmd_input_text text;
    text.text          = prompt.c_str();
    text.add_special   = true;
    text.parse_special = true;

    const mtmd_bitmap* bmp_ptr = bmp;
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    int32_t rc = mtmd_tokenize(ctx_mtmd_, chunks, &text, &bmp_ptr, 1);
    mtmd_bitmap_free(bmp);

    if (rc != 0) {
        std::cerr << "[tinyvlm-jetson] Tokenization failed (rc=" << rc << ")\n";
        mtmd_input_chunks_free(chunks);
        return {};
    }

    auto result = generate_from_chunks(chunks, cb);
    mtmd_input_chunks_free(chunks);
    return result;
}

// ═══════════════════════════════════════════════════════════
//  Vision + text chat (raw RGB pixel data)
//  Used by ROS node to pass camera frames directly
// ═══════════════════════════════════════════════════════════

GenerationResult VLM::chat_with_image_data(const std::string& user_msg,
                                            uint32_t width, uint32_t height,
                                            const unsigned char* rgb_data,
                                            const TokenCallback& cb) {
    // Create mtmd bitmap from raw RGB data (w * h * 3 bytes, RGBRGB...)
    mtmd_bitmap* bmp = mtmd_bitmap_init(width, height, rgb_data);
    if (!bmp) {
        std::cerr << "[tinyvlm-jetson] Failed to create bitmap from data ("
                  << width << "x" << height << ")\n";
        return {};
    }

    std::string prompt = build_prompt(user_msg, true);

    mtmd_input_text text;
    text.text          = prompt.c_str();
    text.add_special   = true;
    text.parse_special = true;

    const mtmd_bitmap* bmp_ptr = bmp;
    mtmd_input_chunks* chunks = mtmd_input_chunks_init();
    int32_t rc = mtmd_tokenize(ctx_mtmd_, chunks, &text, &bmp_ptr, 1);
    mtmd_bitmap_free(bmp);

    if (rc != 0) {
        std::cerr << "[tinyvlm-jetson] Tokenization failed (rc=" << rc << ")\n";
        mtmd_input_chunks_free(chunks);
        return {};
    }

    auto result = generate_from_chunks(chunks, cb);
    mtmd_input_chunks_free(chunks);
    return result;
}

// ═══════════════════════════════════════════════════════════
//  Core generation loop
// ═══════════════════════════════════════════════════════════

GenerationResult VLM::generate_from_chunks(mtmd_input_chunks* chunks,
                                             const TokenCallback& cb) {
    GenerationResult res;

    const llama_vocab* vocab = llama_model_get_vocab(model_);
    const llama_token eos = llama_vocab_eos(vocab);
    const llama_token eot = llama_vocab_eot(vocab);

    // ── Prefill: eval all chunks (text tokens + vision embeddings) ──
    auto t0 = std::chrono::high_resolution_clock::now();

    llama_pos new_n_past = 0;
    int32_t rc = mtmd_helper_eval_chunks(
        ctx_mtmd_, ctx_, chunks,
        n_past_,
        0,                // seq_id
        cfg_.n_batch,
        true,             // logits_last
        &new_n_past
    );
    if (rc != 0) {
        std::cerr << "[tinyvlm-jetson] mtmd_helper_eval_chunks failed (rc="
                  << rc << ")\n";
        return res;
    }
    res.tokens_prompt = (int)(new_n_past - n_past_);
    n_past_ = new_n_past;

    auto t1 = std::chrono::high_resolution_clock::now();
    res.time_prefill_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

    // ── Decode: autoregressive token generation ──
    llama_batch batch = llama_batch_init(1, 0, 1);

    for (int i = 0; i < cfg_.max_tokens; i++) {
        llama_token id = llama_sampler_sample(smpl_, ctx_, -1);
        llama_sampler_accept(smpl_, id);

        if (id == eos || id == eot || llama_vocab_is_eog(vocab, id)) {
            res.stopped_eos = true;
            break;
        }

        char buf[256];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n > 0) {
            std::string piece(buf, n);
            res.text += piece;
            if (cb && !cb(piece)) break;
        }

        res.tokens_generated++;

        common_batch_clear(batch);
        common_batch_add(batch, id, n_past_, {0}, true);
        n_past_++;

        if (llama_decode(ctx_, batch) != 0) {
            std::cerr << "[tinyvlm-jetson] decode error at token " << i << "\n";
            break;
        }
    }

    llama_batch_free(batch);

    auto t2 = std::chrono::high_resolution_clock::now();
    res.time_generate_ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
    if (res.time_generate_ms > 0) {
        res.tokens_per_sec = res.tokens_generated * 1000.0 / res.time_generate_ms;
    }

    return res;
}

// ═══════════════════════════════════════════════════════════
//  Reset
// ═══════════════════════════════════════════════════════════

void VLM::reset() {
    if (ctx_) llama_memory_clear(llama_get_memory(ctx_), true);
    if (smpl_) llama_sampler_reset(smpl_);
    n_past_ = 0;
}

// ═══════════════════════════════════════════════════════════
//  Stats
// ═══════════════════════════════════════════════════════════

void VLM::print_stats() const {
    if (!model_) return;
    std::cout << "[tinyvlm-jetson] Model: Qwen2.5-VL-7B-Instruct Q4_K_M\n";
    std::cout << "[tinyvlm-jetson] Size: "
              << (llama_model_size(model_) / (1024*1024)) << " MB\n";
    std::cout << "[tinyvlm-jetson] Params: "
              << (llama_model_n_params(model_) / 1000000) << " M\n";
    std::cout << "[tinyvlm-jetson] Context: " << cfg_.n_ctx << "\n";
    std::cout << "[tinyvlm-jetson] GPU layers: " << gpu_layers_ << "\n";
    std::cout << "[tinyvlm-jetson] Threads: "
              << (cfg_.n_threads > 0 ? cfg_.n_threads
                  : std::max(1, (int)std::thread::hardware_concurrency()/2))
              << "\n";
}

}  // namespace tinyvlm
