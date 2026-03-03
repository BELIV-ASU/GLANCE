/*
 * main.cpp — tinyvlm: Qwen2.5-VL-7B-Instruct on RTX 5000 Ada (30GB)
 *
 * Usage:
 *   tinyvlm --model model.gguf --mmproj mmproj.gguf
 *   tinyvlm --model model.gguf --mmproj mmproj.gguf --image photo.jpg "What is this?"
 *   tinyvlm --model model.gguf --mmproj mmproj.gguf --interactive
 *   tinyvlm --model model.gguf --mmproj mmproj.gguf --batch images/ "Describe traffic rules"
 */

#include "vlm.h"

#include <iostream>
#include <string>
#include <cstring>
#include <filesystem>
#include <vector>
#include <algorithm>
#include <fstream>

namespace fs = std::filesystem;

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options] [prompt]\n\n"
              << "Options:\n"
              << "  --model PATH       Path to Qwen2.5-VL GGUF model (required)\n"
              << "  --mmproj PATH      Path to mmproj GGUF (required)\n"
              << "  --image PATH       Image file for vision queries\n"
              << "  --batch DIR        Process all images in directory\n"
              << "  --gpu-layers N     Layers to offload to GPU (-1=auto, 0=CPU only)\n"
              << "  --ctx N            Context size (default: 8192)\n"
              << "  --temp FLOAT       Temperature (default: 0.7)\n"
              << "  --max-tokens N     Max tokens to generate (default: 512)\n"
              << "  --threads N        CPU threads (default: auto)\n"
              << "  --no-mmap          Disable memory mapping\n"
              << "  --vision-cpu       Run vision encoder on CPU (saves ~2GB VRAM)\n"
              << "  --low-vram         Optimize for <5GB VRAM (vision on CPU + ctx 4096)\n"
              << "  --interactive      Interactive chat mode\n"
              << "  --json PATH        Save results as JSON to PATH\n"
              << "  -h, --help         Show this help\n";
}

static bool is_image_file(const fs::path& p) {
    static const std::vector<std::string> exts = {
        ".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"
    };
    auto ext = p.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
    return std::find(exts.begin(), exts.end(), ext) != exts.end();
}

// Simple JSON escaping
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 16);
    for (char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:   out += c;
        }
    }
    return out;
}

int main(int argc, char** argv) {
    tinyvlm::VLMConfig cfg;
    std::string image_path;
    std::string batch_dir;
    std::string json_path;
    std::string prompt;
    bool interactive = false;

    // ── Parse args ──
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg == "--model" && i+1 < argc)      cfg.model_path = argv[++i];
        else if (arg == "--mmproj" && i+1 < argc)     cfg.mmproj_path = argv[++i];
        else if (arg == "--image" && i+1 < argc)      image_path = argv[++i];
        else if (arg == "--batch" && i+1 < argc)      batch_dir = argv[++i];
        else if (arg == "--gpu-layers" && i+1 < argc)  cfg.n_gpu_layers = std::atoi(argv[++i]);
        else if (arg == "--ctx" && i+1 < argc)         cfg.n_ctx = std::atoi(argv[++i]);
        else if (arg == "--temp" && i+1 < argc)        cfg.temperature = (float)std::atof(argv[++i]);
        else if (arg == "--max-tokens" && i+1 < argc)  cfg.max_tokens = std::atoi(argv[++i]);
        else if (arg == "--threads" && i+1 < argc)     cfg.n_threads = std::atoi(argv[++i]);
        else if (arg == "--json" && i+1 < argc)        json_path = argv[++i];
        else if (arg == "--no-mmap")                   cfg.use_mmap = false;
        else if (arg == "--vision-cpu")                 cfg.vision_cpu = true;
        else if (arg == "--low-vram") {
            cfg.vision_cpu = true;
            cfg.n_gpu_layers = 20;  // ~20 layers on GPU, rest on CPU
            cfg.n_ctx = 2048;
            cfg.n_batch = 1024;
        }
        else if (arg == "--interactive")               interactive = true;
        else if (arg[0] != '-') {
            if (!prompt.empty()) prompt += " ";
            prompt += arg;
        }
        else {
            std::cerr << "Unknown option: " << arg << "\n";
            print_usage(argv[0]);
            return 1;
        }
    }

    if (cfg.model_path.empty() || cfg.mmproj_path.empty()) {
        std::cerr << "Error: --model and --mmproj are required\n\n";
        print_usage(argv[0]);
        return 1;
    }

    // Default GPU layers for RTX 5000 Ada if not specified
    if (cfg.n_gpu_layers < 0) {
        auto preset = tinyvlm::VLMConfig::for_rtx5000(cfg.model_path, cfg.mmproj_path);
        cfg.n_gpu_layers = preset.n_gpu_layers;
        if (cfg.n_threads == 0) cfg.n_threads = preset.n_threads;
    }

    // ── Load model ──
    tinyvlm::VLM vlm(cfg);
    if (!vlm.load()) {
        std::cerr << "Failed to load model\n";
        return 1;
    }

    // ── Streaming callback ──
    auto stream = [](const std::string& piece) -> bool {
        std::cout << piece << std::flush;
        return true;
    };

    // ═══════════════════════════════════════════════════════
    //  Batch mode: process all images in a directory
    // ═══════════════════════════════════════════════════════
    if (!batch_dir.empty()) {
        if (prompt.empty()) {
            prompt = "Look at this image carefully. Identify any traffic signs, "
                     "road markings, signals or driving-relevant elements visible. Then:\n"
                     "1. Describe what you see.\n"
                     "2. Explain the traffic rule(s) that apply.\n"
                     "3. State the correct driver behaviour.\n"
                     "Be concise but thorough.";
        }

        std::vector<fs::path> images;
        for (auto& entry : fs::directory_iterator(batch_dir)) {
            if (entry.is_regular_file() && is_image_file(entry.path()))
                images.push_back(entry.path());
        }
        std::sort(images.begin(), images.end());

        if (images.empty()) {
            std::cerr << "No images found in " << batch_dir << "\n";
            return 1;
        }

        std::cout << "\n[tinyvlm] Batch mode: " << images.size()
                  << " images in " << batch_dir << "\n";

        // JSON output buffer
        std::string json_buf = "[\n";

        for (size_t idx = 0; idx < images.size(); idx++) {
            vlm.reset();
            auto& img = images[idx];
            std::cout << "\n======================================================================\n"
                      << "  Image " << (idx+1) << "/" << images.size()
                      << ":  " << img.filename().string() << "\n"
                      << "======================================================================\n\n";

            auto res = vlm.chat_with_image(prompt, img.string(), stream);
            std::cout << "\n\n  [" << res.tokens_generated << " tokens, "
                      << res.tokens_per_sec << " t/s, "
                      << "prefill " << res.time_prefill_ms << " ms]\n";

            // Append to JSON
            if (idx > 0) json_buf += ",\n";
            json_buf += "  {\n";
            json_buf += "    \"image\": \"" + json_escape(img.filename().string()) + "\",\n";
            json_buf += "    \"image_path\": \"" + json_escape(img.string()) + "\",\n";
            json_buf += "    \"prompt\": \"" + json_escape(prompt) + "\",\n";
            json_buf += "    \"model_response\": \"" + json_escape(res.text) + "\",\n";
            json_buf += "    \"tokens_generated\": " + std::to_string(res.tokens_generated) + ",\n";
            json_buf += "    \"tokens_per_sec\": " + std::to_string(res.tokens_per_sec) + ",\n";
            json_buf += "    \"prefill_ms\": " + std::to_string(res.time_prefill_ms) + ",\n";
            json_buf += "    \"generate_ms\": " + std::to_string(res.time_generate_ms) + "\n";
            json_buf += "  }";
        }

        json_buf += "\n]\n";

        // Save JSON
        std::string out_path = json_path.empty()
            ? (fs::path(batch_dir).parent_path() / "batch_results.json").string()
            : json_path;
        std::ofstream ofs(out_path);
        if (ofs) {
            ofs << json_buf;
            std::cout << "\n[tinyvlm] Results saved to " << out_path << "\n";
        }

        return 0;
    }

    // ═══════════════════════════════════════════════════════
    //  Interactive mode
    // ═══════════════════════════════════════════════════════
    if (interactive) {
        std::cout << "\n=== tinyvlm interactive (Qwen2.5-VL-7B) ===\n"
                  << "Commands: /image <path> — set image for next query\n"
                  << "          /reset       — clear context\n"
                  << "          /quit        — exit\n\n";

        std::string current_image;

        while (true) {
            std::cout << "> " << std::flush;
            std::string line;
            if (!std::getline(std::cin, line) || line == "/quit") break;

            if (line.empty()) continue;

            if (line.substr(0, 7) == "/image ") {
                current_image = line.substr(7);
                std::cout << "[image set: " << current_image << "]\n";
                continue;
            }
            if (line == "/reset") {
                vlm.reset();
                current_image.clear();
                std::cout << "[context cleared]\n";
                continue;
            }

            tinyvlm::GenerationResult res;
            if (!current_image.empty()) {
                res = vlm.chat_with_image(line, current_image, stream);
                current_image.clear();
            } else {
                res = vlm.chat(line, stream);
            }

            std::cout << "\n[" << res.tokens_generated << " tokens, "
                      << res.tokens_per_sec << " t/s, "
                      << "prefill " << res.time_prefill_ms << " ms]\n\n";

            vlm.reset();
        }
        return 0;
    }

    // ═══════════════════════════════════════════════════════
    //  Single-shot mode
    // ═══════════════════════════════════════════════════════
    if (prompt.empty()) {
        std::cerr << "No prompt provided. Use --interactive or provide a prompt.\n";
        return 1;
    }

    tinyvlm::GenerationResult res;
    if (!image_path.empty()) {
        res = vlm.chat_with_image(prompt, image_path, stream);
    } else {
        res = vlm.chat(prompt, stream);
    }

    std::cout << "\n\n--- Stats ---\n"
              << "Prompt tokens: " << res.tokens_prompt << "\n"
              << "Generated: " << res.tokens_generated << "\n"
              << "Prefill: " << res.time_prefill_ms << " ms\n"
              << "Generate: " << res.time_generate_ms << " ms\n"
              << "Speed: " << res.tokens_per_sec << " tokens/sec\n";

    // Save single result as JSON if requested
    if (!json_path.empty()) {
        std::ofstream ofs(json_path);
        if (ofs) {
            ofs << "{\n"
                << "  \"prompt\": \"" << json_escape(prompt) << "\",\n"
                << "  \"image\": \"" << json_escape(image_path) << "\",\n"
                << "  \"model_response\": \"" << json_escape(res.text) << "\",\n"
                << "  \"tokens_generated\": " << res.tokens_generated << ",\n"
                << "  \"tokens_per_sec\": " << res.tokens_per_sec << "\n"
                << "}\n";
            std::cout << "[tinyvlm] Results saved to " << json_path << "\n";
        }
    }

    return 0;
}
