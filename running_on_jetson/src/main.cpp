/*
 * main.cpp — tinyvlm: Qwen2.5-VL-7B-Instruct on GTX 1650
 *
 * Usage:
 *   tinyvlm --model model.gguf --mmproj mmproj.gguf
 *   tinyvlm --model model.gguf --mmproj mmproj.gguf --image photo.jpg "What is this?"
 *   tinyvlm --model model.gguf --mmproj mmproj.gguf --interactive
 */

#include "vlm.h"

#include <iostream>
#include <string>
#include <cstring>

static void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [options] [prompt]\n\n"
              << "Options:\n"
              << "  --model PATH       Path to Qwen2.5-VL GGUF model (required)\n"
              << "  --mmproj PATH      Path to mmproj GGUF (required)\n"
              << "  --image PATH       Image file for vision queries\n"
              << "  --gpu-layers N     Layers to offload to GPU (-1=auto, 0=CPU only)\n"
              << "  --ctx N            Context size (default: 4096)\n"
              << "  --temp FLOAT       Temperature (default: 0.7)\n"
              << "  --max-tokens N     Max tokens to generate (default: 512)\n"
              << "  --threads N        CPU threads (default: auto)\n"
              << "  --no-mmap          Disable memory mapping\n"
              << "  --interactive      Interactive chat mode\n"
              << "  -h, --help         Show this help\n";
}

int main(int argc, char** argv) {
    tinyvlm::VLMConfig cfg;
    std::string image_path;
    std::string prompt;
    bool interactive = false;

    // ── Parse args ──
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg == "--model" && i+1 < argc) {
            cfg.model_path = argv[++i];
        }
        else if (arg == "--mmproj" && i+1 < argc) {
            cfg.mmproj_path = argv[++i];
        }
        else if (arg == "--image" && i+1 < argc) {
            image_path = argv[++i];
        }
        else if (arg == "--gpu-layers" && i+1 < argc) {
            cfg.n_gpu_layers = std::atoi(argv[++i]);
        }
        else if (arg == "--ctx" && i+1 < argc) {
            cfg.n_ctx = std::atoi(argv[++i]);
        }
        else if (arg == "--temp" && i+1 < argc) {
            cfg.temperature = std::atof(argv[++i]);
        }
        else if (arg == "--max-tokens" && i+1 < argc) {
            cfg.max_tokens = std::atoi(argv[++i]);
        }
        else if (arg == "--threads" && i+1 < argc) {
            cfg.n_threads = std::atoi(argv[++i]);
        }
        else if (arg == "--no-mmap") {
            cfg.use_mmap = false;
        }
        else if (arg == "--interactive") {
            interactive = true;
        }
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

    // Default GPU layers for GTX 1650 if not specified
    if (cfg.n_gpu_layers < 0) {
        cfg = tinyvlm::VLMConfig::for_gtx1650(cfg.model_path, cfg.mmproj_path);
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

    // ── Interactive mode ──
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
                current_image.clear();  // Use image once
            } else {
                res = vlm.chat(line, stream);
            }

            std::cout << "\n[" << res.tokens_generated << " tokens, "
                      << res.tokens_per_sec << " t/s, "
                      << "prefill " << res.time_prefill_ms << " ms]\n\n";

            vlm.reset();  // Reset for next turn
        }
        return 0;
    }

    // ── Single-shot mode ──
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

    return 0;
}
