////////////////////////////////////////////////////////////////////////////////
/// @file   EdgeEvaluator.cpp
/// @brief  Implementation of the edge evaluation / benchmarking harness.
///
/// Project: Empirical Study of VLMs (7B) Performance on Edge
///          in Real-World Traffic Violations
///
/// Evaluates:
///   - Classification accuracy, precision, recall, F1, mIoU
///   - Latency statistics (mean, p50, p95, p99, max)
///   - GPU utilization, memory, power (via NVML)
///   - SNN energy efficiency (spike counts, rates)
///   - Exports results to JSON + CSV
////////////////////////////////////////////////////////////////////////////////

#include "edge_vlm_study/EdgeEvaluator.hpp"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sstream>

// TODO: #include <nvml.h>  // NVIDIA Management Library for GPU profiling

namespace edge_vlm_study {

// ═══════════════════════════════════════════════════════════════════════════
// Construction / Destruction
// ═══════════════════════════════════════════════════════════════════════════

EdgeEvaluator::EdgeEvaluator(const EvaluatorConfig& config)
    : config_(config)
{
    std::cout << "[EdgeEvaluator] Initialized.\n"
              << "  ground_truth: " << config_.ground_truth_path << "\n"
              << "  output_dir:   " << config_.output_dir << "\n"
              << "  warmup_iters: " << config_.warmup_iters << "\n"
              << "  bench_iters:  " << config_.benchmark_iters << "\n";
}

EdgeEvaluator::~EdgeEvaluator() = default;

// ═══════════════════════════════════════════════════════════════════════════
// Setup
// ═══════════════════════════════════════════════════════════════════════════

void EdgeEvaluator::set_engine(std::shared_ptr<VLMInferenceEngine> engine) {
    engine_ = std::move(engine);
    std::cout << "[EdgeEvaluator] Inference engine attached.\n";
}

void EdgeEvaluator::set_snn_runtime(std::shared_ptr<SNNRuntimeInterface> snn) {
    snn_ = std::move(snn);
    std::cout << "[EdgeEvaluator] SNN runtime attached.\n";
}

bool EdgeEvaluator::load_ground_truth() {
    if (config_.ground_truth_path.empty()) {
        std::cerr << "[EdgeEvaluator] No ground_truth_path configured.\n";
        return false;
    }

    std::ifstream ifs(config_.ground_truth_path);
    if (!ifs.is_open()) {
        std::cerr << "[EdgeEvaluator] Failed to open ground truth file: "
                  << config_.ground_truth_path << "\n";
        return false;
    }

    // TODO: Implement ground truth parsing.
    //
    // Expected format (JSON array):
    //   [
    //     {
    //       "class": "red_light_running",
    //       "bbox": [0.1, 0.2, 0.5, 0.7],
    //       "image_path": "data/frame_00001.jpg"
    //     },
    //     ...
    //   ]
    //
    // OR CSV format:
    //   image_path,class,x_min,y_min,x_max,y_max
    //   data/frame_00001.jpg,red_light_running,0.1,0.2,0.5,0.7

    std::cout << "[EdgeEvaluator] Ground truth loading is a STUB.\n";

    // Placeholder: create some dummy samples for testing the pipeline
    for (int i = 0; i < 10; ++i) {
        EvalSample sample;
        sample.gt_class = "red_light_running";
        sample.gt_bbox = {0.1f, 0.2f, 0.5f, 0.7f};
        ground_truth_samples_.push_back(sample);
    }

    std::cout << "[EdgeEvaluator] Loaded " << ground_truth_samples_.size()
              << " ground truth samples (placeholder).\n";
    return true;
}

// ═══════════════════════════════════════════════════════════════════════════
// Benchmarking
// ═══════════════════════════════════════════════════════════════════════════

EvalMetrics EdgeEvaluator::run_benchmark() {
    if (!engine_ || !engine_->is_initialized()) {
        throw std::runtime_error(
            "[EdgeEvaluator] Inference engine not set or not initialized.");
    }

    std::cout << "[EdgeEvaluator] Running full benchmark...\n";

    // ── Warm-up phase ───────────────────────────────────────────────────
    engine_->warmup(config_.warmup_iters);

    // ── Benchmark phase ─────────────────────────────────────────────────
    std::vector<double> latencies_ms;
    latencies_ms.reserve(
        static_cast<size_t>(config_.benchmark_iters));

    int correct = 0;
    int total = 0;
    float iou_sum = 0.0f;
    int iou_count = 0;

    for (int iter = 0; iter < config_.benchmark_iters; ++iter) {
        // TODO: Load actual test images from the dataset.
        //       For now, use a dummy image.
        const int w = 448, h = 448, c = 3;
        std::vector<uint8_t> dummy_image(
            static_cast<size_t>(w * h * c), 128);

        auto result = engine_->infer(dummy_image, w, h, c);

        double lat_ms = static_cast<double>(result.latency.count()) / 1000.0;
        latencies_ms.push_back(lat_ms);

        // TODO: Compare result against ground truth for accuracy.
        //       Match by sample index into ground_truth_samples_.
        if (!ground_truth_samples_.empty()) {
            size_t gt_idx = static_cast<size_t>(iter) % ground_truth_samples_.size();
            const auto& gt = ground_truth_samples_[gt_idx];

            if (result.violation_class == gt.gt_class) {
                ++correct;
            }

            float iou = compute_iou(result.bbox, gt.gt_bbox);
            iou_sum += iou;
            ++iou_count;
            ++total;
        }
    }

    // ── Compute metrics ─────────────────────────────────────────────────
    EvalMetrics metrics;
    metrics.num_samples = config_.benchmark_iters;
    metrics.model_name = "distilled_7b";  // TODO: From engine metadata

    // Accuracy
    if (total > 0) {
        metrics.accuracy = static_cast<float>(correct) / static_cast<float>(total);
        metrics.mean_iou = iou_sum / static_cast<float>(iou_count);
    }

    // TODO: Compute per-class precision, recall, F1 using confusion matrix.
    metrics.precision = metrics.accuracy;  // Placeholder
    metrics.recall = metrics.accuracy;
    metrics.f1_score = metrics.accuracy;

    // Latency statistics
    if (!latencies_ms.empty()) {
        std::sort(latencies_ms.begin(), latencies_ms.end());

        metrics.mean_latency_ms =
            std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0)
            / static_cast<double>(latencies_ms.size());

        auto percentile = [&](double p) -> double {
            size_t idx = static_cast<size_t>(
                p * static_cast<double>(latencies_ms.size() - 1));
            return latencies_ms[idx];
        };

        metrics.p50_latency_ms = percentile(0.50);
        metrics.p95_latency_ms = percentile(0.95);
        metrics.p99_latency_ms = percentile(0.99);
        metrics.max_latency_ms = latencies_ms.back();

        // FPS
        if (metrics.mean_latency_ms > 0.0) {
            metrics.fps = 1000.0 / metrics.mean_latency_ms;
        }
    }

    // GPU profiling
    if (config_.enable_gpu_profiling) {
        query_gpu_metrics(metrics);
    }

    // ── SNN evaluation (optional) ───────────────────────────────────────
    if (config_.evaluate_snn && snn_ && snn_->is_initialized()) {
        std::cout << "[EdgeEvaluator] Running SNN evaluation...\n";

        // TODO: Feed VLM hidden-state features into SNN for evaluation.
        //       For now, run a dummy SNN pass.
        std::vector<float> dummy_features(256, 0.5f);
        auto snn_result = snn_->infer(dummy_features);

        metrics.total_spikes = snn_result.total_spikes;
        metrics.avg_spike_rate = snn_result.avg_spike_rate;
    }

    std::cout << "[EdgeEvaluator] Benchmark complete.\n";
    return metrics;
}

EvalMetrics EdgeEvaluator::run_latency_benchmark() {
    if (!engine_ || !engine_->is_initialized()) {
        throw std::runtime_error(
            "[EdgeEvaluator] Inference engine not set or not initialized.");
    }

    std::cout << "[EdgeEvaluator] Running latency-only benchmark...\n";

    engine_->warmup(config_.warmup_iters);

    std::vector<double> latencies_ms;
    latencies_ms.reserve(
        static_cast<size_t>(config_.benchmark_iters));

    const int w = 448, h = 448, c = 3;
    std::vector<uint8_t> dummy_image(
        static_cast<size_t>(w * h * c), 128);

    for (int iter = 0; iter < config_.benchmark_iters; ++iter) {
        auto result = engine_->infer(dummy_image, w, h, c);
        double lat_ms = static_cast<double>(result.latency.count()) / 1000.0;
        latencies_ms.push_back(lat_ms);
    }

    EvalMetrics metrics;
    metrics.num_samples = config_.benchmark_iters;

    std::sort(latencies_ms.begin(), latencies_ms.end());
    metrics.mean_latency_ms =
        std::accumulate(latencies_ms.begin(), latencies_ms.end(), 0.0)
        / static_cast<double>(latencies_ms.size());

    auto percentile = [&](double p) -> double {
        size_t idx = static_cast<size_t>(
            p * static_cast<double>(latencies_ms.size() - 1));
        return latencies_ms[idx];
    };

    metrics.p50_latency_ms = percentile(0.50);
    metrics.p95_latency_ms = percentile(0.95);
    metrics.p99_latency_ms = percentile(0.99);
    metrics.max_latency_ms = latencies_ms.back();
    if (metrics.mean_latency_ms > 0.0) {
        metrics.fps = 1000.0 / metrics.mean_latency_ms;
    }

    return metrics;
}

// ═══════════════════════════════════════════════════════════════════════════
// Reporting
// ═══════════════════════════════════════════════════════════════════════════

void EdgeEvaluator::export_report(const EvalMetrics& metrics,
                                   const std::string& tag)
{
    // TODO: Create output directory if it doesn't exist
    //       std::filesystem::create_directories(config_.output_dir);

    std::string suffix = tag.empty() ? "" : ("_" + tag);

    // ── JSON report ─────────────────────────────────────────────────────
    {
        std::string json_path = config_.output_dir + "/eval_report"
                              + suffix + ".json";
        std::ofstream ofs(json_path);
        if (!ofs.is_open()) {
            std::cerr << "[EdgeEvaluator] Cannot write JSON to: "
                      << json_path << "\n";
            return;
        }

        ofs << std::fixed << std::setprecision(4);
        ofs << "{\n";
        ofs << "  \"model_name\": \"" << metrics.model_name << "\",\n";
        ofs << "  \"backend\": \"" << metrics.backend_name << "\",\n";
        ofs << "  \"num_samples\": " << metrics.num_samples << ",\n";
        ofs << "  \"accuracy\": " << metrics.accuracy << ",\n";
        ofs << "  \"precision\": " << metrics.precision << ",\n";
        ofs << "  \"recall\": " << metrics.recall << ",\n";
        ofs << "  \"f1_score\": " << metrics.f1_score << ",\n";
        ofs << "  \"mean_iou\": " << metrics.mean_iou << ",\n";
        ofs << "  \"mean_latency_ms\": " << metrics.mean_latency_ms << ",\n";
        ofs << "  \"p50_latency_ms\": " << metrics.p50_latency_ms << ",\n";
        ofs << "  \"p95_latency_ms\": " << metrics.p95_latency_ms << ",\n";
        ofs << "  \"p99_latency_ms\": " << metrics.p99_latency_ms << ",\n";
        ofs << "  \"max_latency_ms\": " << metrics.max_latency_ms << ",\n";
        ofs << "  \"fps\": " << metrics.fps << ",\n";
        ofs << "  \"gpu_utilization_pct\": " << metrics.gpu_utilization_pct << ",\n";
        ofs << "  \"gpu_memory_used_mb\": " << metrics.gpu_memory_used_mb << ",\n";
        ofs << "  \"gpu_power_watts\": " << metrics.gpu_power_watts << ",\n";
        ofs << "  \"total_spikes\": " << metrics.total_spikes << ",\n";
        ofs << "  \"avg_spike_rate\": " << metrics.avg_spike_rate << "\n";
        ofs << "}\n";

        std::cout << "[EdgeEvaluator] JSON report → " << json_path << "\n";
    }

    // ── CSV report (append mode for multi-model comparison) ─────────────
    {
        std::string csv_path = config_.output_dir + "/eval_results"
                             + suffix + ".csv";
        bool write_header = true;
        {
            std::ifstream check(csv_path);
            if (check.good()) write_header = false;
        }

        std::ofstream ofs(csv_path, std::ios::app);
        if (!ofs.is_open()) {
            std::cerr << "[EdgeEvaluator] Cannot write CSV to: "
                      << csv_path << "\n";
            return;
        }

        if (write_header) {
            ofs << "model,backend,samples,accuracy,precision,recall,f1,"
                << "mean_iou,mean_lat_ms,p50_lat_ms,p95_lat_ms,p99_lat_ms,"
                << "max_lat_ms,fps,gpu_util_pct,gpu_mem_mb,gpu_power_w,"
                << "total_spikes,avg_spike_rate\n";
        }

        ofs << std::fixed << std::setprecision(4);
        ofs << metrics.model_name << ","
            << metrics.backend_name << ","
            << metrics.num_samples << ","
            << metrics.accuracy << ","
            << metrics.precision << ","
            << metrics.recall << ","
            << metrics.f1_score << ","
            << metrics.mean_iou << ","
            << metrics.mean_latency_ms << ","
            << metrics.p50_latency_ms << ","
            << metrics.p95_latency_ms << ","
            << metrics.p99_latency_ms << ","
            << metrics.max_latency_ms << ","
            << metrics.fps << ","
            << metrics.gpu_utilization_pct << ","
            << metrics.gpu_memory_used_mb << ","
            << metrics.gpu_power_watts << ","
            << metrics.total_spikes << ","
            << metrics.avg_spike_rate << "\n";

        std::cout << "[EdgeEvaluator] CSV report → " << csv_path << "\n";
    }
}

void EdgeEvaluator::print_summary(const EvalMetrics& metrics) {
    std::cout << "\n"
              << "╔══════════════════════════════════════════════════════════╗\n"
              << "║           Edge VLM Evaluation Summary                   ║\n"
              << "╠══════════════════════════════════════════════════════════╣\n";
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "║ Model:        " << std::setw(40) << metrics.model_name << " ║\n";
    std::cout << "║ Backend:      " << std::setw(40) << metrics.backend_name << " ║\n";
    std::cout << "║ Samples:      " << std::setw(40) << metrics.num_samples << " ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Accuracy:     " << std::setw(40) << metrics.accuracy << " ║\n";
    std::cout << "║ Precision:    " << std::setw(40) << metrics.precision << " ║\n";
    std::cout << "║ Recall:       " << std::setw(40) << metrics.recall << " ║\n";
    std::cout << "║ F1 Score:     " << std::setw(40) << metrics.f1_score << " ║\n";
    std::cout << "║ Mean IoU:     " << std::setw(40) << metrics.mean_iou << " ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Mean Lat (ms):" << std::setw(40) << metrics.mean_latency_ms << " ║\n";
    std::cout << "║ P50  Lat (ms):" << std::setw(40) << metrics.p50_latency_ms << " ║\n";
    std::cout << "║ P95  Lat (ms):" << std::setw(40) << metrics.p95_latency_ms << " ║\n";
    std::cout << "║ P99  Lat (ms):" << std::setw(40) << metrics.p99_latency_ms << " ║\n";
    std::cout << "║ FPS:          " << std::setw(40) << metrics.fps << " ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
    std::cout << "║ GPU Util (%): " << std::setw(40) << metrics.gpu_utilization_pct << " ║\n";
    std::cout << "║ GPU Mem (MB): " << std::setw(40) << metrics.gpu_memory_used_mb << " ║\n";
    std::cout << "║ GPU Power (W):" << std::setw(40) << metrics.gpu_power_watts << " ║\n";
    std::cout << "╠══════════════════════════════════════════════════════════╣\n";
    std::cout << "║ Total Spikes: " << std::setw(40) << metrics.total_spikes << " ║\n";
    std::cout << "║ Avg Spike Rt: " << std::setw(40) << metrics.avg_spike_rate << " ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════╝\n"
              << std::endl;
}

// ═══════════════════════════════════════════════════════════════════════════
// Internal helpers
// ═══════════════════════════════════════════════════════════════════════════

float EdgeEvaluator::compute_iou(const std::vector<float>& a,
                                  const std::vector<float>& b) const {
    if (a.size() < 4 || b.size() < 4) {
        return 0.0f;
    }

    // Boxes: [x_min, y_min, x_max, y_max]
    float x_min = std::max(a[0], b[0]);
    float y_min = std::max(a[1], b[1]);
    float x_max = std::min(a[2], b[2]);
    float y_max = std::min(a[3], b[3]);

    float inter_width  = std::max(0.0f, x_max - x_min);
    float inter_height = std::max(0.0f, y_max - y_min);
    float inter_area   = inter_width * inter_height;

    float area_a = (a[2] - a[0]) * (a[3] - a[1]);
    float area_b = (b[2] - b[0]) * (b[3] - b[1]);
    float union_area = area_a + area_b - inter_area;

    if (union_area <= 0.0f) {
        return 0.0f;
    }

    return inter_area / union_area;
}

void EdgeEvaluator::query_gpu_metrics(EvalMetrics& metrics) const {
    // TODO: Implement NVML-based GPU profiling.
    //
    // Pseudocode:
    //   nvmlInit();
    //   nvmlDevice_t device;
    //   nvmlDeviceGetHandleByIndex(0, &device);
    //
    //   nvmlUtilization_t utilization;
    //   nvmlDeviceGetUtilizationRates(device, &utilization);
    //   metrics.gpu_utilization_pct = static_cast<float>(utilization.gpu);
    //
    //   nvmlMemory_t mem_info;
    //   nvmlDeviceGetMemoryInfo(device, &mem_info);
    //   metrics.gpu_memory_used_mb = mem_info.used / (1024 * 1024);
    //
    //   unsigned int power_mw;
    //   nvmlDeviceGetPowerUsage(device, &power_mw);
    //   metrics.gpu_power_watts = static_cast<float>(power_mw) / 1000.0f;
    //
    //   nvmlShutdown();

    std::cout << "[EdgeEvaluator] GPU profiling STUB – "
              << "implement with NVML for actual metrics.\n";

    metrics.gpu_utilization_pct = 0.0f;
    metrics.gpu_memory_used_mb = 0;
    metrics.gpu_power_watts = 0.0f;
}

}  // namespace edge_vlm_study
