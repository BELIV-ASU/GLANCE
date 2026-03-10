////////////////////////////////////////////////////////////////////////////////
/// @file   EdgeEvaluator.hpp
/// @brief  Benchmarking and evaluation harness for edge-deployed VLMs.
///
/// Project: Empirical Study of VLMs (7B) Performance on Edge
///          in Real-World Traffic Violations
///
/// Measures:
///   - End-to-end latency (ingestion → inference → result)
///   - GPU utilization and memory consumption
///   - Throughput (frames per second)
///   - Semantic accuracy (against ground-truth annotations)
///   - Power consumption (via NVML or external power monitors)
///   - SNN spike efficiency metrics
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <fstream>
#include <optional>

#include "edge_vlm_study/VLMInferenceEngine.hpp"
#include "edge_vlm_study/SNNRuntimeInterface.hpp"

namespace edge_vlm_study {

/// ──────────────────────────────────────────────────────────────────────────
/// Single evaluation sample (ground truth + prediction)
/// ──────────────────────────────────────────────────────────────────────────
struct EvalSample {
    /// Ground-truth violation class
    std::string gt_class;

    /// Predicted violation class (from InferenceResult)
    std::string pred_class;

    /// Ground-truth bounding box [x_min, y_min, x_max, y_max]
    std::vector<float> gt_bbox;

    /// Predicted bounding box
    std::vector<float> pred_bbox;

    /// Prediction confidence
    float confidence{0.0f};

    /// Inference latency for this sample
    std::chrono::microseconds latency{0};
};

/// ──────────────────────────────────────────────────────────────────────────
/// Aggregated evaluation metrics
/// ──────────────────────────────────────────────────────────────────────────
struct EvalMetrics {
    // ── Accuracy ────────────────────────────────────────────────────────
    float accuracy{0.0f};            ///< Top-1 classification accuracy
    float precision{0.0f};           ///< Weighted precision
    float recall{0.0f};              ///< Weighted recall
    float f1_score{0.0f};            ///< Weighted F1
    float mean_iou{0.0f};            ///< Mean IoU for bounding boxes

    // ── Latency ─────────────────────────────────────────────────────────
    double mean_latency_ms{0.0};     ///< Average latency (ms)
    double p50_latency_ms{0.0};      ///< Median latency
    double p95_latency_ms{0.0};      ///< 95th percentile
    double p99_latency_ms{0.0};      ///< 99th percentile
    double max_latency_ms{0.0};      ///< Worst-case latency

    // ── Throughput ──────────────────────────────────────────────────────
    double fps{0.0};                 ///< Frames per second

    // ── GPU ─────────────────────────────────────────────────────────────
    float gpu_utilization_pct{0.0f}; ///< Average GPU utilization %
    size_t gpu_memory_used_mb{0};    ///< Peak GPU memory (MB)
    float gpu_power_watts{0.0f};     ///< Average power draw (W)

    // ── SNN-specific ────────────────────────────────────────────────────
    uint64_t total_spikes{0};        ///< Aggregate spike count
    float avg_spike_rate{0.0f};      ///< Average spike rate

    // ── Model metadata ──────────────────────────────────────────────────
    std::string model_name;
    std::string backend_name;
    int num_samples{0};
};

/// ──────────────────────────────────────────────────────────────────────────
/// Evaluator configuration
/// ──────────────────────────────────────────────────────────────────────────
struct EvaluatorConfig {
    /// Path to ground-truth annotation file (JSON / CSV)
    std::string ground_truth_path{};

    /// Directory to write evaluation reports
    std::string output_dir{"./eval_results"};

    /// Number of warm-up iterations before benchmarking
    int warmup_iters{10};

    /// Number of benchmark iterations
    int benchmark_iters{100};

    /// Whether to query GPU metrics via NVML
    bool enable_gpu_profiling{true};

    /// IoU threshold for correct detection
    float iou_threshold{0.5f};

    /// Whether to also evaluate SNN path
    bool evaluate_snn{false};
};

/// ──────────────────────────────────────────────────────────────────────────
/// @class EdgeEvaluator
/// @brief Orchestrates benchmarking of VLM inference on edge hardware.
///
/// Usage:
///   1. Construct with config.
///   2. Call set_engine() to attach the VLMInferenceEngine to evaluate.
///   3. Optionally set_snn_runtime() for SNN evaluation.
///   4. Call run_benchmark() to execute.
///   5. Call export_report() to persist results.
/// ──────────────────────────────────────────────────────────────────────────
class EdgeEvaluator {
public:
    explicit EdgeEvaluator(const EvaluatorConfig& config);
    ~EdgeEvaluator();

    /// @brief Attach the inference engine to evaluate.
    void set_engine(std::shared_ptr<VLMInferenceEngine> engine);

    /// @brief Optionally attach an SNN runtime for evaluation.
    void set_snn_runtime(std::shared_ptr<SNNRuntimeInterface> snn);

    /// @brief Load ground-truth annotations from file.
    /// @return true on success.
    bool load_ground_truth();

    /// @brief Run the full benchmark suite.
    /// @return Aggregated metrics.
    [[nodiscard]] EvalMetrics run_benchmark();

    /// @brief Run a quick latency-only benchmark (no accuracy computation).
    /// @return Latency metrics only.
    [[nodiscard]] EvalMetrics run_latency_benchmark();

    /// @brief Export evaluation report to disk (JSON + CSV).
    /// @param metrics  The metrics to export.
    /// @param tag      Optional tag appended to filenames.
    void export_report(const EvalMetrics& metrics,
                       const std::string& tag = "");

    /// @brief Print a human-readable summary to stdout.
    static void print_summary(const EvalMetrics& metrics);

private:
    // ── Internal helpers ────────────────────────────────────────────────
    float compute_iou(const std::vector<float>& a,
                      const std::vector<float>& b) const;
    void query_gpu_metrics(EvalMetrics& metrics) const;

    // TODO: Implement NVML GPU profiling (nvmlInit, nvmlDeviceGetUtilizationRates, etc.)
    // TODO: Implement power measurement integration

    // ── State ───────────────────────────────────────────────────────────
    EvaluatorConfig config_;
    std::shared_ptr<VLMInferenceEngine> engine_;
    std::shared_ptr<SNNRuntimeInterface> snn_;
    std::vector<EvalSample> ground_truth_samples_;
};

}  // namespace edge_vlm_study
