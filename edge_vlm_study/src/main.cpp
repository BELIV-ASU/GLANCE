////////////////////////////////////////////////////////////////////////////////
/// @file   main.cpp
/// @brief  Entry point for the Edge VLM deployment node.
///
/// Project: Empirical Study of VLMs (7B) Performance on Edge
///          in Real-World Traffic Violations
///
/// This executable:
///   1. Initializes the ROS 2 runtime.
///   2. Creates the DataIngestionNode for camera/depth/semantic feeds.
///   3. Creates the VLMInferenceEngine with the chosen backend and model.
///   4. Optionally initializes the SNNRuntimeInterface.
///   5. Wires ingestion callbacks to inference.
///   6. Optionally runs the EdgeEvaluator benchmark.
///   7. Spins the ROS 2 event loop.
////////////////////////////////////////////////////////////////////////////////

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>

#include "edge_vlm_study/DataIngestionNode.hpp"
#include "edge_vlm_study/VLMInferenceEngine.hpp"
#include "edge_vlm_study/SNNRuntimeInterface.hpp"
#include "edge_vlm_study/EdgeEvaluator.hpp"

using namespace edge_vlm_study;

int main(int argc, char* argv[]) {
    // ── Initialize ROS 2 ────────────────────────────────────────────────
    rclcpp::init(argc, argv);

    std::cout << "╔══════════════════════════════════════════════════════════╗\n"
              << "║   Edge VLM 7B – Traffic Violation Detection Node        ║\n"
              << "║   Empirical Study on Edge Deployment                    ║\n"
              << "╚══════════════════════════════════════════════════════════╝\n\n";

    // ── Configure data ingestion ────────────────────────────────────────
    IngestionConfig ingest_cfg;
    // TODO: Load from YAML config file or ROS 2 parameters
    ingest_cfg.camera_topics = {"/camera/front/image_raw"};
    ingest_cfg.depth_topic   = "/depth/points";
    ingest_cfg.semantic_topic = "/semantic/annotations";
    ingest_cfg.target_width  = 448;
    ingest_cfg.target_height = 448;

    auto ingestion_node = std::make_shared<DataIngestionNode>(ingest_cfg);

    // ── Configure VLM inference engine ──────────────────────────────────
    InferenceEngineConfig engine_cfg;
    // TODO: Load from YAML config or command-line args
    engine_cfg.backend       = InferenceBackend::TENSORRT;
    engine_cfg.model_path    = "models/qwen2_5_7b_distilled.engine";
    engine_cfg.model_id      = ModelID::QWEN2_5_7B;
    engine_cfg.device_id     = 0;
    engine_cfg.max_batch_size = 1;
    engine_cfg.fp16          = true;

    auto inference_engine = std::make_shared<VLMInferenceEngine>(engine_cfg);
    if (!inference_engine->initialize()) {
        RCLCPP_ERROR(ingestion_node->get_logger(),
            "Failed to initialize inference engine!");
        rclcpp::shutdown();
        return EXIT_FAILURE;
    }

    // ── Configure SNN runtime (optional) ────────────────────────────────
    std::shared_ptr<SNNRuntimeInterface> snn_runtime;
#ifdef ENABLE_SNN
    SNNConfig snn_cfg;
    snn_cfg.backend       = SNNBackend::GPU_CUDA;
    snn_cfg.encoding      = SpikeEncoding::RATE_CODING;
    snn_cfg.num_timesteps = 100;
    snn_cfg.weight_path   = "models/snn_traffic_classifier.pt";

    snn_runtime = std::make_shared<SNNRuntimeInterface>(snn_cfg);
    if (!snn_runtime->initialize()) {
        RCLCPP_WARN(ingestion_node->get_logger(),
            "SNN runtime failed to initialize – continuing without SNN.");
        snn_runtime.reset();
    }
#endif

    // ── Wire ingestion → inference ──────────────────────────────────────
    ingestion_node->register_frame_callback(
        [&inference_engine, &ingestion_node, &snn_runtime](
            std::shared_ptr<FrameBundle> frame)
        {
            // Run VLM inference on the ingested frame
            auto result = inference_engine->infer(
                frame->image_data,
                frame->image_width,
                frame->image_height,
                frame->image_channels,
                frame->semantic_text);

            RCLCPP_INFO(ingestion_node->get_logger(),
                "Frame %ld → class=%s conf=%.3f latency=%ld µs",
                frame->timestamp_ns,
                result.violation_class.c_str(),
                result.confidence,
                result.latency.count());

            // TODO: Publish result to a ROS 2 topic for downstream consumers
            // TODO: Optionally run SNN pass on VLM hidden features
            // TODO: Log to evaluation database
        });

    // ── Optionally run offline benchmark ────────────────────────────────
    // TODO: Gate this behind a command-line flag (e.g., --benchmark)
    bool run_benchmark = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--benchmark") {
            run_benchmark = true;
        }
    }

    if (run_benchmark) {
        EvaluatorConfig eval_cfg;
        eval_cfg.ground_truth_path = "data/ground_truth.json";
        eval_cfg.output_dir        = "eval_results";
        eval_cfg.warmup_iters      = 10;
        eval_cfg.benchmark_iters   = 100;
        eval_cfg.evaluate_snn      = (snn_runtime != nullptr);

        EdgeEvaluator evaluator(eval_cfg);
        evaluator.set_engine(inference_engine);
        if (snn_runtime) {
            evaluator.set_snn_runtime(snn_runtime);
        }

        auto metrics = evaluator.run_benchmark();
        EdgeEvaluator::print_summary(metrics);
        evaluator.export_report(metrics, "benchmark_run");

        rclcpp::shutdown();
        return EXIT_SUCCESS;
    }

    // ── Spin the ROS 2 event loop ───────────────────────────────────────
    RCLCPP_INFO(ingestion_node->get_logger(),
        "Edge VLM node is running. Waiting for data...");
    rclcpp::spin(ingestion_node);

    rclcpp::shutdown();
    return EXIT_SUCCESS;
}
