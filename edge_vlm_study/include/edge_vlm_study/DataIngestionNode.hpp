////////////////////////////////////////////////////////////////////////////////
/// @file   DataIngestionNode.hpp
/// @brief  ROS 2 node for real-time data ingestion from RosBags, camera
///         feeds, depth maps, and text/semantic inputs.
///
/// Project: Empirical Study of VLMs (7B) Performance on Edge
///          in Real-World Traffic Violations
///
/// This node subscribes to live camera topics, depth-map topics, and can
/// replay from RosBag2 files. It publishes preprocessed tensors to
/// downstream inference nodes.
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <functional>
#include <optional>

// ROS 2
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <std_msgs/msg/string.hpp>

namespace edge_vlm_study {

/// ──────────────────────────────────────────────────────────────────────────
/// Configuration for data ingestion sources
/// ──────────────────────────────────────────────────────────────────────────
struct IngestionConfig {
    /// Path to RosBag2 directory (empty string = live mode)
    std::string rosbag_path{};

    /// Camera image topic names
    std::vector<std::string> camera_topics{"/camera/front/image_raw",
                                            "/camera/rear/image_raw"};

    /// Depth map / point cloud topic
    std::string depth_topic{"/depth/points"};

    /// Text / semantic annotation topic
    std::string semantic_topic{"/semantic/annotations"};

    /// Target image dimensions for preprocessing
    int target_width{640};
    int target_height{480};

    /// Maximum queue depth for subscriptions
    int queue_depth{10};
};

/// ──────────────────────────────────────────────────────────────────────────
/// Preprocessed frame bundle passed downstream
/// ──────────────────────────────────────────────────────────────────────────
struct FrameBundle {
    /// Timestamp of the frame (nanoseconds, ROS clock)
    int64_t timestamp_ns{0};

    /// Raw camera image bytes (H x W x C, uint8)
    std::vector<uint8_t> image_data;
    int image_width{0};
    int image_height{0};
    int image_channels{3};

    /// Depth map (H x W, float32 in meters)
    std::vector<float> depth_data;

    /// Optional semantic / text annotation
    std::optional<std::string> semantic_text;
};

/// ──────────────────────────────────────────────────────────────────────────
/// Callback type for downstream consumers of FrameBundles
/// ──────────────────────────────────────────────────────────────────────────
using FrameCallback = std::function<void(std::shared_ptr<FrameBundle>)>;

/// ──────────────────────────────────────────────────────────────────────────
/// @class DataIngestionNode
/// @brief ROS 2 node handling camera, depth, and semantic data ingestion.
///
/// Responsibilities:
///   1. Subscribe to live topics OR replay from RosBag2.
///   2. Synchronize multi-modal inputs (camera + depth + text).
///   3. Preprocess images (resize, normalize) for downstream inference.
///   4. Expose a callback mechanism so VLMInferenceEngine can consume frames.
/// ──────────────────────────────────────────────────────────────────────────
class DataIngestionNode : public rclcpp::Node {
public:
    /// @brief Construct the ingestion node with the given config.
    /// @param config  Ingestion configuration (topics, rosbag path, etc.)
    explicit DataIngestionNode(
        const IngestionConfig& config = IngestionConfig{});

    /// @brief Register a callback invoked whenever a new FrameBundle is ready.
    /// @param cb  Callback receiving a shared_ptr to the assembled FrameBundle.
    void register_frame_callback(FrameCallback cb);

    /// @brief Begin playback from a RosBag2 file (non-live mode).
    /// @return true if playback started successfully.
    bool start_rosbag_playback();

    /// @brief Get total frames ingested since node start.
    [[nodiscard]] uint64_t frames_ingested() const noexcept;

private:
    // ── Internal helpers ────────────────────────────────────────────────
    void on_camera_image(const sensor_msgs::msg::Image::SharedPtr msg);
    void on_depth_data(const sensor_msgs::msg::PointCloud2::SharedPtr msg);
    void on_semantic_text(const std_msgs::msg::String::SharedPtr msg);
    void try_assemble_frame();

    // TODO: Add time-synchronization logic (e.g., message_filters::Synchronizer)

    // ── State ───────────────────────────────────────────────────────────
    IngestionConfig config_;
    std::vector<FrameCallback> callbacks_;

    // ROS 2 subscriptions
    std::vector<rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr>
        camera_subs_;
    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr
        depth_sub_;
    rclcpp::Subscription<std_msgs::msg::String>::SharedPtr
        semantic_sub_;

    // Latest received data (guarded by internal synchronization)
    std::shared_ptr<sensor_msgs::msg::Image> latest_image_;
    std::shared_ptr<sensor_msgs::msg::PointCloud2> latest_depth_;
    std::optional<std::string> latest_semantic_;

    uint64_t frame_count_{0};
};

}  // namespace edge_vlm_study
