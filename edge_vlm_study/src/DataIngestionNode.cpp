////////////////////////////////////////////////////////////////////////////////
/// @file   DataIngestionNode.cpp
/// @brief  Implementation of the ROS 2 data ingestion node.
///
/// Project: Empirical Study of VLMs (7B) Performance on Edge
///          in Real-World Traffic Violations
///
/// Handles:
///   - Subscribing to camera image topics (multiple cameras)
///   - Subscribing to depth / point-cloud topics for localization
///   - Subscribing to semantic / text annotation topics
///   - RosBag2 replay for offline evaluation
///   - Assembling synchronized FrameBundles for downstream inference
////////////////////////////////////////////////////////////////////////////////

#include "edge_vlm_study/DataIngestionNode.hpp"

#include <algorithm>
#include <chrono>
#include <mutex>
#include <stdexcept>

// TODO: #include <rosbag2_cpp/reader.hpp>  // Uncomment when rosbag2 is linked

namespace edge_vlm_study {

// ═══════════════════════════════════════════════════════════════════════════
// Construction
// ═══════════════════════════════════════════════════════════════════════════

DataIngestionNode::DataIngestionNode(const IngestionConfig& config)
    : Node("data_ingestion_node"), config_(config)
{
    RCLCPP_INFO(this->get_logger(),
        "Initializing DataIngestionNode (cameras=%zu, depth=%s, semantic=%s)",
        config_.camera_topics.size(),
        config_.depth_topic.c_str(),
        config_.semantic_topic.c_str());

    // ── Subscribe to each camera topic ──────────────────────────────────
    for (const auto& topic : config_.camera_topics) {
        auto sub = this->create_subscription<sensor_msgs::msg::Image>(
            topic,
            rclcpp::QoS(config_.queue_depth),
            [this](sensor_msgs::msg::Image::SharedPtr msg) {
                this->on_camera_image(msg);
            });
        camera_subs_.push_back(sub);
        RCLCPP_INFO(this->get_logger(), "  Subscribed to camera topic: %s",
                     topic.c_str());
    }

    // ── Subscribe to depth topic ────────────────────────────────────────
    depth_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
        config_.depth_topic,
        rclcpp::QoS(config_.queue_depth),
        [this](sensor_msgs::msg::PointCloud2::SharedPtr msg) {
            this->on_depth_data(msg);
        });
    RCLCPP_INFO(this->get_logger(), "  Subscribed to depth topic: %s",
                 config_.depth_topic.c_str());

    // ── Subscribe to semantic text topic ────────────────────────────────
    semantic_sub_ = this->create_subscription<std_msgs::msg::String>(
        config_.semantic_topic,
        rclcpp::QoS(config_.queue_depth),
        [this](std_msgs::msg::String::SharedPtr msg) {
            this->on_semantic_text(msg);
        });
    RCLCPP_INFO(this->get_logger(), "  Subscribed to semantic topic: %s",
                 config_.semantic_topic.c_str());

    RCLCPP_INFO(this->get_logger(), "DataIngestionNode initialized.");
}

// ═══════════════════════════════════════════════════════════════════════════
// Public API
// ═══════════════════════════════════════════════════════════════════════════

void DataIngestionNode::register_frame_callback(FrameCallback cb) {
    callbacks_.push_back(std::move(cb));
    RCLCPP_INFO(this->get_logger(),
        "Registered frame callback (total=%zu)", callbacks_.size());
}

bool DataIngestionNode::start_rosbag_playback() {
    if (config_.rosbag_path.empty()) {
        RCLCPP_ERROR(this->get_logger(),
            "Cannot start rosbag playback: rosbag_path is empty.");
        return false;
    }

    RCLCPP_INFO(this->get_logger(),
        "Starting RosBag2 playback from: %s", config_.rosbag_path.c_str());

    // TODO: Implement RosBag2 playback using rosbag2_cpp::Reader
    //
    // Pseudocode:
    //   auto reader = std::make_unique<rosbag2_cpp::Reader>();
    //   reader->open(config_.rosbag_path);
    //   while (reader->has_next()) {
    //       auto msg = reader->read_next();
    //       // deserialize based on topic name
    //       // feed into on_camera_image / on_depth_data / on_semantic_text
    //   }

    RCLCPP_WARN(this->get_logger(),
        "RosBag2 playback is a STUB – implement with rosbag2_cpp::Reader.");
    return false;
}

uint64_t DataIngestionNode::frames_ingested() const noexcept {
    return frame_count_;
}

// ═══════════════════════════════════════════════════════════════════════════
// Subscription callbacks
// ═══════════════════════════════════════════════════════════════════════════

void DataIngestionNode::on_camera_image(
    const sensor_msgs::msg::Image::SharedPtr msg)
{
    // TODO: Add mutex protection for thread-safe access to latest_image_
    latest_image_ = msg;

    RCLCPP_DEBUG(this->get_logger(),
        "Received camera image: %dx%d encoding=%s",
        msg->width, msg->height, msg->encoding.c_str());

    // Attempt to assemble a complete frame bundle
    try_assemble_frame();
}

void DataIngestionNode::on_depth_data(
    const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
    // TODO: Add mutex protection
    latest_depth_ = msg;

    RCLCPP_DEBUG(this->get_logger(),
        "Received depth data: %d points, frame_id=%s",
        msg->width * msg->height, msg->header.frame_id.c_str());

    try_assemble_frame();
}

void DataIngestionNode::on_semantic_text(
    const std_msgs::msg::String::SharedPtr msg)
{
    // TODO: Add mutex protection
    latest_semantic_ = msg->data;

    RCLCPP_DEBUG(this->get_logger(),
        "Received semantic text: \"%s\"",
        msg->data.substr(0, 80).c_str());

    try_assemble_frame();
}

// ═══════════════════════════════════════════════════════════════════════════
// Frame assembly
// ═══════════════════════════════════════════════════════════════════════════

void DataIngestionNode::try_assemble_frame() {
    // We require at minimum a camera image to proceed.
    // Depth and semantic data are optional but preferred.
    if (!latest_image_) {
        return;
    }

    // TODO: Implement proper time-synchronized frame assembly.
    //       Use message_filters::TimeSynchronizer or ApproximateTimeSynchronizer
    //       to align camera + depth + semantic messages by timestamp.

    auto bundle = std::make_shared<FrameBundle>();

    // ── Populate image data ─────────────────────────────────────────────
    bundle->timestamp_ns = latest_image_->header.stamp.sec * 1'000'000'000LL
                         + latest_image_->header.stamp.nanosec;
    bundle->image_width  = static_cast<int>(latest_image_->width);
    bundle->image_height = static_cast<int>(latest_image_->height);
    bundle->image_channels = 3;  // TODO: Derive from encoding
    bundle->image_data   = latest_image_->data;

    // TODO: Resize image to config_.target_width x config_.target_height
    //       using cv::resize (via cv_bridge) if dimensions don't match.

    // ── Populate depth data ─────────────────────────────────────────────
    if (latest_depth_) {
        // TODO: Convert PointCloud2 → dense depth map (H x W float32).
        //       For now, just reserve space.
        size_t depth_size = static_cast<size_t>(
            bundle->image_width * bundle->image_height);
        bundle->depth_data.resize(depth_size, 0.0f);

        RCLCPP_DEBUG(this->get_logger(),
            "Depth data included (stub: %zu zeros)", depth_size);
    }

    // ── Populate semantic text ──────────────────────────────────────────
    bundle->semantic_text = latest_semantic_;

    // ── Dispatch to all registered callbacks ────────────────────────────
    for (const auto& cb : callbacks_) {
        cb(bundle);
    }

    ++frame_count_;

    // Clear consumed data to avoid re-dispatching the same frame
    latest_image_.reset();

    if (frame_count_ % 100 == 0) {
        RCLCPP_INFO(this->get_logger(),
            "Frames ingested: %lu", frame_count_);
    }
}

}  // namespace edge_vlm_study
