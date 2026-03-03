/*
 * ros2_node.cpp — ROS2 real-time inference node for Jetson AGX Orin
 *
 * Subscribes to /sensing/camera/camera0/image_rect_color (sensor_msgs/Image),
 * runs Qwen2.5-VL-7B inference via llama.cpp, and displays results in an
 * OpenCV window with the camera feed + model response overlay.
 *
 * Architecture (3 threads for low latency):
 *   1. ROS callback  → receives frames, skips N, signals preprocess thread
 *   2. Preprocess thread → CUDA NPP BGR→RGB+resize, double-buffers, signals inference
 *   3. Inference thread  → VLM inference on latest preprocessed frame, no idle wait
 *
 * The display runs on a wall timer at ~30 FPS, always showing the latest
 * camera frame with the most recent model response overlaid.
 *
 * Usage (after colcon build):
 *   ros2 run tinyvlm_jetson ros2_inference_node --ros-args \
 *       -p model_path:=/path/to/qwen2.5-vl-7b-Q4_K_M.gguf \
 *       -p mmproj_path:=/path/to/qwen2.5-vl-7b-mmproj-f16.gguf \
 *       -p skip_frames:=3 -p max_tokens:=128
 */

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>

#include "vlm.h"
#include "cuda_preprocess.cuh"

#include <mutex>
#include <thread>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <sstream>
#include <iomanip>
#include <vector>
#include <string>
#include <cstring>

using namespace std::chrono_literals;

// ═══════════════════════════════════════════════════════════
//  Overlay helper: wrap + render text on frame
// ═══════════════════════════════════════════════════════════

struct OverlayStats {
    std::string response_text;
    std::string inference_stats;  // tok/s, prefill, gen timings
    bool        inference_running = false;
    int         frame_count      = 0;
    double      display_fps      = 0.0;
    double      camera_fps       = 0.0;
    int         inference_count   = 0;
    int         skip_frames       = 3;
    double      last_preprocess_ms = 0.0;
    bool        recording         = false;
};

static void draw_text_overlay(cv::Mat& frame, const OverlayStats& s) {
    // Semi-transparent background bar at bottom
    int bar_height = std::min(frame.rows / 3, 300);
    cv::Mat overlay = frame.clone();
    cv::rectangle(overlay,
                  cv::Point(0, frame.rows - bar_height),
                  cv::Point(frame.cols, frame.rows),
                  cv::Scalar(20, 20, 20), cv::FILLED);
    cv::addWeighted(overlay, 0.75, frame, 0.25, 0, frame);

    // ── Top bar: status + HUD stats ──
    // Status indicator (left)
    cv::Scalar status_color = s.inference_running
        ? cv::Scalar(0, 165, 255)   // Orange = processing
        : cv::Scalar(0, 200, 80);   // Green = ready
    std::string status_text = s.inference_running ? "ANALYZING..." : "READY";
    cv::circle(frame, cv::Point(25, 30), 10, status_color, cv::FILLED);
    cv::putText(frame, status_text, cv::Point(45, 37),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2);

    // Recording indicator (red dot blinking)
    if (s.recording) {
        bool blink = (s.frame_count / 15) % 2 == 0;  // blink at ~1 Hz
        if (blink) {
            cv::circle(frame, cv::Point(frame.cols - 30, 30), 10,
                       cv::Scalar(0, 0, 220), cv::FILLED);
        }
        cv::putText(frame, "REC", cv::Point(frame.cols - 80, 37),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 220), 2);
    }

    // FPS + frame stats (top center-right)
    {
        std::ostringstream hud;
        hud << std::fixed << std::setprecision(1)
            << "Display: " << s.display_fps << " FPS | "
            << "Camera: " << s.camera_fps << " FPS | "
            << "Frame #" << s.frame_count << " | "
            << "Skip: 1/" << s.skip_frames;
        cv::putText(frame, hud.str(), cv::Point(200, 37),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(200, 200, 200), 1);
    }

    // Inference count + preprocess latency (second row)
    {
        std::ostringstream row2;
        row2 << "Inferences: " << s.inference_count
             << " | Preprocess: " << std::fixed << std::setprecision(1)
             << s.last_preprocess_ms << " ms";
        cv::putText(frame, row2.str(), cv::Point(25, 62),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(170, 170, 180), 1);
    }

    // ── Bottom overlay: title + inference stats + response ──
    cv::putText(frame, "SafetyVLM - Jetson AGX Orin [RT]",
                cv::Point(10, frame.rows - bar_height + 25),
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(80, 200, 120), 2);

    if (!s.inference_stats.empty()) {
        cv::putText(frame, s.inference_stats,
                    cv::Point(10, frame.rows - bar_height + 50),
                    cv::FONT_HERSHEY_SIMPLEX, 0.45, cv::Scalar(180, 180, 190), 1);
    }

    // Word-wrap the model response text
    if (!s.response_text.empty()) {
        int max_chars_per_line = frame.cols / 11;
        std::vector<std::string> lines;
        std::istringstream iss(s.response_text);
        std::string word, current_line;

        while (iss >> word) {
            if (current_line.empty()) {
                current_line = word;
            } else if ((int)(current_line.size() + 1 + word.size()) <= max_chars_per_line) {
                current_line += " " + word;
            } else {
                lines.push_back(current_line);
                current_line = word;
            }
        }
        if (!current_line.empty()) lines.push_back(current_line);

        int y_start = frame.rows - bar_height + 72;
        int line_height = 22;
        int max_lines = (bar_height - 80) / line_height;

        for (int i = 0; i < std::min((int)lines.size(), max_lines); i++) {
            cv::putText(frame, lines[i],
                        cv::Point(15, y_start + i * line_height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.48,
                        cv::Scalar(230, 230, 230), 1);
        }
        if ((int)lines.size() > max_lines) {
            cv::putText(frame, "...",
                        cv::Point(15, y_start + max_lines * line_height),
                        cv::FONT_HERSHEY_SIMPLEX, 0.48,
                        cv::Scalar(180, 180, 180), 1);
        }
    }
}

// ═══════════════════════════════════════════════════════════
//  ROS2 Node — Real-time 3-thread architecture
// ═══════════════════════════════════════════════════════════

class InferenceNode : public rclcpp::Node {
public:
    InferenceNode() : Node("safetyvlm_inference") {
        // ── Declare parameters ──
        this->declare_parameter<std::string>("model_path", "");
        this->declare_parameter<std::string>("mmproj_path", "");
        this->declare_parameter<std::string>("prompt",
            "Briefly identify traffic signs, markings, signals in this image. "
            "State the key rule and correct driver action in 2-3 sentences.");
        this->declare_parameter<int>("max_tokens", 128);
        this->declare_parameter<int>("target_width", 480);
        this->declare_parameter<int>("target_height", 320);
        this->declare_parameter<int>("skip_frames", 3);
        this->declare_parameter<std::string>("topic",
            "/sensing/camera/camera0/image_rect_color");
        this->declare_parameter<bool>("use_cuda_preprocess", true);
        this->declare_parameter<bool>("record", true);
        this->declare_parameter<std::string>("record_path", "/home/beliv/GLANCE/recordings");
        this->declare_parameter<double>("record_fps", 20.0);

        // ── Read parameters ──
        std::string model_path  = this->get_parameter("model_path").as_string();
        std::string mmproj_path = this->get_parameter("mmproj_path").as_string();
        prompt_       = this->get_parameter("prompt").as_string();
        target_w_     = this->get_parameter("target_width").as_int();
        target_h_     = this->get_parameter("target_height").as_int();
        skip_frames_  = this->get_parameter("skip_frames").as_int();
        use_cuda_pp_  = this->get_parameter("use_cuda_preprocess").as_bool();
        record_        = this->get_parameter("record").as_bool();
        record_path_   = this->get_parameter("record_path").as_string();
        record_fps_    = this->get_parameter("record_fps").as_double();
        std::string topic = this->get_parameter("topic").as_string();
        int max_tokens = this->get_parameter("max_tokens").as_int();

        if (model_path.empty() || mmproj_path.empty()) {
            RCLCPP_FATAL(this->get_logger(),
                "model_path and mmproj_path parameters are required!");
            rclcpp::shutdown();
            return;
        }

        // ── Initialize VLM ──
        RCLCPP_INFO(this->get_logger(), "Loading VLM model...");
        auto cfg = tinyvlm::VLMConfig::for_jetson_orin(model_path, mmproj_path);
        cfg.max_tokens = max_tokens;
        vlm_ = std::make_unique<tinyvlm::VLM>(cfg);

        if (!vlm_->load()) {
            RCLCPP_FATAL(this->get_logger(), "Failed to load VLM model!");
            rclcpp::shutdown();
            return;
        }

        // ── Initialize CUDA preprocessing ──
        if (use_cuda_pp_) {
            if (!tinyvlm::cuda::preprocess_init(1920, 1080, target_w_, target_h_)) {
                RCLCPP_WARN(this->get_logger(),
                    "CUDA preprocessing init failed, falling back to CPU");
                use_cuda_pp_ = false;
            }
        }

        // Allocate double-buffered RGB output
        rgb_buf_a_.resize(target_w_ * target_h_ * 3);
        rgb_buf_b_.resize(target_w_ * target_h_ * 3);
        preprocess_out_ = rgb_buf_a_.data();
        inference_in_    = rgb_buf_b_.data();

        // ── Subscribe to camera topic ──
        sub_ = this->create_subscription<sensor_msgs::msg::Image>(
            topic, rclcpp::SensorDataQoS(),
            std::bind(&InferenceNode::image_callback, this, std::placeholders::_1));

        RCLCPP_INFO(this->get_logger(), "Subscribed to: %s", topic.c_str());
        RCLCPP_INFO(this->get_logger(), "Skip frames: %d (analyze every %d-th)",
                     skip_frames_, skip_frames_);
        RCLCPP_INFO(this->get_logger(), "Target: %dx%d | max_tokens: %d",
                     target_w_, target_h_, max_tokens);
        RCLCPP_INFO(this->get_logger(), "CUDA preprocess: %s",
                     use_cuda_pp_ ? "ON" : "OFF");
        RCLCPP_INFO(this->get_logger(), "Recording: %s",
                     record_ ? "ON" : "OFF");

        // ── Start worker threads ──
        preprocess_thread_ = std::thread(&InferenceNode::preprocess_loop, this);
        inference_thread_  = std::thread(&InferenceNode::inference_loop, this);

        // ── OpenCV display timer (30 FPS) ──
        display_timer_ = this->create_wall_timer(
            33ms, std::bind(&InferenceNode::display_callback, this));

        RCLCPP_INFO(this->get_logger(), "Node ready (3-thread RT mode). Waiting for images...");
    }

    ~InferenceNode() override {
        shutdown_ = true;
        preprocess_cv_.notify_all();
        inference_cv_.notify_all();
        if (preprocess_thread_.joinable()) preprocess_thread_.join();
        if (inference_thread_.joinable())  inference_thread_.join();

        // Finalize video recording
        if (video_writer_.isOpened()) {
            video_writer_.release();
            RCLCPP_INFO(this->get_logger(), "Recording saved: %s",
                        video_file_path_.c_str());
        }

        tinyvlm::cuda::preprocess_cleanup();
        cv::destroyAllWindows();
    }

private:
    // ─────────────────────────────────────────────────────
    //  Thread 1 (ROS callback): receive frames, skip N
    // ─────────────────────────────────────────────────────
    void image_callback(const sensor_msgs::msg::Image::ConstSharedPtr& msg) {
        frame_counter_++;

        try {
            cv_bridge::CvImageConstPtr cv_ptr;
            if (msg->encoding == "bgr8") {
                cv_ptr = cv_bridge::toCvShare(msg, "bgr8");
            } else if (msg->encoding == "rgb8") {
                cv_ptr = cv_bridge::toCvShare(msg, "rgb8");
            } else {
                cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
            }

            // Always update display frame (full framerate)
            {
                std::lock_guard<std::mutex> lock(display_mutex_);
                display_frame_ = cv_ptr->image.clone();
                has_display_frame_ = true;
            }

            // Only send every N-th frame for preprocessing
            if (frame_counter_ % skip_frames_ != 0) return;

            // Don't queue if preprocess is still busy
            if (preprocess_busy_) return;

            {
                std::lock_guard<std::mutex> lock(raw_mutex_);
                raw_frame_ = cv_ptr->image.clone();
                raw_encoding_ = msg->encoding;
                has_raw_frame_ = true;
            }
            preprocess_cv_.notify_one();

        } catch (const cv_bridge::Exception& e) {
            RCLCPP_ERROR_THROTTLE(this->get_logger(), *this->get_clock(), 5000,
                "cv_bridge error: %s", e.what());
        }
    }

    // ─────────────────────────────────────────────────────
    //  Thread 2: CUDA preprocessing (BGR→RGB + resize)
    // ─────────────────────────────────────────────────────
    void preprocess_loop() {
        RCLCPP_INFO(this->get_logger(), "Preprocess thread started");

        while (!shutdown_ && rclcpp::ok()) {
            // Wait for a raw frame
            {
                std::unique_lock<std::mutex> lock(raw_mutex_);
                preprocess_cv_.wait(lock, [this] {
                    return has_raw_frame_ || shutdown_;
                });
                if (shutdown_) break;
            }

            preprocess_busy_ = true;

            cv::Mat frame;
            std::string encoding;
            {
                std::lock_guard<std::mutex> lock(raw_mutex_);
                frame = std::move(raw_frame_);
                encoding = raw_encoding_;
                has_raw_frame_ = false;
            }

            if (frame.empty()) {
                preprocess_busy_ = false;
                continue;
            }

            auto t0 = std::chrono::high_resolution_clock::now();

            // Preprocess into the output buffer
            bool ok = false;
            if (use_cuda_pp_ && encoding == "bgr8") {
                ok = tinyvlm::cuda::preprocess_bgr_to_rgb_resize(
                    frame.data, frame.cols, frame.rows,
                    preprocess_out_, target_w_, target_h_);
            }

            if (!ok) {
                // CPU fallback
                cv::Mat resized;
                cv::resize(frame, resized, cv::Size(target_w_, target_h_),
                           0, 0, cv::INTER_LINEAR);
                cv::Mat rgb;
                if (encoding == "rgb8") {
                    rgb = resized;
                } else {
                    cv::cvtColor(resized, rgb, cv::COLOR_BGR2RGB);
                }
                if (rgb.isContinuous()) {
                    std::memcpy(preprocess_out_, rgb.data,
                                target_w_ * target_h_ * 3);
                } else {
                    for (int r = 0; r < target_h_; r++) {
                        std::memcpy(preprocess_out_ + r * target_w_ * 3,
                                    rgb.ptr(r), target_w_ * 3);
                    }
                }
            }

            auto t1 = std::chrono::high_resolution_clock::now();
            last_preprocess_ms_ = std::chrono::duration<double, std::milli>(t1 - t0).count();

            // Swap buffers: preprocess_out_ becomes inference_in_
            {
                std::lock_guard<std::mutex> lock(swap_mutex_);
                std::swap(preprocess_out_, inference_in_);
                has_preprocessed_ = true;
            }
            inference_cv_.notify_one();

            preprocess_busy_ = false;
        }

        RCLCPP_INFO(this->get_logger(), "Preprocess thread stopped");
    }

    // ─────────────────────────────────────────────────────
    //  Thread 3: VLM inference (runs continuously)
    // ─────────────────────────────────────────────────────
    void inference_loop() {
        RCLCPP_INFO(this->get_logger(), "Inference thread started");

        while (!shutdown_ && rclcpp::ok()) {
            // Wait for preprocessed frame
            {
                std::unique_lock<std::mutex> lock(swap_mutex_);
                inference_cv_.wait(lock, [this] {
                    return has_preprocessed_ || shutdown_;
                });
                if (shutdown_) break;
                has_preprocessed_ = false;
            }

            {
                std::lock_guard<std::mutex> lock(result_mutex_);
                inference_running_ = true;
            }

            auto t_start = std::chrono::high_resolution_clock::now();

            // Run VLM on the inference buffer
            vlm_->reset();

            // Read pointer under lock (swap_mutex_ protects inference_in_)
            uint8_t* data_ptr;
            {
                std::lock_guard<std::mutex> lock(swap_mutex_);
                data_ptr = inference_in_;
            }

            auto result = vlm_->chat_with_image_data(
                prompt_,
                static_cast<uint32_t>(target_w_),
                static_cast<uint32_t>(target_h_),
                data_ptr);

            auto t_end = std::chrono::high_resolution_clock::now();
            double total_ms = std::chrono::duration<double, std::milli>(
                t_end - t_start).count();

            // Update results for display
            {
                std::lock_guard<std::mutex> lock(result_mutex_);
                last_response_ = result.text;
                inference_running_ = false;
                inference_count_++;

                std::ostringstream ss;
                ss << "#" << inference_count_ << " | "
                   << result.tokens_generated << " tok | "
                   << std::fixed << std::setprecision(1)
                   << result.tokens_per_sec << " tok/s | "
                   << "pre " << (int)last_preprocess_ms_ << "ms | "
                   << "pfill " << (int)result.time_prefill_ms << "ms | "
                   << "gen " << (int)result.time_generate_ms << "ms | "
                   << "total " << (int)total_ms << "ms";
                last_stats_ = ss.str();
            }

            RCLCPP_INFO(this->get_logger(),
                "[#%d] %d tok, %.1f tok/s, %.0fms (pre:%.0f pfill:%.0f gen:%.0f)",
                inference_count_.load(), result.tokens_generated,
                result.tokens_per_sec, total_ms,
                last_preprocess_ms_.load(),
                result.time_prefill_ms, result.time_generate_ms);
        }

        RCLCPP_INFO(this->get_logger(), "Inference thread stopped");
    }

    // ─────────────────────────────────────────────────────
    //  Display callback (~30 FPS on main thread)
    // ─────────────────────────────────────────────────────
    void display_callback() {
        cv::Mat display;
        {
            std::lock_guard<std::mutex> lock(display_mutex_);
            if (!has_display_frame_) return;
            display = display_frame_.clone();
        }

        // Compute display FPS
        display_frame_count_++;
        auto now = std::chrono::steady_clock::now();
        double elapsed_sec = std::chrono::duration<double>(
            now - display_fps_start_).count();
        if (elapsed_sec >= 1.0) {
            display_fps_ = display_frame_count_ / elapsed_sec;
            display_frame_count_ = 0;
            display_fps_start_ = now;
        }

        // Compute camera FPS (frames received per second)
        int cur_frame = frame_counter_.load();
        double cam_elapsed = std::chrono::duration<double>(
            now - camera_fps_start_).count();
        if (cam_elapsed >= 1.0) {
            camera_fps_ = (cur_frame - camera_fps_last_count_) / cam_elapsed;
            camera_fps_last_count_ = cur_frame;
            camera_fps_start_ = now;
        }

        // Build overlay stats struct
        OverlayStats os;
        {
            std::lock_guard<std::mutex> lock(result_mutex_);
            os.response_text    = last_response_;
            os.inference_stats  = last_stats_;
            os.inference_running = inference_running_;
        }
        os.frame_count       = frame_counter_;
        os.display_fps       = display_fps_;
        os.camera_fps        = camera_fps_;
        os.inference_count   = inference_count_;
        os.skip_frames       = skip_frames_;
        os.last_preprocess_ms = last_preprocess_ms_;
        os.recording         = record_ && video_writer_.isOpened();

        draw_text_overlay(display, os);

        // ── Record to MP4 ──
        if (record_) {
            if (!video_writer_.isOpened()) {
                init_video_writer(display.cols, display.rows);
            }
            if (video_writer_.isOpened()) {
                video_writer_.write(display);
            }
        }

        cv::imshow("SafetyVLM - Jetson AGX Orin [RT]", display);
        int key = cv::waitKey(1);
        if (key == 27 || key == 'q') {
            RCLCPP_INFO(this->get_logger(), "Quit requested");
            rclcpp::shutdown();
        }
    }

    // ─────────────────────────────────────────────────────
    //  Video writer initialization
    // ─────────────────────────────────────────────────────
    void init_video_writer(int width, int height) {
        // Create output directory
        std::string mkdir_cmd = "mkdir -p " + record_path_;
        (void)system(mkdir_cmd.c_str());

        // Generate timestamped filename
        auto now_t = std::chrono::system_clock::now();
        auto time_t_now = std::chrono::system_clock::to_time_t(now_t);
        std::tm tm_buf;
        localtime_r(&time_t_now, &tm_buf);

        std::ostringstream fname;
        fname << record_path_ << "/safetyvlm_"
              << std::put_time(&tm_buf, "%Y%m%d_%H%M%S") << ".mp4";
        video_file_path_ = fname.str();

        // Try hardware-accelerated H.264 first (Jetson NVENC), then software fallback
        int fourcc = cv::VideoWriter::fourcc('a', 'v', 'c', '1');  // H.264
        video_writer_.open(video_file_path_, fourcc, record_fps_,
                           cv::Size(width, height), true);

        if (!video_writer_.isOpened()) {
            // Fallback: XVID in AVI container
            video_file_path_ = video_file_path_.substr(
                0, video_file_path_.size() - 4) + ".avi";
            fourcc = cv::VideoWriter::fourcc('X', 'V', 'I', 'D');
            video_writer_.open(video_file_path_, fourcc, record_fps_,
                               cv::Size(width, height), true);
        }

        if (video_writer_.isOpened()) {
            RCLCPP_INFO(this->get_logger(), "Recording to: %s (%dx%d @ %.0f FPS)",
                        video_file_path_.c_str(), width, height, record_fps_);
        } else {
            RCLCPP_ERROR(this->get_logger(),
                "Failed to open video writer! Recording disabled.");
            record_ = false;
        }
    }

    // ── Members ──────────────────────────────────────────
    std::unique_ptr<tinyvlm::VLM> vlm_;

    rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr sub_;
    rclcpp::TimerBase::SharedPtr display_timer_;

    // Display frame (full framerate)
    std::mutex display_mutex_;
    cv::Mat    display_frame_;
    bool       has_display_frame_ = false;

    // Raw frame for preprocessing (only every N-th)
    std::mutex              raw_mutex_;
    std::condition_variable preprocess_cv_;
    cv::Mat                 raw_frame_;
    std::string             raw_encoding_ = "bgr8";
    bool                    has_raw_frame_ = false;

    // Double-buffered preprocessed RGB data
    std::mutex              swap_mutex_;
    std::condition_variable inference_cv_;
    std::vector<uint8_t>    rgb_buf_a_;
    std::vector<uint8_t>    rgb_buf_b_;
    uint8_t*                preprocess_out_ = nullptr;  // being written by preprocess
    uint8_t*                inference_in_   = nullptr;   // being read by inference
    bool                    has_preprocessed_ = false;

    // Inference results (read by display)
    std::mutex  result_mutex_;
    std::string last_response_;
    std::string last_stats_;
    bool        inference_running_ = false;

    // Worker threads
    std::thread        preprocess_thread_;
    std::thread        inference_thread_;
    std::atomic<bool>  shutdown_{false};
    std::atomic<bool>  preprocess_busy_{false};

    // Counters
    std::atomic<int>    frame_counter_{0};
    std::atomic<int>    inference_count_{0};
    std::atomic<double> last_preprocess_ms_{0.0};

    // Display FPS tracking
    int    display_frame_count_ = 0;
    double display_fps_ = 0.0;
    std::chrono::steady_clock::time_point display_fps_start_ =
        std::chrono::steady_clock::now();

    // Camera FPS tracking
    double camera_fps_ = 0.0;
    int    camera_fps_last_count_ = 0;
    std::chrono::steady_clock::time_point camera_fps_start_ =
        std::chrono::steady_clock::now();

    // Video recording
    cv::VideoWriter video_writer_;
    std::string     video_file_path_;
    bool            record_ = true;
    std::string     record_path_;
    double          record_fps_ = 20.0;

    // Config
    std::string prompt_;
    int         target_w_ = 480;
    int         target_h_ = 320;
    int         skip_frames_ = 3;
    bool        use_cuda_pp_ = true;
};

// ═══════════════════════════════════════════════════════════
//  main
// ═══════════════════════════════════════════════════════════

int main(int argc, char** argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<InferenceNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
