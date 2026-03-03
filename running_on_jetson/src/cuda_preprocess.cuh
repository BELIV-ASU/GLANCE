/*
 * cuda_preprocess.cuh — GPU image preprocessing for Jetson AGX Orin
 *
 * Uses NVIDIA NPP (Performance Primitives) for:
 *   - BGR → RGB color conversion (GPU)
 *   - Bilinear resize (GPU)
 *
 * This avoids CPU round-trips for image preprocessing between
 * the ROS camera topic and the VLM vision encoder.
 *
 * On Jetson's unified memory architecture, CUDA managed memory
 * means the "upload/download" is essentially free (same physical RAM).
 * We still use NPP for the actual compute (color swap + resize).
 */

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace tinyvlm {
namespace cuda {

// ─── GPU Image Buffer ──────────────────────────────────────
// Manages a CUDA-allocated image buffer. On Jetson unified memory,
// this is accessible from both CPU and GPU without explicit copies.
struct GpuImage {
    uint8_t* data   = nullptr;
    int      width  = 0;
    int      height = 0;
    int      channels = 3;    // Always RGB or BGR (3 channels)
    size_t   pitch  = 0;      // Row stride in bytes

    size_t byte_size() const { return pitch * height; }
    bool   valid()     const { return data != nullptr && width > 0 && height > 0; }

    // Allocate pitched GPU memory
    bool alloc(int w, int h, int ch = 3);
    void free();

    ~GpuImage() { free(); }

    // Non-copyable, moveable
    GpuImage() = default;
    GpuImage(const GpuImage&) = delete;
    GpuImage& operator=(const GpuImage&) = delete;
    GpuImage(GpuImage&& o) noexcept;
    GpuImage& operator=(GpuImage&& o) noexcept;
};

// ─── Preprocessing Pipeline ────────────────────────────────
// All-in-one: BGR→RGB + bilinear resize on GPU via NPP.
//
// src_bgr:    Host pointer to BGR8 image (from ROS sensor_msgs::Image)
// src_w/h:    Source dimensions
// dst_rgb:    Output host buffer (must be >= dst_w * dst_h * 3)
// dst_w/h:    Target dimensions
// stream:     Optional CUDA stream (nullptr = default stream)
//
// Returns true on success.
bool preprocess_bgr_to_rgb_resize(
    const uint8_t* src_bgr, int src_w, int src_h,
    uint8_t* dst_rgb, int dst_w, int dst_h,
    cudaStream_t stream = nullptr);

// Initialize/cleanup the preprocessing pipeline
// Call once at startup/shutdown. Allocates internal GPU buffers.
bool preprocess_init(int max_src_w, int max_src_h,
                     int dst_w, int dst_h);
void preprocess_cleanup();

}  // namespace cuda
}  // namespace tinyvlm
