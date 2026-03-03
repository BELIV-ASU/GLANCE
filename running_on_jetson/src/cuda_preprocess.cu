/*
 * cuda_preprocess.cu — GPU image preprocessing via NPP
 *
 * Jetson AGX Orin (CUDA 12.2, sm_87, NPP 12.2).
 *
 * Pipeline: BGR8 (ROS) → upload → NPP BGR→RGB → NPP resize → download → host RGB
 *
 * On Jetson's unified memory, the upload/download are zero-copy when using
 * cudaMallocManaged or pinned memory. We use pitched device memory + explicit
 * memcpy for maximum compatibility with NPP's ROI-based API.
 */

#include "cuda_preprocess.cuh"

#include <npp.h>
#include <nppi_color_conversion.h>
#include <nppi_geometry_transforms.h>
#include <nppi_data_exchange_and_initialization.h>

#include <iostream>
#include <cstring>

namespace tinyvlm {
namespace cuda {

// ═══════════════════════════════════════════════════════════
//  GpuImage implementation
// ═══════════════════════════════════════════════════════════

bool GpuImage::alloc(int w, int h, int ch) {
    free();
    size_t pitch_out = 0;
    cudaError_t err = cudaMallocPitch(&data, &pitch_out, w * ch, h);
    if (err != cudaSuccess) {
        std::cerr << "[cuda_preprocess] cudaMallocPitch failed: "
                  << cudaGetErrorString(err) << "\n";
        data = nullptr;
        return false;
    }
    width    = w;
    height   = h;
    channels = ch;
    pitch    = pitch_out;
    return true;
}

void GpuImage::free() {
    if (data) {
        cudaFree(data);
        data = nullptr;
    }
    width = height = 0;
    pitch = 0;
}

GpuImage::GpuImage(GpuImage&& o) noexcept
    : data(o.data), width(o.width), height(o.height),
      channels(o.channels), pitch(o.pitch) {
    o.data = nullptr;
    o.width = o.height = 0;
    o.pitch = 0;
}

GpuImage& GpuImage::operator=(GpuImage&& o) noexcept {
    if (this != &o) {
        free();
        data     = o.data;
        width    = o.width;
        height   = o.height;
        channels = o.channels;
        pitch    = o.pitch;
        o.data   = nullptr;
        o.width  = o.height = 0;
        o.pitch  = 0;
    }
    return *this;
}

// ═══════════════════════════════════════════════════════════
//  Internal state (persistent GPU buffers)
// ═══════════════════════════════════════════════════════════

static GpuImage g_src_gpu;     // Source BGR on GPU
static GpuImage g_rgb_gpu;     // After BGR→RGB
static GpuImage g_dst_gpu;     // After resize (final RGB)
static bool     g_initialized = false;

bool preprocess_init(int max_src_w, int max_src_h,
                     int dst_w, int dst_h) {
    if (g_initialized) preprocess_cleanup();

    if (!g_src_gpu.alloc(max_src_w, max_src_h, 3)) return false;
    if (!g_rgb_gpu.alloc(max_src_w, max_src_h, 3)) return false;
    if (!g_dst_gpu.alloc(dst_w, dst_h, 3))         return false;

    g_initialized = true;
    std::cout << "[cuda_preprocess] Init: src buffer " << max_src_w << "x"
              << max_src_h << ", dst " << dst_w << "x" << dst_h << "\n";
    return true;
}

void preprocess_cleanup() {
    g_src_gpu.free();
    g_rgb_gpu.free();
    g_dst_gpu.free();
    g_initialized = false;
}

// ═══════════════════════════════════════════════════════════
//  BGR→RGB + Resize pipeline
// ═══════════════════════════════════════════════════════════

bool preprocess_bgr_to_rgb_resize(
    const uint8_t* src_bgr, int src_w, int src_h,
    uint8_t* dst_rgb, int dst_w, int dst_h,
    cudaStream_t stream)
{
    if (!src_bgr || !dst_rgb || src_w <= 0 || src_h <= 0 || dst_w <= 0 || dst_h <= 0) {
        std::cerr << "[cuda_preprocess] Invalid parameters\n";
        return false;
    }

    // Allocate/reallocate if needed (lazy init)
    if (!g_initialized || g_src_gpu.width < src_w || g_src_gpu.height < src_h ||
        g_dst_gpu.width != dst_w || g_dst_gpu.height != dst_h) {
        if (!preprocess_init(src_w, src_h, dst_w, dst_h)) {
            return false;
        }
    }

    // Step 1: Upload BGR8 to GPU
    cudaError_t err = cudaMemcpy2DAsync(
        g_src_gpu.data, g_src_gpu.pitch,
        src_bgr, src_w * 3,
        src_w * 3, src_h,
        cudaMemcpyHostToDevice, stream);
    if (err != cudaSuccess) {
        std::cerr << "[cuda_preprocess] Upload failed: " << cudaGetErrorString(err) << "\n";
        return false;
    }

    // Step 2: NPP BGR → RGB (swap channels 0↔2)
    NppiSize src_roi = { src_w, src_h };
    // Swap order: dst[0]=src[2], dst[1]=src[1], dst[2]=src[0]
    const int channel_order[3] = { 2, 1, 0 };

    NppStatus npp_status = nppiSwapChannels_8u_C3R(
        g_src_gpu.data, static_cast<int>(g_src_gpu.pitch),
        g_rgb_gpu.data, static_cast<int>(g_rgb_gpu.pitch),
        src_roi, channel_order);
    if (npp_status != NPP_SUCCESS) {
        std::cerr << "[cuda_preprocess] nppiSwapChannels failed: " << npp_status << "\n";
        return false;
    }

    // Step 3: NPP bilinear resize
    NppiSize src_size = { src_w, src_h };
    NppiRect src_rect = { 0, 0, src_w, src_h };
    NppiSize dst_size = { dst_w, dst_h };
    NppiRect dst_rect = { 0, 0, dst_w, dst_h };

    npp_status = nppiResize_8u_C3R(
        g_rgb_gpu.data, static_cast<int>(g_rgb_gpu.pitch), src_size, src_rect,
        g_dst_gpu.data, static_cast<int>(g_dst_gpu.pitch), dst_size, dst_rect,
        NPPI_INTER_LINEAR);
    if (npp_status != NPP_SUCCESS) {
        std::cerr << "[cuda_preprocess] nppiResize failed: " << npp_status << "\n";
        return false;
    }

    // Step 4: Download result to host
    err = cudaMemcpy2DAsync(
        dst_rgb, dst_w * 3,
        g_dst_gpu.data, g_dst_gpu.pitch,
        dst_w * 3, dst_h,
        cudaMemcpyDeviceToHost, stream);
    if (err != cudaSuccess) {
        std::cerr << "[cuda_preprocess] Download failed: " << cudaGetErrorString(err) << "\n";
        return false;
    }

    // Synchronize to ensure the output buffer is ready
    if (stream) {
        cudaStreamSynchronize(stream);
    } else {
        cudaDeviceSynchronize();
    }

    return true;
}

}  // namespace cuda
}  // namespace tinyvlm
