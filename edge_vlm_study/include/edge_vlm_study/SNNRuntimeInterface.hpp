////////////////////////////////////////////////////////////////////////////////
/// @file   SNNRuntimeInterface.hpp
/// @brief  Interface and stubs for Spiking Neural Network (SNN) execution
///         on edge hardware.
///
/// Project: Empirical Study of VLMs (7B) Performance on Edge
///          in Real-World Traffic Violations
///
/// This module provides a C++ interface for:
///   - Encoding conventional tensors into spike trains
///   - Executing SNN layers on neuromorphic / GPU hardware
///   - Decoding spike trains back to conventional output tensors
///
/// The actual SNN kernels will be plugged in later; this file defines the
/// contract and placeholder implementations.
////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include <chrono>
#include <functional>

namespace edge_vlm_study {

/// ──────────────────────────────────────────────────────────────────────────
/// SNN backend targets
/// ──────────────────────────────────────────────────────────────────────────
enum class SNNBackend {
    GPU_CUDA,       ///< Spike simulation on CUDA GPU
    NEUROMORPHIC,   ///< Intel Loihi / BrainChip Akida / SpiNNaker
    CPU_REFERENCE   ///< Pure CPU reference implementation (slow, for tests)
};

/// ──────────────────────────────────────────────────────────────────────────
/// Spike encoding schemes
/// ──────────────────────────────────────────────────────────────────────────
enum class SpikeEncoding {
    RATE_CODING,        ///< Probability of spike ∝ input magnitude
    TEMPORAL_CODING,    ///< First-spike-time ∝ 1/input magnitude
    LATENCY_CODING,     ///< Latency from reference spike
    DELTA_MODULATION    ///< Spike on significant change
};

/// ──────────────────────────────────────────────────────────────────────────
/// Configuration for the SNN runtime
/// ──────────────────────────────────────────────────────────────────────────
struct SNNConfig {
    /// Target backend hardware
    SNNBackend backend{SNNBackend::GPU_CUDA};

    /// Encoding method for converting analog → spikes
    SpikeEncoding encoding{SpikeEncoding::RATE_CODING};

    /// Number of simulation time steps per inference pass
    int num_timesteps{100};

    /// Membrane potential threshold for LIF neurons
    float threshold{1.0f};

    /// Membrane time constant (tau) in ms
    float tau_mem{10.0f};

    /// Synaptic time constant in ms
    float tau_syn{5.0f};

    /// Path to pre-trained SNN weight file (e.g., from snnTorch export)
    std::string weight_path{};

    /// GPU device ID (for CUDA backend)
    int device_id{0};
};

/// ──────────────────────────────────────────────────────────────────────────
/// Spike train container
/// ──────────────────────────────────────────────────────────────────────────
struct SpikeTrain {
    /// Shape: [timesteps, neurons]
    std::vector<uint8_t> spikes;  // 0 or 1 per neuron per timestep
    int num_timesteps{0};
    int num_neurons{0};
};

/// ──────────────────────────────────────────────────────────────────────────
/// SNN inference result
/// ──────────────────────────────────────────────────────────────────────────
struct SNNResult {
    /// Decoded output tensor (class logits or feature vector)
    std::vector<float> output;

    /// Total spikes fired across all layers (energy proxy)
    uint64_t total_spikes{0};

    /// Inference (simulation) latency
    std::chrono::microseconds latency{0};

    /// Average spike rate across output neurons
    float avg_spike_rate{0.0f};
};

/// ──────────────────────────────────────────────────────────────────────────
/// @class SNNRuntimeInterface
/// @brief Abstract-ish interface for executing SNN models on edge hardware.
///
/// Lifecycle:
///   1. Construct with config.
///   2. Call initialize() to allocate resources / load weights.
///   3. encode() → run() → decode()  per inference pass.
///   4. Destructor releases resources.
///
/// TODO: When neuromorphic hardware SDK is available, create a derived class
///       (e.g., LoihiSNNRuntime) that overrides the virtual methods.
/// ──────────────────────────────────────────────────────────────────────────
class SNNRuntimeInterface {
public:
    explicit SNNRuntimeInterface(const SNNConfig& config);
    virtual ~SNNRuntimeInterface();

    // Non-copyable
    SNNRuntimeInterface(const SNNRuntimeInterface&) = delete;
    SNNRuntimeInterface& operator=(const SNNRuntimeInterface&) = delete;

    /// @brief Initialize the runtime (allocate spike buffers, load weights).
    /// @return true on success.
    virtual bool initialize();

    /// @brief Encode a floating-point tensor into a spike train.
    /// @param input  Analog input tensor (e.g., VLM hidden-state features).
    /// @return Encoded spike train.
    [[nodiscard]] virtual SpikeTrain encode(const std::vector<float>& input);

    /// @brief Run the SNN simulation on the given spike train.
    /// @param input_spikes  Encoded input spike train.
    /// @return Output spike train after simulation.
    [[nodiscard]] virtual SpikeTrain run(const SpikeTrain& input_spikes);

    /// @brief Decode an output spike train back into an analog tensor.
    /// @param output_spikes  SNN output spike train.
    /// @return Decoded floating-point tensor.
    [[nodiscard]] virtual std::vector<float> decode(
        const SpikeTrain& output_spikes);

    /// @brief Convenience: encode → run → decode in one call.
    /// @param input  Analog input features.
    /// @return SNNResult with decoded output, spike counts, latency.
    [[nodiscard]] SNNResult infer(const std::vector<float>& input);

    /// @brief Query initialization state.
    [[nodiscard]] bool is_initialized() const noexcept;

    /// @brief Get configured backend.
    [[nodiscard]] SNNBackend backend() const noexcept;

protected:
    SNNConfig config_;
    bool initialized_{false};

    // TODO: Add backend-specific handles (e.g., CUDA streams, Loihi contexts)
};

}  // namespace edge_vlm_study
