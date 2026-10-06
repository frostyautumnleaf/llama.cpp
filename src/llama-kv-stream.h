// llama-kv-stream.h - KV streaming from RAM with resident window (Strata's --kv-resident)
//
// When KV streaming is enabled, the full KV cache is stored in pinned CPU memory
// and only a sliding window of N entries is kept resident in VRAM. Before each
// attention operation, the needed KV entries are streamed from CPU to VRAM.
//
// This allows much longer contexts on GPUs with limited VRAM.

#pragma once

#include "llama-kv-cache.h"

#include <memory>
#include <vector>

// KV streaming state for a single layer
struct llama_kv_stream_layer {
    // Resident window in VRAM (sliding window of most recent entries)
    ggml_tensor * k_resident = nullptr;
    ggml_tensor * v_resident = nullptr;
    ggml_backend_buffer_t k_buf = nullptr;
    ggml_backend_buffer_t v_buf = nullptr;

    // Window position
    uint32_t window_start = 0;
    uint32_t window_size = 0;
};

// KV streaming state for all layers
class llama_kv_stream {
public:
    llama_kv_stream() = default;
    ~llama_kv_stream();

    llama_kv_stream(const llama_kv_stream&) = delete;
    llama_kv_stream& operator=(const llama_kv_stream&) = delete;

    // Initialize KV streaming
    // kv: the KV cache (must be in CPU memory)
    // window_size: number of entries to keep resident in VRAM
    // dev: GPU device for the resident window
    bool init(llama_kv_cache * kv, uint32_t window_size, ggml_backend_dev_t dev);

    void close();

    bool valid() const { return layers_.size() > 0; }
    uint32_t window_size() const { return window_size_; }

    // Stream the needed KV entries from CPU to VRAM before attention
    // il: layer index
    // n_kv: number of KV entries to stream (most recent n_kv entries)
    void stream_layer(int32_t il, uint32_t n_kv);

    // Get the resident window tensors for a layer
    ggml_tensor * get_k(int32_t il) const;
    ggml_tensor * get_v(int32_t il) const;

private:
    llama_kv_cache * kv_ = nullptr;
    ggml_backend_dev_t dev_ = nullptr;
    ggml_context * ctx_ = nullptr;
    uint32_t window_size_ = 0;
    std::vector<llama_kv_stream_layer> layers_;
};
