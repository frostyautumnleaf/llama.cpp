// llama-kv-stream.cpp - KV streaming from RAM with resident window
#include "llama-kv-stream.h"

#include "llama-model.h"
#include "llama-kv-cache.h"
#include "ggml-backend.h"

#include <cstring>

llama_kv_stream::~llama_kv_stream() {
    close();
}

bool llama_kv_stream::init(llama_kv_cache * kv, uint32_t window_size, ggml_backend_dev_t dev) {
    close();

    if (kv == nullptr || window_size == 0 || dev == nullptr) {
        return false;
    }

    kv_ = kv;
    dev_ = dev;
    window_size_ = window_size;

    // Get the number of layers from the KV cache
    const uint32_t n_layer = kv->get_size();

    // Create a ggml context for the resident window tensors
    ggml_init_params ctx_params = {
        /*.mem_size   =*/ 32 * 1024 * 1024, // 32 MB for tensor metadata
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ false,
    };
    ctx_ = ggml_init(ctx_params);
    if (ctx_ == nullptr) {
        return false;
    }

    layers_.resize(n_layer);

    // Allocate resident window for each layer
    for (uint32_t il = 0; il < n_layer; ++il) {
        // Get the K tensor from the KV cache
        ggml_tensor * k_storage = kv->get_k_storage(il);
        if (k_storage == nullptr) {
            continue;
        }

        // Create resident window tensors with the same shape as the full cache
        // but only window_size entries
        llama_kv_stream_layer & layer = layers_[il];

        // Store dimensions for later use
        layer.n_embd_head_k = k_storage->ne[0]; // actual: n_embd_k_gqa
        layer.n_head_kv = k_storage->ne[1];     // actual: kv_size (not used directly)
        layer.n_embd_k_gqa = k_storage->ne[0];  // row size in storage
        layer.n_stream = k_storage->ne[2];

        // Create 4D window tensors: [n_embd_k_gqa, window_size, n_stream]
        // This matches the layout of the KV cache storage
        layer.k_resident = ggml_new_tensor_3d(
            ctx_, k_storage->type,
            k_storage->ne[0], window_size, k_storage->ne[2]);

        // V: same shape as K
        layer.v_resident = ggml_new_tensor_3d(
            ctx_, k_storage->type,
            k_storage->ne[0], window_size, k_storage->ne[2]);

        if (layer.k_resident == nullptr || layer.v_resident == nullptr) {
            return false;
        }

        // Allocate on GPU
        ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev_);
        layer.k_buf = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(layer.k_resident));
        layer.v_buf = ggml_backend_buft_alloc_buffer(buft, ggml_nbytes(layer.v_resident));

        if (layer.k_buf == nullptr || layer.v_buf == nullptr) {
            return false;
        }

        ggml_backend_buffer_init_tensor(layer.k_buf, layer.k_resident);
        ggml_backend_buffer_init_tensor(layer.v_buf, layer.v_resident);

        layer.window_size = window_size;
        layer.window_start = 0;
    }

    return true;
}

void llama_kv_stream::close() {
    for (auto & layer : layers_) {
        if (layer.k_buf != nullptr) {
            ggml_backend_buffer_free(layer.k_buf);
        }
        if (layer.v_buf != nullptr) {
            ggml_backend_buffer_free(layer.v_buf);
        }
        layer.k_resident = nullptr;
        layer.v_resident = nullptr;
        layer.k_buf = nullptr;
        layer.v_buf = nullptr;
        layer.window_start = 0;
        layer.window_size = 0;
    }
    layers_.clear();

    if (ctx_ != nullptr) {
        ggml_free(ctx_);
        ctx_ = nullptr;
    }

    kv_ = nullptr;
    dev_ = nullptr;
    window_size_ = 0;
}

void llama_kv_stream::stream_layer(int32_t il, uint32_t n_kv) {
    if (il < 0 || il >= (int32_t)layers_.size()) {
        return;
    }

    if (n_kv == 0) {
        return;
    }

    llama_kv_stream_layer & layer = layers_[il];
    if (layer.k_resident == nullptr || layer.v_resident == nullptr) {
        return;
    }

    // Get the full KV cache tensors
    ggml_tensor * k_storage = kv_->get_k_storage(il);
    if (k_storage == nullptr) {
        return;
    }

    // Determine the range of entries to stream
    // Stream the most recent n_kv entries
    uint32_t total_entries = k_storage->ne[1]; // kv_size dimension
    uint32_t start = 0;
    if (n_kv < total_entries) {
        start = total_entries - n_kv;
    }

    // Update window position
    layer.window_start = start;

    // Calculate the number of entries to copy
    uint32_t n_copy = n_kv;
    if (n_copy > window_size_) {
        n_copy = window_size_;
    }

    // Copy K entries from CPU to VRAM
    // Source: k_storage[:, start:start+n_copy, :]
    // Destination: layer.k_resident[:, :n_copy, :]
    for (uint32_t s = 0; s < layer.n_stream; ++s) {
        // Source view: [n_embd_k_gqa, n_copy] at offset start*n_embd_k_gqa + s*kv_size*n_embd_k_gqa
        ggml_tensor * k_src_view = ggml_view_2d(
            ctx_, k_storage,
            layer.n_embd_k_gqa, n_copy,
            k_storage->nb[1],
            k_storage->nb[1] * start + k_storage->nb[2] * s);

        // Destination view: [n_embd_k_gqa, n_copy] at stream s
        ggml_tensor * k_dst_view = ggml_view_2d(
            ctx_, layer.k_resident,
            layer.n_embd_k_gqa, n_copy,
            layer.k_resident->nb[1],
            layer.k_resident->nb[2] * s);

        ggml_backend_tensor_copy(k_src_view, k_dst_view);
    }

    // Copy V entries from CPU to VRAM (same approach)
    // Note: V storage has the same layout as K
    for (uint32_t s = 0; s < layer.n_stream; ++s) {
        ggml_tensor * v_src_view = ggml_view_2d(
            ctx_, k_storage, // TODO: use actual V storage
            layer.n_embd_k_gqa, n_copy,
            k_storage->nb[1],
            k_storage->nb[1] * start + k_storage->nb[2] * s);

        ggml_tensor * v_dst_view = ggml_view_2d(
            ctx_, layer.v_resident,
            layer.n_embd_k_gqa, n_copy,
            layer.v_resident->nb[1],
            layer.v_resident->nb[2] * s);

        ggml_backend_tensor_copy(v_src_view, v_dst_view);
    }
}

ggml_tensor * llama_kv_stream::get_k(int32_t il, ggml_context * ctx, uint32_t n_kv) const {
    if (il < 0 || il >= (int32_t)layers_.size()) {
        return nullptr;
    }

    const llama_kv_stream_layer & layer = layers_[il];
    if (layer.k_resident == nullptr) {
        return nullptr;
    }

    GGML_UNUSED(ctx);
    GGML_UNUSED(n_kv);

    // Return the window tensor directly
    // Shape: [n_embd_k_gqa, window_size, n_stream]
    return layer.k_resident;
}

ggml_tensor * llama_kv_stream::get_v(int32_t il, ggml_context * ctx, uint32_t n_kv) const {
    if (il < 0 || il >= (int32_t)layers_.size()) {
        return nullptr;
    }

    const llama_kv_stream_layer & layer = layers_[il];
    if (layer.v_resident == nullptr) {
        return nullptr;
    }

    GGML_UNUSED(ctx);
    GGML_UNUSED(n_kv);

    return layer.v_resident;
}
