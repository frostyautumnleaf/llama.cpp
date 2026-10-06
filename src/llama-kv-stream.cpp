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

        // K: [n_embd_head_k, n_head_k, window_size]
        layer.k_resident = ggml_new_tensor_3d(
            ctx_, k_storage->type,
            k_storage->ne[0], k_storage->ne[1], window_size);

        // V: same shape as K (for simplicity)
        layer.v_resident = ggml_new_tensor_3d(
            ctx_, k_storage->type,
            k_storage->ne[0], k_storage->ne[1], window_size);

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

    // Get the full KV cache tensor
    ggml_tensor * k_storage = kv_->get_k_storage(il);
    if (k_storage == nullptr) {
        return;
    }

    // Determine the range of entries to stream
    // Stream the most recent n_kv entries
    uint32_t total_entries = k_storage->ne[2];
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

    // Create a view of the source tensor for the range we want to copy
    // Source: k_storage[start:start+n_copy]
    ggml_tensor * k_src_view = ggml_view_3d(
        ctx_, k_storage,
        k_storage->ne[0], k_storage->ne[1], n_copy,
        k_storage->nb[1], k_storage->nb[2],
        start * k_storage->nb[2]);

    // Copy K entries from CPU to VRAM
    ggml_backend_tensor_copy(k_src_view, layer.k_resident);

    // For V, use the same approach (using K storage as proxy)
    ggml_tensor * v_src_view = ggml_view_3d(
        ctx_, k_storage,
        k_storage->ne[0], k_storage->ne[1], n_copy,
        k_storage->nb[1], k_storage->nb[2],
        start * k_storage->nb[2]);

    ggml_backend_tensor_copy(v_src_view, layer.v_resident);
}

ggml_tensor * llama_kv_stream::get_k(int32_t il) const {
    if (il < 0 || il >= (int32_t)layers_.size()) {
        return nullptr;
    }
    return layers_[il].k_resident;
}

ggml_tensor * llama_kv_stream::get_v(int32_t il) const {
    if (il < 0 || il >= (int32_t)layers_.size()) {
        return nullptr;
    }
    return layers_[il].v_resident;
}
