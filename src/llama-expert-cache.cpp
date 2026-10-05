// llama-expert-cache.cpp - Adaptive VRAM expert tier implementation
#include "llama-expert-cache.h"

#include "ggml.h"
#include "ggml-backend.h"

#include <cstdio>
#include <cstring>
#include <algorithm>

llama_expert_cache::~llama_expert_cache() {
    close();
}

llama_expert_cache::llama_expert_cache(llama_expert_cache&& other) noexcept {
    slot_tensor_ = other.slot_tensor_;
    residency_tensor_ = other.residency_tensor_;
    base_ = other.base_;
    residency_ = std::move(other.residency_);
    slots_ = other.slots_;
    n_layers_ = other.n_layers_;
    n_expert_ = other.n_expert_;
    blob_ = other.blob_;
    next_free_ = other.next_free_;
    fills_ = other.fills_;
    per_layer_ = other.per_layer_;
    layer_next_ = std::move(other.layer_next_);
    off_ = std::move(other.off_);
    admitted_ = other.admitted_;

    other.slot_tensor_ = nullptr;
    other.residency_tensor_ = nullptr;
    other.base_ = nullptr;
    other.slots_ = 0;
    other.n_layers_ = 0;
    other.n_expert_ = 0;
    other.blob_ = 0;
    other.next_free_ = 0;
    other.fills_ = 0;
    other.per_layer_ = false;
    other.admitted_ = 0;
}

llama_expert_cache& llama_expert_cache::operator=(llama_expert_cache&& other) noexcept {
    if (this != &other) {
        close();
        slot_tensor_ = other.slot_tensor_;
        residency_tensor_ = other.residency_tensor_;
        base_ = other.base_;
        residency_ = std::move(other.residency_);
        slots_ = other.slots_;
        n_layers_ = other.n_layers_;
        n_expert_ = other.n_expert_;
        blob_ = other.blob_;
        next_free_ = other.next_free_;
        fills_ = other.fills_;
        per_layer_ = other.per_layer_;
        layer_next_ = std::move(other.layer_next_);
        off_ = std::move(other.off_);
        admitted_ = other.admitted_;

        other.slot_tensor_ = nullptr;
        other.residency_tensor_ = nullptr;
        other.base_ = nullptr;
        other.slots_ = 0;
        other.n_layers_ = 0;
        other.n_expert_ = 0;
        other.blob_ = 0;
        other.next_free_ = 0;
        other.fills_ = 0;
        other.per_layer_ = false;
        other.admitted_ = 0;
    }
    return *this;
}

bool llama_expert_cache::open(int64_t n_slots, int64_t n_layers, int64_t n_expert, int64_t blob_bytes,
                              struct ggml_context* ctx, std::string& err) {
    close();
    if (n_slots <= 0) {
        err = "llama_expert_cache: n_slots must be positive";
        return false;
    }
    if (n_layers <= 0 || n_expert <= 0 || blob_bytes <= 0) {
        err = "llama_expert_cache: n_layers, n_expert and blob_bytes must all be positive";
        return false;
    }
    if (ctx == nullptr) {
        err = "llama_expert_cache: ctx must not be null";
        return false;
    }

    ctx_ = ctx;
    const uint64_t want = (uint64_t)n_slots * (uint64_t)blob_bytes;

    // Allocate a single tensor for the entire slot arena
    // This is a flat byte buffer that we'll partition into slots
    // Use I8 as a byte type (GGML_TYPE_U8 doesn't exist)
    slot_tensor_ = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t)want);
    if (slot_tensor_ == nullptr) {
        char buf[256];
        snprintf(buf, sizeof(buf), "llama_expert_cache: failed to allocate %.2f GiB for expert cache",
                 (double)want / 1073741824.0);
        err = buf;
        return false;
    }
    base_ = slot_tensor_->data;

    // Zero the arena so unfilled slots are deterministic
    std::memset(base_, 0, (size_t)want);

    residency_.assign((size_t)(n_layers * n_expert), LLAMA_EXPERT_NOT_RESIDENT);
    slots_ = n_slots;
    n_layers_ = n_layers;
    n_expert_ = n_expert;
    blob_ = blob_bytes;
    next_free_ = 0;
    fills_ = 0;
    admitted_ = 0;

    // Initialize per-layer slot ranges
    layer_next_.assign((size_t)n_layers, 0);
    for (int64_t l = 0; l < n_layers; ++l) {
        int64_t lo = 0, hi = 0;
        layer_slot_range(l, lo, hi);
        layer_next_[(size_t)l] = (int32_t)lo;
    }

    // Create a GGML tensor that wraps the residency table
    // This is a view tensor that points to the residency_ vector data
    residency_tensor_ = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_layers * n_expert);
    residency_tensor_->data = residency_.data();

    return true;
}

bool llama_expert_cache::open_sized(const std::vector<int64_t>& slot_bytes, int64_t n_layers,
                                    int64_t n_expert, struct ggml_context* ctx, std::string& err) {
    if (slot_bytes.empty()) {
        err = "llama_expert_cache: no slots";
        return false;
    }

    int64_t mx = 0;
    std::vector<uint64_t> off(slot_bytes.size() + 1, 0);
    for (size_t i = 0; i < slot_bytes.size(); ++i) {
        // 256-byte aligned slots
        off[i + 1] = off[i] + ((uint64_t)slot_bytes[i] + 255) / 256 * 256;
        mx = slot_bytes[i] > mx ? slot_bytes[i] : mx;
    }

    if (!open((int64_t)off.back(), n_layers, n_expert, 1, ctx, err)) {
        return false;
    }

    slots_ = (int64_t)slot_bytes.size();
    blob_ = mx;
    off_ = std::move(off);

    layer_next_.assign((size_t)n_layers, 0);
    return true;
}

void llama_expert_cache::close() {
    ctx_ = nullptr;
    slot_tensor_ = nullptr;
    residency_tensor_ = nullptr;
    base_ = nullptr;
    off_.clear();
    residency_.clear();
    slots_ = 0;
    n_layers_ = 0;
    n_expert_ = 0;
    blob_ = 0;
    next_free_ = 0;
    fills_ = 0;
    admitted_ = 0;
    layer_next_.clear();
}

void llama_expert_cache::layer_slot_range(int64_t layer, int64_t& lo, int64_t& hi) const {
    lo = 0;
    hi = 0;
    if (n_layers_ <= 0 || slots_ <= 0 || layer < 0 || layer >= n_layers_) {
        return;
    }
    const int64_t q = slots_ / n_layers_;
    lo = layer * q;
    hi = (layer == n_layers_ - 1) ? slots_ : (layer + 1) * q;
}

int32_t llama_expert_cache::slot_of(int64_t layer, int64_t expert) const {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) {
        return LLAMA_EXPERT_NOT_RESIDENT;
    }
    return residency_[(size_t)(layer * n_expert_ + expert)];
}

int32_t llama_expert_cache::admit(int64_t layer, int64_t expert) {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) {
        return LLAMA_EXPERT_NOT_RESIDENT;
    }
    const size_t at = (size_t)(layer * n_expert_ + expert);
    if (residency_[at] != LLAMA_EXPERT_NOT_RESIDENT) {
        return residency_[at];
    }

    if (per_layer_) {
        if (layer_next_.empty()) {
            return LLAMA_EXPERT_NOT_RESIDENT;
        }
        int64_t lo = 0, hi = 0;
        layer_slot_range(layer, lo, hi);
        if ((int64_t)layer_next_[(size_t)layer] >= hi) {
            return LLAMA_EXPERT_NOT_RESIDENT;  // this layer's quota is full
        }
        residency_[at] = layer_next_[(size_t)layer]++;
        ++admitted_;
        return residency_[at];
    }

    if (next_free_ >= slots_) {
        return LLAMA_EXPERT_NOT_RESIDENT;  // full: no eviction
    }
    residency_[at] = (int32_t)next_free_;
    return (int32_t)next_free_++;
}

void* llama_expert_cache::device_slot(int32_t slot) {
    if (slot < 0 || slot >= slots_) {
        return nullptr;
    }
    if (!off_.empty()) {
        return (uint8_t*)base_ + off_[(size_t)slot];
    }
    return (uint8_t*)base_ + (size_t)slot * (size_t)blob_;
}

const void* llama_expert_cache::device_slot(int32_t slot) const {
    if (slot < 0 || slot >= slots_) {
        return nullptr;
    }
    if (!off_.empty()) {
        return (const uint8_t*)base_ + off_[(size_t)slot];
    }
    return (const uint8_t*)base_ + (size_t)slot * (size_t)blob_;
}

bool llama_expert_cache::fill_slot_blocking(int32_t slot, const void* src,
                                            std::string& err, int64_t bytes) {
    const size_t n = (size_t)(bytes > 0 && bytes <= blob_ ? bytes : blob_);
    void* dst = device_slot(slot);
    if (dst == nullptr) {
        err = "llama_expert_cache::fill_slot_blocking: slot outside the arena";
        return false;
    }
    if (src == nullptr) {
        err = "llama_expert_cache::fill_slot_blocking: source is null";
        return false;
    }

    // In a real implementation, this would use cudaMemcpyAsync or similar
    // For now, we use memcpy which works for both host and device memory
    // when the backend supports it (e.g., unified memory)
    std::memcpy(dst, src, n);
    ++fills_;
    return true;
}

struct ggml_tensor* llama_expert_cache::slot_tensor(struct ggml_context* graph_ctx, const ggml_tensor* w) const {
    if (w == nullptr) {
        return slot_tensor_;
    }

    // Create a properly-shaped view of the slot arena matching w's layout.
    // w has shape [ne0, ne1, n_expert] and we want [ne0, ne1, n_slots].
    // The strides nb[0], nb[1] are copied from w; nb[2] is the size of one slot.
    //
    // The view tensor is allocated in the graph context so it has the same
    // lifetime as the rest of the graph.

    const int64_t ne0 = w->ne[0];
    const int64_t ne1 = w->ne[1];
    const int64_t n_slots = slots_;

    // The size of one slot in bytes is blob_ (for uniform slots)
    // or off_[1] - off_[0] (for sized slots).
    int64_t slot_bytes = blob_;
    if (!off_.empty()) {
        slot_bytes = (int64_t)(off_[1] - off_[0]);
    }

    // Create the view tensor. We use ggml_new_tensor_3d to allocate the tensor
    // metadata, then overwrite the data pointer and strides.
    // The element type must match w's type so the kernel can interpret the data correctly.
    struct ggml_tensor* view = ggml_new_tensor_3d(graph_ctx, w->type, ne0, ne1, n_slots);
    if (view == nullptr) {
        return slot_tensor_;
    }

    // Set the data pointer to the slot arena
    view->data = slot_tensor_->data;

    // Set the strides to match w's layout, but with n_slots instead of n_expert
    view->nb[0] = w->nb[0];
    view->nb[1] = w->nb[1];
    view->nb[2] = slot_bytes;
    view->nb[3] = slot_bytes * n_slots;

    // Mark as a view so ggml doesn't try to free the data
    view->op = GGML_OP_VIEW;
    view->view_src = slot_tensor_;

    return view;
}

bool llama_read_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                               std::vector<std::pair<int32_t, int32_t>>& ranked,
                               int64_t& slots, std::string& err) {
    FILE* f = fopen(path.c_str(), "rb");
    if (f == nullptr) {
        err = "llama_read_expert_profile: cannot open " + path;
        return false;
    }

    char magic[4] = {0, 0, 0, 0};
    uint32_t hdr[5] = {0, 0, 0, 0, 0};
    if (fread(magic, 1, 4, f) != 4 || fread(hdr, 4, 5, f) != 5) {
        fclose(f);
        err = "llama_read_expert_profile: " + path + " is too short to hold a header";
        return false;
    }

    if (memcmp(magic, "STRP", 4) != 0) {
        fclose(f);
        err = "llama_read_expert_profile: " + path + " does not start with STRP";
        return false;
    }

    const uint32_t version = hdr[0];
    const uint32_t nl = hdr[1];
    const uint32_t ne = hdr[2];
    const uint32_t want = hdr[3];
    const uint32_t n_ranked = hdr[4];

    if ((int64_t)nl != n_layers || (int64_t)ne != n_expert) {
        fclose(f);
        char buf[256];
        snprintf(buf, sizeof(buf),
                 "llama_read_expert_profile: %s is %ux%u but this model is %lldx%lld",
                 path.c_str(), nl, ne, (long long)n_layers, (long long)n_expert);
        err = buf;
        return false;
    }

    if (n_ranked > want) {
        fclose(f);
        err = "llama_read_expert_profile: the header claims more ranked pairs than slots";
        return false;
    }

    ranked.assign(n_ranked, {0, 0});
    std::vector<uint16_t> raw((size_t)n_ranked * 2);
    if (n_ranked > 0 && fread(raw.data(), 2, (size_t)n_ranked * 2, f) != (size_t)n_ranked * 2) {
        fclose(f);
        err = "llama_read_expert_profile: the ranked list is truncated";
        return false;
    }
    fclose(f);

    for (uint32_t i = 0; i < n_ranked; ++i) {
        const int32_t l = (int32_t)raw[(size_t)i * 2];
        const int32_t e = (int32_t)raw[(size_t)i * 2 + 1];
        if (l < 0 || l >= n_layers || e < 0 || e >= n_expert) {
            char buf[256];
            snprintf(buf, sizeof(buf),
                     "llama_read_expert_profile: pair %u is (layer %d, expert %d), out of range",
                     i, l, e);
            err = buf;
            return false;
        }
        ranked[(size_t)i] = {l, e};
    }

    slots = (int64_t)want;
    (void)version;  // future format versioning
    return true;
}

