// llama-expert-cache.cpp - Adaptive VRAM expert tier implementation
// Multi-GPU support: each tier is a separate expert cache on a different GPU
#include "llama-expert-cache.h"

#include "ggml.h"
#include "ggml-backend.h"

#include "llama-impl.h"

#include <cstdio>
#include <cstring>
#include <algorithm>
#include <functional>
#include <utility>

// ============================================================================
// llama_expert_cache_tier: single-GPU expert cache
// ============================================================================

llama_expert_cache_tier::~llama_expert_cache_tier() {
    close();
}

llama_expert_cache_tier::llama_expert_cache_tier(llama_expert_cache_tier&& other) noexcept {
    ctx_ = other.ctx_;
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

    other.ctx_ = nullptr;
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

llama_expert_cache_tier& llama_expert_cache_tier::operator=(llama_expert_cache_tier&& other) noexcept {
    if (this != &other) {
        close();
        ctx_ = other.ctx_;
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

        other.ctx_ = nullptr;
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

bool llama_expert_cache_tier::open(int64_t n_slots, int64_t n_layers, int64_t n_expert,
                                   int64_t blob_bytes, struct ggml_context* ctx, std::string& err) {
    close();
    if (n_slots <= 0) {
        err = "llama_expert_cache_tier: n_slots must be positive";
        return false;
    }
    if (n_layers <= 0 || n_expert <= 0 || blob_bytes <= 0) {
        err = "llama_expert_cache_tier: n_layers, n_expert and blob_bytes must all be positive";
        return false;
    }
    if (ctx == nullptr) {
        err = "llama_expert_cache_tier: ctx must not be null";
        return false;
    }

    ctx_ = ctx;
    const uint64_t want = (uint64_t)n_slots * (uint64_t)blob_bytes;

    // Allocate a single tensor for the entire slot arena
    // Use I8 as a byte type (GGML_TYPE_U8 doesn't exist)
    slot_tensor_ = ggml_new_tensor_1d(ctx, GGML_TYPE_I8, (int64_t)want);
    if (slot_tensor_ == nullptr) {
        char buf[256];
        snprintf(buf, sizeof(buf), "llama_expert_cache_tier: failed to allocate %.2f GiB for expert cache tier",
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
    residency_tensor_ = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_layers * n_expert);
    residency_tensor_->data = residency_.data();

    return true;
}

void llama_expert_cache_tier::close() {
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

void llama_expert_cache_tier::layer_slot_range(int64_t layer, int64_t& lo, int64_t& hi) const {
    lo = 0;
    hi = 0;
    if (n_layers_ <= 0 || slots_ <= 0 || layer < 0 || layer >= n_layers_) {
        return;
    }
    const int64_t q = slots_ / n_layers_;
    lo = layer * q;
    hi = (layer == n_layers_ - 1) ? slots_ : (layer + 1) * q;
}

int32_t llama_expert_cache_tier::slot_of(int64_t layer, int64_t expert) const {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) {
        return LLAMA_EXPERT_NOT_RESIDENT;
    }
    return residency_[(size_t)(layer * n_expert_ + expert)];
}

int32_t llama_expert_cache_tier::admit(int64_t layer, int64_t expert) {
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

int32_t llama_expert_cache_tier::evict(int64_t layer, int64_t expert) {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) {
        return LLAMA_EXPERT_NOT_RESIDENT;
    }
    const size_t at = (size_t)(layer * n_expert_ + expert);
    if (residency_[at] == LLAMA_EXPERT_NOT_RESIDENT) {
        return LLAMA_EXPERT_NOT_RESIDENT;
    }
    const int32_t slot = residency_[at];
    residency_[at] = LLAMA_EXPERT_NOT_RESIDENT;
    --admitted_;
    return slot;
}

void llama_expert_cache_tier::set_residency(int64_t layer, int64_t expert, int32_t slot) {
    if (layer < 0 || layer >= n_layers_ || expert < 0 || expert >= n_expert_) {
        return;
    }
    const size_t at = (size_t)(layer * n_expert_ + expert);
    if (residency_[at] == slot) {
        return; // no change
    }
    residency_[at] = slot;
}

void* llama_expert_cache_tier::device_slot(int32_t slot) {
    if (slot < 0 || slot >= slots_) {
        return nullptr;
    }
    if (!off_.empty()) {
        return (uint8_t*)base_ + off_[(size_t)slot];
    }
    return (uint8_t*)base_ + (size_t)slot * (size_t)blob_;
}

const void* llama_expert_cache_tier::device_slot(int32_t slot) const {
    if (slot < 0 || slot >= slots_) {
        return nullptr;
    }
    if (!off_.empty()) {
        return (const uint8_t*)base_ + off_[(size_t)slot];
    }
    return (const uint8_t*)base_ + (size_t)slot * (size_t)blob_;
}

bool llama_expert_cache_tier::fill_slot_blocking(int32_t slot, const void* src,
                                                 std::string& err, int64_t bytes) {
    const size_t n = (size_t)(bytes > 0 && bytes <= blob_ ? bytes : blob_);
    void* dst = device_slot(slot);
    if (dst == nullptr) {
        err = "llama_expert_cache_tier::fill_slot_blocking: slot outside the arena";
        return false;
    }
    if (src == nullptr) {
        err = "llama_expert_cache_tier::fill_slot_blocking: source is null";
        return false;
    }

    std::memcpy(dst, src, n);
    ++fills_;
    return true;
}

struct ggml_tensor* llama_expert_cache_tier::slot_tensor(struct ggml_context* graph_ctx,
                                                         const ggml_tensor* w) const {
    if (w == nullptr) {
        return slot_tensor_;
    }

    // Create a properly-shaped view of the slot arena matching w's layout.
    // w has shape [ne0, ne1, n_expert] and we want [ne0, ne1, n_slots].
    const int64_t ne0 = w->ne[0];
    const int64_t ne1 = w->ne[1];
    const int64_t n_slots = slots_;

    int64_t slot_bytes = blob_;
    if (!off_.empty()) {
        slot_bytes = (int64_t)(off_[1] - off_[0]);
    }

    struct ggml_tensor* view = ggml_new_tensor_3d(graph_ctx, w->type, ne0, ne1, n_slots);
    if (view == nullptr) {
        return slot_tensor_;
    }

    view->data = slot_tensor_->data;
    view->nb[0] = w->nb[0];
    view->nb[1] = w->nb[1];
    view->nb[2] = slot_bytes;
    view->nb[3] = slot_bytes * n_slots;
    view->op = GGML_OP_VIEW;
    view->view_src = slot_tensor_;

    return view;
}

// ============================================================================
// llama_expert_cache: multi-tier manager
// ============================================================================

llama_expert_cache::~llama_expert_cache() {
    close();
}

llama_expert_cache::llama_expert_cache(llama_expert_cache&& other) noexcept {
    for (int i = 0; i < LLAMA_EXPERT_MAX_TIERS; ++i) {
        tiers_[i] = std::move(other.tiers_[i]);
    }
    num_tiers_ = other.num_tiers_;
    other.num_tiers_ = 0;
}

llama_expert_cache& llama_expert_cache::operator=(llama_expert_cache&& other) noexcept {
    if (this != &other) {
        close();
        for (int i = 0; i < LLAMA_EXPERT_MAX_TIERS; ++i) {
            tiers_[i] = std::move(other.tiers_[i]);
        }
        num_tiers_ = other.num_tiers_;
        other.num_tiers_ = 0;
    }
    return *this;
}

bool llama_expert_cache::open(int64_t n_slots, int64_t n_layers, int64_t n_expert,
                              int64_t blob_bytes, struct ggml_context* ctx, std::string& err) {
    close();
    if (!tiers_[0].open(n_slots, n_layers, n_expert, blob_bytes, ctx, err)) {
        return false;
    }
    num_tiers_ = 1;
    return true;
}

int llama_expert_cache::open_tier(int64_t n_slots, struct ggml_context* ctx, int device_idx,
                                  std::string& err) {
    (void)device_idx; // TODO: use for P2P device selection
    if (num_tiers_ <= 0) {
        err = "llama_expert_cache: tier 0 must be opened first";
        return -1;
    }
    if (num_tiers_ >= LLAMA_EXPERT_MAX_TIERS) {
        err = "llama_expert_cache: maximum number of tiers reached";
        return -1;
    }
    if (n_slots <= 0) {
        err = "llama_expert_cache: n_slots must be positive";
        return -1;
    }

    const int tier_idx = num_tiers_;
    // Reuse tier 0's parameters for layers/experts/blob size
    if (!tiers_[tier_idx].open(n_slots, tiers_[0].n_layers(), tiers_[0].n_expert(),
                               tiers_[0].blob_bytes(), ctx, err)) {
        return -1;
    }
    tiers_[tier_idx].set_per_layer_admission(tiers_[0].per_layer_admission());
    num_tiers_ = tier_idx + 1;
    return tier_idx;
}

void llama_expert_cache::close() {
    for (int i = 0; i < num_tiers_; ++i) {
        tiers_[i].close();
    }
    num_tiers_ = 0;
}

int64_t llama_expert_cache::total_resident() const {
    int64_t total = 0;
    for (int i = 0; i < num_tiers_; ++i) {
        total += tiers_[i].resident();
    }
    return total;
}

void llama_expert_cache::slot_of(int64_t layer, int64_t expert, int& tier, int32_t& slot) const {
    tier = -1;
    slot = LLAMA_EXPERT_NOT_RESIDENT;
    for (int i = 0; i < num_tiers_; ++i) {
        const int32_t s = tiers_[i].slot_of(layer, expert);
        if (s != LLAMA_EXPERT_NOT_RESIDENT) {
            tier = i;
            slot = s;
            return;
        }
    }
}

int llama_expert_cache::admit(int64_t layer, int64_t expert) {
    for (int i = 0; i < num_tiers_; ++i) {
        const int32_t s = tiers_[i].admit(layer, expert);
        if (s != LLAMA_EXPERT_NOT_RESIDENT) {
            return i;
        }
    }
    return -1;
}

void llama_expert_cache::set_per_layer_admission(bool on) {
    for (int i = 0; i < num_tiers_; ++i) {
        tiers_[i].set_per_layer_admission(on);
    }
}

void* llama_expert_cache::device_slot(int tier, int32_t slot) {
    if (tier < 0 || tier >= num_tiers_) return nullptr;
    return tiers_[tier].device_slot(slot);
}

const void* llama_expert_cache::device_slot(int tier, int32_t slot) const {
    if (tier < 0 || tier >= num_tiers_) return nullptr;
    return tiers_[tier].device_slot(slot);
}

bool llama_expert_cache::fill_slot_blocking(int tier, int32_t slot, const void* src,
                                            std::string& err, int64_t bytes) {
    if (tier < 0 || tier >= num_tiers_) {
        err = "llama_expert_cache: invalid tier";
        return false;
    }
    return tiers_[tier].fill_slot_blocking(slot, src, err, bytes);
}

struct ggml_tensor* llama_expert_cache::slot_tensor(int tier, struct ggml_context* graph_ctx,
                                                    const ggml_tensor* w) const {
    if (tier < 0 || tier >= num_tiers_) return nullptr;
    return tiers_[tier].slot_tensor(graph_ctx, w);
}

int64_t llama_expert_cache::promote_tier_to(int src_tier, int dst_tier) {
    if (src_tier < 0 || src_tier >= num_tiers_ || dst_tier < 0 || dst_tier >= num_tiers_) {
        return 0;
    }
    if (src_tier == dst_tier) {
        return 0;
    }

    int64_t copied = 0;
    const int64_t n = n_layers() * n_expert();
    const int32_t* src_table = tiers_[src_tier].residency_table();

    // Iterate over all (layer, expert) pairs in the source tier
    for (int64_t i = 0; i < n; ++i) {
        const int32_t src_slot = src_table[i];
        if (src_slot < 0) {
            continue; // Not resident in source tier
        }

        // Decompose index into layer and expert
        const int64_t layer = i / n_expert();
        const int64_t expert = i % n_expert();

        // Try to admit this expert to the destination tier
        const int32_t dst_slot = tiers_[dst_tier].admit(layer, expert);
        if (dst_slot < 0) {
            continue; // Destination tier is full
        }

        // Copy the expert weights from source to destination
        const void* src_data = tiers_[src_tier].device_slot(src_slot);
        if (src_data == nullptr) {
            continue;
        }

        std::string err;
        if (!tiers_[dst_tier].fill_slot_blocking(dst_slot, src_data, err, blob_bytes())) {
            continue;
        }

        // Remove from source tier by clearing its residency entry
        // (we can't easily remove from the source tier's residency table,
        //  so we just leave it; the combined table will prefer the dst tier)
        ++copied;
    }

    return copied;
}

struct ggml_tensor* llama_expert_cache::combined_residency_table(struct ggml_context* ctx) const {
    if (num_tiers_ <= 0) return nullptr;

    const int64_t n = n_layers() * n_expert();
    struct ggml_tensor* table = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n);
    if (table == nullptr) return nullptr;

    int32_t* data = (int32_t*)table->data;
    std::fill(data, data + n, LLAMA_EXPERT_NOT_RESIDENT);

    // Tier 0: slot >= 0
    if (tiers_[0].valid()) {
        const int32_t* t0 = tiers_[0].residency_table();
        for (int64_t i = 0; i < n; ++i) {
            if (t0[i] >= 0) {
                data[i] = t0[i];
            }
        }
    }

    // Tier 1: encode as -slot - 2 (only if not already in tier 0)
    // Tier 2: encode as -(slot + tier1_slots) - 2 (only if not in tier 0 or 1)
    // Kernel decodes: val = -enc - 2; if val < tier1_slots → tier 1, else tier 2
    const int64_t tier1_slots = (num_tiers_ > 1) ? tiers_[1].slots() : 0;

    if (num_tiers_ > 1 && tiers_[1].valid()) {
        const int32_t* t1 = tiers_[1].residency_table();
        for (int64_t i = 0; i < n; ++i) {
            if (t1[i] >= 0 && data[i] < 0) {
                data[i] = -t1[i] - 2;
            }
        }
    }

    if (num_tiers_ > 2 && tiers_[2].valid()) {
        const int32_t* t2 = tiers_[2].residency_table();
        for (int64_t i = 0; i < n; ++i) {
            if (t2[i] >= 0 && data[i] < 0) {
                data[i] = -(t2[i] + (int32_t)tier1_slots) - 2;
            }
        }
    }

    return table;
}

// ============================================================================
// Runtime admission and eviction (Strata's adaptive_tier.cpp)
// ============================================================================

bool llama_expert_cache::enable_adaptation(const llama_expert_adapt_params& params,
                                           std::string& err) {
    if (!valid()) {
        err = "llama_expert_cache: adaptation requires an open cache";
        return false;
    }
    if (adapt_params_ != nullptr) {
        err = "llama_expert_cache: adaptation already enabled";
        return false;
    }
    if (params.copy_fn == nullptr) {
        err = "llama_expert_cache: adaptation requires a copy_fn callback";
        return false;
    }

    adapt_params_ = std::make_unique<llama_expert_adapt_params>(params);
    usage_.assign(n_layers() * n_expert(), 0.0f);
    tokens_since_adapt_ = 0;
    adapt_count_ = 0;
    runtime_admitted_ = 0;
    runtime_evicted_ = 0;
    runtime_swapped_ = 0;

    LLAMA_LOG_INFO("%s: runtime adaptation enabled (max_moves=%d, threshold=%.1f, swap_margin=%.1f, decay=%.1f, interval=%d)\n",
                   __func__, params.max_moves, params.usage_threshold, params.swap_margin,
                   params.decay_factor, params.adapt_interval);

    return true;
}

void llama_expert_cache::record_usage(int64_t layer, int64_t expert) {
    if (usage_.empty() || layer < 0 || layer >= n_layers() || expert < 0 || expert >= n_expert()) {
        return;
    }
    usage_[(size_t)(layer * n_expert() + expert)] += 1.0f;
}

void llama_expert_cache::record_usage_layer(int64_t layer, const int32_t* ids, int64_t n_entries) {
    if (usage_.empty() || layer < 0 || layer >= n_layers() || ids == nullptr) {
        return;
    }
    const size_t base = (size_t)layer * n_expert();
    for (int64_t i = 0; i < n_entries; ++i) {
        const int64_t e = ids[i];
        if (e >= 0 && e < n_expert()) {
            usage_[base + (size_t)e] += 1.0f;
        }
    }
}

void llama_expert_cache::record_usage_all_layers(const int32_t* const* per_layer_ids, int64_t n_entries_per_layer) {
    if (usage_.empty() || per_layer_ids == nullptr) {
        return;
    }
    for (int64_t l = 0; l < n_layers(); ++l) {
        record_usage_layer(l, per_layer_ids[l], n_entries_per_layer);
    }
}

bool llama_expert_cache::should_adapt() const {
    if (adapt_params_ == nullptr) {
        return false;
    }
    if (adapt_params_->adapt_interval <= 0) {
        return false; // manual only
    }
    return tokens_since_adapt_ >= adapt_params_->adapt_interval;
}

float llama_expert_cache::get_usage(int64_t layer, int64_t expert) const {
    if (usage_.empty() || layer < 0 || layer >= n_layers() || expert < 0 || expert >= n_expert()) {
        return 0.0f;
    }
    return usage_[(size_t)(layer * n_expert() + expert)];
}

int64_t llama_expert_cache::adapt(std::string& err) {
    (void)err;
    if (adapt_params_ == nullptr || !valid()) {
        return 0;
    }

    const int64_t nl = n_layers();
    const int64_t ne = n_expert();
    const float threshold = adapt_params_->usage_threshold;
    const float swap_margin = adapt_params_->swap_margin;
    const int max_moves = adapt_params_->max_moves;

    int64_t total_moves = 0;

    // For each layer, identify candidates (non-resident, high usage) and victims (resident, low usage)
    for (int64_t l = 0; l < nl; ++l) {
        if (total_moves >= max_moves) break;

        const size_t base = (size_t)l * ne;

        // Collect candidates: non-resident experts with usage >= threshold
        std::vector<std::pair<float, int64_t>> candidates;
        // Collect victims: resident experts (with their slot)
        std::vector<std::pair<float, std::pair<int64_t, int32_t>>> victims;

        for (int64_t e = 0; e < ne; ++e) {
            const float usage = usage_[base + (size_t)e];
            const int32_t slot = tiers_[0].slot_of(l, e);

            if (slot == LLAMA_EXPERT_NOT_RESIDENT) {
                if (usage >= threshold) {
                    candidates.emplace_back(usage, e);
                }
            } else {
                victims.emplace_back(std::make_pair(usage, std::make_pair(e, slot)));
            }
        }

        if (candidates.empty()) continue;

        // Sort candidates by usage (descending) - highest usage first
        std::sort(candidates.begin(), candidates.end(),
                  [](const auto& a, const auto& b) { return a.first > b.first; });

        // Sort victims by usage (ascending) - lowest usage first (best eviction candidates)
        std::sort(victims.begin(), victims.end(),
                  [](const auto& a, const auto& b) { return a.first < b.first; });

        // First, try to fill free slots
        size_t ci = 0;
        while (ci < candidates.size() && total_moves < max_moves) {
            const int64_t expert = candidates[ci].second;
            const int32_t slot = tiers_[0].admit(l, expert);
            if (slot == LLAMA_EXPERT_NOT_RESIDENT) {
                break; // no more free slots
            }

            // Copy expert weights from model tensor to slot via callback
            if (adapt_params_->copy_fn != nullptr) {
                void* slot_ptr = tiers_[0].device_slot(slot);
                adapt_params_->copy_fn(l, expert, slot_ptr, blob_bytes(), adapt_params_->copy_user_data);
            }

            ++runtime_admitted_;
            ++total_moves;
            ++ci;
        }

        // Then, swap out low-usage residents for high-usage non-residents
        size_t vi = 0;
        while (ci < candidates.size() && vi < victims.size() && total_moves < max_moves) {
            const float cand_usage = candidates[ci].first;
            const float vict_usage = victims[vi].first;
            const int64_t vict_expert = victims[vi].second.first;
            const int32_t vict_slot = victims[vi].second.second;

            // Swap only if candidate usage exceeds victim usage by the margin
            if (cand_usage < vict_usage + swap_margin) {
                break; // no more worthwhile swaps
            }

            const int64_t cand_expert = candidates[ci].second;

            // Evict victim: clear its residency entry
            tiers_[0].evict(l, vict_expert);
            ++runtime_evicted_;

            // Admit candidate to the freed slot
            tiers_[0].set_residency(l, cand_expert, vict_slot);
            tiers_[0].increment_admitted();

            // Copy expert weights from model tensor to slot via callback
            if (adapt_params_->copy_fn != nullptr) {
                void* slot_ptr = tiers_[0].device_slot(vict_slot);
                adapt_params_->copy_fn(l, cand_expert, slot_ptr, blob_bytes(), adapt_params_->copy_user_data);
            }

            ++runtime_swapped_;
            ++total_moves;
            ++ci;
            ++vi;
        }
    }

    // Decay usage counts after adaptation
    if (!usage_.empty() && adapt_params_->decay_factor > 0.0f && adapt_params_->decay_factor < 1.0f) {
        const float decay = adapt_params_->decay_factor;
        for (float& u : usage_) {
            u *= decay;
        }
    }

    tokens_since_adapt_ = 0;
    ++adapt_count_;

    if (total_moves > 0) {
        LLAMA_LOG_INFO("%s: adapted %lld experts (admitted=%lld, evicted=%lld, swapped=%lld)\n",
                       __func__, (long long)total_moves, (long long)runtime_admitted_,
                       (long long)runtime_evicted_, (long long)runtime_swapped_);
    }

    return total_moves;
}

void llama_expert_cache::capture_expert_usage(int64_t layer, struct ggml_tensor* selected_experts) {
    if (adapt_params_ == nullptr || selected_experts == nullptr) {
        return;
    }
    captured_usage_.push_back({layer, selected_experts});
}

void llama_expert_cache::apply_captured_usage() {
    if (captured_usage_.empty() || adapt_params_ == nullptr) {
        return;
    }

    for (const auto& entry : captured_usage_) {
        ggml_tensor* sel = entry.selected_experts;
        if (sel == nullptr) continue;

        const int64_t n_expert_used = sel->ne[0];
        const int64_t n_tokens = sel->ne[1];
        const int64_t n_entries = n_expert_used * n_tokens;

        if (n_entries <= 0) continue;

        // Copy tensor data to host
        std::vector<int32_t> ids(n_entries);
        memcpy(ids.data(), sel->data, n_entries * sizeof(int32_t));

        // Update usage counters
        record_usage_layer(entry.layer, ids.data(), n_entries);
    }

    captured_usage_.clear();
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

