// llama-expert-cache.h - Adaptive VRAM expert tier for MoE models
//
// Implements Strata's R4 design: a pool of VRAM slots that hold the most-frequently
// routed experts, with a residency table mapping (layer, expert) -> slot or -1.
//
// Multi-GPU tiers (Strata's "second GPU as another expert tier"):
//   - Tier 0: main GPU (always present when cache is enabled)
//   - Tier 1: second GPU (optional, holds warm experts not in tier 0)
//   - Tier 2: third GPU (optional, holds cool experts not in tier 0 or 1)
//   - Each tier has its own slot arena and residency table
//   - The kernel checks tiers in order; first hit wins
//
// Key design decisions (from Strata measurements):
//   - Per-layer slot ranges are MANDATORY: a shared pool starves all but the first few
//     layers (2.97% hit rate for shared pool vs 21.4% for 8 slots/layer)
//   - The kernel must handle a per-row hit/miss split because only 4.6% of (layer,token)
//     pairs have all 10 experts resident
//   - Eviction policy is a measured question (LFU-decay vs LRU sweep), not a placeholder
//
// The cache is populated at startup from a routing profile (tools/make_profile.py in Strata)
// and can be adapted at runtime based on observed routing patterns.

#pragma once

#include <cstdint>
#include <cstddef>
#include <string>
#include <vector>
#include <memory>

struct ggml_tensor;
struct ggml_backend_dev;

// Slot index or -1 if not resident
static constexpr int32_t LLAMA_EXPERT_NOT_RESIDENT = -1;

// Maximum number of expert cache tiers (GPUs)
static constexpr int LLAMA_EXPERT_MAX_TIERS = 3;

// A single expert cache tier (one GPU)
class llama_expert_cache_tier {
public:
    llama_expert_cache_tier() = default;
    ~llama_expert_cache_tier();

    llama_expert_cache_tier(const llama_expert_cache_tier&) = delete;
    llama_expert_cache_tier& operator=(const llama_expert_cache_tier&) = delete;

    llama_expert_cache_tier(llama_expert_cache_tier&& other) noexcept;
    llama_expert_cache_tier& operator=(llama_expert_cache_tier&& other) noexcept;

    // Open the tier with uniform slot sizes
    bool open(int64_t n_slots, int64_t n_layers, int64_t n_expert, int64_t blob_bytes,
              struct ggml_context* ctx, std::string& err);

    void close();

    bool valid() const { return slot_tensor_ != nullptr; }
    int64_t slots() const { return slots_; }
    int64_t resident() const { return per_layer_ ? admitted_ : next_free_; }
    int64_t bytes() const { return off_.empty() ? slots_ * blob_ : (int64_t)off_.back(); }

    // (layer, expert) -> slot index, or LLAMA_EXPERT_NOT_RESIDENT
    int32_t slot_of(int64_t layer, int64_t expert) const;

    // Claim the next free slot for (layer, expert)
    int32_t admit(int64_t layer, int64_t expert);

    void set_per_layer_admission(bool on) { per_layer_ = on; }
    bool per_layer_admission() const { return per_layer_; }
    void layer_slot_range(int64_t layer, int64_t& lo, int64_t& hi) const;

    void* device_slot(int32_t slot);
    const void* device_slot(int32_t slot) const;

    bool fill_slot_blocking(int32_t slot, const void* src, std::string& err, int64_t bytes = 0);

    int64_t fills() const { return fills_; }

    const int32_t* residency_table() const { return residency_.data(); }
    struct ggml_tensor* residency_table_tensor() const { return residency_tensor_; }

    // Get the slot tensor (for kernel access)
    struct ggml_tensor* slot_tensor(struct ggml_context* graph_ctx, const ggml_tensor* w = nullptr) const;

    int64_t n_layers() const { return n_layers_; }
    int64_t n_expert() const { return n_expert_; }
    int64_t blob_bytes() const { return blob_; }

private:
    struct ggml_context* ctx_ = nullptr;
    struct ggml_tensor* slot_tensor_ = nullptr;
    struct ggml_tensor* residency_tensor_ = nullptr;
    void* base_ = nullptr;
    std::vector<int32_t> residency_;   // [n_layers * n_expert] -> slot or LLAMA_EXPERT_NOT_RESIDENT
    int64_t slots_ = 0;
    int64_t n_layers_ = 0;
    int64_t n_expert_ = 0;
    int64_t blob_ = 0;
    int64_t next_free_ = 0;
    int64_t fills_ = 0;
    bool per_layer_ = false;
    std::vector<int32_t> layer_next_;  // [n_layers] -> that layer's next free slot
    std::vector<uint64_t> off_;        // slot offsets when sized
    int64_t admitted_ = 0;
};

// Expert cache: multi-tier slot storage and residency tables
class llama_expert_cache {
public:
    llama_expert_cache() = default;
    ~llama_expert_cache();

    llama_expert_cache(const llama_expert_cache&) = delete;
    llama_expert_cache& operator=(const llama_expert_cache&) = delete;

    llama_expert_cache(llama_expert_cache&& other) noexcept;
    llama_expert_cache& operator=(llama_expert_cache&& other) noexcept;

    // Open tier 0 (main GPU). Must be called before any other tier.
    bool open(int64_t n_slots, int64_t n_layers, int64_t n_expert, int64_t blob_bytes,
              struct ggml_context* ctx, std::string& err);

    // Open an additional tier on a different GPU. Returns the tier index (1 or 2).
    // device_idx: CUDA device index for this tier's GPU
    int open_tier(int64_t n_slots, struct ggml_context* ctx, int device_idx, std::string& err);

    void close();

    bool valid() const { return tiers_[0].valid(); }
    int num_tiers() const { return num_tiers_; }
    int64_t slots(int tier = 0) const { return tier < num_tiers_ ? tiers_[tier].slots() : 0; }
    int64_t resident(int tier = 0) const { return tier < num_tiers_ ? tiers_[tier].resident() : 0; }
    int64_t total_resident() const;
    int64_t bytes(int tier = 0) const { return tier < num_tiers_ ? tiers_[tier].bytes() : 0; }

    // (layer, expert) -> (tier, slot) or (-1, LLAMA_EXPERT_NOT_RESIDENT)
    // Searches tiers in order; first hit wins
    void slot_of(int64_t layer, int64_t expert, int& tier, int32_t& slot) const;

    // Claim a slot for (layer, expert), trying tiers in order
    // Returns the tier index, or -1 if all full
    int admit(int64_t layer, int64_t expert);

    void set_per_layer_admission(bool on);
    bool per_layer_admission() const { return tiers_[0].per_layer_admission(); }

    void* device_slot(int tier, int32_t slot);
    const void* device_slot(int tier, int32_t slot) const;

    bool fill_slot_blocking(int tier, int32_t slot, const void* src, std::string& err, int64_t bytes = 0);

    int64_t fills(int tier = 0) const { return tier < num_tiers_ ? tiers_[tier].fills() : 0; }

    // Access tier's residency table directly
    const int32_t* residency_table(int tier) const { return tier < num_tiers_ ? tiers_[tier].residency_table() : nullptr; }

    // Get tier's residency table as a GGML tensor
    struct ggml_tensor* residency_table_tensor(int tier) const { return tier < num_tiers_ ? tiers_[tier].residency_table_tensor() : nullptr; }

    // Combined residency table encoding (tier, slot) for multi-tier kernel:
    //   value >= 0:  tier 0, slot = value
    //   value == -1: not resident
    //   value <= -2: val = -value - 2; if val < tier1_slots → tier 1 slot=val;
    //                else → tier 2 slot=val-tier1_slots
    // Must be called after all tiers are populated. Creates a combined table
    // in the given ggml context.
    struct ggml_tensor* combined_residency_table(struct ggml_context* ctx) const;

    // Get tier's slot tensor
    struct ggml_tensor* slot_tensor(int tier, struct ggml_context* graph_ctx, const ggml_tensor* w = nullptr) const;

    // Copy experts from a source tier to a destination tier (for P2P fallback).
    // Experts that can't fit in the destination tier remain in the source tier.
    // Returns the number of experts copied.
    int64_t promote_tier_to(int src_tier, int dst_tier);

    int64_t n_layers() const { return tiers_[0].n_layers(); }
    int64_t n_expert() const { return tiers_[0].n_expert(); }
    int64_t blob_bytes() const { return tiers_[0].blob_bytes(); }

    // Tier access for kernel use
    const llama_expert_cache_tier& tier(int i) const { return tiers_[i]; }

private:
    llama_expert_cache_tier tiers_[LLAMA_EXPERT_MAX_TIERS];
    int num_tiers_ = 0;
};

// Read an expert profile from a file (Strata's STRP format)
// Returns the ranked list of (layer, expert) pairs
bool llama_read_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                               std::vector<std::pair<int32_t, int32_t>>& ranked, int64_t& slots, std::string& err);

