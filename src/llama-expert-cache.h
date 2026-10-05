// llama-expert-cache.h - Adaptive VRAM expert tier for MoE models
//
// Implements Strata's R4 design: a pool of VRAM slots that hold the most-frequently
// routed experts, with a residency table mapping (layer, expert) -> slot or -1.
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

// Slot index or -1 if not resident
static constexpr int32_t LLAMA_EXPERT_NOT_RESIDENT = -1;

// Expert cache: slot storage and residency table
class llama_expert_cache {
public:
    llama_expert_cache() = default;
    ~llama_expert_cache();

    llama_expert_cache(const llama_expert_cache&) = delete;
    llama_expert_cache& operator=(const llama_expert_cache&) = delete;

    // Open the cache with uniform slot sizes
    // n_slots: total number of expert slots
    // n_layers: number of MoE layers
    // n_expert: experts per layer
    // blob_bytes: size of each expert's weights in bytes
    // ctx: ggml context for allocation (must outlive the cache)
    bool open(int64_t n_slots, int64_t n_layers, int64_t n_expert, int64_t blob_bytes,
              struct ggml_context* ctx, std::string& err);

    // Open the cache with per-slot sizes (for native packs where experts differ in size)
    bool open_sized(const std::vector<int64_t>& slot_bytes, int64_t n_layers, int64_t n_expert,
                    struct ggml_context* ctx, std::string& err);

    void close();

    bool valid() const { return slot_tensor_ != nullptr; }
    int64_t slots() const { return slots_; }
    int64_t resident() const { return per_layer_ ? admitted_ : next_free_; }
    int64_t bytes() const { return off_.empty() ? slots_ * blob_ : (int64_t)off_.back(); }

    // (layer, expert) -> slot index, or LLAMA_EXPERT_NOT_RESIDENT
    int32_t slot_of(int64_t layer, int64_t expert) const;

    // Claim the next free slot for (layer, expert)
    // Returns the slot index, or LLAMA_EXPERT_NOT_RESIDENT if full
    int32_t admit(int64_t layer, int64_t expert);

    // Per-layer admission: layer l uses slots [l*q, (l+1)*q) where q = slots/n_layers
    void set_per_layer_admission(bool on) { per_layer_ = on; }
    bool per_layer_admission() const { return per_layer_; }
    void layer_slot_range(int64_t layer, int64_t& lo, int64_t& hi) const;

    // Get device pointer for a slot
    void* device_slot(int32_t slot);
    const void* device_slot(int32_t slot) const;

    // Copy expert weights from source tensor into a slot (blocking)
    // src: pointer to the expert's weights in the original tensor
    bool fill_slot_blocking(int32_t slot, const void* src, std::string& err, int64_t bytes = 0);

    // Number of slots filled so far
    int64_t fills() const { return fills_; }

    // Access residency table directly (for kernel use)
    const int32_t* residency_table() const { return residency_.data(); }

    // Get the slot tensor (for kernel access)
    struct ggml_tensor* slot_tensor() const { return slot_tensor_; }

    // Number of layers and experts (for kernel use)
    int64_t n_layers() const { return n_layers_; }
    int64_t n_expert() const { return n_expert_; }

private:
    struct ggml_tensor* slot_tensor_ = nullptr;
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

// Read an expert profile from a file (Strata's STRP format)
// Returns the ranked list of (layer, expert) pairs
bool llama_read_expert_profile(const std::string& path, int64_t n_layers, int64_t n_expert,
                               std::vector<std::pair<int32_t, int32_t>>& ranked, int64_t& slots, std::string& err);

