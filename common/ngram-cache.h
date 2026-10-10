#pragma once

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

#define LLAMA_NGRAM_MIN    1
#define LLAMA_NGRAM_MAX    4
#define LLAMA_NGRAM_STATIC 2

// Data structures to map n-grams to empirical token probabilities:
//
// The upstream implementation was a map of maps:
//
//     std::unordered_map<common_ngram, std::unordered_map<llama_token, int32_t>, ...>
//
// which is slow to build, slow to draft with, and expensive to keep: every n-gram paid for a bucket array, a node
// per follower and a pointer chase per lookup. The structures here replace it (see docs/3x-optimizations.md):
//
//   * the outer map is a flat, open-addressed table with a one-byte fingerprint per slot (a SwissTable-style
//     metadata array), so a miss is answered from one cache line and a hit chases no pointers;
//   * the followers of one n-gram live in a single arena, sorted by token, and are searched with a branchless,
//     fixed-length binary search - 64% of n-grams have one follower, so a hash map per n-gram was pure overhead;
//   * every n-gram keeps its total count and its most frequent follower, which lets a draft be rejected before
//     the followers are scored at all;
//   * the same layout is written to and read back from disk in three bulk copies, so loading a static cache
//     parses nothing but its header.

struct common_ngram {
    llama_token tokens[LLAMA_NGRAM_MAX];

    common_ngram() {
        for (int i = 0; i < LLAMA_NGRAM_MAX; ++i) {
            tokens[i] = LLAMA_TOKEN_NULL;
        }
    }

    common_ngram(const llama_token * input, const int ngram_size) {
        for (int i = 0; i < LLAMA_NGRAM_MAX; ++i) {
            tokens[i] = i < ngram_size ? input[i] : LLAMA_TOKEN_NULL;
        }
    }

    bool operator==(const common_ngram & other) const {
        for (int i = 0; i < LLAMA_NGRAM_MAX; ++i) {
            if (tokens[i] != other.tokens[i]) {
                return false;
            }
        }
        return true;
    }
};

struct common_token_hash_function {
    size_t operator()(const llama_token token) const {
        // see https://probablydance.com/2018-06-16/fibonacci-hashing-the-optimization-that-the-world-forgot-or-a-better-alternative-to-integer-modulo/
        return token * 11400714819323198485llu;
    }
};

struct common_ngram_hash_function {
    size_t operator()(const common_ngram & ngram) const {
        size_t hash = common_token_hash_function{}(ngram.tokens[0]);
        for (int i = 1; i < LLAMA_NGRAM_MAX; ++i) {
            hash ^= common_token_hash_function{}(ngram.tokens[i]);
        }
        return hash;
    }
};

// one follower of an n-gram: the token and how often it followed
struct common_ngram_entry {
    llama_token token;
    int32_t     count;
};

// The followers of one n-gram, in ascending token order, as a read-only view.
// `total` and the most frequent follower are kept with the group so that a candidate can be dismissed without
// touching any entry (see common_ngram_cache_draft).
struct common_ngram_part {
    const common_ngram_entry * entries = nullptr;
    uint32_t    n         = 0;
    int32_t     total     = 0;
    int32_t     max_count = 0;
    llama_token max_token = LLAMA_TOKEN_NULL;

    bool empty() const {
        return n == 0;
    }

    // Branchless lower_bound over `entries`. The remaining length shrinks by the same amount whatever the
    // comparison returns, so `n > 1` never depends on a value read from memory: the search for the next
    // candidate can start while this one's loads are still in flight (std::lower_bound cannot).
    size_t lower_bound(llama_token token) const {
        const common_ngram_entry * base = entries;
        size_t                     n    = this->n;

        while (n > 1) {
            const size_t half = n / 2;
            base = base[half].token < token ? base + half : base;
            n -= half;
        }

        return (base - entries) + (base->token < token ? 1 : 0);
    }

    // the entry for `token`, or nullptr; `n == 1` needs no loop and `n == 0` has nothing to search
    const common_ngram_entry * find(llama_token token) const {
        if (n == 0) {
            return nullptr;
        }
        const size_t i = lower_bound(token);
        return i < n && entries[i].token == token ? entries + i : nullptr;
    }

    int32_t get(llama_token token) const {
        const common_ngram_entry * e = find(token);
        return e != nullptr ? e->count : 0;
    }
};

// n-gram -> empirical distribution of following tokens.
// Move-only: a cache is shared with std::shared_ptr instead of being copied per sequence.
class common_ngram_cache {
public:
    common_ngram_cache() = default;
    ~common_ngram_cache();

    common_ngram_cache(common_ngram_cache && other) noexcept;
    common_ngram_cache & operator=(common_ngram_cache && other) noexcept;

    common_ngram_cache(const common_ngram_cache &) = delete;
    common_ngram_cache & operator=(const common_ngram_cache &) = delete;

    // number of stored n-grams
    size_t size() const {
        return n_parts;
    }

    bool empty() const {
        return n_parts == 0;
    }

    // total number of (token, count) pairs in the arena: after compaction or loading, the live ones
    size_t n_entries() const {
        return arena.size();
    }

    // the live followers, wherever they sit in the arena (a group that outgrew its region leaves a copy behind)
    size_t n_live_entries() const;

    // rough footprint of the table plus the arena, for reporting
    size_t size_bytes() const;

    void clear();

    // read-only lookup; an empty part when the n-gram is absent
    common_ngram_part find(const common_ngram & ngram) const;

    // count one occurrence of `token` after `ngram`
    void add(const common_ngram & ngram, llama_token token, int32_t count = 1);

    // visit every stored n-gram in an unspecified order
    template <typename Callable>
    void for_each(Callable && fn) const {
        for (size_t i = 0; i < slots.size(); ++i) {
            if (meta[i] == 0) {
                continue;
            }
            fn(key_of(slots[i]), part_of(slots[i]));
        }
    }

    // grow the table to hold at least `n` n-grams
    void reserve(size_t n);

    // rebuild the arena so it holds only the live followers: a group that outgrew its reserved region leaves its
    // old copy behind, so compaction is what makes a cache as small as its contents
    void compact();

    // used by common_ngram_cache_save/load: the three tables are written and read as they lie in memory
    uint64_t save_index_bits() const;
    void     save_tables(std::ostream & out) const;
    void     load_tables(uint64_t index_bits, uint64_t n_parts_hint, uint64_t n_entries, std::istream & in);

private:
    // 40 bytes: the packed n-gram key, the group's place in the arena, and the aggregates a draft needs
    struct slot {
        uint64_t  k0;        // tokens[0] | tokens[1] << 32
        uint64_t  k1;        // tokens[2] | tokens[3] << 32
        uint32_t  off;       // index of the group in the arena
        uint32_t  n;         // followers in the group
        uint32_t  cap;       // slots reserved for the group at `off`
        int32_t   total;     // sum of the counts
        int32_t   max_count; // largest count
        llama_token max_token;
    };
    static_assert(sizeof(slot) == 40, "the slot layout is the on-disk layout");

    std::vector<uint8_t> meta;  // one fingerprint per slot, 0 = empty
    std::vector<slot>    slots; // same length as meta
    std::vector<common_ngram_entry> arena;

    size_t n_parts     = 0;
    size_t n_used      = 0;  // slots in use
    size_t index_cap   = 0;  // meta/slots length (power of two)
    size_t index_bits_ = 0;

    static uint64_t key_part(const llama_token * tokens, int base) {
        uint64_t lo = (uint64_t) (uint32_t) tokens[base + 0];
        uint64_t hi = (uint64_t) (uint32_t) tokens[base + 1];
        return lo | (hi << 32);
    }

    static common_ngram key_of(const slot & s) {
        common_ngram ngram;
        ngram.tokens[0] = (llama_token) (uint32_t) (s.k0 & 0xffffffffull);
        ngram.tokens[1] = (llama_token) (uint32_t) (s.k0 >> 32);
        ngram.tokens[2] = (llama_token) (uint32_t) (s.k1 & 0xffffffffull);
        ngram.tokens[3] = (llama_token) (uint32_t) (s.k1 >> 32);
        return ngram;
    }

    common_ngram_part part_of(const slot & s) const {
        common_ngram_part part;
        part.entries   = arena.data() + s.off;
        part.n         = s.n;
        part.total     = s.total;
        part.max_count = s.max_count;
        part.max_token = s.max_token;
        return part;
    }

    // index of the slot holding the key, or of the first free slot of its probe chain when inserting
    // SIZE_MAX when the table holds no such key and offers no free slot
    size_t find_slot(uint64_t k0, uint64_t k1, uint8_t fp, bool insert) const;

    void rehash(size_t index_bits);
};

using common_ngram_cache_ptr = std::shared_ptr<const common_ngram_cache>;

// Update an ngram cache with tokens.
// ngram_cache:         the cache to modify.
// ngram_min/ngram_max: the min/max size of the ngrams to extract from inp_data.
// inp_data:            the token sequence with which to update ngram_cache.
// nnew:                how many new tokens have been appended to inp_data since the last call to this function.
// print_progress:      whether to print progress to stderr.
//
// In order to get correct results inp_data can ONLY BE APPENDED TO.
// Changes in the middle need a complete rebuild.
void common_ngram_cache_update(
    common_ngram_cache & ngram_cache, int ngram_min, int ngram_max, std::vector<llama_token> & inp_data, int nnew, bool print_progress);

// Try to draft tokens from ngram caches.
// inp:                the tokens generated so far.
// draft:              the token sequence to draft. Expected to initially contain the previously sampled token.
// n_draft:            maximum number of tokens to add to draft.
// ngram_min/gram_max: the min/max size of the ngrams in nc_context and nc_dynamic.
// nc_context:         ngram cache based on current context.
// nc_dynamic:         ngram cache based on previous user generations.
// nc_static:          ngram cache generated from a large text corpus, used for validation.
void common_ngram_cache_draft(
    std::vector<llama_token> & inp, std::vector<llama_token> & draft, int n_draft, int ngram_min, int ngram_max,
    const common_ngram_cache & nc_context, const common_ngram_cache & nc_dynamic, const common_ngram_cache & nc_static);

// Save an ngram cache to a file.
// ngram_cache: the ngram cache to save.
// filename:    the path under which to save the ngram cache.
void common_ngram_cache_save(const common_ngram_cache & ngram_cache, const std::string & filename);

// Load an ngram cache saved with common_ngram_cache_save. Caches written by older versions of llama.cpp (the
// per-n-gram record format) are parsed as before.
// filename: the path from which to load the ngram cache.
// returns:  an ngram cache containing the information saved to filename.
common_ngram_cache common_ngram_cache_load(const std::string & filename);

// Load a static cache without keeping it alive: the returned cache owns the buffer it was read into.
// The shared_ptr lets every sequence of a server share one read-only copy instead of copying the table per slot.
common_ngram_cache_ptr common_ngram_cache_load_shared(const std::string & filename);

// Merge two ngram caches.
// ngram_cache_target: the cache to which to add the information from ngram_cache_add.
// ngram_cache_add:    the cache to add to ngram_cache_target.
void common_ngram_cache_merge(common_ngram_cache & ngram_cache_target, const common_ngram_cache & ngram_cache_add);
