#include "ngram-cache.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <fstream>

//
// hashing
//

namespace {

uint64_t ngram_mix(uint64_t x) {
    // splitmix64 finalizer
    x ^= x >> 30;
    x *= 0xbf58476d1ce4e5b9ull;
    x ^= x >> 27;
    x *= 0x94d049bb133111ebull;
    return x ^ (x >> 31);
}

uint64_t ngram_hash(uint64_t k0, uint64_t k1) {
    return ngram_mix(k0 + 0x9e3779b97f4a7c15ull * (k1 + 1));
}

// 0 marks an empty slot, so the fingerprint is kept in [1, 255]
uint8_t ngram_fingerprint(uint64_t hash) {
    return (uint8_t) ((hash >> 56) | 1ull);
}

constexpr size_t npos = ~(size_t) 0;

// keep the load factor low enough that linear probing stays short: one cache line of fingerprints holds 64 slots
constexpr double ngram_max_load = 0.70;

// the smallest index the table ever gets: 1024 n-grams, so a short conversation never rehashes
constexpr size_t ngram_min_index_bits = 10;

} // namespace

//
// common_ngram_cache
//

common_ngram_cache::~common_ngram_cache() = default;

common_ngram_cache::common_ngram_cache(common_ngram_cache && other) noexcept = default;

common_ngram_cache & common_ngram_cache::operator=(common_ngram_cache && other) noexcept = default;

size_t common_ngram_cache::size_bytes() const {
    return meta.capacity() + slots.capacity() * sizeof(slot) + arena.capacity() * sizeof(common_ngram_entry);
}

void common_ngram_cache::clear() {
    meta.clear();
    slots.clear();
    arena.clear();
    n_parts     = 0;
    n_used      = 0;
    index_cap   = 0;
    index_bits_ = 0;
}

void common_ngram_cache::reserve(size_t n) {
    size_t bits = ngram_min_index_bits;
    while (((double) ((size_t) 1 << bits)) * ngram_max_load < (double) (n + 1)) {
        ++bits;
    }
    if (index_cap != 0 && bits <= index_bits_) {
        return;
    }
    rehash(bits);
}

// only the index is rebuilt: the groups stay where they are in the arena, so a rehash invalidates nothing
void common_ngram_cache::rehash(size_t index_bits) {
    std::vector<uint8_t> old_meta  = std::move(meta);
    std::vector<slot>    old_slots = std::move(slots);

    index_bits_ = index_bits;
    index_cap   = (size_t) 1 << index_bits;
    meta.assign(index_cap, 0);
    slots.assign(index_cap, slot{});

    n_parts = 0;
    n_used  = 0;

    for (size_t i = 0; i < old_slots.size(); ++i) {
        if (old_meta[i] == 0) {
            continue;
        }
        const slot s = old_slots[i];

        const uint8_t fp = ngram_fingerprint(ngram_hash(s.k0, s.k1));
        size_t        j  = (size_t) ngram_hash(s.k0, s.k1) & (index_cap - 1);
        while (meta[j] != 0) {
            j = (j + 1) & (index_cap - 1);
        }

        meta[j]  = fp;
        slots[j] = s;

        ++n_parts;
        ++n_used;
    }
}

size_t common_ngram_cache::find_slot(uint64_t k0, uint64_t k1, uint8_t fp, bool insert) const {
    if (index_cap == 0) {
        return npos;
    }

    const size_t mask = index_cap - 1;
    size_t       i    = (size_t) ngram_hash(k0, k1) & mask;

    // the table is never full (the load factor stays below 0.7), so a probe ends at a free slot
    for (size_t probe = 0; probe < index_cap; ++probe, i = (i + 1) & mask) {
        const uint8_t m = meta[i];
        if (m == 0) {
            return insert ? i : npos;
        }
        if (m == fp) {
            const slot & s = slots[i];
            if (s.k0 == k0 && s.k1 == k1) {
                return i;
            }
        }
    }

    return npos;
}

common_ngram_part common_ngram_cache::find(const common_ngram & ngram) const {
    const uint64_t k0 = key_part(ngram.tokens, 0);
    const uint64_t k1 = key_part(ngram.tokens, 2);

    const size_t i = find_slot(k0, k1, ngram_fingerprint(ngram_hash(k0, k1)), false);
    if (i == npos) {
        return {};
    }
    return part_of(slots[i]);
}

void common_ngram_cache::add(const common_ngram & ngram, llama_token token, int32_t count) {
    const uint64_t k0 = key_part(ngram.tokens, 0);
    const uint64_t k1 = key_part(ngram.tokens, 2);
    const uint8_t  fp = ngram_fingerprint(ngram_hash(k0, k1));

    size_t i = find_slot(k0, k1, fp, true);

    // no table yet, or the group is new and the table would get too full to stay fast: grow, then look again
    if (index_cap == 0 || (i != npos && meta[i] == 0 && (double) (n_used + 1) > ngram_max_load * (double) index_cap)) {
        rehash(index_cap == 0 ? ngram_min_index_bits : index_bits_ + 1);
        i = find_slot(k0, k1, fp, true);
        if (i == npos) {
            return;
        }
    }

    slot & s = slots[i];

    if (meta[i] == 0) {
        meta[i]       = fp;
        s.k0          = k0;
        s.k1          = k1;
        s.off         = (uint32_t) arena.size();
        s.n           = 0;
        s.cap         = 0;
        s.total       = 0;
        s.max_count   = 0;
        s.max_token   = LLAMA_TOKEN_NULL;
        ++n_parts;
        ++n_used;
    }

    if (s.n == 0) {
        arena.push_back(common_ngram_entry{ token, count });
        s.off       = (uint32_t) (arena.size() - 1);
        s.n         = 1;
        s.cap       = 1;
        s.total     = count;
        s.max_count = count;
        s.max_token = token;
        return;
    }

    // the group is searched through an index, never a pointer, so growing the arena cannot dangle it
    const common_ngram_part part = part_of(s);
    const size_t            idx  = part.lower_bound(token);

    if (idx < s.n && part.entries[idx].token == token) {
        int32_t * dst = &arena[s.off + idx].count;
        *dst += count;
        s.total += count;
        if (*dst > s.max_count) {
            s.max_count = *dst;
            s.max_token = token;
        }
        return;
    }

    if (s.n < s.cap) {
        // the group's reserved tail is still its own: open a hole and step into it
        if (idx < s.n) {
            std::memmove(&arena[s.off + idx + 1], &arena[s.off + idx], (s.n - idx) * sizeof(common_ngram_entry));
        }
        arena[s.off + idx] = common_ngram_entry{ token, count };
        s.n += 1;
    } else {
        // no room left in its region: move the group to a fresh, larger one at the end of the arena
        const uint32_t cap = std::max<uint32_t>(2 * s.cap, 4);
        const uint32_t off = (uint32_t) arena.size();
        const uint32_t n   = s.n;
        const uint32_t old = s.off;

        arena.resize(arena.size() + cap);
        std::memcpy(&arena[off], &arena[old], n * sizeof(common_ngram_entry));
        if (idx < n) {
            std::memmove(&arena[off + idx + 1], &arena[off + idx], (n - idx) * sizeof(common_ngram_entry));
        }
        arena[off + idx] = common_ngram_entry{ token, count };

        // the old region is dead space until the cache is cleared; growth doubles, so it stays bounded
        s.off = off;
        s.n   = n + 1;
        s.cap = cap;
    }

    s.total += count;
    if (count > s.max_count) {
        s.max_count = count;
        s.max_token = token;
    }
}

//
// building
//

void common_ngram_cache_update(common_ngram_cache & ngram_cache, int ngram_min, int ngram_max,
                              std::vector<llama_token> & inp, int nnew, bool print_progress) {
    const int64_t t_start_ms = (int64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count();
    const int64_t inp_size = inp.size();

    const int64_t n_todo = inp_size * (ngram_max - ngram_min + 1);
    int64_t n_done = 0;


    for (int64_t ngram_size = ngram_min; ngram_size <= ngram_max; ++ngram_size) {
        const int64_t i_start = std::max(inp_size - nnew, ngram_size);
        for (int64_t i = i_start; i < inp_size; ++i) {
            const int64_t ngram_start = i - ngram_size;
            common_ngram ngram(&inp[ngram_start], (int) ngram_size);
            const llama_token token = inp[i];

            ngram_cache.add(ngram, token, 1);
            ++n_done;

            if (print_progress && n_done % 10000000 == 0) {
                const int64_t t_now_ms = (int64_t) std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();
                const int64_t eta_ms   = (inp_size*(ngram_max-ngram_min+1) - n_done) * (t_now_ms - t_start_ms) / n_done;
                const int64_t eta_min  = eta_ms / (60*1000);
                const int64_t eta_s    = (eta_ms - 60*1000*eta_min) / 1000;

                fprintf(stderr, "%s: %" PRId64 "/%" PRId64 " done, ETA: %02" PRId64 ":%02" PRId64 "\n", __func__, n_done, n_todo, eta_min, eta_s);
            }
        }
    }
}

//
// drafting
//

// If sample size or percentage are below these thresholds the draft is aborted early:
constexpr int    draft_min_sample_size_lax[LLAMA_NGRAM_MAX] = { 2,  2,  1,  1};
constexpr int        draft_min_percent_lax[LLAMA_NGRAM_MAX] = {66, 50, 50, 50};
constexpr int draft_min_sample_size_strict[LLAMA_NGRAM_MAX] = { 4,  3,  2,  2};
constexpr int     draft_min_percent_strict[LLAMA_NGRAM_MAX] = {75, 66, 66, 66};

// Helper function that tries to draft a token from only the static ngram cache:
// the group carries its total and its most frequent follower, so both thresholds and the candidate are read
// without scanning the followers at all
static llama_token try_draft(const common_ngram_cache & nc_static, const common_ngram & ngram_static) {
    const common_ngram_part part_static = nc_static.find(ngram_static);
    if (part_static.empty()) {
        return LLAMA_TOKEN_NULL;
    }

    if (part_static.total < draft_min_sample_size_lax[LLAMA_NGRAM_STATIC-1]) {
        return LLAMA_TOKEN_NULL;
    }
    if (100*part_static.max_count < draft_min_percent_lax[LLAMA_NGRAM_STATIC-1]*part_static.total) {
        return LLAMA_TOKEN_NULL;
    }
    return part_static.max_token;
}

// One candidate n-gram of a drafting step, with the index its thresholds live at
struct draft_ngram {
    common_ngram ngram;
    int          k; // n - 1, into the sample-size and percentage tables
};

// Try to draft a token from primary cache (context/dynamic), validate with static cache:
static llama_token try_draft(
    const common_ngram_cache & nc_primary, const draft_ngram * ngrams_primary, size_t n_ngrams,
    const common_ngram_part & part_static, const int * min_sample_size, const int * min_percent) {

    for (size_t i = n_ngrams; i-- > 0;) {
        const int k = ngrams_primary[i].k;
        const common_ngram_part part_primary = nc_primary.find(ngrams_primary[i].ngram);
        if (part_primary.empty()) {
            continue;
        }

        // the two thresholds before any follower is scored. the sample size is one of the group's aggregates;
        // and since every candidate's primary count is at most max_count, a max_count under the ratio rules the
        // whole group out whatever the static cache says - the test is always on the selected candidate's
        // primary count, never on the weighted product
        if (part_primary.total < min_sample_size[k]) {
            continue;
        }
        if (100*part_primary.max_count < min_percent[k]*part_primary.total) {
            continue;
        }

        llama_token max_token         = LLAMA_TOKEN_NULL;
        int64_t     max_count_primary = 0;
        int64_t     max_count_static  = 0;
        int64_t     max_product       = 0;

        if (part_static.empty()) {
            // nothing to reweight with: the argmax of the scores is the group's most frequent follower
            max_token = part_primary.max_token;
        } else {
            const common_ngram_entry * e = part_primary.entries;
            for (uint32_t j = 0; j < part_primary.n; ++j, ++e) {
                const common_ngram_entry * st = part_static.find(e->token);

                const int64_t count_primary = e->count;
                const int64_t count_static  = st ? 100*st->count : 1;
                const int64_t product       = count_primary*count_static;

                // among equally scored candidates take the one with the strongest primary count: the ratio
                // threshold is tested on that count, so with the upstream rule (the first maximum in the bucket
                // order of a hash map) the same cache could pass or fail the threshold depending on hashing
                if (product > max_product || (product == max_product && count_primary > max_count_primary)) {
                    max_token         = e->token;
                    max_count_primary = count_primary;
                    max_count_static  = count_static;
                    max_product       = product;
                }
            }

            // the sample-size threshold is a test on the group total (the aggregate check above covered it);
            // the ratio test is on the count of the candidate the weighting picked, so it needs re-checking here
            if (max_token == LLAMA_TOKEN_NULL || 100*max_count_primary < min_percent[k]*part_primary.total) {
                continue;
            }
        }

        return max_token;
    }

    return LLAMA_TOKEN_NULL;
}

// Helper function to get a token from the combined, speculative sequence of inp and draft.
static llama_token get_token(const std::vector<llama_token> & inp, const std::vector<llama_token> & draft, const size_t i) {
    return i < inp.size() ? inp[i] : draft[1 + i - inp.size()];
}

void common_ngram_cache_draft(
    std::vector<llama_token> & inp, std::vector<llama_token> & draft, int n_draft, int ngram_min, int ngram_max,
    const common_ngram_cache & nc_context, const common_ngram_cache & nc_dynamic, const common_ngram_cache & nc_static
) {
    // upstream required the draft to hold exactly the sampled token; a longer prefix is fine too and lets a
    // caller ask for one more drafted token at a time (see tests/test-ngram-cache.cpp)
    GGML_ASSERT(draft.size() >= 1);
    const int inp_size = inp.size();

    if (inp_size < LLAMA_NGRAM_STATIC) {
        return;
    }

    // one stack buffer for every n-gram size of the step: the drafting loop allocates nothing
    draft_ngram ngrams[LLAMA_NGRAM_MAX + 1];

    while ((int) draft.size()-1 < n_draft) {
        llama_token drafted_token = LLAMA_TOKEN_NULL;

        const int ngram_start_static = inp_size-LLAMA_NGRAM_STATIC + (int) draft.size()-1;
        common_ngram ngram_static;
        for (int j = ngram_start_static; j < ngram_start_static + LLAMA_NGRAM_STATIC; ++j) {
            ngram_static.tokens[j-ngram_start_static] = get_token(inp, draft, j);
        }
        const common_ngram_part part_static = nc_static.find(ngram_static);

        // an n-gram needs its whole window inside the sequence, so the sizes that do not fit yet are skipped
        // (the upstream loop asked for the token before the start of the sequence instead)
        int n_ngrams = 0;
        for (int ngram_size_cd = ngram_min; ngram_size_cd <= ngram_max; ++ngram_size_cd) {
            const int ngram_start_cd = inp_size - ngram_size_cd + (int) draft.size()-1;
            if (ngram_start_cd < 0) {
                continue;
            }

            draft_ngram & cd = ngrams[n_ngrams];
            cd.k = ngram_size_cd - 1;
            for (int j = 0; j < LLAMA_NGRAM_MAX; ++j) {
                cd.ngram.tokens[j] = LLAMA_TOKEN_NULL;
            }
            for (int j = ngram_start_cd; j < ngram_start_cd + ngram_size_cd; ++j) {
                cd.ngram.tokens[j-ngram_start_cd] = get_token(inp, draft, j);
            }
            ++n_ngrams;
        }

        if (drafted_token == LLAMA_TOKEN_NULL) {
            drafted_token = try_draft(nc_context, ngrams, n_ngrams, part_static,
                                      draft_min_sample_size_lax, draft_min_percent_lax);
        }
        if (drafted_token == LLAMA_TOKEN_NULL) {
            drafted_token = try_draft(nc_dynamic, ngrams, n_ngrams, part_static,
                                      draft_min_sample_size_strict, draft_min_percent_strict);
        }
        if (drafted_token == LLAMA_TOKEN_NULL) {
            drafted_token = try_draft(nc_static, ngram_static);
        }

        if (drafted_token == LLAMA_TOKEN_NULL) {
            break;
        }

        draft.push_back(drafted_token);
    }
}

//
// persistence
//
// v2 stores the three tables of the cache as they lie in memory: the fingerprint array, the slot array and the
// arena of followers. Loading is the header plus three bulk reads - no parsing, no hashing, no re-adding - and
// the lookups run straight out of what was read.

namespace {

constexpr char     ngram_magic[8] = { 'N', 'G', 'M', 'C', 'a', 'c', 'h', '2' };
constexpr uint32_t ngram_version  = 2;

struct ngram_file_header {
    char     magic[8];
    uint32_t version;
    uint32_t flags;
    uint64_t index_bits;
    uint64_t n_parts;
    uint64_t n_entries;
    uint64_t reserved[3];
};
static_assert(sizeof(ngram_file_header) == 64, "the header is one cache line");

} // namespace

uint64_t common_ngram_cache::save_index_bits() const {
    return index_bits_;
}

size_t common_ngram_cache::n_live_entries() const {
    size_t live = 0;
    for (size_t i = 0; i < index_cap; ++i) {
        if (meta[i]) {
            live += slots[i].n;
        }
    }
    return live;
}

void common_ngram_cache::compact() {
    std::vector<common_ngram_entry> packed;
    packed.reserve(n_live_entries());

    for (size_t i = 0; i < index_cap; ++i) {
        if (!meta[i]) {
            continue;
        }
        slot &         s    = slots[i];
        const uint32_t from = s.off;
        s.off = (uint32_t) packed.size();
        packed.insert(packed.end(), arena.begin() + from, arena.begin() + from + s.n);
        s.cap = s.n; // the region is exactly the group again: the next insert grows it
    }

    arena.swap(packed);
}

// the file holds the live followers and nothing else: a group that outgrew its reserved region left its old copy
// in the arena, and a table that grew by rehashing holds empty slots, so the slots are written with new offsets
void common_ngram_cache::save_tables(std::ostream & out) const {
    std::vector<common_ngram_entry> packed;
    packed.reserve(n_live_entries());

    std::vector<slot>      copy    = slots;
    std::vector<uint32_t>  new_off(index_cap, 0);

    for (size_t i = 0; i < index_cap; ++i) {
        if (!meta[i]) {
            continue;
        }
        const slot & s = slots[i];
        new_off[i]     = (uint32_t) packed.size();
        packed.insert(packed.end(), arena.begin() + s.off, arena.begin() + s.off + s.n);
    }
    for (size_t i = 0; i < index_cap; ++i) {
        if (meta[i]) {
            copy[i].off = new_off[i];
        }
    }

    if (!meta.empty()) {
        out.write(reinterpret_cast<const char *>(meta.data()), (std::streamsize) meta.size());
    }
    if (!copy.empty()) {
        out.write(reinterpret_cast<const char *>(copy.data()), (std::streamsize) copy.size() * sizeof(slot));
    }
    if (!packed.empty()) {
        out.write(reinterpret_cast<const char *>(packed.data()),
                  (std::streamsize) packed.size() * sizeof(common_ngram_entry));
    }
}

void common_ngram_cache::load_tables(uint64_t index_bits, uint64_t n_parts_hint, uint64_t n_entries, std::istream & in) {
    if (index_bits == 0 || index_bits > 40) {
        throw std::runtime_error("ngram cache: bad index size in the cache file");
    }

    // the groups are addressed by offset, so the file's slot array is the table: nothing to rebuild
    index_bits_ = (size_t) index_bits;
    index_cap   = (size_t) 1 << index_bits_;

    meta.assign(index_cap, 0);
    slots.assign(index_cap, slot{});
    arena.resize((size_t) n_entries);

    in.read(reinterpret_cast<char *>(meta.data()),  (std::streamsize) meta.size());
    in.read(reinterpret_cast<char *>(slots.data()), (std::streamsize) slots.size() * sizeof(slot));
    if (n_entries > 0) {
        in.read(reinterpret_cast<char *>(arena.data()), (std::streamsize) arena.size() * sizeof(common_ngram_entry));
    }
    if (!in) {
        throw std::runtime_error("ngram cache: truncated cache file");
    }

    n_parts = 0;
    n_used  = 0;
    for (size_t i = 0; i < index_cap; ++i) {
        if (meta[i] != 0) {
            ++n_used;
            ++n_parts;
        }
    }
    if (n_parts_hint != 0 && n_parts != n_parts_hint) {
        // not fatal: the table is consistent with itself, the hint only came from the header
        fprintf(stderr, "%s: header says %" PRIu64 " n-grams, the table holds %zu\n", __func__, n_parts_hint, n_parts);
    }
}

void common_ngram_cache_save(const common_ngram_cache & ngram_cache, const std::string & filename) {
    std::ofstream file_out(filename, std::ios::binary);
    if (!file_out) {
        throw std::ios_base::failure("Unable to write ngram cache to file " + filename);
    }

    ngram_file_header header{};
    std::memcpy(header.magic, ngram_magic, sizeof(ngram_magic));
    header.version    = ngram_version;
    header.flags      = 0;
    header.index_bits = ngram_cache.save_index_bits();
    header.n_parts    = ngram_cache.size();
    header.n_entries  = ngram_cache.n_live_entries();

    file_out.write(reinterpret_cast<const char *>(&header), sizeof(header));
    ngram_cache.save_tables(file_out);
}

common_ngram_cache common_ngram_cache_load(const std::string & filename) {
    std::ifstream hashmap_file(filename, std::ios::binary);
    if (!hashmap_file) {
        throw std::ifstream::failure("Unable to open file " + filename);
    }

    ngram_file_header header{};
    if (!hashmap_file.read(reinterpret_cast<char *>(&header), sizeof(header))) {
        throw std::ifstream::failure("Unable to read the header of " + filename);
    }

    common_ngram_cache ngram_cache;

    if (std::memcmp(header.magic, ngram_magic, sizeof(ngram_magic)) == 0 && header.version == ngram_version) {
        ngram_cache.load_tables(header.index_bits, header.n_parts, header.n_entries, hashmap_file);
        return ngram_cache;
    }

    // the pre-v2 format: one record per n-gram, [key][ntokens][token, count]*. read from the start
    hashmap_file.seekg(0, std::ios::beg);

    common_ngram ngram;
    int32_t      ntokens;
    llama_token  token;
    int32_t      count;

    while (hashmap_file.read(reinterpret_cast<char *>(&ngram), sizeof(common_ngram))) {
        if (!hashmap_file.read(reinterpret_cast<char *>(&ntokens), sizeof(int32_t))) {
            break;
        }
        if (ntokens <= 0) {
            break;
        }
        for (int i = 0; i < ntokens; ++i) {
            if (!hashmap_file.read(reinterpret_cast<char *>(&token), sizeof(llama_token))) {
                break;
            }
            if (!hashmap_file.read(reinterpret_cast<char *>(&count), sizeof(int32_t))) {
                break;
            }
            if (count <= 0) {
                continue;
            }
            ngram_cache.add(ngram, token, count);
        }
    }

    return ngram_cache;
}

common_ngram_cache_ptr common_ngram_cache_load_shared(const std::string & filename) {
    return std::make_shared<common_ngram_cache>(common_ngram_cache_load(filename));
}

void common_ngram_cache_merge(common_ngram_cache & ngram_cache_target, const common_ngram_cache & ngram_cache_add) {
    ngram_cache_target.reserve(ngram_cache_target.size() + ngram_cache_add.size());

    ngram_cache_add.for_each([&](const common_ngram & ngram, const common_ngram_part part) {
        const common_ngram_entry * e = part.entries;
        for (uint32_t i = 0; i < part.n; ++i, ++e) {
            if (e->count > 0) {
                ngram_cache_target.add(ngram, e->token, e->count);
            }
        }
    });
}
