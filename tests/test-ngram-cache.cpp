// Tests and benchmarks for the n-gram (prompt lookup) caches.
//
// part 1  unit tests of the flat cache: the branchless search, the aggregates the drafting thresholds read, and
//         the on-disk round trip (both the new format and the one older llama.cpp wrote).
// part 2  equivalence against the upstream map-of-maps algorithm, transcribed here over std::unordered_map:
//         every n-gram, follower count and aggregate must match, and every drafted decision must be one the
//         upstream rule can produce. Upstream picks the first maximum in the bucket order of a hash map, so two
//         runs of it can disagree; the flat cache instead takes the strongest primary count among equally scored
//         candidates, which is what the reference here checks against.
// part 3  the benchmark: build, save, load and drafting latency, one variant per process so each peak RSS is its
//         own.
//
// build:
//   g++ -std=c++17 -O2 -I common -I include -I ggml/include
//       tests/test-ngram-cache.cpp common/ngram-cache.cpp -o test-ngram-cache
// run:
//   ./test-ngram-cache                      # unit + equivalence tests
//   ./test-ngram-cache --bench ref|new      # benchmark

#include "ngram-cache.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

// Standalone builds (see the header of this file) compile ngram-cache.cpp next to the test and link nothing, so
// the assert hook comes from here. In the CMake build the hook comes from libggml with llama-common.
#ifndef TEST_NGRAM_COMMON_LINKED
extern "C" void ggml_abort(const char * file, int line, const char * fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "abort at %s:%d: ", file, line);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, "\n");
    abort();
}
#endif // TEST_NGRAM_COMMON_LINKED

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++failures; \
    } \
} while (0)

static int failures = 0;

// ------------------------------------------------------------------
// part 1: the flat cache
// ------------------------------------------------------------------

static void test_branchless_lower_bound() {
    std::mt19937_64 rng(12345);

    for (int trial = 0; trial < 20000; ++trial) {
        const int n = 1 + (int) (rng() % 48);
        std::vector<llama_token> tokens;
        for (int i = 0; i < n; ++i) {
            tokens.push_back((llama_token) (rng() % 500));
        }
        std::sort(tokens.begin(), tokens.end());
        tokens.erase(std::unique(tokens.begin(), tokens.end()), tokens.end());

        std::vector<common_ngram_entry> entries;
        for (auto t : tokens) {
            entries.push_back(common_ngram_entry{ t, 1 });
        }
        common_ngram_part part;
        part.entries = entries.data();
        part.n       = (uint32_t) entries.size();

        for (int probe = 0; probe < 8; ++probe) {
            const llama_token token = (llama_token) ((int) (rng() % 520) - 10);
            const size_t expected = std::lower_bound(tokens.begin(), tokens.end(), token) - tokens.begin();
            CHECK(part.lower_bound(token) == expected);
            CHECK((part.find(token) != nullptr) == std::binary_search(tokens.begin(), tokens.end(), token));
        }
    }
}

struct cache_model {
    std::unordered_map<uint64_t, std::vector<std::pair<llama_token, int32_t>>> m;
    static uint64_t key(const common_ngram & n) {
        uint64_t h = 0x9e3779b97f4a7c15ull;
        for (int i = 0; i < LLAMA_NGRAM_MAX; ++i) {
            h = (h ^ (uint64_t) (uint32_t) n.tokens[i]) * 0x100000001b3ull;
        }
        return h;
    }
};

static void test_cache_matches_model() {
    std::mt19937_64 rng(777);

    common_ngram_cache cache;
    cache_model        model;

    for (int step = 0; step < 200000; ++step) {
        common_ngram n;
        const int    size = 1 + (int) (rng() % LLAMA_NGRAM_MAX);
        for (int i = 0; i < LLAMA_NGRAM_MAX; ++i) {
            n.tokens[i] = i < size ? (llama_token) (rng() % (4 + step / 1000)) : LLAMA_TOKEN_NULL;
        }
        const llama_token token = (llama_token) (rng() % 5000);

        cache.add(n, token, 1);
        model.m[cache_model::key(n)].push_back({ token, 1 });

        if (step % 997 == 0) {
            const common_ngram_part part = cache.find(n);
            auto &                  ref  = model.m[cache_model::key(n)];

            int32_t total = 0, max_count = 0;
            std::unordered_map<llama_token, int32_t> counts;
            for (auto & tc : ref) {
                counts[tc.first]++;
            }
            for (auto & kv : counts) {
                total += kv.second;
                if (kv.second > max_count) {
                    max_count = kv.second;
                }
                CHECK(part.get(kv.first) == kv.second);
            }
            CHECK((int32_t) counts.size() == (int32_t) part.n);
            CHECK(part.total == total);
            CHECK(part.max_count == max_count);
            CHECK(part.get(part.max_token) == part.max_count);

            for (uint32_t i = 1; i < part.n; ++i) { // the group must stay sorted for the search
                CHECK(part.entries[i - 1].token < part.entries[i].token);
            }
        }
    }

    CHECK(cache.size() == model.m.size());
}

static void test_save_load_roundtrip() {
    std::mt19937_64 rng(4242);

    common_ngram_cache cache;
    for (int step = 0; step < 40000; ++step) {
        common_ngram n;
        const int    size = 1 + (int) (rng() % LLAMA_NGRAM_MAX);
        for (int i = 0; i < LLAMA_NGRAM_MAX; ++i) {
            n.tokens[i] = i < size ? (llama_token) (rng() % 300) : LLAMA_TOKEN_NULL;
        }
        cache.add(n, (llama_token) (rng() % 1000), 1 + (int) (rng() % 3));
    }

    const char * path = "test-ngram-cache-v2.bin";
    common_ngram_cache_save(cache, path);

    common_ngram_cache loaded = common_ngram_cache_load(path);
    CHECK(loaded.size() == cache.size());
    // the file holds the live followers, so the loaded arena is smaller than a built one that grew in place
    CHECK(loaded.n_entries() == cache.n_live_entries());

    size_t checked = 0;
    cache.for_each([&](const common_ngram & n, const common_ngram_part a) {
        const common_ngram_part b = loaded.find(n);
        CHECK(a.n == b.n);
        CHECK(a.total == b.total);
        CHECK(a.max_count == b.max_count);
        for (uint32_t i = 0; i < a.n; ++i) {
            CHECK(a.entries[i].token == b.entries[i].token);
            CHECK(a.entries[i].count == b.entries[i].count);
        }
        ++checked;
    });
    CHECK(checked == cache.size());

    // a loaded table is a table: writing it again must not lose anything
    const char * path2 = "test-ngram-cache-v2b.bin";
    common_ngram_cache_save(loaded, path2);
    common_ngram_cache again = common_ngram_cache_load(path2);
    CHECK(again.size() == cache.size());
    CHECK(again.n_entries() == cache.n_live_entries());

    std::remove(path);
    std::remove(path2);
}

// the pre-v2 file format, written by hand: [ngram:4 x int32][ntokens:int32][token,count]*
static void test_load_v1() {
    const char * path = "test-ngram-cache-v1.bin";

    std::ofstream out(path, std::ios::binary);
    for (int n = 0; n < 200; ++n) {
        common_ngram ng;
        ng.tokens[0] = (llama_token) n;
        const int32_t ntok = 3;
        out.write(reinterpret_cast<const char *>(&ng), sizeof(common_ngram));
        out.write(reinterpret_cast<const char *>(&ntok), sizeof(int32_t));
        for (int32_t t = 0; t < ntok; ++t) {
            const int32_t token = 1000 + 7*n + t;
            const int32_t count = 1 + t;
            out.write(reinterpret_cast<const char *>(&token), sizeof(int32_t));
            out.write(reinterpret_cast<const char *>(&count), sizeof(int32_t));
        }
    }
    out.close();

    common_ngram_cache loaded = common_ngram_cache_load(path);
    CHECK(loaded.size() == 200);
    for (int n = 0; n < 200; ++n) {
        common_ngram ng;
        ng.tokens[0] = (llama_token) n;
        const common_ngram_part part = loaded.find(ng);
        CHECK(part.n == 3);
        CHECK(part.total == 6);
        CHECK(part.max_count == 3);
        CHECK(part.get(1000 + 7*n + 2) == 3);
    }
    std::remove(path);
}

// ------------------------------------------------------------------
// part 2: the upstream algorithm, transcribed over std::unordered_map
// ------------------------------------------------------------------

namespace upstream {

struct ngram {
    std::array<llama_token, LLAMA_NGRAM_MAX> t;
    ngram() {
        t.fill(LLAMA_TOKEN_NULL);
    }
    bool operator==(const ngram & o) const {
        return t == o.t;
    }
};

struct ngram_hash {
    size_t operator()(const ngram & n) const {
        size_t h = 1469598103934665603ull;
        for (auto tok : n.t) {
            h = (h ^ (uint64_t) tok) * 1099511628211ull;
        }
        return h;
    }
};

using part  = std::unordered_map<llama_token, int32_t>;
using cache = std::unordered_map<ngram, part, ngram_hash>;

static void update(cache & c, int ngram_min, int ngram_max, std::vector<llama_token> & inp, int nnew) {
    const int64_t inp_size = inp.size();
    for (int64_t size = ngram_min; size <= ngram_max; ++size) {
        const int64_t i_start = std::max(inp_size - nnew, size);
        for (int64_t i = i_start; i < inp_size; ++i) {
            ngram ng;
            for (int64_t j = 0; j < size; ++j) {
                ng.t[j] = inp[i - size + j];
            }
            c[ng][inp[i]] += 1;
        }
    }
}

static const int min_sample_lax[LLAMA_NGRAM_MAX]     = { 2,  2,  1,  1};
static const int min_percent_lax[LLAMA_NGRAM_MAX]    = {66, 50, 50, 50};
static const int min_sample_strict[LLAMA_NGRAM_MAX]  = { 4,  3,  2,  2};
static const int min_percent_strict[LLAMA_NGRAM_MAX] = {75, 66, 66, 66};

struct best {
    bool        passes  = false;
    llama_token token   = LLAMA_TOKEN_NULL;
    int64_t     product = 0;
    int64_t     sum     = 0;
};

// the upstream rule: the first candidate whose score exceeds the running maximum wins
static best best_of(const part & primary, const part & st, int n_index, const int * a, const int * p) {
    best out;
    int64_t max_primary = 0, max_static = 0;
    for (auto & kv : primary) {
        const int64_t count_primary = kv.second;
        const auto    it            = st.find(kv.first);
        const int64_t count_static  = it != st.end() ? 100*it->second : 1;
        if (count_primary*count_static > max_primary*max_static) {
            out.token   = kv.first;
            max_primary = count_primary;
            max_static  = count_static;
        }
        out.sum += count_primary;
    }
    out.product = max_primary*max_static;
    if (out.sum < a[n_index]) {
        return out;
    }
    if (100*max_primary < p[n_index]*out.sum) {
        return out;
    }
    out.passes = true;
    return out;
}

// the same rule with the deterministic tie-break the flat cache uses: among equally scored candidates, the one
// with the strongest primary count (the threshold is tested on that count)
static best best_of_strongest(const part & primary, const part & st, int n_index, const int * a, const int * p) {
    best out;
    int64_t max_primary = 0, max_product = 0;
    for (auto & kv : primary) {
        const int64_t count_primary = kv.second;
        const auto    it            = st.find(kv.first);
        const int64_t count_static  = it != st.end() ? 100*it->second : 1;
        const int64_t product       = count_primary*count_static;
        if (product > max_product || (product == max_product && count_primary > max_primary)) {
            out.token   = kv.first;
            max_primary = count_primary;
            max_product = product;
        }
        out.sum += count_primary;
    }
    out.product = max_product;
    if (out.sum < a[n_index] || 100*max_primary < p[n_index]*out.sum) {
        return out;
    }
    out.passes = true;
    return out;
}

} // namespace upstream

// the token at index i of the sequence formed by `inp` followed by the drafted tokens
static llama_token token_at(const std::vector<llama_token> & inp, const std::vector<llama_token> & draft, size_t i) {
    return i < inp.size() ? inp[i] : draft[1 + i - inp.size()];
}

struct ref_decision {
    bool        used   = false;
    bool        is_dyn = false;
    bool        is_st  = false;
    int         n      = 0;
    llama_token token  = LLAMA_TOKEN_NULL;
};

// the upstream drafting rule, with either tie-break
static ref_decision upstream_decide(const upstream::cache & ctx, const upstream::cache & dyn, const upstream::cache & st,
                                    const std::vector<llama_token> & inp, const std::vector<llama_token> & draft_sofar,
                                    bool strongest_tie) {
    ref_decision out;
    const int    n_draft = (int) draft_sofar.size() - 1;

    upstream::part part_static;
    const int      start_static = (int) inp.size() - LLAMA_NGRAM_STATIC + n_draft;
    if (start_static >= 0) {
        upstream::ngram ng;
        ng.t[0] = token_at(inp, draft_sofar, start_static);
        ng.t[1] = token_at(inp, draft_sofar, start_static + 1);
        const auto it = st.find(ng);
        if (it != st.end()) {
            part_static = it->second;
        }
    }

    for (int pass = 0; pass < 3; ++pass) {
        const upstream::cache * primary = pass == 0 ? &ctx : pass == 1 ? &dyn : &st;
        const int * a  = pass == 1 ? upstream::min_sample_strict  : upstream::min_sample_lax;
        const int * p  = pass == 1 ? upstream::min_percent_strict : upstream::min_percent_lax;
        const int   n_min = pass == 2 ? LLAMA_NGRAM_STATIC : LLAMA_NGRAM_MIN;
        const int   n_max = pass == 2 ? LLAMA_NGRAM_STATIC : LLAMA_NGRAM_MAX;

        for (int n = n_max; n >= n_min; --n) {
            const int start = (int) inp.size() - n + n_draft;
            if (start < 0) {
                continue; // the window does not fit in the sequence yet
            }
            upstream::ngram ng;
            for (int j = 0; j < n; ++j) {
                ng.t[j] = token_at(inp, draft_sofar, start + j);
            }
            const auto it = primary->find(ng);
            if (it == primary->end()) {
                continue;
            }
            const upstream::part empty;
            const upstream::best b = strongest_tie
                ? upstream::best_of_strongest(it->second, pass == 2 ? empty : part_static, n - 1, a, p)
                : upstream::best_of(it->second, pass == 2 ? empty : part_static, n - 1, a, p);
            if (b.passes) {
                out.used    = true;
                out.is_dyn  = pass == 1;
                out.is_st   = pass == 2;
                out.n       = n;
                out.token   = b.token;
                return out;
            }
        }
    }
    return out;
}

// the score of a candidate in the group of the n-gram at the current draft position
static int64_t upstream_score(const upstream::cache & cache, const upstream::cache & st,
                              const std::vector<llama_token> & inp, const std::vector<llama_token> & draft_sofar,
                              int n, llama_token token, bool use_static_part) {
    const int n_draft = (int) draft_sofar.size() - 1;
    const int start   = (int) inp.size() - n + n_draft;
    if (start < 0) {
        return -1;
    }
    upstream::ngram ng;
    for (int j = 0; j < n; ++j) {
        ng.t[j] = token_at(inp, draft_sofar, start + j);
    }
    const auto it = cache.find(ng);
    if (it == cache.end()) {
        return -1;
    }
    int64_t weight = 1;
    if (use_static_part) {
        const int start_static = (int) inp.size() - LLAMA_NGRAM_STATIC + n_draft;
        if (start_static >= 0) {
            upstream::ngram sung;
            sung.t[0] = token_at(inp, draft_sofar, start_static);
            sung.t[1] = token_at(inp, draft_sofar, start_static + 1);
            const auto ist = st.find(sung);
            if (ist != st.end()) {
                const auto it_st = ist->second.find(token);
                if (it_st != ist->second.end()) {
                    weight = 100*it_st->second;
                }
            }
        }
    }
    const auto it_t = it->second.find(token);
    if (it_t == it->second.end()) {
        return -1;
    }
    return (int64_t) it_t->second * weight;
}

// ------------------------------------------------------------------
// corpus
// ------------------------------------------------------------------

// a synthetic token stream with the two properties prompt lookup cares about: a Zipf-distributed vocabulary and
// long stretches that repeat earlier text (what makes a lookup draft worth its cost)
static std::vector<llama_token> make_corpus(size_t n, uint64_t seed, int vocab) {
    std::mt19937_64 rng(seed);

    std::vector<double> w(vocab);
    for (int r = 0; r < vocab; ++r) {
        w[r] = 1.0 / std::pow((double) r + 1.0, 0.62); // a spread like a real tokenizer: many rare tokens, a heavy head
    }
    std::discrete_distribution<int> pick(w.begin(), w.end());
    auto token = [&]() { return (llama_token) pick(rng); };

    std::vector<std::vector<llama_token>> phrases;
    phrases.reserve(4000);
    for (int p = 0; p < 4000; ++p) {
        const int len = 3 + (int) (rng() % 12);
        std::vector<llama_token> phrase;
        phrase.reserve(len);
        for (int i = 0; i < len; ++i) {
            phrase.push_back(token());
        }
        phrases.push_back(std::move(phrase));
    }

    std::vector<llama_token> out;
    out.reserve(n);
    while (out.size() < n) {
        const uint64_t r = rng() % 100;
        if (r < 55) {
            const auto & phrase = phrases[rng() % phrases.size()];
            out.insert(out.end(), phrase.begin(), phrase.end());
        } else if (r < 85 && out.size() > 2000) {
            const size_t span = 8 + rng() % 60;
            const size_t from = rng() % (out.size() - 2000);
            for (size_t i = 0; i < span && out.size() < n; ++i) {
                out.push_back(out[from + i]);
            }
        } else {
            out.push_back(token());
        }
    }
    out.resize(n);
    return out;
}

// ------------------------------------------------------------------
// the flat cache against the upstream map, entry by entry
// ------------------------------------------------------------------

static void test_cache_equals_upstream(const std::vector<llama_token> & corpus) {
    common_ngram_cache flat;
    upstream::cache    up;

    std::vector<llama_token> grow;
    for (size_t i = 0; i < corpus.size(); ++i) {
        grow.push_back(corpus[i]);
        common_ngram_cache_update(flat, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, grow, 1, false);
    }
    std::vector<llama_token> bulk = grow;
    upstream::update(up, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, bulk, (int) bulk.size());

    size_t entries = 0;
    for (auto & kv : up) {
        for (auto & tc : kv.second) {
            entries += tc.second;
        }
    }

    // verify, compact, verify again: compaction moves the groups but must not change a single count
    for (int pass = 0; pass < 2; ++pass) {
        if (pass) {
            const size_t live = flat.n_live_entries();
            flat.compact();
            CHECK(flat.n_entries() == live);
            CHECK(flat.n_live_entries() == live);
        }

        size_t mismatch = 0;
        CHECK(flat.size() == up.size());

        for (auto & kv : up) {
            common_ngram ng;
            for (int j = 0; j < LLAMA_NGRAM_MAX; ++j) {
                ng.tokens[j] = kv.first.t[j];
            }
            const common_ngram_part part = flat.find(ng);
            if (part.n != kv.second.size()) {
                ++mismatch;
                continue;
            }
            int32_t total = 0, max_count = 0;
            for (auto & tc : kv.second) {
                if (part.get(tc.first) != tc.second) {
                    ++mismatch;
                }
                total += tc.second;
                if (tc.second > max_count) {
                    max_count = tc.second;
                }
            }
            if (part.total != total || part.max_count != max_count || part.get(part.max_token) != max_count) {
                ++mismatch;
            }
        }

        printf("flat cache vs upstream map (pass %d): %zu n-grams, %zu follower counts, %zu mismatches\n",
               pass, flat.size(), entries, mismatch);
        CHECK(mismatch == 0);
    }
}

// ------------------------------------------------------------------
// the drafting decisions against the upstream rule
// ------------------------------------------------------------------

static void run_drafts_equiv(const std::vector<llama_token> & corpus, int n_draft_max, bool verbose) {
    common_ngram_cache ctx_new, dyn_new, st_new;
    upstream::cache    ctx_ref, dyn_ref, st_ref;

    std::vector<llama_token> train(corpus.begin(), corpus.begin() + corpus.size() / 2);
    std::vector<llama_token> train_a = train, train_b = train;
    common_ngram_cache_update(st_new, LLAMA_NGRAM_STATIC, LLAMA_NGRAM_STATIC, train_a, (int) train_a.size(), false);
    upstream::update(st_ref, LLAMA_NGRAM_STATIC, LLAMA_NGRAM_STATIC, train_b, (int) train_b.size());

    size_t checked = 0, mismatch = 0, ties = 0, drafted = 0, accepted = 0;

    std::vector<llama_token> inp;
    for (size_t i = 0; i < corpus.size(); ++i) {
        inp.push_back(corpus[i]);

        common_ngram_cache_update(ctx_new, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, inp, 1, false);
        std::vector<llama_token> inp_up = inp;
        upstream::update(ctx_ref, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, inp_up, 1);

        std::vector<llama_token> prefix = { corpus[i] };
        for (int k = 1; k <= n_draft_max; ++k) {
            std::vector<llama_token> d_new = prefix;
            common_ngram_cache_draft(inp, d_new, k, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, ctx_new, dyn_new, st_new);
            const llama_token t_new = d_new.size() > prefix.size() ? d_new.back() : LLAMA_TOKEN_NULL;

            const ref_decision dec = upstream_decide(ctx_ref, dyn_ref, st_ref, inp, prefix, true);
            const llama_token  t_ref = dec.used ? dec.token : LLAMA_TOKEN_NULL;

            if ((t_new == LLAMA_TOKEN_NULL) != (t_ref == LLAMA_TOKEN_NULL)) {
                ++mismatch;
                if (verbose && mismatch < 4) {
                    fprintf(stderr, "  pos %zu step %d: new drafted %d, the reference drafts %d\n", i, k, t_new, t_ref);
                }
                break;
            }
            if (t_new == LLAMA_TOKEN_NULL) {
                break; // neither side drafts
            }
            // the token must be an argmax of the group the rule used: upstream's own pick among equal scores is
            // the bucket order of a hash map
            const upstream::cache & which = dec.is_dyn ? dyn_ref : dec.is_st ? st_ref : ctx_ref;
            const int64_t s_new  = upstream_score(which, st_ref, inp, prefix, dec.n, t_new,  !dec.is_st);
            const int64_t s_ref  = upstream_score(which, st_ref, inp, prefix, dec.n, t_ref,  !dec.is_st);
            if (s_new != s_ref || s_new <= 0) {
                ++mismatch;
                if (verbose && mismatch < 4) {
                    fprintf(stderr, "  pos %zu step %d: new %d scores %lld, best candidate %d scores %lld (n=%d)\n",
                            i, k, t_new, (long long) s_new, t_ref, (long long) s_ref, dec.n);
                }
                break;
            }
            if (t_new != t_ref) {
                ++ties;
            }
            ++drafted;
            if (i + (size_t) k < corpus.size() && t_new == corpus[i + k]) {
                ++accepted;
            }
            prefix.push_back(t_new);
        }

        ++checked;
    }

    printf("draft equivalence: %zu positions, %zu drafted, %zu accepted (%.4f), %zu ties, %zu mismatches\n",
           checked, drafted, accepted, drafted ? (double) accepted / drafted : 0.0, ties, mismatch);
    CHECK(mismatch == 0);
}

// ------------------------------------------------------------------
// part 3: benchmark (one variant per process, so each peak RSS is its own)
// ------------------------------------------------------------------

static double rss_mb(const char * key) {
    std::ifstream f("/proc/self/status");
    std::string   line;
    while (std::getline(f, line)) {
        if (line.rfind(key, 0) == 0) {
            return atol(line.c_str() + strlen(key)) / 1024.0;
        }
    }
    return -1.0;
}

static double peak_rss_mb() {
    return rss_mb("VmHWM:");
}

// write the cache in the pre-v2 format the old loader had to parse, so the two load paths can be timed on the
// same contents
static void write_v1_file(const common_ngram_cache & cache, const char * path) {
    std::ofstream out(path, std::ios::binary);
    cache.for_each([&](const common_ngram & ngram, const common_ngram_part part) {
        const int32_t ntok = (int32_t) part.n;
        out.write(reinterpret_cast<const char *>(ngram.tokens), sizeof(ngram.tokens));
        out.write(reinterpret_cast<const char *>(&ntok), sizeof(int32_t));
        for (uint32_t i = 0; i < part.n; ++i) {
            const int32_t token = part.entries[i].token;
            const int32_t count = part.entries[i].count;
            out.write(reinterpret_cast<const char *>(&token), sizeof(int32_t));
            out.write(reinterpret_cast<const char *>(&count), sizeof(int32_t));
        }
    });
}

static int bench(const std::string & which, size_t n_corpus, size_t n_static, int vocab, int n_draft_max) {
    const auto t0 = std::chrono::steady_clock::now();
    std::vector<llama_token> train  = make_corpus(n_static, 1, vocab);
    std::vector<llama_token> corpus = make_corpus(n_corpus, 2, vocab);
    printf("corpus: %zu tokens (train %zu, vocab %d) in %.2f s, RSS now %.1f MB\n", corpus.size(), train.size(), vocab,
           std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), rss_mb("VmRSS:"));

    if (which == "new") {
        std::vector<llama_token> tr = train;
        const auto b0 = std::chrono::steady_clock::now();
        common_ngram_cache st;
        common_ngram_cache_update(st, LLAMA_NGRAM_STATIC, LLAMA_NGRAM_STATIC, tr, (int) tr.size(), false);
        const auto b1 = std::chrono::steady_clock::now();

        const char * path = "bench-ngram-static.bin";
        common_ngram_cache_save(st, path);
        const auto b2 = std::chrono::steady_clock::now();
        common_ngram_cache loaded = common_ngram_cache_load(path);
        const auto b3 = std::chrono::steady_clock::now();

        printf("static build: %.3f s | save: %.3f s | load: %.3f s\n",
               std::chrono::duration<double>(b1 - b0).count(),
               std::chrono::duration<double>(b2 - b1).count(),
               std::chrono::duration<double>(b3 - b2).count());
        printf("static cache: %zu n-grams, %zu followers, %.1f MB in memory, RSS now %.1f MB\n",
               loaded.size(), loaded.n_entries(), loaded.size_bytes() / 1e6, rss_mb("VmRSS:"));

        // the old format, parsed n-gram by n-gram, on the same contents
        const char * path_v1 = "bench-ngram-static-v1.bin";
        write_v1_file(st, path_v1);
        const auto v0 = std::chrono::steady_clock::now();
        common_ngram_cache loaded_v1 = common_ngram_cache_load(path_v1);
        const double v1_secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - v0).count();
        printf("load the same contents in the old format: %.3f s (%zu n-grams)\n",
               v1_secs, loaded_v1.size());
        std::remove(path_v1);

        common_ngram_cache ctx, dyn;
        std::vector<llama_token> inp;
        int64_t drafted = 0;
        const auto d0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < corpus.size(); ++i) {
            inp.push_back(corpus[i]);
            common_ngram_cache_update(ctx, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, inp, 1, false);
            std::vector<llama_token> d = { corpus[i] };
            common_ngram_cache_draft(inp, d, n_draft_max, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, ctx, dyn, loaded);
            drafted += (int64_t) d.size() - 1;
        }
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - d0).count();
        printf("drafting: %.3f s, %lld drafted tokens -> %.1f ns/draft, %.0f context tokens/s\n",
               secs, (long long) drafted, 1e9 * secs / std::max<int64_t>(drafted, 1), corpus.size() / secs);
        std::remove(path);
    } else {
        std::vector<llama_token> tr = train;
        const auto b0 = std::chrono::steady_clock::now();
        upstream::cache st;
        upstream::update(st, LLAMA_NGRAM_STATIC, LLAMA_NGRAM_STATIC, tr, (int) tr.size());
        const auto b1 = std::chrono::steady_clock::now();

        size_t entries = 0;
        for (auto & kv : st) {
            entries += kv.second.size();
        }
        printf("static build: %.3f s\n", std::chrono::duration<double>(b1 - b0).count());
        printf("static cache: %zu n-grams, %zu followers, RSS now %.1f MB\n",
               st.size(), entries, rss_mb("VmRSS:"));

        upstream::cache ctx, dyn;
        std::vector<llama_token> inp;
        int64_t drafted = 0;
        const auto d0 = std::chrono::steady_clock::now();
        for (size_t i = 0; i < corpus.size(); ++i) {
            inp.push_back(corpus[i]);
            upstream::update(ctx, LLAMA_NGRAM_MIN, LLAMA_NGRAM_MAX, inp, 1);
            std::vector<llama_token> d = { corpus[i] };
            while ((int) d.size() - 1 < n_draft_max) {
                const ref_decision dec = upstream_decide(ctx, dyn, st, inp, d, false);
                if (!dec.used) {
                    break;
                }
                d.push_back(dec.token);
            }
            drafted += (int64_t) d.size() - 1;
        }
        const double secs = std::chrono::duration<double>(std::chrono::steady_clock::now() - d0).count();
        printf("drafting: %.3f s, %lld drafted tokens -> %.1f ns/draft, %.0f context tokens/s\n",
               secs, (long long) drafted, 1e9 * secs / std::max<int64_t>(drafted, 1), corpus.size() / secs);
    }

    printf("peak RSS: %.1f MB\n", peak_rss_mb());
    return 0;
}

int main(int argc, char ** argv) {
    std::string which = "new";
    size_t      n_corpus = 400000, n_static = 400000;
    int         vocab = 100000, n_draft_max = 5;

    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--bench")) {
            which = i + 1 < argc ? argv[++i] : "new";
        } else if (!strcmp(argv[i], "--corpus")) {
            n_corpus = atol(argv[++i]);
        } else if (!strcmp(argv[i], "--static")) {
            n_static = atol(argv[++i]);
        } else if (!strcmp(argv[i], "--vocab")) {
            vocab = atoi(argv[++i]);
        } else if (!strcmp(argv[i], "--draft")) {
            n_draft_max = atoi(argv[++i]);
        }
    }

    if (argc > 1) {
        return bench(which, n_corpus, n_static, vocab, n_draft_max);
    }

    test_branchless_lower_bound();
    printf("branchless lower_bound: ok\n");
    test_cache_matches_model();
    printf("flat cache vs model: %s\n", failures ? "FAILED" : "ok");
    test_save_load_roundtrip();
    printf("save/load v2: %s\n", failures ? "FAILED" : "ok");
    test_load_v1();
    printf("load v1: %s\n", failures ? "FAILED" : "ok");

    test_cache_equals_upstream(make_corpus(120000, 5, 20000));
    run_drafts_equiv(make_corpus(30000, 5, 20000), 5, true);

    printf("%s\n", failures ? "TESTS FAILED" : "ALL TESTS PASSED");
    return failures ? 1 : 0;
}
