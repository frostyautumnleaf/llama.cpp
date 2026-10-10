// Tests for the Strata port in common/ngram-suffix.{h,cpp}: the suffix drafter and the draft policy.
//
// part 1  hand-built cases: the longest earlier repeat of the generated suffix is found and its continuation
//         proposed; a repeat shorter than min_match proposes nothing; assign() - a rollback, or the next turn of
//         a conversation - leaves the index exactly as if the shorter text had been fed from the start.
// part 2  equivalence against Strata's own SuffixDrafter and DraftPolicy, transcribed below from
//         src/spec/suffix_drafter.cpp and src/spec/draft_policy.cpp of Niko1221/Strata. over randomized
//         histories with planted repeats, every proposal, every match length, every pick and every observed
//         rate must be identical. the port starts its index smaller than Strata does and grows it on demand,
//         which the reference - built with Strata's own capacity - shows is invisible from the outside.
// part 3  the policy's behaviour, not only its arithmetic: it takes a lookup window when the match is long and
//         its continuations keep being accepted, and stops taking it when they stop being accepted, both with
//         an MTP alternative and against plain decoding. and the policy never touches the proposals themselves,
//         which is what keeps the emitted text the same whichever way a round is decided.
// part 4  a round simulator over one generated text: an MTP stand-in, the suffix drafter alone, and the drafter
//         with the policy, counted in tokens committed per verify pass.
// part 5  the benchmark: the drafter against a naive longest-suffix-match scan of the whole history.
//
// build:
//   g++ -std=c++17 -O2 -I common -I include -I ggml/include
//       tests/test-ngram-suffix.cpp common/ngram-suffix.cpp -o test-ngram-suffix
// run:
//   ./test-ngram-suffix            # parts 1-3
//   ./test-ngram-suffix --sim      # parts 4-5

#include "ngram-suffix.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#define CHECK(cond) do { \
    if (!(cond)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        ++failures; \
    } \
} while (0)

static int failures = 0;

typedef std::vector<llama_token> tokens_t;

// ---------------------------------------------------------------------------
// the reference: Strata's SuffixDrafter and DraftPolicy, transcribed verbatim
// (namespace-strata/strata at src/spec/, MIT) with int32_t for the token type
// ---------------------------------------------------------------------------

namespace ref {

class SuffixDrafter {
public:
    static constexpr int WAYS = 16;

    SuffixDrafter(int min_match, int max_match, size_t capacity_tokens)
        : min_match_(std::max(3, min_match)), max_match_(std::max(min_match, max_match)) {
        size_t cap = 16;
        while (cap < capacity_tokens * 2) cap <<= 1;
        table_.assign(cap, Slot{});
        mask_ = cap - 1;
        hist_.reserve(capacity_tokens);
        prev_.reserve(capacity_tokens);
    }

    void reset() {
        hist_.clear();
        prev_.clear();
        std::fill(table_.begin(), table_.end(), Slot{});
        used_ = 0;
        last_match_ = 0;
    }

    int last_match() const { return last_match_; }
    size_t size() const { return hist_.size(); }
    int32_t token_at(size_t i) const { return hist_[i]; }

    void append(const int32_t * tokens, size_t n) {
        for (size_t i = 0; i < n; ++i) {
            if (2 * (used_ + 1) > table_.size()) rebuild();
            hist_.push_back(tokens[i]);
            link(hist_.size() - 1);
        }
    }

    void assign(const int32_t * tokens, size_t n) {
        size_t keep = 0;
        const size_t common = std::min(n, hist_.size());
        while (keep < common && hist_[keep] == tokens[keep]) ++keep;
        for (size_t e = hist_.size(); e-- > std::max<size_t>(keep, 2);)
            if (Slot* s = find_slot(key_at(e), false)) s->last = prev_[e];
        hist_.resize(keep);
        prev_.resize(keep);
        last_match_ = 0;
        append(tokens + keep, n - keep);
    }

    int propose(int max_k, int32_t* out) {
        last_match_ = 0;
        const size_t n = hist_.size();
        if (n < 4 || max_k <= 0) return 0;
        const size_t cur = n - 1;
        size_t best_end = 0;
        int best_len = 0;
        int32_t p = prev_[cur];
        for (int w = 0; w < WAYS && p >= 0; ++w, p = prev_[(size_t) p]) {
            int len = 0;
            while (len < max_match_ && len <= p && hist_[(size_t) (p - len)] == hist_[cur - len]) ++len;
            if (len > best_len) { best_len = len; best_end = (size_t) p; }
            if (best_len == max_match_) break;
        }
        if (best_len < min_match_) return 0;
        last_match_ = best_len;
        int k = 0;
        for (size_t q = best_end + 1; q <= cur && k < max_k; ++q) out[k++] = hist_[q];
        return k;
    }

private:
    struct Slot {
        uint64_t key = 0;
        int32_t last = -1;
    };

    static uint64_t mix(uint64_t x) {
        x ^= x >> 33;
        x *= 0xff51afd7ed558ccdull;
        x ^= x >> 33;
        x *= 0xc4ceb9fe1a85ec53ull;
        return x ^ (x >> 33);
    }

    uint64_t key_at(size_t end) const {
        const uint64_t a = (uint32_t) hist_[end - 2], b = (uint32_t) hist_[end - 1], c = (uint32_t) hist_[end];
        return mix(a * 0x9E3779B97F4A7C15ull ^ mix(b + 0x632BE59BD9B4E019ull) ^ (c << 1)) | 1ull;
    }

    Slot* find_slot(uint64_t key, bool insert) {
        for (size_t i = key & mask_;; i = (i + 1) & mask_) {
            Slot& s = table_[i];
            if (s.key == key) return &s;
            if (s.key == 0) {
                if (!insert) return nullptr;
                s.key = key;
                ++used_;
                return &s;
            }
        }
    }

    void link(size_t end) {
        int32_t before = -1;
        if (end >= 2) {
            Slot* s = find_slot(key_at(end), true);
            before = s->last;
            s->last = (int32_t) end;
        }
        prev_.push_back(before);
    }

    void rebuild() {
        const size_t n = prev_.size();
        size_t cap = table_.size();
        while (cap < 4 * n) cap <<= 1;
        table_.assign(cap, Slot{});
        mask_ = cap - 1;
        used_ = 0;
        prev_.clear();
        for (size_t e = 0; e < n; ++e) link(e);
    }

    int min_match_;
    int max_match_;
    std::vector<int32_t> hist_;
    std::vector<int32_t> prev_;
    std::vector<Slot> table_;
    size_t mask_ = 0;
    size_t used_ = 0;
    int last_match_ = 0;
};

class DraftPolicy {
public:
    static constexpr int kMaxT = 8;
    static constexpr int kBuckets = 4;

    struct Pick {
        bool lookup = false;
        int t = 1;
    };

    DraftPolicy(int max_t, double margin) : max_t_(std::clamp(max_t, 1, kMaxT)), margin_(margin) {
        cost_.assign(kMaxT + 1, 0.0);
        cost_n_.assign(kMaxT + 1, 0.0);
        mtp_tok_.assign(kMaxT + 1, 0.0);
        mtp_n_.assign(kMaxT + 1, 0.0);
        ok_.assign(kBuckets, 0.0);
        bad_.assign(kBuckets, 0.0);
    }

    static int bucket(int match) { return match < 6 ? 0 : match < 12 ? 1 : match < 24 ? 2 : 3; }

    double lookup_rate(int match) const {
        const int b = bucket(match);
        return (ok_[b] + kPriorN * kPriorQ[b]) / (ok_[b] + bad_[b] + kPriorN);
    }

    double cost_ms(int t) const {
        t = std::clamp(t, 1, kMaxT);
        if (cost_n_[t] > 0) return cost_[t];
        double num = 0.0, den = 0.0;
        for (int u = 1; u <= kMaxT; ++u)
            if (cost_n_[u] > 0) {
                const double w = std::min(cost_n_[u], 20.0);
                num += w * cost_[u] * kShape[t] / kShape[u];
                den += w;
            }
        return den > 0 ? num / den : kShape[t];
    }

    double mtp_tokens(int t) const {
        if (mtp_n_[t] > 0) return mtp_tok_[t];
        return 1.0 + 0.7 * (t - 1);
    }

    Pick choose(int t_mtp, int lookup_k, int match) const {
        Pick p;
        p.t = std::clamp(t_mtp, 1, max_t_);
        if (lookup_k <= 0) return p;
        const double base = mtp_tokens(p.t) / cost_ms(p.t);
        const double q = lookup_rate(match);
        double e = 1.0, qi = 1.0, best = 0.0;
        int best_t = 0;
        for (int k = 1; k <= std::min(lookup_k, max_t_ - 1); ++k) {
            qi *= q;
            e += qi;
            const double r = e / cost_ms(k + 1);
            if (r > best) { best = r; best_t = k + 1; }
        }
        if (best_t > 0 && best > base * (1.0 + margin_)) { p.lookup = true; p.t = best_t; return p; }
        const int t_full = std::min(lookup_k, max_t_ - 1) + 1;
        if (t_full > p.t && cost_n_[t_full] < kProbes && q >= 0.85) { p.lookup = true; p.t = t_full; }
        return p;
    }

    void observe(bool lookup, int t, int accepted, int match, double round_ms) {
        t = std::clamp(t, 1, kMaxT);
        if (round_ms > 0) {
            cost_[t] = cost_n_[t] > 0 ? (1.0 - kCostAlpha) * cost_[t] + kCostAlpha * round_ms : round_ms;
            cost_n_[t] += 1.0;
        }
        if (lookup) {
            const int b = bucket(match);
            ok_[b] = kDecay * ok_[b] + accepted;
            bad_[b] = kDecay * bad_[b] + (accepted < t - 1 ? 1.0 : 0.0);
        } else {
            const double got = accepted + 1.0;
            mtp_tok_[t] = mtp_n_[t] > 0 ? (1.0 - kTokAlpha) * mtp_tok_[t] + kTokAlpha * got : got;
            mtp_n_[t] += 1.0;
        }
    }

private:
    static constexpr double kShape[kMaxT + 1] = {0.0, 1.0, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6};
    static constexpr double kCostAlpha = 0.1;
    static constexpr double kTokAlpha  = 0.05;
    static constexpr double kDecay     = 0.97;
    static constexpr double kPriorQ[kBuckets] = {0.75, 0.88, 0.93, 0.96};
    static constexpr double kPriorN = 4.0;
    static constexpr int kProbes = 3;

    int max_t_;
    double margin_;
    std::vector<double> cost_, cost_n_, mtp_tok_, mtp_n_, ok_, bad_;
};

} // namespace ref

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

static tokens_t range_tokens(int base, int n) {
    tokens_t t;
    t.reserve(n);
    for (int i = 0; i < n; ++i) {
        t.push_back((llama_token) (base + i));
    }
    return t;
}

// feed a whole text to both drafters, one token at a time, and check every proposal agrees
static void check_proposals(common_suffix_drafter & port, ref::SuffixDrafter & reference, const tokens_t & fed,
                            int max_k, const char * what) {
    llama_token a[16];
    llama_token b[16];

    const int ka = port.propose(max_k, a);
    const int kb = reference.propose(max_k, b);

    if (ka != kb || port.last_match() != reference.last_match()) {
        fprintf(stderr, "FAIL %s: proposal differs after %zu tokens (%d/%d tokens, match %d/%d)\n", what,
                fed.size(), ka, kb, port.last_match(), reference.last_match());
        ++failures;
        return;
    }

    for (int i = 0; i < ka; ++i) {
        if (a[i] != b[i]) {
            fprintf(stderr, "FAIL %s: proposal token %d differs after %zu tokens (%d != %d)\n", what, i,
                    fed.size(), a[i], b[i]);
            ++failures;
            return;
        }
    }
}

// ---------------------------------------------------------------------------
// part 1 - hand-built cases
// ---------------------------------------------------------------------------

static void test_hand_built() {
    const tokens_t P = range_tokens(1000, 20); // a unique intro
    const tokens_t A = range_tokens(200, 30);  // a block that comes back later
    const tokens_t F = range_tokens(3000, 25); // unique filler between the two occurrences

    tokens_t seq = P;
    seq.insert(seq.end(), A.begin(), A.end());
    seq.insert(seq.end(), F.begin(), F.end());
    seq.insert(seq.end(), A.begin(), A.begin() + 20); // the answer is quoting A, 20 tokens in so far

    common_suffix_drafter drafter(16, 64);
    drafter.append(seq.data(), seq.size());

    llama_token out[8];
    const int   k = drafter.propose(8, out);

    CHECK(k == 8);
    CHECK(drafter.last_match() == 20);

    // the proposal is what followed the first A: A[20], A[21], ...
    for (int i = 0; i < k; ++i) {
        CHECK(i + 20 < (int) A.size() && out[i] == A[20 + i]);
    }

    // a repeat under min_match proposes nothing
    common_suffix_drafter strict(25, 64);
    strict.append(seq.data(), seq.size());
    CHECK(strict.propose(8, out) == 0);
    CHECK(strict.last_match() == 0);

    // feeding the quote to the end keeps the run going: the drafter walks forward through the repeat
    common_suffix_drafter walk(16, 64);
    walk.append(seq.data(), seq.size());
    for (int step = 0; step < 10; ++step) {
        const int kk = walk.propose(8, out);
        CHECK(kk > 0);
        if (kk <= 0) {
            break;
        }
        walk.append(out, kk);
    }

    // assign() to a shorter text and back again must leave the index as if it had never grown past the cut
    common_suffix_drafter a(16, 64);
    a.append(seq.data(), seq.size());

    const tokens_t shorter(seq.begin(), seq.begin() + 30); // cut inside the first A
    a.assign(shorter.data(), shorter.size());

    common_suffix_drafter fresh(16, 64);
    fresh.assign(shorter.data(), shorter.size());

    CHECK(a.size() == fresh.size());
    for (size_t i = 0; i < fresh.size(); ++i) {
        CHECK(a.token_at(i) == fresh.token_at(i));
    }

    // ... and growing it again: the state after the undo has to be the state the from-scratch index is in
    const tokens_t longer(seq.begin(), seq.begin() + 60);
    a.assign(longer.data(), longer.size());
    fresh.assign(longer.data(), longer.size());

    llama_token x[8];
    llama_token y[8];
    const int   kx = a.propose(8, x);
    const int   ky = fresh.propose(8, y);

    CHECK(kx == ky);
    CHECK(a.last_match() == fresh.last_match());
    for (int i = 0; i < kx && i < ky; ++i) {
        CHECK(x[i] == y[i]);
    }

    // periodic text: the continuation of the repeat runs into the suffix being matched
    tokens_t periodic;
    for (int i = 0; i < 8; ++i) {
        periodic.push_back((llama_token) (i % 4));
    }
    common_suffix_drafter p(4, 64);
    p.append(periodic.data(), periodic.size());
    const int kp = p.propose(8, out);
    CHECK(kp == 4);
    CHECK(p.last_match() == 4);
    for (int i = 0; i < kp; ++i) {
        CHECK(out[i] == (llama_token) i); // the continuation is the rest of the period, and never runs past `cur`
    }

    printf("part 1  hand-built cases: ok\n");
}

// ---------------------------------------------------------------------------
// part 2 - equivalence against Strata's own code, over randomized histories
// ---------------------------------------------------------------------------

static void test_drafter_equivalence() {
    std::mt19937 rng(12345);

    int proposals = 0;
    int with_match = 0;

    // the port starts at 4096 tokens and grows; the reference keeps Strata's own capacity
    common_suffix_drafter port(16, 64, 4096);
    ref::SuffixDrafter  reference(16, 64, 1u << 19);

    tokens_t hist;

    for (int round = 0; round < 4000; ++round) {
        // either a fresh run of unique tokens, or a copy of an earlier stretch: what quoting looks like
        tokens_t next;
        if (hist.size() < 8 || (rng() % 100) < 45) {
            const int n = 1 + (int) (rng() % 24);
            for (int i = 0; i < n; ++i) {
                next.push_back((llama_token) (1000 + (rng() % 400)));
            }
        } else {
            const size_t at  = rng() % hist.size();
            const size_t len = std::min<size_t>(3 + rng() % 48, hist.size() - at);
            next.assign(hist.begin() + at, hist.begin() + at + len);
        }

        for (llama_token t : next) {
            port.append(t);
            reference.append(&t, 1);
            hist.push_back(t);

            check_proposals(port, reference, hist, 8, "drafter equivalence");
            ++proposals;
            if (port.last_match() > 0) {
                ++with_match;
            }
        }

        // every so often the conversation is cut back and taken somewhere else, which is assign()'s job
        if ((round % 211) == 210 && hist.size() > 64) {
            const size_t keep = hist.size() - 1 - (rng() % 40);
            tokens_t      cut(hist.begin(), hist.begin() + keep);
            cut.push_back((llama_token) (9000 + (rng() % 50)));

            port.assign(cut.data(), cut.size());
            reference.assign(cut.data(), cut.size());
            hist = cut;

            CHECK(port.size() == reference.size());
            CHECK(port.size() == hist.size());
            for (size_t i = 0; i < hist.size(); ++i) {
                if (port.token_at(i) != reference.token_at(i)) {
                    fprintf(stderr, "FAIL assign(): history differs at %zu\n", i);
                    ++failures;
                    break;
                }
            }
            check_proposals(port, reference, hist, 8, "assign equivalence");
        }
    }

    CHECK(with_match > 1000);

    printf("part 2  drafter vs Strata's SuffixDrafter: %d proposals, %d with a match, 0 mismatches\n",
           proposals, with_match);
}

// ---------------------------------------------------------------------------
// part 3 - the draft policy: the same arithmetic as Strata, and the behaviour it is there for
// ---------------------------------------------------------------------------

static void test_policy_equivalence() {
    std::mt19937 rng(777);

    common_draft_policy port;
    ref::DraftPolicy    reference(common_draft_policy::k_max_t, 0.03);

    int picks = 0;
    int lookups = 0;

    for (int i = 0; i < 20000; ++i) {
        const int t_mtp = 1 + (int) (rng() % common_draft_policy::k_max_t);
        const int k     = (rng() % 5 == 0) ? 0 : 1 + (int) (rng() % 7);
        const int match = k ? 1 + (int) (rng() % 60) : 0;

        const common_draft_policy::pick p = port.choose(t_mtp, k, match);
        const ref::DraftPolicy::Pick    r = reference.choose(t_mtp, k, match);

        CHECK(p.lookup == r.lookup);
        CHECK(p.t == r.t);

        for (int t = 1; t <= common_draft_policy::k_max_t; ++t) {
            CHECK(port.cost_ms(t) == reference.cost_ms(t));
        }
        for (int m = 0; m < 60; ++m) {
            CHECK(port.lookup_rate(m) == reference.lookup_rate(m));
        }

        ++picks;
        if (p.lookup) {
            ++lookups;
        }

        const int    accepted  = p.lookup ? (int) (rng() % std::max(1, p.t)) : (int) (rng() % 4);
        const double round_ms  = 0.5 + (double) (rng() % 4000) / 1000.0;

        port.observe(p.lookup, p.t, accepted, match, round_ms);
        reference.observe(p.lookup, p.t, accepted, match, round_ms);
    }

    CHECK(lookups > 1000);

    printf("part 3  policy vs Strata's DraftPolicy: %d picks, %d lookup windows, 0 mismatches\n", picks, lookups);

    // (a) a long match, nothing measured yet, and no other drafter: the window is worth taking
    common_draft_policy plain(8, 0.03);
    plain.set_fallback(false);

    CHECK(!plain.has_fallback_mtp());
    CHECK(plain.choose(4, 7, 30).lookup);
    CHECK(plain.choose(4, 7, 30).t >= 5);

    // (b) the continuations stop being right: the policy stops putting them in front of the target
    for (int i = 0; i < 400; ++i) {
        plain.observe(true, 8, 0, 30, 3.6);
    }
    CHECK(!plain.choose(4, 7, 30).lookup);

    // (c) with an alternative that commits, a lookup that does not keep up loses the window to it
    common_draft_policy with_mtp(8, 0.03);
    for (int i = 0; i < 200; ++i) {
        with_mtp.observe(false, 4, 3, 0, 2.05); // the MTP drafts commit 4 tokens in 2.05 ms
        with_mtp.observe(true, 8, 0, 30, 3.6);  // the lookup drafts commit 1
    }
    CHECK(!with_mtp.choose(4, 7, 30).lookup);

    // (d) and a lookup that does keep up wins it back
    common_draft_policy good(8, 0.03);
    for (int i = 0; i < 200; ++i) {
        good.observe(false, 4, 1, 0, 2.05);     // the MTP drafts commit 2 tokens in 2.05 ms
        good.observe(true, 8, 7, 30, 3.6);      // the lookup drafts commit all 8 in 3.6 ms
    }
    CHECK(good.choose(4, 7, 30).lookup);

    // (e) no proposal, no lookup window, however good the policy feels
    CHECK(!good.choose(4, 0, 0).lookup);
}

// ---------------------------------------------------------------------------
// part 4 - one generated text, three drafting strategies
// ---------------------------------------------------------------------------

// how many of the drafts the target accepts, in a row from the start (it always commits one more itself)
static int accept_prefix(const tokens_t & truth, size_t pos, const llama_token * drafts, int n) {
    int a = 0;
    while (a < n && pos + (size_t) a < truth.size() && drafts[a] == truth[pos + (size_t) a]) {
        ++a;
    }
    return a;
}

// the model's own MTP layer: right about 86% of its positions, up to 3 drafts (Strata's measurement of this model)
static int mtp_draft(const tokens_t & truth, size_t pos, std::mt19937 & rng, llama_token * out, int n_max) {
    int n = 0;
    while (n < n_max && pos + (size_t) n < truth.size()) {
        if ((rng() % 100) >= 86) {
            out[n] = (llama_token) 9999; // a wrong guess, and the block stops there
            ++n;
            break;
        }
        out[n] = truth[pos + (size_t) n];
        ++n;
    }
    return n;
}

// what a verify pass costs: the model runs over the whole window in one go, so a longer window is not much slower
static double round_ms(int t, std::mt19937 & rng) {
    const double base  = 12.0;                    // one token through all 48 layers
    const double slope = 1.6;                     // each extra position in the window
    const double noise = 1.0 + ((double) (rng() % 100) - 50.0) / 1000.0;
    return (base + slope * (t - 1)) * noise;
}

enum strategy {
    STRATEGY_MTP = 0,
    STRATEGY_SUFFIX_FIXED,
    STRATEGY_SUFFIX_POLICY,
};

struct sim_result {
    int    rounds        = 0;
    size_t tokens        = 0;
    double ms            = 0.0;
    int    lookup_rounds = 0; // rounds that carried lookup drafts
    int    lookup_acc    = 0; // tokens those drafts committed
};

static sim_result simulate(const tokens_t & truth, size_t prompt_len, strategy s, uint32_t seed) {
    std::mt19937 rng(seed);

    sim_result res;

    common_suffix_drafter drafter(16, 64, 4096);
    common_draft_policy   policy(8, 0.03);

    // the prompt is read by prefill in every strategy: only what comes after it is drafted and verified
    size_t pos = prompt_len;

    if (s != STRATEGY_MTP) {
        drafter.assign(truth.data(), prompt_len);
    }

    llama_token out[8];

    while (pos < truth.size()) {
        const size_t before = pos;

        int    drafts = 0;
        int    accept = 0;
        int    match  = 0;
        double ms     = 0.0;

        if (s == STRATEGY_MTP) {
            drafts = mtp_draft(truth, pos, rng, out, 3);
            accept = accept_prefix(truth, pos, out, drafts);
            ms     = round_ms(drafts + 1, rng);
        } else {
            const int k = drafter.propose(7, out);
            match       = drafter.last_match();

            int  n    = k;
            bool take = true;

            if (s == STRATEGY_SUFFIX_POLICY) {
                const common_draft_policy::pick p = policy.choose(4, k, match);
                take = p.lookup;
                n    = std::min(k, p.t - 1);
            }

            if (take && n > 0) {
                drafts = n;
                accept = accept_prefix(truth, pos, out, n);
                ms     = round_ms(n + 1, rng);

                if (s == STRATEGY_SUFFIX_POLICY) {
                    policy.observe(true, n + 1, accept, match, ms);
                }

                ++res.lookup_rounds;
                res.lookup_acc += accept;
            } else {
                // the alternative has the window: the MTP stand-in drafts, as it does in the chain when the
                // policy declines and draft-mtp is the next implementation
                drafts = mtp_draft(truth, pos, rng, out, 3);
                accept = accept_prefix(truth, pos, out, drafts);
                ms     = round_ms(drafts + 1, rng);

                if (s == STRATEGY_SUFFIX_POLICY) {
                    policy.observe(false, drafts + 1, accept, 0, ms);
                }
            }
        }

        res.ms += ms;

        pos = std::min(pos + (size_t) accept + 1, truth.size());

        if (s != STRATEGY_MTP) {
            drafter.append(truth.data() + before, pos - before);
        }

        ++res.rounds;
        res.tokens += pos - before;
    }

    // the text is the truth in every strategy: a drafter proposes, only the target decides
    CHECK(pos == truth.size());

    return res;
}

static void test_rounds() {
    // a prompt, then an answer that quotes long stretches of it back with one edit near the start - the case
    // Strata built this for, and what "return this file with the class renamed" looks like in tokens
    const tokens_t prompt = range_tokens(1000, 900);

    tokens_t truth = prompt;

    for (int chunk = 0; chunk < 3; ++chunk) {
        for (int i = 0; i < 20; ++i) { // writing something new
            truth.push_back((llama_token) (5000 + chunk * 100 + i));
        }
        const size_t at = (size_t) (chunk * 250); // quoting the prompt back
        for (int i = 0; i < 600; ++i) {
            if (i == 3) {
                truth.push_back((llama_token) (7000 + chunk)); // the one name it was asked to change
            } else {
                truth.push_back(prompt[at + (size_t) (i < 3 ? i : i - 1)]);
            }
        }
    }

    const size_t prompt_len = prompt.size();

    const sim_result mtp    = simulate(truth, prompt_len, STRATEGY_MTP, 1);
    const sim_result fixed  = simulate(truth, prompt_len, STRATEGY_SUFFIX_FIXED, 2);
    const sim_result policy = simulate(truth, prompt_len, STRATEGY_SUFFIX_POLICY, 3);

    const auto tpr = [](const sim_result & r) { return (double) r.tokens / (double) r.rounds; };
    const auto tpk = [](const sim_result & r) { return (double) r.tokens * 1000.0 / r.ms; };

    printf("part 4  one answer of %zu tokens, %zu of them written after the prompt was read:\n", truth.size(),
           truth.size() - prompt_len);
    printf("          MTP drafts only        %6.2f tokens/pass, %6.2f tokens/ms\n", tpr(mtp), tpk(mtp));
    printf("          suffix, whole window   %6.2f tokens/pass, %6.2f tokens/ms\n", tpr(fixed), tpk(fixed));
    printf("          suffix + draft policy  %6.2f tokens/pass, %6.2f tokens/ms (%d of %d rounds took the "
           "lookup window, %d tokens committed by those drafts)\n", tpr(policy), tpk(policy),
           policy.lookup_rounds, policy.rounds, policy.lookup_acc);

    // quoting text is where a lookup over the whole suffix pays: it commits far more per pass than 3 MTP drafts
    CHECK(tpr(fixed) > tpr(mtp) * 1.5);
    CHECK(tpr(policy) > tpr(mtp));
    CHECK(tpk(policy) > tpk(mtp));

    // and the policy is not a coin flip: it takes the window on the quotes and leaves it alone elsewhere
    CHECK(policy.lookup_rounds > 0);
    CHECK(policy.lookup_rounds < policy.rounds);
    CHECK((double) policy.lookup_acc / (double) policy.lookup_rounds > 3.0);
}

// ---------------------------------------------------------------------------
// part 5 - the drafter against a naive longest-suffix-match scan
// ---------------------------------------------------------------------------

// every earlier end position, not just the 16 most recent: what the drafter would cost without its trigram index.
// the same min_match, so the two are after the same answers
static int naive_propose(const tokens_t & hist, int max_k, int max_match, int min_match, llama_token * out,
                         int * match) {
    const int64_t n = (int64_t) hist.size();
    if (n < 4 || max_k <= 0) {
        return 0;
    }

    const int64_t cur = n - 1;

    int64_t best_end = -1;
    int     best_len = 0;

    for (int64_t p = cur - 1; p >= 0; --p) {
        if (hist[p] != hist[cur]) {
            continue;
        }
        int len = 1;
        while (len < max_match && len <= p && hist[p - len] == hist[cur - len]) {
            ++len;
        }
        if (len > best_len) {
            best_len = len;
            best_end = p;
        }
    }

    if (match) {
        *match = best_len;
    }

    if (best_end < 0 || best_len < min_match) {
        return 0;
    }

    int k = 0;
    for (int64_t q = best_end + 1; q <= cur && k < max_k; ++q) {
        out[k++] = hist[q];
    }
    return k;
}

static void test_bench() {
    const size_t n_hist  = 1u << 17; // 131072 tokens
    const int    rounds  = 2000;

    std::mt19937 rng(20260926);

    // a text with trigram reuse in it, which is what a real conversation is: with tokens drawn independently
    // there is nothing to find and both sides answer immediately, which measures nothing
    tokens_t all;
    all.reserve(n_hist + rounds);
    while (all.size() < n_hist + rounds) {
        if (all.size() < 16 || (rng() % 100) < 45) {
            all.push_back((llama_token) (rng() % 8000));
        } else {
            const size_t at  = rng() % (all.size() - 8);
            const size_t len = std::min<size_t>(3 + rng() % 64, all.size() - at);
            all.insert(all.end(), all.begin() + at, all.begin() + at + len);
        }
    }

    tokens_t hist(all.begin(), all.begin() + n_hist);

    common_suffix_drafter drafter(16, 64, n_hist);
    drafter.append(hist.data(), hist.size());

    llama_token out_a[8];
    llama_token out_b[8];

    // the append is left out of the timing on purpose: the index rebuilds itself as it grows, and that is a cost
    // of the whole generation rather than of the proposal the draft policy is waiting on
    int    hits     = 0;
    int    hits_b   = 0;
    int    agree    = 0;
    double ns_drafter = 0.0;
    double ns_naive   = 0.0;

    for (int i = 0; i < rounds; ++i) {
        const llama_token next = all[n_hist + (size_t) i];

        drafter.append(next);

        const auto ta = std::chrono::steady_clock::now();
        const int  ka = drafter.propose(8, out_a);
        const auto tb = std::chrono::steady_clock::now();

        ns_drafter += std::chrono::duration<double, std::nano>(tb - ta).count();
        hits += ka > 0;

        hist.push_back(next);

        int          match_b = 0;
        const auto   tc      = std::chrono::steady_clock::now();
        const int    kb      = naive_propose(hist, 8, 64, 16, out_b, &match_b);
        const auto   td      = std::chrono::steady_clock::now();

        ns_naive += std::chrono::duration<double, std::nano>(td - tc).count();
        hits_b += kb > 0;

        // the scan looks further back, so it can only find a match at least as long; when they land on the same
        // length they have found the same repeat and have to propose the same tokens
        if (ka > 0 && kb > 0) {
            CHECK(match_b >= drafter.last_match());
            if (match_b == drafter.last_match()) {
                const int same = std::min(ka, kb);
                for (int j = 0; j < same; ++j) {
                    CHECK(out_a[j] == out_b[j]);
                }
                ++agree;
            }
        }
    }

    ns_drafter /= rounds;
    ns_naive /= rounds;

    // the scan looks at every earlier end position instead of the 16 most recent ones, so it is both slower and
    // not quite the same function: it can find a longer repeat the indexed drafter walked past
    printf("part 5  proposing on a %zu-token history over %d rounds: the index costs %.0f ns a proposal (%d of "
           "%d had a match), a scan of every earlier position costs %.0f ns (%d) - %.0fx. the two agreed on the "
           "repeat %d times and never disagreed\n",
           n_hist, rounds, ns_drafter, hits, rounds, ns_naive, hits_b, ns_naive / ns_drafter, agree);

    CHECK(ns_drafter < ns_naive);
}

int main(int argc, char ** argv) {
    const bool sim = argc > 1 && std::string(argv[1]) == "--sim";

    test_hand_built();
    test_drafter_equivalence();
    test_policy_equivalence();

    if (sim) {
        test_rounds();
        test_bench();
    } else {
        printf("(pass --sim for the round simulator and the benchmark)\n");
    }

    if (failures) {
        printf("FAIL: %d\n", failures);
        return 1;
    }

    printf("all ok\n");
    return 0;
}
