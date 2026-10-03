#include "ngram-suffix.h"

#include <algorithm>

// Ported from Strata (MIT), src/spec/suffix_drafter.cpp and src/spec/draft_policy.cpp.

// ---------------------------------------------------------------------------
// common_suffix_drafter
// ---------------------------------------------------------------------------

common_suffix_drafter::common_suffix_drafter(int min_match, int max_match, size_t capacity_tokens)
    : min_match_(std::max(3, min_match)), max_match_(std::max(min_match_, max_match)) {
    size_t cap = 16;
    while (cap < capacity_tokens * 2) {
        cap <<= 1; // load factor <= 0.5 at the nominal capacity
    }
    table_.assign(cap, slot{});
    mask_ = cap - 1;
    hist_.reserve(capacity_tokens);
    prev_.reserve(capacity_tokens);
}

void common_suffix_drafter::reset() {
    hist_.clear();
    prev_.clear();
    std::fill(table_.begin(), table_.end(), slot{});
    used_       = 0;
    last_match_ = 0;
}

uint64_t common_suffix_drafter::mix(uint64_t x) {
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdull;
    x ^= x >> 33;
    x *= 0xc4ceb9fe1a85ec53ull;
    return x ^ (x >> 33);
}

// the hash of the trigram ending at `end`, with the low bit set so that a stored key is never 0
uint64_t common_suffix_drafter::key_at(size_t end) const {
    const uint64_t a = (uint32_t) hist_[end - 2];
    const uint64_t b = (uint32_t) hist_[end - 1];
    const uint64_t c = (uint32_t) hist_[end];
    return mix(a * 0x9E3779B97F4A7C15ull ^ mix(b + 0x632BE59BD9B4E019ull) ^ (c << 1)) | 1ull;
}

common_suffix_drafter::slot * common_suffix_drafter::find_slot(uint64_t key, bool insert) {
    for (size_t i = key & mask_; true; i = (i + 1) & mask_) { // never full: rebuilt at half load
        slot & s = table_[i];
        if (s.key == key) {
            return &s;
        }
        if (s.key == 0) {
            if (!insert) {
                return nullptr;
            }
            s.key = key;
            ++used_;
            return &s;
        }
    }
}

// index hist_[end], the newest token: it joins its trigram's chain of end positions
void common_suffix_drafter::link(size_t end) {
    int32_t before = -1;
    if (end >= 2) {
        slot * s   = find_slot(key_at(end), true);
        before     = s->last;
        s->last    = (int32_t) end;
    }
    prev_.push_back(before);
}

// the indexed positions again, without the slots of undone ones, in a table that holds them at a quarter load
void common_suffix_drafter::rebuild() {
    const size_t n = prev_.size();

    size_t cap = table_.size();
    while (cap < 4 * n) {
        cap <<= 1;
    }

    table_.assign(cap, slot{});
    mask_ = cap - 1;
    used_ = 0;

    prev_.clear();
    for (size_t e = 0; e < n; ++e) {
        link(e);
    }
}

void common_suffix_drafter::append(const llama_token * tokens, size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (2 * (used_ + 1) > table_.size()) {
            rebuild();
        }
        hist_.push_back(tokens[i]);
        link(hist_.size() - 1);
    }
}

void common_suffix_drafter::assign(const llama_token * tokens, size_t n) {
    size_t keep = 0;
    const size_t common = std::min(n, hist_.size());
    while (keep < common && hist_[keep] == tokens[keep]) {
        ++keep;
    }

    // newest first: each trigram's `last` goes back to its previous end position
    for (size_t e = hist_.size(); e-- > std::max<size_t>(keep, 2);) {
        slot * s = find_slot(key_at(e), false);
        if (s) {
            s->last = prev_[e];
        }
    }

    hist_.resize(keep);
    prev_.resize(keep);
    last_match_ = 0;

    append(tokens + keep, n - keep);
}

int common_suffix_drafter::propose(int max_k, llama_token * out) {
    last_match_ = 0;

    const size_t n = hist_.size();
    if (n < 4 || max_k <= 0) {
        return 0;
    }

    const size_t cur = n - 1;

    size_t best_end = 0;
    int    best_len = 0;

    // the current trigram's earlier occurrences, newest first
    int32_t p = prev_[cur];
    for (int w = 0; w < WAYS && p >= 0; ++w, p = prev_[(size_t) p]) {
        int len = 0;
        while (len < max_match_ && len <= p && hist_[(size_t) (p - len)] == hist_[cur - len]) {
            ++len;
        }
        if (len > best_len) { // most recent first, so ties keep the newer occurrence
            best_len = len;
            best_end = (size_t) p;
        }
        if (best_len == max_match_) {
            break;
        }
    }

    if (best_len < min_match_) {
        return 0;
    }

    last_match_ = best_len;

    int k = 0;
    // the continuation may run into the current suffix (periodic text); reading history up to `cur` is valid
    for (size_t q = best_end + 1; q <= cur && k < max_k; ++q) {
        out[k++] = hist_[q];
    }

    return k;
}

// ---------------------------------------------------------------------------
// common_draft_policy
// ---------------------------------------------------------------------------

namespace {

// The shape of a round's cost by window size, relative to one token, used only for sizes not measured yet
// (the measured round times replace it): +10 ms per token with every missed expert on the CPU, flatter with
// a default CPU/DMA split (Strata, bench/results/2026-09-27-spec/window-cost).
constexpr double kShape[common_draft_policy::k_max_t + 1] = { 0.0, 1.0, 1.35, 1.7, 2.05, 2.45, 2.85, 3.25, 3.6 };

constexpr double kCostAlpha = 0.1;  // EMA weight of a new round time
constexpr double kTokAlpha  = 0.05; // EMA weight of a new MTP window outcome
constexpr double kDecay     = 0.97; // lookup counts: older windows fade

// Before a bucket has data: the longer the match, the likelier its continuation (llama.cpp's lookup decoding
// gates on the same thing); worth 4 observations, so a few real windows override it.
constexpr double kPriorQ[common_draft_policy::k_buckets] = { 0.75, 0.88, 0.93, 0.96 };
constexpr double kPriorN = 4.0;

// a lookup window size is tried this often before its guessed cost can veto it
constexpr int kProbes = 3;

} // namespace

common_draft_policy::common_draft_policy(int max_t, double margin)
    : max_t_(std::clamp(max_t, 1, k_max_t)), margin_(margin) {
    cost_.assign(k_max_t + 1, 0.0);
    cost_n_.assign(k_max_t + 1, 0.0);
    mtp_tok_.assign(k_max_t + 1, 0.0);
    mtp_n_.assign(k_max_t + 1, 0.0);
    ok_.assign(k_buckets, 0.0);
    bad_.assign(k_buckets, 0.0);
}

int common_draft_policy::bucket(int match) {
    return match < 6 ? 0 : match < 12 ? 1 : match < 24 ? 2 : 3;
}

double common_draft_policy::lookup_rate(int match) const {
    const int b = bucket(match);
    return (ok_[b] + kPriorN * kPriorQ[b]) / (ok_[b] + bad_[b] + kPriorN);
}

double common_draft_policy::cost_ms(int t) const {
    t = std::clamp(t, 1, k_max_t);
    if (cost_n_[t] > 0) {
        return cost_[t];
    }
    // scale from the measured sizes, weighting each by how often it was seen
    double num = 0.0;
    double den = 0.0;
    for (int u = 1; u <= k_max_t; ++u) {
        if (cost_n_[u] > 0) {
            const double w = std::min(cost_n_[u], 20.0);
            num += w * cost_[u] * kShape[t] / kShape[u];
            den += w;
        }
    }
    return den > 0 ? num / den : kShape[t];
}

double common_draft_policy::mtp_tokens(int t) const {
    if (mtp_n_[t] > 0) {
        return mtp_tok_[t];
    }
    return 1.0 + 0.7 * (t - 1); // before any MTP window of this size: a typical acceptance
}

common_draft_policy::pick common_draft_policy::choose(int t_mtp, int lookup_k, int match) const {
    pick p;
    p.t = std::clamp(t_mtp, 1, max_t_);
    if (lookup_k <= 0) {
        return p;
    }

    const double base = mtp_tokens(p.t) / cost_ms(p.t);
    const double q    = lookup_rate(match);

    double e     = 1.0;
    double qi    = 1.0;
    double best  = 0.0;
    int    best_t = 0;

    for (int k = 1; k <= std::min(lookup_k, max_t_ - 1); ++k) {
        qi *= q;
        e += qi;
        const double r = e / cost_ms(k + 1);
        if (r > best) {
            best   = r;
            best_t = k + 1;
        }
    }

    if (best_t > 0 && best > base * (1.0 + margin_)) {
        p.lookup = true;
        p.t      = best_t;
        return p;
    }

    // a guessed cost can keep the policy from ever measuring a size: the first few times a confident lookup
    // would need a size not measured yet, it is tried (verification keeps the output; only that round's
    // speed is at stake)
    const int t_full = std::min(lookup_k, max_t_ - 1) + 1;
    if (t_full > p.t && cost_n_[t_full] < kProbes && q >= 0.85) {
        p.lookup = true;
        p.t      = t_full;
    }

    return p;
}

void common_draft_policy::observe(bool lookup, int t, int accepted, int match, double round_ms) {
    t = std::clamp(t, 1, k_max_t);

    if (round_ms > 0) {
        cost_[t] = cost_n_[t] > 0 ? (1.0 - kCostAlpha) * cost_[t] + kCostAlpha * round_ms : round_ms;
        cost_n_[t] += 1.0;
    }

    if (lookup) {
        const int b = bucket(match);
        ok_[b]  = kDecay * ok_[b] + accepted;
        bad_[b] = kDecay * bad_[b] + (accepted < t - 1 ? 1.0 : 0.0);
    } else {
        const double got = accepted + 1.0;
        mtp_tok_[t] = mtp_n_[t] > 0 ? (1.0 - kTokAlpha) * mtp_tok_[t] + kTokAlpha * got : got;
        mtp_n_[t] += 1.0;
    }
}
