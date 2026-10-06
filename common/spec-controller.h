#pragma once

// Refined speculative decoding window policy, ported from the Strata inference engine
// (https://github.com/Niko1221/Strata, MIT) - src/spec/controller.{hpp,cpp}.
//
// This is Strata's "plan v0.3 P6" controller: it maximizes E[tokens committed] / T(step) over the
// choices {none (k=0) | suffix lookup with k <= proposal | MTP with k <= K_MAX}, using running
// estimates of per-position acceptance and a cost model of a (k+1)-token verify pass. It keeps
// k=0 (plain decoding) unless the best choice beats it by `min_gain` (default 5%).
//
// Acceptance is tracked per position for MTP (position i's conditional acceptance p_i, learned
// online with EMA) and per match-length bucket for lookup drafts. The cost model accounts for
// dense computation scaling, CPU expert miss costs, sync overhead, and draft time - all with
// defaults measured on Strata's RTX 5070 with Qwen3.8-Flash-Next.
//
// This complements the simpler common_draft_policy (from Strata's DraftPolicy) already in
// common/ngram-suffix.h. The controller is more sophisticated: it has an explicit cost model
// with per-expert CPU miss accounting, per-position MTP acceptance tracking, and an expansion
// mechanism that lets the window grow when all drafted tokens are accepted.

#include <algorithm>
#include <array>
#include <cstdint>

namespace spec_ctrl {

inline constexpr int K_MAX = 8;

// Cost model for a (k+1)-token verify pass plus drafting k tokens.
// Defaults are Strata's 23 Sep measurements on RTX 5070 with Qwen3.8-Flash-Next:
//   dense ratio: bench/results/2026-09-23-mmvq-multi
//   distinct experts: bench/results/2026-09-23-spec-economics
//   extra-token CPU cost: bench/results/2026-09-23-cpu-expert-multi
struct cost_model {
    double dense_ms = 11.0;      // one-token dense pass (Strata plan P3 target)
    std::array<double, K_MAX + 1> dense_ratio{
        1.0, 1.05, 1.3, 1.45, 1.85, 2.2, 2.6, 3.0, 3.3
    };                            // by n = k+1 (index n-1)
    double cpu_all_miss_ms = 15.8; // 480 experts on the CPU
    double hit_rate = 0.55;       // share of distinct experts served from VRAM
    std::array<double, K_MAX + 1> distinct_ratio{
        1.0, 1.70, 2.31, 2.88, 3.40, 3.89, 4.35, 4.80, 5.2
    };                            // U(n)/U(1): distinct experts per window
    double extra_use_cost = 0.25; // each extra token on an expert, fraction of a read
    double sync_ms = 2.4;         // device sync overhead
    double mtp_draft_ms = 1.2;    // per MTP draft token
    double lookup_draft_ms = 0.01; // per lookup draft token (negligible)

    // Step time for verifying n = k + 1 tokens, plus drafting k tokens with the given source.
    // n = k+1: the verify pass processes the last real token plus k drafted tokens.
    // CPU misses: distinct experts grow as U(n); each extra token routed to a missed expert
    // adds a fraction of a read. Tokens per distinct expert = n / U(n)-ratio.
    double step_ms(int k, bool mtp) const {
        const int n = std::clamp(k + 1, 1, K_MAX + 1);
        const double dense = dense_ms * dense_ratio[n - 1];
        const double u = distinct_ratio[n - 1];
        const double uses_per_expert = (double) n / u;
        const double cpu = (1.0 - hit_rate) * cpu_all_miss_ms * u * (1.0 + extra_use_cost * (uses_per_expert - 1.0));
        const double draft = k * (mtp ? mtp_draft_ms : lookup_draft_ms);
        return dense + cpu + sync_ms + draft;
    }
};

enum class source : uint8_t { none = 0, lookup = 1, mtp = 2 };

struct choice {
    source src = source::none;
    int k = 0;                    // number of draft tokens
    double expected_tokens = 1.0; // 1 (the verify pass always commits one) + expected accepted drafts
    double tokens_per_ms = 0.0;
};

class controller {
public:
    explicit controller(cost_model cost = {}, double min_gain = 0.05, double ema = 0.05)
        : cost_(cost), min_gain_(min_gain), ema_(ema) {
        // Priors: MTP per-position acceptance from Strata's measurement on Qwen3.8-Flash-Next
        // (0.86 per draft); lookup by match length from the offline replay
        // (2026-09-23-suffix-lookup). Both are then learned per session.
        mtp_p_.fill(0.86);
        lookup_q_ = {0.35, 0.6, 0.8, 0.92};
    }

    // `lookup_available` tokens the suffix drafter can propose now, from a match of `lookup_match` tokens.
    // `mtp_ready` is true if the model's own MTP drafter can provide drafts.
    choice choose(int lookup_available, int lookup_match, bool mtp_ready) const {
        choice best;
        best.tokens_per_ms = 1.0 / cost_.step_ms(0, false);
        const double baseline = best.tokens_per_ms;

        auto consider = [&](source s, int k, double e) {
            const double rate = e / cost_.step_ms(k, s == source::mtp);
            if (rate > best.tokens_per_ms) {
                best = choice{s, k, e, rate};
            }
        };

        const int lk = std::min(lookup_available, K_MAX);
        const double q = lookup_q_[bucket(lookup_match)];
        for (int k = 1; k <= lk; ++k) {
            consider(source::lookup, k, expected(&q, k, true));
        }
        if (mtp_ready) {
            for (int k = 1; k <= K_MAX; ++k) {
                consider(source::mtp, k, expected(mtp_p_.data(), k, false));
            }
        }

        if (best.src != source::none && best.tokens_per_ms < baseline * (1.0 + min_gain_)) {
            choice none_choice;
            none_choice.tokens_per_ms = baseline;
            return none_choice;
        }
        return best;
    }

    // After verification: `accepted` of the `k` drafted tokens matched (a prefix).
    void observe(const choice& c, int accepted, int lookup_match) {
        if (c.src == source::none || c.k <= 0) return;
        accepted = std::clamp(accepted, 0, c.k);
        // Positions 0..accepted-1 were accepted given their prefix; position `accepted` (if drafted) was rejected.
        const int seen = std::min(c.k, accepted + 1);
        for (int i = 0; i < seen; ++i) {
            const double hit = i < accepted ? 1.0 : 0.0;
            if (c.src == source::mtp) {
                mtp_p_[i] += ema_ * (hit - mtp_p_[i]);
            } else {
                lookup_q_[bucket(lookup_match)] += ema_ * (hit - lookup_q_[bucket(lookup_match)]);
            }
        }
        // Everything drafted was accepted: the positions beyond the window were never tested, so without this
        // they keep their prior forever and the window can never grow. Pull them toward the deepest observed rate.
        if (c.src == source::mtp && accepted == c.k && c.k < K_MAX) {
            for (int i = c.k; i < K_MAX; ++i) {
                mtp_p_[i] += ema_ * (mtp_p_[c.k - 1] - mtp_p_[i]);
            }
        }
    }

    double mtp_accept(int position) const { return mtp_p_[position]; }
    double lookup_accept(int match) const { return lookup_q_[bucket(match)]; }
    const cost_model& cost() const { return cost_; }

private:
    static int bucket(int match) { return match >= 16 ? 3 : match >= 8 ? 2 : match >= 5 ? 1 : 0; }

    // Expected committed tokens: the verify pass always commits one token, plus the expected number of
    // accepted drafts. For lookup, all positions share the same conditional acceptance q. For MTP, each
    // position i has its own conditional acceptance p[i].
    static double expected(const double* p, int k, bool conditional_same) {
        double e = 1.0, run = 1.0;
        for (int i = 0; i < k; ++i) {
            run *= conditional_same ? p[0] : p[i];
            e += run;
        }
        return e;
    }

    cost_model cost_;
    double min_gain_;
    double ema_;
    std::array<double, K_MAX> mtp_p_{};      // per-position MTP acceptance
    std::array<double, 4> lookup_q_{};       // lookup acceptance by match bucket
};

} // namespace spec_ctrl
