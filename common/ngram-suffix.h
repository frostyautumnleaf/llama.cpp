#pragma once

// Suffix-lookup drafting and the adaptive window policy, ported from the Strata inference engine
// (https://github.com/Niko1221/Strata, MIT) - src/spec/suffix_drafter.{hpp,cpp} and
// src/spec/draft_policy.{hpp,cpp}. Strata runs Qwen3.8-Flash-Next; these are the two pieces of its
// speculation that carry over to llama.cpp without touching a kernel.

#include "llama.h"

#include <cstddef>
#include <cstdint>
#include <vector>

// Longest-match prompt lookup: proposes the tokens that followed the longest earlier occurrence of the
// sequence's current suffix. When the answer quotes its input - an edit, a refactored file, a repeated
// block of code - the continuation is usually exact, so one verify pass accepts many tokens.
//
// The index is an open-addressing table mapping every trigram of the history to its most recent end
// position, and each position links to the previous end position of the same trigram, so appending is O(1)
// and the table is rebuilt (larger) at half load. A proposal walks the trigram's WAYS most recent earlier
// occurrences, extends each one backwards up to max_match, and keeps the longest (the most recent on
// ties). Matches shorter than min_match propose nothing - a short repeat is not evidence.
//
// assign() replaces the history with a text that may share a prefix with it (the next request of a
// conversation): the positions past the common prefix are undone newest first, which leaves the index
// exactly as appending the prefix alone would have, and the rest is appended.
struct common_suffix_drafter {
    // earlier occurrences a proposal checks; in code a trigram recurs often (Strata: 16)
    static constexpr int WAYS = 16;

    explicit common_suffix_drafter(int min_match = 16, int max_match = 64, size_t capacity_tokens = 1u << 19);

    void reset();

    // add tokens to the history: the prompt, then every accepted token
    void append(const llama_token * tokens, size_t n);
    void append(llama_token token) { append(&token, 1); }

    // make the history exactly `tokens`, keeping the shared prefix (see above)
    void assign(const llama_token * tokens, size_t n);

    // write up to max_k proposed next tokens to out; returns how many (0 = no match of at least min_match)
    int propose(int max_k, llama_token * out);

    // the length of the match behind the last propose(), 0 if there was none
    int last_match() const { return last_match_; }

    size_t size() const { return hist_.size(); }

    // the history, for callers that need to check they are still in sync with it
    llama_token token_at(size_t i) const { return hist_[i]; }

private:
    struct slot {
        uint64_t key  = 0;   // trigram hash, never 0; 0 marks a free slot
        int32_t  last = -1;  // the trigram's most recent end position (-1: none left)
    };

    static uint64_t mix(uint64_t x);

    slot * find_slot(uint64_t key, bool insert);
    uint64_t key_at(size_t end) const;
    void link(size_t end);
    void rebuild();

    int min_match_;
    int max_match_;

    std::vector<llama_token> hist_;   // the tokens
    std::vector<int32_t>     prev_;   // per end position: the previous end position of its trigram
    std::vector<slot>        table_;
    size_t mask_ = 0;
    size_t used_ = 0;
    int    last_match_ = 0;
};

// Which window to verify this round: the MTP block's drafts, or the suffix lookup's.
//
// Taken whenever the lookup proposes more than the MTP, prompt lookup lost 2-8% on ordinary text: a long
// lookup window costs much more to verify than the MTP's usual 3-4 tokens, and it only pays when enough of
// it is accepted. llama.cpp's n-gram caches answer the same problem with fixed confidence thresholds;
// Strata learns both sides online and compares expected committed tokens per millisecond:
//
//   MTP     E = the mean tokens a window of that size has committed (an EMA), at the measured cost of that size
//   lookup  E(k) = 1 + q + q^2 + ... + q^k for k <= the proposal, where q is the acceptance rate of lookup
//           drafts whose match was about as long (4 buckets of match length, decayed counts), at the measured
//           cost of a window of k + 1
//
// and takes the lookup window only when its best E/cost beats the MTP's by `margin`. Costs are the measured
// round times per window size (an EMA; a size not measured yet is scaled from the measured ones by a prior
// shape), so the policy adapts to the machine and to the context length. It only chooses which drafts to
// verify - the target model still decides every emitted token.
struct common_draft_policy {
    static constexpr int k_max_t   = 8;  // window size = 1 + drafts
    static constexpr int k_buckets = 4;  // match-length buckets

    struct pick {
        bool lookup = false;
        int  t      = 1;  // window size (1 + drafts)
    };

    explicit common_draft_policy(int max_t = k_max_t, double margin = 0.03);

    // t_mtp: the MTP's window size; lookup_k: the lookup proposal's length (0 = none); match: its length
    pick choose(int t_mtp, int lookup_k, int match) const;

    // after the round: the window it used, how many drafts were accepted, the match length and the round time
    void observe(bool lookup, int t, int accepted, int match, double round_ms);

    double lookup_rate(int match) const;  // the current q for a match length
    double cost_ms(int t) const;          // measured or scaled round time of a window of t tokens

private:
    static int bucket(int match);
    double mtp_tokens(int t) const;

    int    max_t_;
    double margin_;

    std::vector<double> cost_;     // round ms by window size
    std::vector<double> cost_n_;
    std::vector<double> mtp_tok_;  // tokens committed by MTP windows of that size
    std::vector<double> mtp_n_;
    std::vector<double> ok_;       // lookup drafts accepted, decayed, by match bucket
    std::vector<double> bad_;      // lookup windows cut short, decayed, by match bucket
};
