// Tests for the wiring of --spec-type ngram-suffix: the priority chain, the draft policy's decline, the
// per-sequence state, the resync after a rollback, and the round reporting that teaches the policy what the rest
// of the chain commits.
//
// The drafter and the policy themselves are tested against Strata's own implementations in test-ngram-suffix.cpp.
// What this file covers is the part that has no model in it but had no test: common_speculative driving them.
// The implementations it runs - ngram-suffix, ngram-mod, ngram-simple - read nothing but the token sequences they
// are handed, so a target model is simulated here instead: a round is
//
//   draft params -> common_speculative_draft -> verify against the reference text -> common_speculative_accept
//
// and the cost of a round is simulated by spinning in proportion to its window, which is what gives the draft
// policy a cost curve like a machine's rather than the microsecond clock of a test process.
//
// What each run below checks: fixed - the drafts are the continuation of the sequence the drafter was fed and most
// of what is offered is accepted; policy - the window is taken on quoting text and nothing is offered where no
// repeat reaches min_match; cost curve - on a machine where a wide verify pass costs much more than a narrow one
// the policy stops using wide windows and commits more per millisecond; chain - an implementation ahead in the
// priority list that abstains leaves the window to ngram-suffix, and ngram-suffix with nothing to offer does not
// stall the rest of the list; rollback - a sequence that shrinks resyncs the drafter instead of drafting
// nonsense; slot reuse - begin() with another prompt drafts the new sequence; sequences - two sequences drafted in
// one round keep separate indices while sharing the policy, which is what a policy of machine timings must do;
// n_max - caps the window and the draft budget the caller sizes its batch from.
//
// build (in the CMake build with tests on):  cmake --build build --target test-spec-suffix
//   TEST_SPEC_SUFFIX_DEBUG=1 ./build/bin/test-spec-suffix    # prints every round of every run

#include "speculative.h"

#include "ggml.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static int g_fail = 0;
static int g_line = 0;

#define CHECK(cond, ...)                                                                                             \
    do {                                                                                                             \
        g_line = __LINE__;                                                                                           \
        if (!(cond)) {                                                                                               \
            ++g_fail;                                                                                                \
            printf("  FAIL line %d: ", g_line);                                                                      \
            printf(__VA_ARGS__);                                                                                     \
            printf("\n");                                                                                            \
        }                                                                                                            \
    } while (0)

// ---------------------------------------------------------------------------
// the simulated target
// ---------------------------------------------------------------------------

// a round's cost, so the policy measures something: a window of t costs base_us * (1 + slope*(t-1)). the slope is
// how much wider passes cost more - on a machine whose experts are mostly on the CPU it is steep, and the draft
// policy is supposed to find that out for itself rather than assume a shape
static double spin_round_us(double base_us, int t, double slope) {
    const double us = base_us * (1.0 + slope * (t - 1));
    const int64_t  t0 = ggml_time_us();
    while ((double) (ggml_time_us() - t0) < us) {}
    return us / 1000.0;
}

// tokens are plain integers, and every corpus is built from disjoint ranges, so a repeat is always the block it
// was written to be and never an accident of the generator
static llama_tokens block(int base, int n) {
    llama_tokens t(n);
    for (int i = 0; i < n; ++i) {
        t[i] = (llama_token) (base + i);
    }
    return t;
}

static llama_tokens cat(const std::vector<llama_tokens> & parts) {
    llama_tokens t;
    for (const auto & p : parts) {
        t.insert(t.end(), p.begin(), p.end());
    }
    return t;
}

struct round_stats {
    int rounds        = 0; // rounds that ran
    int drafted       = 0; // rounds that carried at least one draft
    int drafts        = 0; // draft tokens offered
    int accepted      = 0; // draft tokens the target accepted
    int wrong         = 0; // rounds whose first draft token was not the continuation: the pass bought nothing
    int max_window    = 0;
    int widest        = 0; // rounds that used the longest window the drafter can offer
    int tokens        = 0; // tokens committed, which is rounds + accepted
    double ms         = 0; // the cost the simulated machine charged for the rounds

    double tokens_per_round() const { return rounds ? (double) tokens / rounds : 0.0; }
    double drafted_fraction() const { return rounds ? (double) drafted / rounds : 0.0; }
    double tokens_per_ms() const { return ms > 0 ? tokens / ms : 0.0; }
};

// one sequence of the simulated target: the committed tokens, the last sampled one, and where the reference
// generation has got to
struct seq_state {
    llama_tokens seq;        // what the server hands over as dparams.prompt: the prompt plus what has been emitted
    llama_token  id_last = 0;
    size_t       in_gen = 0; // index in gen of the next token the target would sample; gen[0] is the first sampled
    bool         done   = false;
};

static void seq_init(seq_state & s, const llama_tokens & prompt, const llama_tokens & gen) {
    s.seq    = prompt;
    s.id_last = gen.at(0);
    s.in_gen  = 1;
    s.done    = false;
}

// how many of the draft the target would accept: the longest prefix that is the reference continuation
static int verify(const llama_tokens & draft, const llama_tokens & gen, size_t in_gen) {
    int n = 0;
    while (n < (int) draft.size() && in_gen + n < gen.size() && draft[n] == gen[in_gen + n]) {
        ++n;
    }
    return n;
}

// one round for one sequence; `truncate` throws away the tail of the draft before verifying, which is how a
// rejection looks from here
static void round_one(
        common_speculative * spec,
        llama_seq_id seq_id,
        seq_state & s,
        const llama_tokens & gen,
        int n_max,
        round_stats & st,
        int truncate = -1,
        double cost_slope = 0.35) {
    llama_tokens draft;

    auto & dp = common_speculative_get_draft_params(spec, seq_id);

    dp.drafting = true;
    dp.n_max    = n_max;
    dp.pos0     = (llama_pos) s.seq.size();
    dp.id_last  = s.id_last;
    dp.prompt   = &s.seq;
    dp.result   = &draft;
    dp.result_q = nullptr;
    dp.temp     = 0.0f;
    dp.seed     = LLAMA_DEFAULT_SEED;

    common_speculative_draft(spec);

    if (truncate >= 0 && (int) draft.size() > truncate) {
        draft.resize(truncate);
    }

    const int n_acc = verify(draft, gen, s.in_gen);

    if (std::getenv("TEST_SPEC_SUFFIX_DEBUG") != nullptr && st.rounds < 40) {
        printf("    round %3d: seq %d, %zu drafted, %d accepted, in_gen %zu\n", st.rounds, (int) seq_id, draft.size(),
                n_acc, s.in_gen);
    }

    // the round is counted against the drafter when its first token is not the continuation, which is the case
    // where the verify pass bought nothing; a proposal that ran into a divergence two tokens later still committed
    // those two
    if (!draft.empty() && (s.in_gen >= gen.size() || draft[0] != gen[s.in_gen])) {
        ++st.wrong;
    }

    st.rounds++;
    st.drafts += (int) draft.size();
    st.accepted += n_acc;
    if (!draft.empty()) {
        st.drafted++;
    }
    st.max_window = std::max(st.max_window, (int) draft.size() + 1);
    if ((int) draft.size() == n_max) {
        st.widest++;
    }

    // commit. the token sampled in the previous round joins the sequence now, then the accepted drafts; the token
    // the target samples this round is id_last and joins next round - server-context.cpp,
    // handle_last_sampled_token(): prompt.tokens.push_back(sampled) happens when the batch is made, after the
    // draft params were read. so dparams.prompt never contains id_last, which is what the drafters assume
    s.seq.push_back(s.id_last);
    for (int i = 0; i < n_acc; ++i) {
        s.seq.push_back(draft[i]);
    }
    const llama_token bonus = gen.at(s.in_gen + n_acc);
    s.id_last = bonus;
    s.in_gen += (size_t) n_acc + 1;
    s.done = s.in_gen >= gen.size();

    st.tokens = (int) s.in_gen;

    st.ms += spin_round_us(120.0, (int) draft.size() + 1, cost_slope);

    common_speculative_accept(spec, seq_id, (uint16_t) n_acc);
}

// run a whole generation; the committed text must be exactly prompt + gen whatever the drafts were
static round_stats simulate(
        common_speculative * spec,
        llama_seq_id seq_id,
        const llama_tokens & prompt,
        const llama_tokens & gen,
        int n_max,
        int max_rounds = 100000,
        double cost_slope = 0.35) {
    seq_state s;
    seq_init(s, prompt, gen);

    common_speculative_begin(spec, seq_id, prompt);

    round_stats st;
    const int   fails0 = g_fail; // only a failure of this run stops it, not one of an earlier test

    while (!s.done && st.rounds < max_rounds) {
        round_one(spec, seq_id, s, gen, n_max, st, -1, cost_slope);
        CHECK(s.seq == cat({ prompt, { gen.begin(), gen.begin() + (long) (s.in_gen - 1) } }),
                "seq %d: the committed text diverged from the reference at round %d", (int) seq_id, st.rounds);
        if (g_fail > fails0) {
            break;
        }
    }
    CHECK(st.rounds < max_rounds, "seq %d: the generation did not finish in %d rounds", (int) seq_id, max_rounds);
    return st;
}

// ---------------------------------------------------------------------------
// the corpora
// ---------------------------------------------------------------------------

struct corpus {
    const char *   name;
    llama_tokens   prompt;
    llama_tokens   gen;   // gen[0] is the token sampled after the prefill
    const char *   what;
};

static std::vector<corpus> make_corpora() {
    std::vector<corpus> out;

    // quoting: an answer that returns a slice of the prompt, twice, with one token changed in the second copy
    {
        llama_tokens prompt = block(1000, 240);

        llama_tokens quote1(prompt.begin() + 20, prompt.begin() + 140);
        llama_tokens quote2(prompt.begin() + 20, prompt.begin() + 80);
        llama_tokens tail(prompt.begin() + 81, prompt.begin() + 140);

        llama_tokens gen = cat({ { 500 }, quote1, { 501 }, quote2, { 9000 }, tail, block(7000, 12) });

        out.push_back({ "quote", prompt, gen, "an answer that quotes 120 prompt tokens, edits one, quotes on" });
    }

    // novel: nothing repeats, so nothing is worth drafting
    {
        llama_tokens prompt = block(2000, 120);
        llama_tokens gen    = cat({ { 600 }, block(20000, 60) });

        out.push_back({ "novel", prompt, gen, "an answer of nothing but new tokens" });
    }

    // short: eight tokens recur with a different separator each time, which is under the 16-token evidence bar
    {
        llama_tokens unit = block(30000, 8);
        llama_tokens prompt;
        llama_tokens gen = { 700 };
        for (int i = 0; i < 14; ++i) {
            prompt.insert(prompt.end(), unit.begin(), unit.end());
            prompt.push_back((llama_token) (40000 + i));
        }
        for (int i = 0; i < 10; ++i) {
            gen.insert(gen.end(), unit.begin(), unit.end());
            gen.push_back((llama_token) (41000 + i));
        }
        out.push_back({ "short", prompt, gen, "a repeat of 8 tokens behind a separator: under min_match" });
    }

    // mid: a repeat of exactly 20 tokens, which is over the lookup drafter's bar and under ngram-mod's
    {
        llama_tokens unit = block(50000, 20);
        llama_tokens prompt = cat({ block(55000, 40), unit, block(56000, 40) });
        llama_tokens gen = cat({ { 800 }, unit, block(57000, 24), unit, block(58000, 8) });

        out.push_back({ "mid", prompt, gen, "a repeat of 20 tokens: over 16, under ngram-mod's 24" });
    }

    // editfile: the answer is the prompt returned, with one token changed in the middle - the case Strata's
    // prompt lookup is tuned for, and the one an n-gram drafter of fixed window length does worst on
    {
        llama_tokens prompt = block(8000, 600);

        llama_tokens head(prompt.begin(), prompt.begin() + 300);
        llama_tokens tail(prompt.begin() + 301, prompt.end());

        llama_tokens gen = cat({ { 8500 }, head, { 8999 }, tail, block(9000, 10) });

        out.push_back({ "editfile", prompt, gen, "the prompt returned with one token changed" });
    }

    // twelve: a repeat of exactly 12 tokens - the lookup drafter has nothing to say, ngram-simple has plenty
    {
        llama_tokens unit = block(60000, 12);
        llama_tokens prompt = cat({ block(61000, 40), unit, block(62000, 40) });
        llama_tokens gen = cat({ { 900 }, unit, block(63000, 30), unit, block(64000, 8) });

        out.push_back({ "twelve", prompt, gen, "a repeat of 12 tokens: under the lookup bar, over ngram-simple's" });
    }

    return out;
}

static const corpus * find(const std::vector<corpus> & corpora, const char * name) {
    for (const auto & c : corpora) {
        if (0 == std::strcmp(c.name, name)) {
            return &c;
        }
    }
    return nullptr;
}

// a lookup proposal is only wrong where the answer diverged from its own earlier occurrence, or where what came
// after the earlier occurrence is not what comes after this one; both are the design, not a fault. what is worth
// asserting is that they stay a minority of the drafted rounds
static bool proposals_mostly_right(const round_stats & st, int pct = 75) {
    return st.drafted == 0 || 100 * (st.drafted - st.wrong) / st.drafted >= pct;
}

// ---------------------------------------------------------------------------
// the configurations
// ---------------------------------------------------------------------------

static common_speculative_ptr init_spec(
        const std::vector<common_speculative_type> & types,
        uint32_t n_seq,
        int n_max,
        bool adaptive) {
    common_params_speculative sp;
    sp.types                      = types;
    sp.ngram_suffix.n_max         = n_max;
    sp.ngram_suffix.adaptive      = adaptive;

    common_speculative * raw = common_speculative_init(sp, n_seq);
    if (raw == nullptr) {
        return nullptr;
    }
    return common_speculative_ptr(raw);
}

// ---------------------------------------------------------------------------
// the tests
// ---------------------------------------------------------------------------

// the drafter drafts the whole proposal when the policy is off, and every token of it is the reference
static void test_fixed(const std::vector<corpus> & corpora) {
    printf("fixed (--spec-ngram-suffix-fixed)\n");

    const corpus * quote = find(corpora, "quote");
    CHECK(quote != nullptr, "corpus quote");
    if (!quote || g_fail) {
        return;
    }

    auto spec = init_spec({ COMMON_SPECULATIVE_TYPE_NGRAM_SUFFIX }, 1, 7, /* adaptive */ false);
    CHECK(spec != nullptr, "common_speculative_init returned null");
    if (!spec) {
        return;
    }

    CHECK(common_speculative_n_max(spec.get()) == 7, "n_max of the initialized implementations is %d, want 7",
            common_speculative_n_max(spec.get()));

    round_stats st = simulate(spec.get(), 0, quote->prompt, quote->gen, 7);

    CHECK(proposals_mostly_right(st), "%d of %d drafted rounds diverged from the reference somewhere", st.wrong,
            st.drafted);
    CHECK(st.accepted * 5 >= st.drafts * 4, "only %.2f of the offered draft tokens were accepted", st.drafted ?
            (double) st.accepted / st.drafts : 0.0);
    CHECK(st.max_window == 8, "the widest window is %d, want 8 (n_max 7 + the target's own token)", st.max_window);
    CHECK(st.drafted_fraction() > 0.25, "the drafter offered nothing in %.2f of %d rounds", st.drafted_fraction(),
            st.rounds);
    CHECK(st.tokens_per_round() > 2.5, "%.2f tokens a round on quoting text, want more than 2.5",
            st.tokens_per_round());

    printf("  quote:    %d rounds, %.2f tokens a round, %d/%d drafted, %d/%d offered accepted\n", st.rounds,
            st.tokens_per_round(), st.drafted, st.rounds, st.accepted, st.drafts);

    // the case Strata's lookup is tuned for: the prompt returned with one token changed. the rounds are long here
    // and the pass over them short, which is the whole point of drafting from the suffix rather than an n-gram
    const corpus * editfile = find(corpora, "editfile");
    CHECK(editfile != nullptr, "corpus editfile");
    if (editfile) {
        auto spec2 = init_spec({ COMMON_SPECULATIVE_TYPE_NGRAM_SUFFIX }, 1, 7, false);
        if (!spec2) {
            CHECK(false, "common_speculative_init returned null");
            return;
        }
        round_stats e = simulate(spec2.get(), 0, editfile->prompt, editfile->gen, 7);
        // the 16-token evidence bar costs about 16 single-token rounds after the answer starts and after every
        // divergence from what it quoted; on 611 tokens that is where the measured 5.4 comes from
        CHECK(e.tokens_per_round() > 5.0, "%.2f tokens a round when the answer is the prompt returned, want over 5",
                e.tokens_per_round());
        CHECK(proposals_mostly_right(e), "%d of %d drafted rounds diverged from the reference", e.wrong, e.drafted);
        printf("  editfile: %d rounds, %.2f tokens a round, %d/%d drafted, %d/%d offered accepted\n", e.rounds,
                e.tokens_per_round(), e.drafted, e.rounds, e.accepted, e.drafts);
    }
}

// the policy: takes the lookup window where it pays, declines where it does not, and a declined round is a round
// with no draft in it at all
static void test_policy(const std::vector<corpus> & corpora) {
    printf("policy (--spec-type ngram-suffix, adaptive)\n");

    const corpus * quote = find(corpora, "quote");
    const corpus * novel = find(corpora, "novel");
    const corpus * short_ = find(corpora, "short");
    CHECK(quote && novel && short_, "corpora");
    if (!(quote && novel && short_) || g_fail) {
        return;
    }

    auto spec = init_spec({ COMMON_SPECULATIVE_TYPE_NGRAM_SUFFIX }, 1, 7, /* adaptive */ true);
    CHECK(spec != nullptr, "common_speculative_init returned null");
    if (!spec) {
        return;
    }

    round_stats a = simulate(spec.get(), 0, quote->prompt, quote->gen, 7);
    CHECK(proposals_mostly_right(a), "quote: %d of %d drafted rounds diverged from the reference", a.wrong,
            a.drafted);
    // the policy takes every window the drafter offers here, which is what it should do when they are accepted
    CHECK(a.drafted_fraction() > 0.25, "quote: the policy took the lookup window in only %.2f of %d rounds",
            a.drafted_fraction(), a.rounds);
    CHECK(a.tokens_per_round() > 2.5, "quote: %.2f tokens a round, want more than 2.5", a.tokens_per_round());

    // a fresh generation on the same spec: the policy keeps what it learned, which is what it is for
    round_stats n = simulate(spec.get(), 0, novel->prompt, novel->gen, 7);
    CHECK(n.wrong == 0, "novel: %d rounds diverged from the reference where nothing repeats", n.wrong);
    CHECK(n.drafted == 0, "novel: %d draft tokens offered where nothing repeats", n.drafted);
    CHECK(n.tokens_per_round() < 1.05, "novel: %.2f tokens a round, want one", n.tokens_per_round());

    round_stats s = simulate(spec.get(), 0, short_->prompt, short_->gen, 7);
    CHECK(s.drafted == 0, "short: %d draft tokens offered over a repeat of 8, want none (min_match is 16)", s.drafted);

    printf("  quote: %d rounds, %.2f tokens a round, %d/%d drafted\n", a.rounds, a.tokens_per_round(), a.drafted,
            a.rounds);
    printf("  novel: %d rounds, %.2f tokens a round, %d drafted\n", n.rounds, n.tokens_per_round(), n.drafted);
    printf("  short: %d rounds, %.2f tokens a round, %d drafted\n", s.rounds, s.tokens_per_round(), s.drafted);
}

// the policy's ground is the machine it measures. on a machine where a wide verify pass costs much more than a
// narrow one - Strata's own cost shape is +10 ms a token with the missed experts left on the CPU - drafting the
// whole proposal is the wrong call even when every draft is accepted, and the policy is meant to find that out by
// measuring, not assume it. the probe clause is what lets it measure a window size it has not tried yet.
static void test_policy_cost_curve(const std::vector<corpus> & corpora) {
    printf("cost curve (a machine where a wide window is expensive)\n");

    const corpus * editfile = find(corpora, "editfile");
    CHECK(editfile != nullptr, "corpus editfile");
    if (!editfile) {
        return;
    }

    const double slope = 1.5;  // a window of 8 costs 11.5 times a window of 1

    auto fixed = init_spec({ COMMON_SPECULATIVE_TYPE_NGRAM_SUFFIX }, 1, 7, /* adaptive */ false);
    auto adapt = init_spec({ COMMON_SPECULATIVE_TYPE_NGRAM_SUFFIX }, 1, 7, /* adaptive */ true);
    if (!fixed || !adapt) {
        CHECK(false, "common_speculative_init returned null");
        return;
    }

    round_stats f = simulate(fixed.get(), 0, editfile->prompt, editfile->gen, 7, 100000, slope);
    round_stats h = simulate(adapt.get(), 0, editfile->prompt, editfile->gen, 7, 100000, slope);

    CHECK(f.tokens == h.tokens, "the policy changed what was emitted: %d tokens instead of %d", h.tokens, f.tokens);
    // the policy probes a window size it has not measured at most kProbes times, and stops using it once it has
    // priced it; drafting the whole proposal uses the widest window whenever there is a proposal at all
    CHECK(h.widest <= 4, "the policy used the widest window in %d rounds after pricing it", h.widest);
    CHECK(f.widest > 20, "drafting the whole proposal used the widest window in only %d of %d rounds", f.widest,
            f.rounds);
    CHECK(h.tokens_per_ms() > f.tokens_per_ms(), "%.3f tokens per ms with the policy against %.3f without",
            h.tokens_per_ms(), f.tokens_per_ms());

    printf("  fixed:  %d rounds, %.2f tokens a round, %.3f tokens per ms, widest window in %d rounds\n", f.rounds,
            f.tokens_per_round(), f.tokens_per_ms(), f.widest);
    printf("  policy: %d rounds, %.2f tokens a round, %.3f tokens per ms, widest window in %d rounds\n", h.rounds,
            h.tokens_per_round(), h.tokens_per_ms(), h.widest);
}

// the chain: an implementation ahead in the priority list abstains, so the window reaches ngram-suffix
static void test_chain_after_abstention(const std::vector<corpus> & corpora) {
    printf("chain (ngram-mod abstains, ngram-suffix gets the window)\n");

    const corpus * mid = find(corpora, "mid");
    CHECK(mid != nullptr, "corpus mid");
    if (!mid || g_fail) {
        return;
    }

    // on its own, ngram-mod cannot draft from a repeat of 20: it matches 24 and wants 48 after it
    {
        auto only_mod = init_spec({ COMMON_SPECULATIVE_TYPE_NGRAM_MOD }, 1, 7, true);
        CHECK(only_mod != nullptr, "init ngram-mod alone");
        if (only_mod) {
            round_stats st = simulate(only_mod.get(), 0, mid->prompt, mid->gen, 7);
            CHECK(st.drafted == 0, "ngram-mod alone drafted %d tokens over a repeat of 20, want none", st.drafted);
            CHECK(st.tokens_per_round() < 1.05, "ngram-mod alone: %.2f tokens a round, want one",
                    st.tokens_per_round());
            printf("  ngram-mod alone: %d rounds, %d drafted\n", st.rounds, st.drafted);
        }
    }

    {
        auto both = init_spec({ COMMON_SPECULATIVE_TYPE_NGRAM_MOD, COMMON_SPECULATIVE_TYPE_NGRAM_SUFFIX }, 1, 7,
                /* adaptive */ false);
        CHECK(both != nullptr, "init ngram-mod,ngram-suffix");
        if (!both) {
            return;
        }
        round_stats st = simulate(both.get(), 0, mid->prompt, mid->gen, 7);
        CHECK(proposals_mostly_right(st), "%d of %d drafted rounds diverged from the reference", st.wrong,
                st.drafted);
        CHECK(st.drafted > 0, "the chain reached neither drafter: %d rounds, nothing offered", st.rounds);
        // the repeat is 20 tokens, so the drafter has something to propose only over its last few tokens: the
        // point of the test is that the window reached it at all, not how much it made of this corpus
        CHECK(st.tokens > st.rounds, "%d tokens over %d rounds: the drafted rounds committed nothing", st.tokens,
                st.rounds);
        printf("  both: %d rounds, %.2f tokens a round, %d/%d drafted\n", st.rounds, st.tokens_per_round(), st.drafted,
                st.rounds);
    }
}

// the chain the other way: ngram-suffix has nothing to offer and the rest of the list drafts anyway. note that
// ngram-simple sits ahead of ngram-suffix in the priority list, so this is about the window not stalling rather
// than about falling through to a later implementation - test_chain_after_abstention covers that direction
static void test_chain_after_decline(const std::vector<corpus> & corpora) {
    printf("chain (ngram-suffix has no match, the rest of the list drafts)\n");

    const corpus * twelve = find(corpora, "twelve");
    CHECK(twelve != nullptr, "corpus twelve");
    if (!twelve || g_fail) {
        return;
    }

    {
        auto alone = init_spec({ COMMON_SPECULATIVE_TYPE_NGRAM_SUFFIX }, 1, 7, false);
        if (!alone) {
            CHECK(false, "init ngram-suffix alone");
            return;
        }
        round_stats a = simulate(alone.get(), 0, twelve->prompt, twelve->gen, 7);
        CHECK(a.drafted == 0, "ngram-suffix alone offered %d tokens over a repeat of 12, want none", a.drafted);
        printf("  ngram-suffix alone: %d rounds, %d drafted\n", a.rounds, a.drafted);
    }

    {
        auto chain = init_spec({ COMMON_SPECULATIVE_TYPE_NGRAM_SUFFIX, COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE }, 1, 7,
                true);
        if (!chain) {
            CHECK(false, "init ngram-suffix,ngram-simple");
            return;
        }
        round_stats st = simulate(chain.get(), 0, twelve->prompt, twelve->gen, 7);
        CHECK(st.drafted > 0, "the chain stalled: %d rounds and nothing offered, although a 12-token repeat is "
                              "ngram-simple's to take", st.rounds);
        printf("  with ngram-simple behind it: %d rounds, %d/%d drafted, widest window %d\n", st.rounds, st.drafted,
                st.rounds, st.max_window);
    }
}

// a rollback: the tail the drafter has already seen is thrown away, as when the target rejects a draft and a
// checkpoint is restored. The next round must be right again, not merely safe.
static void test_rollback(const std::vector<corpus> & corpora) {
    printf("rollback (the sequence shrinks under the drafter)\n");

    const corpus * quote = find(corpora, "quote");
    CHECK(quote != nullptr, "corpus quote");
    if (!quote || g_fail) {
        return;
    }

    auto spec = init_spec({ COMMON_SPECULATIVE_TYPE_NGRAM_SUFFIX }, 1, 7, false);
    CHECK(spec != nullptr, "common_speculative_init returned null");
    if (!spec) {
        return;
    }

    seq_state s;
    seq_init(s, quote->prompt, quote->gen);
    common_speculative_begin(spec.get(), 0, quote->prompt);

    round_stats st;
    int rounds_until_rollback = 12;
    const int fails0          = g_fail;

    while (!s.done && st.rounds < 100000) {
        // throw away the second half of the answer once, and again later: the drafter has indexed tokens that
        // the sequence no longer has
        if (st.rounds == rounds_until_rollback) {
            // a checkpoint restore never takes back the prompt, only what the target generated
            const size_t n_back = std::min<size_t>(12, s.in_gen - 1);

            s.in_gen -= n_back;
            s.seq.resize(quote->prompt.size() + s.in_gen - 1);
            s.id_last = quote->gen[s.in_gen - 1];

            rounds_until_rollback += 40;
        }

        round_one(spec.get(), 0, s, quote->gen, 7, st);

        CHECK(s.seq == cat({ quote->prompt, { quote->gen.begin(), quote->gen.begin() + (long) (s.in_gen - 1) } }),
                "the committed text diverged from the reference at round %d", st.rounds);
        if (g_fail > fails0) {
            break;
        }
    }

    CHECK(proposals_mostly_right(st), "%d of %d drafted rounds diverged from the reference after rollbacks",
            st.wrong, st.drafted);
    // each rollback throws away part of what the answer had quoted, so the drafter must be shown the evidence again
    // from there: about min_match rounds per rollback, which is where the measured number lands
    CHECK(st.tokens_per_round() > 2.0, "%.2f tokens a round after rollbacks, want more than 2.0",
            st.tokens_per_round());

    printf("  %d rounds with 2 rollbacks, %.2f tokens a round, %d/%d drafted\n", st.rounds, st.tokens_per_round(),
            st.drafted, st.rounds);
}

// a slot handed back to the pool: begin() again with a different prompt, on the same sequence id
static void test_slot_reuse(const std::vector<corpus> & corpora) {
    printf("slot reuse (begin again with another prompt on the same sequence id)\n");

    const corpus * quote = find(corpora, "quote");
    const corpus * mid   = find(corpora, "mid");
    CHECK(quote && mid, "corpora");
    if (!(quote && mid) || g_fail) {
        return;
    }

    auto spec = init_spec({ COMMON_SPECULATIVE_TYPE_NGRAM_SUFFIX }, 1, 7, false);
    CHECK(spec != nullptr, "common_speculative_init returned null");
    if (!spec) {
        return;
    }

    round_stats a = simulate(spec.get(), 0, quote->prompt, quote->gen, 7);
    round_stats b = simulate(spec.get(), 0, mid->prompt, mid->gen, 7);   // same seq id, nothing in common

    CHECK(proposals_mostly_right(a) && proposals_mostly_right(b), "draft rounds that diverged from the reference: %d + %d",
            a.wrong, b.wrong);
    CHECK(b.drafted > 0, "the second task on the slot got no drafts: the first one's index was in the way");
    CHECK(b.tokens_per_round() > 1.0, "%.2f tokens a round on the second task: nothing was drafted for it",
            b.tokens_per_round());

    printf("  task 1: %.2f tokens a round, task 2: %.2f tokens a round, %d/%d drafted\n", a.tokens_per_round(),
            b.tokens_per_round(), b.drafted, b.rounds);
}

// several sequences, drafted in one round each, with a rejection in one of them
static void test_multi_seq(const std::vector<corpus> & corpora) {
    printf("sequences (--parallel 2)\n");

    const corpus * quote = find(corpora, "quote");
    const corpus * mid   = find(corpora, "mid");
    CHECK(quote && mid, "corpora");
    if (!(quote && mid) || g_fail) {
        return;
    }

    auto spec = init_spec({ COMMON_SPECULATIVE_TYPE_NGRAM_SUFFIX }, 2, 7, false);
    CHECK(spec != nullptr, "common_speculative_init returned null");
    if (!spec) {
        return;
    }

    seq_state a, b;
    seq_init(a, quote->prompt, quote->gen);
    seq_init(b, mid->prompt, mid->gen);

    common_speculative_begin(spec.get(), 0, quote->prompt);
    common_speculative_begin(spec.get(), 1, mid->prompt);

    round_stats sa, sb;
    int n = 0;
    const int fails0 = g_fail;

    while ((!a.done || !b.done) && n < 100000) {
        if (!a.done) {
            round_one(spec.get(), 0, a, quote->gen, 7, sa);
        }
        if (!b.done) {
            // every fourth round the target keeps only the first draft token
            round_one(spec.get(), 1, b, mid->gen, 7, sb, (n % 4 == 3) ? 1 : -1);
        }
        ++n;

        CHECK(a.seq == cat({ quote->prompt, { quote->gen.begin(), quote->gen.begin() + (long) (a.in_gen - 1) } }),
                "sequence 0 diverged from its reference at round %d", n);
        CHECK(b.seq == cat({ mid->prompt, { mid->gen.begin(), mid->gen.begin() + (long) (b.in_gen - 1) } }),
                "sequence 1 diverged from its reference at round %d", n);
        if (g_fail > fails0) {
            break;
        }
    }

    CHECK(proposals_mostly_right(sa) && proposals_mostly_right(sb), "draft rounds that diverged: %d + %d", sa.wrong,
            sb.wrong);
    CHECK(sa.drafted > 0 && sb.drafted > 0, "one of the two sequences was never drafted: %d + %d", sa.drafted,
            sb.drafted);

    printf("  seq 0: %.2f tokens a round, seq 1: %.2f tokens a round (one rejection in four rounds)\n",
            sa.tokens_per_round(), sb.tokens_per_round());
}

// the length the rest of the system has to size its batch by
static void test_n_max() {
    printf("the draft budget\n");

    {
        common_params_speculative sp;
        sp.types              = { COMMON_SPECULATIVE_TYPE_NGRAM_SUFFIX };
        sp.ngram_suffix.n_max = 5;
        CHECK(common_speculative_n_max(&sp) == 5, "n_max for ngram-suffix n_max=5 is %d",
                common_speculative_n_max(&sp));
    }

    {
        common_params_speculative sp;
        sp.types                      = { COMMON_SPECULATIVE_TYPE_NGRAM_SUFFIX, COMMON_SPECULATIVE_TYPE_DRAFT_MTP };
        sp.ngram_suffix.n_max         = 3;
        sp.draft.n_max                = 4;
        CHECK(common_speculative_n_max(&sp) == 4, "n_max with a draft model that goes further is %d, want 4",
                common_speculative_n_max(&sp));
    }

    auto spec = init_spec({ COMMON_SPECULATIVE_TYPE_NGRAM_SUFFIX }, 1, 2, false);
    CHECK(spec != nullptr, "common_speculative_init returned null");
    if (!spec) {
        return;
    }

    const std::vector<corpus> corpora = make_corpora();
    const corpus * quote              = find(corpora, "quote");
    round_stats st                    = simulate(spec.get(), 0, quote->prompt, quote->gen, 2);

    CHECK(st.max_window == 3, "a window of %d with n_max 2, want 3", st.max_window);
    CHECK(proposals_mostly_right(st), "%d of %d drafted rounds diverged from the reference", st.wrong, st.drafted);
    printf("  n_max 2: %.2f tokens a round, widest window %d\n", st.tokens_per_round(), st.max_window);
}

// the text is the target's, whichever way each round is decided
static void test_output_invariance(const std::vector<corpus> & corpora) {
    printf("output invariance\n");

    const std::vector<std::pair<std::vector<common_speculative_type>, bool>> configs = {
        { { COMMON_SPECULATIVE_TYPE_NGRAM_SUFFIX }, true },
        { { COMMON_SPECULATIVE_TYPE_NGRAM_SUFFIX }, false },
        { { COMMON_SPECULATIVE_TYPE_NGRAM_SUFFIX, COMMON_SPECULATIVE_TYPE_NGRAM_SIMPLE }, true },
        { { COMMON_SPECULATIVE_TYPE_NGRAM_MOD, COMMON_SPECULATIVE_TYPE_NGRAM_SUFFIX }, true },
    };

    for (const auto & c : corpora) {
        for (const auto & cfg : configs) {
            auto spec = init_spec(cfg.first, 1, 7, cfg.second);
            if (!spec) {
                CHECK(false, "common_speculative_init returned null for %s", c.name);
                continue;
            }
            round_stats st = simulate(spec.get(), 0, c.prompt, c.gen, 7);
            CHECK(st.tokens == (int) c.gen.size(), "%s with %zu spec types: %d of %zu generated tokens committed",
                    c.name, cfg.first.size(), st.tokens, c.gen.size());
        }
    }
    printf("  %zu corpora x %zu configurations, every committed token the reference's\n", corpora.size(),
            configs.size());
}

int main(void) {
    const std::vector<corpus> corpora = make_corpora();

    printf("corpora:\n");
    for (const auto & c : corpora) {
        printf("  %-8s %4zu prompt tokens, %4zu generated - %s\n", c.name, c.prompt.size(), c.gen.size(), c.what);
    }
    printf("\n");

    test_fixed(corpora);
    test_policy(corpora);
    test_policy_cost_curve(corpora);
    test_chain_after_abstention(corpora);
    test_chain_after_decline(corpora);
    test_rollback(corpora);
    test_slot_reuse(corpora);
    test_multi_seq(corpora);
    test_n_max();
    test_output_invariance(corpora);

    if (g_fail) {
        printf("\nFAILED: %d\n", g_fail);
        return 1;
    }
    printf("\nall ok\n");
    return 0;
}
