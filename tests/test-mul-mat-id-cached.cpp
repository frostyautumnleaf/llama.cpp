// test-mul-mat-id-cached
//
// MUL_MAT_ID_CACHED is the expert-cache form of mul_mat_id (Strata's expert cache): before it
// multiplies an expert it asks a residency table where that expert's weights live - a slot in one
// of the cache's tier arenas, or the model weights. Two things are easy to get wrong here and
// invisible without a test, so this is the test for them:
//
//   1. A slot holds *quantized* bytes, so the kernel has to run the same per-type vec_dot that
//      mul_mat_id runs. The first port of this kernel read the bytes as floats, which is right for
//      F32 and silently multiplies garbage for everything else - and every real model of this shape
//      is quantized. (Checked by mutation: with the vec_dot forced to F32 this test reports NaN.)
//   2. The residency table has one row per layer, so a node has to know which layer it is. Index
//      the table by expert alone and every layer reads layer 0's residency. The two rows here are
//      deliberately different, including for the same expert id.
//
// The reference is plain mul_mat_id over the model weights with the expert ids rewritten through
// the residency table: expert e becomes whichever expert the slot actually holds. That is the same
// bytes through the same kind of dot product, so the outputs have to agree.

#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <string>
#include <vector>

static const int n_embd   = 256;  // a multiple of every quant block size (Q4_K wants 256)
static const int n_ff     = 48;
static const int n_expert = 8;
static const int n_layers = 2;
static const int n_exp_u  = 3;    // n_expert_used
static const int n_tokens = 5;

// The slots are one pool shared by all layers, so each slot belongs to one (layer, expert) pair and
// the layers must not name the same slot. Tier 0: slots 0-3 are layer 0's, 4-7 are layer 1's. Tier
// 1: slots 0-1 are layer 0's, 2-3 are layer 1's.
static const int n_slots0 = 8;
static const int n_slots1 = 4;

static const int  slot_layer0[n_slots0]  = { 0, 0, 0, 0, 1, 1, 1, 1 };
static const int slot_expert0[n_slots0] = { 5, 6, 7, 2, 2, 7, 5, 4 };
static const int  slot_layer1[n_slots1]  = { 0, 0, 1, 1 };
static const int slot_expert1[n_slots1] = { 4, 1, 1, 0 };

// residency[(layer * n_expert) + expert]:
//   >= 0 : tier 0 slot          -1 : not resident, read the model weights
//   <= -2: tier 1 slot -enc-2   (a value below -2 - n_slots1 would name tier 2, which this test
//                                leaves out so that a null tier 2 arena stays legal)
static const int32_t residency[n_layers * n_expert] = {
    /* layer 0 */   0, -1,  1, -2, -1,  2, -3, -1,
    /* layer 1 */   4, -1, -1,  5,  6, -1, -1, -4,
};

static const int32_t ids[n_exp_u * n_tokens] = {
    0, 1, 2, 3, 4,
    5, 6, 7, 0, 2,
    4, 7, 1, 6, 3,
};

static ggml_backend_t g_backend = nullptr;

static void fill_random(float * dst, int64_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.0f, 1.0f / std::sqrt((float) n_embd));
    for (int64_t i = 0; i < n; ++i) {
        dst[i] = dist(rng);
    }
}

// which expert a residency entry resolves to, plus which layer that expert belongs to; -1 means the
// model weights of the layer being asked about
static int resolve_expert(int32_t enc, int * out_slot_layer) {
    if (enc == -1) {
        *out_slot_layer = -1;
        return -1;
    }
    if (enc >= 0) {
        *out_slot_layer = slot_layer0[enc];
        return slot_expert0[enc];
    }
    const int32_t val = -enc - 2;
    if (val >= n_slots1) {
        return -2; // tier 2, not part of this test
    }
    *out_slot_layer = slot_layer1[val];
    return slot_expert1[val];
}

// The cached path never takes the tiled mul_mat_id, so a match against plain mul_mat_id is judged
// with a tolerance rather than bit for bit. NaN has to fail: every comparison against NaN is false,
// which is how a kernel reading quantized bytes as floats can otherwise look like a perfect match.
static bool compare(const char * tname, const char * what, int il,
                    const std::vector<float> & got, const std::vector<float> & ref) {
    double scale = 0.0;
    for (float v : ref) {
        scale = std::max(scale, (double) std::fabs(v));
    }
    if (!(scale > 0.0)) {
        printf("  [%s] FAIL: the reference is all zeros - the test weights are wrong\n", tname);
        return false;
    }

    const double tol = 1e-3 * scale;
    double       max = 0.0;
    int            n = 0;
    int      first   = -1;

    for (size_t i = 0; i < got.size(); ++i) {
        const double d = std::fabs((double) got[i] - (double) ref[i]);
        if (!(d <= tol)) {
            if (first < 0) {
                first = (int) i;
            }
        }
        if (!(d <= max)) {
            max = d;
        }
        ++n;
    }

    if (first >= 0) {
        printf("  [%s] FAIL: layer %d %s differs at element %d (%g vs %g, tol %g)\n",
               tname, il, what, first, got[first], ref[first], tol);
        return false;
    }

    printf("  [%s] layer %d %s: %d outputs match the reference (max abs err %.2g, tol %.2g)\n",
           tname, il, what, n, max, tol);
    return true;
}

static bool differs(const char * tname, int il,
                    const std::vector<float> & a, const std::vector<float> & b) {
    for (size_t i = 0; i < a.size(); ++i) {
        if (a[i] != b[i]) {
            return true;
        }
    }
    printf("  [%s] FAIL: layer %d produced the same output with and without the residency table - "
           "the table was never consulted\n", tname, il);
    return false;
}

static bool run_case(enum ggml_type type) {
    const char * tname = ggml_type_name(type);

    struct ggml_init_params params = { /* mem_size = */ 16 * 1024 * 1024, /* mem_buffer = */ nullptr, /* no_alloc = */ true };
    struct ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        printf("  [%s] FAIL: ggml_init\n", tname);
        return false;
    }
    ggml_gallocr_t g_alloc = ggml_gallocr_new(ggml_backend_get_default_buffer_type(g_backend));

    const int64_t ne_w[4]  = { n_embd, n_ff, n_expert, 1 };
    const int64_t ne_a0[4] = { n_embd, n_ff, n_slots0, 1 };
    const int64_t ne_a1[4] = { n_embd, n_ff, n_slots1, 1 };
    const int64_t ne_in[4] = { n_embd, 1,    n_tokens, 1 };

    const size_t blob_bytes = (size_t) ggml_row_size(type, (int64_t) n_embd) * n_ff; // one expert
    const size_t nb1        = (size_t) ggml_row_size(type, (int64_t) n_embd);

    struct ggml_tensor * as_w[n_layers];      // the model weights
    struct ggml_tensor * as_f32[n_layers];    // the same weights, for the quantizing copy
    for (int il = 0; il < n_layers; ++il) {
        as_w[il]   = ggml_new_tensor(ctx, type, 4, ne_w);
        as_f32[il] = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_w);
        char nm[32];
        snprintf(nm, sizeof(nm), "as.%d", il);
        ggml_set_name(as_w[il], nm);
    }

    struct ggml_tensor * src1   = ggml_new_tensor(ctx, GGML_TYPE_F32, 4, ne_in);
    struct ggml_tensor * ids_t  = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_exp_u, n_tokens);
    struct ggml_tensor * res_t  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_layers * n_expert);
    struct ggml_tensor * res_em = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_layers * n_expert);
    struct ggml_tensor * arena0 = ggml_new_tensor(ctx, type, 4, ne_a0);
    struct ggml_tensor * arena1 = ggml_new_tensor(ctx, type, 4, ne_a1);
    ggml_set_name(arena0, "arena0");
    ggml_set_name(arena1, "arena1");

    struct ggml_tensor * ref_ids[n_layers];
    for (int il = 0; il < n_layers; ++il) {
        ref_ids[il] = ggml_new_tensor_2d(ctx, GGML_TYPE_I32, n_exp_u, n_tokens);
    }

    struct ggml_cgraph * gf = ggml_new_graph(ctx);

    // 1. quantize the weights: CPY is the public way to get valid quantized bytes, and it is what a
    //    model loader effectively does
    for (int il = 0; il < n_layers; ++il) {
        ggml_build_forward_expand(gf, ggml_cpy(ctx, as_f32[il], as_w[il]));
    }

    // 2. fill the arenas: a slot is a byte-for-byte copy of one expert of one layer, which is what
    //    the cache's copy_fn writes
    auto fill = [&](struct ggml_tensor * arena, const int * slot_layer, const int * slot_expert, int n_slots) {
        for (int s = 0; s < n_slots; ++s) {
            struct ggml_tensor * src = ggml_view_3d(ctx, as_w[slot_layer[s]], n_embd, n_ff, 1, nb1, blob_bytes,
                                                    (size_t) slot_expert[s] * blob_bytes);
            struct ggml_tensor * dst = ggml_view_3d(ctx, arena, n_embd, n_ff, 1, nb1, blob_bytes,
                                                    (size_t) s * blob_bytes);
            ggml_build_forward_expand(gf, ggml_cpy(ctx, src, dst));
        }
    };
    fill(arena0, slot_layer0, slot_expert0, n_slots0);
    fill(arena1, slot_layer1, slot_expert1, n_slots1);

    // 3. the nodes under test and their references
    struct ggml_tensor * got[n_layers];       // the table as written
    struct ggml_tensor * got_em[n_layers];    // same node, nothing resident
    struct ggml_tensor * ref[n_layers];       // mul_mat_id with the ids resolved through the table
    struct ggml_tensor * ref0[n_layers];      // mul_mat_id with the original ids
    std::vector<int32_t> want[n_layers];

    for (int il = 0; il < n_layers; ++il) {
        got[il]    = ggml_mul_mat_id_cached(ctx, as_w[il], src1, ids_t, arena0, res_t,  il, arena1, nullptr, n_slots1);
        got_em[il] = ggml_mul_mat_id_cached(ctx, as_w[il], src1, ids_t, arena0, res_em, il, arena1, nullptr, n_slots1);
        ref0[il]   = ggml_mul_mat_id(ctx, as_w[il], src1, ids_t);
        ggml_build_forward_expand(gf, got[il]);
        ggml_build_forward_expand(gf, got_em[il]);
        ggml_build_forward_expand(gf, ref0[il]);
    }

    bool ok = true;
    for (int il = 0; il < n_layers && ok; ++il) {
        want[il].resize(n_exp_u * n_tokens);
        for (int i = 0; i < n_exp_u * n_tokens; ++i) {
            int slot_layer  = -1;
            const int32_t r = resolve_expert(residency[il * n_expert + ids[i]], &slot_layer);
            if (r == -2 || (slot_layer != -1 && slot_layer != il)) {
                printf("  [%s] FAIL: layer %d expert %d resolves into layer %d - the table and the "
                       "slot contents disagree\n", tname, il, ids[i], slot_layer);
                ok = false;
                break;
            }
            want[il][i] = r == -1 ? ids[i] : r;
        }
        if (!ok) {
            break;
        }
        ref[il] = ggml_mul_mat_id(ctx, as_w[il], src1, ref_ids[il]);
        ggml_build_forward_expand(gf, ref[il]);
    }

    // Every result has to keep its own buffer: without the output flag the allocator may alias a
    // node under test onto its own reference, and the comparison would prove nothing.
    for (int il = 0; il < n_layers; ++il) {
        ggml_set_output(got[il]);
        ggml_set_output(got_em[il]);
        ggml_set_output(ref0[il]);
        if (ok) {
            ggml_set_output(ref[il]);
        }
    }

    // 4. allocate, then fill the leaf tensors
    if (ok && !ggml_gallocr_alloc_graph(g_alloc, gf)) {
        printf("  [%s] FAIL: could not allocate the graph\n", tname);
        ok = false;
    }

    if (ok) {
        for (int il = 0; il < n_layers; ++il) {
            std::vector<float> w((size_t) ggml_nelements(as_f32[il]));
            fill_random(w.data(), (int64_t) w.size(), 4242 + 101 * il);
            ggml_backend_tensor_set(as_f32[il], w.data(), 0, ggml_nbytes(as_f32[il]));
            ggml_backend_tensor_set(ref_ids[il], want[il].data(), 0, ggml_nbytes(ref_ids[il]));
        }
        std::vector<float> in((size_t) ggml_nelements(src1));
        fill_random(in.data(), (int64_t) in.size(), 777);
        ggml_backend_tensor_set(src1, in.data(), 0, ggml_nbytes(src1));
        ggml_backend_tensor_set(ids_t, ids, 0, ggml_nbytes(ids_t));
        ggml_backend_tensor_set(res_t, residency, 0, ggml_nbytes(res_t));
        std::vector<int32_t> empty(n_layers * n_expert, -1);
        ggml_backend_tensor_set(res_em, empty.data(), 0, ggml_nbytes(res_em));

        if (ggml_backend_graph_compute(g_backend, gf) != GGML_STATUS_SUCCESS) {
            printf("  [%s] FAIL: computing the graph\n", tname);
            ok = false;
        }
    }

    // 5. compare
    if (ok) {
        const size_t n_out = (size_t) ggml_nelements(got[0]);
        std::vector<float> a(n_out), b(n_out), c(n_out), d(n_out);

        for (int il = 0; il < n_layers && ok; ++il) {
            ggml_backend_tensor_get(got[il],    a.data(), 0, ggml_nbytes(got[il]));
            ggml_backend_tensor_get(ref[il],    b.data(), 0, ggml_nbytes(ref[il]));
            ggml_backend_tensor_get(got_em[il], c.data(), 0, ggml_nbytes(got_em[il]));
            ggml_backend_tensor_get(ref0[il],   d.data(), 0, ggml_nbytes(ref0[il]));

            ok = compare(tname, "resolved through the table", il, a, b) && ok;
            // with nothing resident the same node has to read the model weights
            ok = compare(tname, "with nothing resident  ", il, c, d) && ok;
            // and with the table it has to produce something else, or the check above proves nothing
            ok = differs(tname, il, a, c) && ok;
        }
    }

    ggml_gallocr_free(g_alloc);
    ggml_free(ctx);
    return ok;
}

int main(void) {
    printf("test-mul-mat-id-cached: expert-cache slots and per-layer residency\n");

    g_backend = ggml_backend_cpu_init();
    if (!g_backend) {
        printf("FAIL: could not create the CPU backend\n");
        return 1;
    }

    const enum ggml_type types[] = { GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_Q8_0, GGML_TYPE_Q4_K };

    int failures = 0;
    for (enum ggml_type type : types) {
        if (!run_case(type)) {
            ++failures;
        }
    }

    ggml_backend_free(g_backend);

    if (failures) {
        printf("FAIL: %d type(s)\n", failures);
        return 1;
    }
    printf("PASS\n");
    return 0;
}
