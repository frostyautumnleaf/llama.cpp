# Strata port on `3x` - what works, what does not

Measured on this branch against `master` (cb7934c52) and Strata's own sources.

## Working, and verified

**Prompt-lookup drafting (the 42x work) - `--spec-type ngram-suffix`**

- `common/ngram-cache.{h,cpp}`: the nested `std::unordered_map` caches replaced by flat caches; inner follower
  maps replaced by sorted vectors searched with the branchless binary search from the write-up (length drives
  the iteration count, not the compared value, so searches pipeline). Verified against the upstream map:
  105 999 n-grams, 479 990 follower counts, 0 mismatches; 30 000 draft positions, 0 acceptance mismatches.
- `common/ngram-suffix.{h,cpp}`: Strata's suffix drafter and draft policy. Verified against Strata's own
  `SuffixDrafter` and `DraftPolicy`: 81 047 proposals and 20 000 policy picks, 0 mismatches.
- `common/spec-controller.h`: the refined window policy (Strata plan v0.3 P6 controller).
- `common/speculative.cpp`: wired into the spec chain above the model drafters, so a declined round falls
  through to `draft-mtp` rather than losing the step.
- `tests/test-ngram-cache`, `test-ngram-suffix`, `test-spec-controller`, `test-spec-suffix` all pass.
  `test-spec-suffix` checks output invariance: every committed token identical to the reference across 6
  corpora x 4 configurations.

**Model-loading fixes** (`qwen4exp` MTP block): the MTP layer's compress ratio, `out_ids` built only when the
batch has outputs, `k_idxs` kept in the graph, the saver writing `n_layer_all` for per-layer arrays. These are
what let `draft-mtp` run at all on this model.

**The expert-cache kernel itself** (`GGML_OP_MUL_MAT_ID_CACHED`), CPU path - `tests/test-mul-mat-id-cached`.
The cache is still inert (nothing hands it a tensor yet, see below); this is the op underneath it, which is
now correct and testable on its own. The reference is plain `mul_mat_id` over the model weights with the
expert ids resolved through the residency table - the same bytes through the same kind of dot product - for
F32/F16/Q8_0/Q4_K, two layers with deliberately different residency rows, 720 outputs each, matching exactly.
Three mutations were run against it to check it fails for the right reasons: ignoring the residency table
(fails, all four types), indexing the table by expert instead of by `(layer, expert)` (fails at layer 1, all
four), and reading the slot bytes as floats (fails for Q8_0 and Q4_K). The comparison has to be NaN-aware -
the first version of this test passed all three mutations, because every comparison against NaN is false.

## Ported but not functional - arguments accepted and ignored

`--expert-cache`, `--expert-cache-secondary`, `--expert-cache-tertiary`, `--expert-adapt`,
`--expert-adapt-interval`, `--expert-adapt-max-moves`, `--expert-prefetch`, `--expert-profile`,
`--kv-resident`.

These now warn once and are ignored. Before that they were fatal, which is why the model would not load:

1. **Segfault on the first graph build.** `combined_residency_table()` made a tensor on the graph context and
   then filled `table->data`. The graph context is created with `no_alloc` (`llama-graph.cpp:1355`), so that
   pointer is null. Reproduced: `--expert-cache 4` on a small model dies with SIGSEGV right after the cache
   reports itself initialised.
2. **The arena is host RAM, not VRAM.** `llama_expert_cache_tier::open()` allocates with
   `ggml_new_tensor_1d` on a `ggml_init({..., no_alloc = false})` context, which is a plain malloc'd pool.
   Nothing routes it to a GPU buffer - so the tier cannot do its one job. Handing it to `mul_mat_id` as a
   weight makes the backend stage a host buffer per MoE layer.
3. **Slot size is wrong, and the copy overruns it.** `blob_bytes` is `ggml_nbytes(ffn_gate_exps) / n_expert` -
   one projection - but `copy_fn` then memcpy's gate **and** up **and** down into that slot: a 2.9x overrun of
   every slot. On the AtomicChat Q5_K_M file a slot is 0.53 MiB while the copy writes 1.50 MiB.
4. **One arena cannot serve every MoE projection.** `build_lora_mm_id` routes *all* MoE matmuls - gate_up and
   down, which differ in shape - through one `slot_tensor`, so `cached->nb[2]` is the wrong stride for at
   least one of them.
5. ~~**The kernel reads the wrong residency rows.**~~ **Fixed** (see `test-mul-mat-id-cached`): the table is
   `[n_layers * n_expert]` and both kernels used to index it as `residency_host[i02]` over
   `ne02 == n_expert`, so every layer read layer 0's residency. A node now carries its layer
   (`ggml_mul_mat_id_cached(..., layer, ...)`, `op_params[1]`) and reads that row.
6. **Tier 1 and tier 2 are never reachable.** `ggml_cuda_set_expert_cache_tier1_slot_tensor`,
   `..._tier2_...` and `..._tier1_slots` are defined and never called from anywhere, so `cached1`/`cached2`
   are null in the kernel and tier 1/2 experts silently fall back to the original weights.
7. **`GGML_OP_MUL_MAT_ID_CACHED` is missing its plumbing.** Partly fixed:
   - ~~the CPU path computes in a scalar loop that only handles F32/F16 - for a quantized `src0` it reads
     quantized bytes as floats~~ **Fixed.** The CPU kernel is now the `mul_mat_id` body with the per-expert
     base pointer resolved through the residency table, so it keeps the per-type `vec_dot`, the `src1`
     conversion and the threading. Registered in the CPU work-size switch and the task-count switch (it
     shares `mul_mat_id`'s layout); the tiled path is skipped when the weights come from a slot.
   - still absent from CUDA `supports_op`, which ends in `default: return false`, so no CUDA backend
     claims the op today. (`ggml_op_alloc_size_may_expand` turned out not to need it: that list only
     allows for a backend whose alloc size exceeds `ggml_nbytes`, and this op's output is plain F32.)
   - both kernels now reject a residency entry that names a slot the arena does not have, instead of
     reading past the end of it.
8. **`--kv-resident` corrupts output.** `llama_kv_stream::stream_layer` copies K storage into the V window
   (`k_storage`, marked `// TODO: use actual V storage`), and `get_k`/`get_v` return the whole fixed window
   regardless of `n_kv`, so attention reads stale and wrong cells. It also sizes its layer array from
   `kv->get_size()`, which is the cell count, not the layer count.

Strata's own design is not this shape: `docs/DETAILS.md` puts it at ~700 experts per GB, which matches a
gate+up+down slot of ~1.5 MiB, and its hit path is a grouped kernel over device-resident experts
(`moe_hit_grouped_s2`), not a generic `mul_mat_id` over a host arena. `--expert-cache-per-layer` exists
because a shared pool measured 2.97% hit rate against 21.4% at 8 slots/layer.

## What a real port needs

- Allocate the arena on a device buffer (`ggml_backend_buft_alloc_buffer` on the GPU buffer type), one arena
  **per projection**, each sized `n_slots * ggml_row_size(type, ne0*ne1)`, with residency per (layer, expert)
  resolved against that projection.
- Keep the residency table in cache-owned memory and hand the kernel a per-layer `[n_expert]` view, so layer
  `il` reads layer `il`'s rows.
- Copy through the backend (`ggml_backend_tensor_set`), not `memcpy` on a `t->data` pointer that is null for
  device tensors.
- Register the op properly: `supports_op` on each backend, the alloc-size list, the CPU work-size case, and a
  CPU path that dequantizes - or restrict the cache to devices that implement it and leave others on
  `mul_mat_id`.
- For `--kv-resident`, stream the real V tensor and return a window sized to the attention's actual `n_kv`,
  or leave it off.

## Notes for whoever picks this up

- `common/common.cpp` keeps the original cache-construction blocks behind `if (false)` as a starting point.
- The user's reported `invalid argument: 32` was a separate, simpler bug: `--expert-adapt` was declared with a
  value hint `""` but a `handler_string`, so the parser treated it as taking a value and swallowed the `32`
  meant for `--expert-adapt-interval`. Fixed by declaring it as a flag.
