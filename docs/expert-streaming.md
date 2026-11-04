# Expert streaming: running a Mixture-of-Experts model bigger than RAM

A Mixture-of-Experts model is enormous on disk and small per token. Kimi K2 has a trillion
parameters and touches about 32 billion of them for each token; the Qwen3.5-35B-A3B model below
keeps 8.98 GB of routed experts on disk and uses 8 of 256 per layer. Only a few percent of the
weights matter to any one token, and which ones those are is decided by the router a few
microseconds before they are read.

When the file does not fit in memory, llama.cpp leaves it to the operating system. The kernel
does not know what an expert is: it pages bytes in 4 KiB at a time, with no idea which expert
comes next, and it evicts whatever its generic policy prefers.

Expert streaming makes llama.cpp do it on purpose. The graph names the router output of every
MoE layer `ffn_moe_topk-<il>`; the backend scheduler is asked for that node, which makes it
compute up to the router, wait for the result and hand it over. At that instant the experts this
ubatch needs are known and the mat-muls that read them have not started. So:

- **whole experts are read in parallel** with `readahead(2)` and mapped in one go with
  `MADV_POPULATE_READ`, `--expert-io-threads` of them at a time, instead of being faulted in
  page by page by the mat-mul;
- **a byte budget** (`--expert-budget`) decides how much RAM the routed experts may occupy;
- **a residency policy** decides what stays. Plain LRU, or `admit` (default), which lets a
  missed expert in only when its decayed per-token usage beats the expert it would evict —
  a one-off expert cannot push out a hot one;
- **eviction runs in the background** with `MADV_DONTNEED` + `POSIX_FADV_DONTNEED`; doing it
  synchronously with `MADV_PAGEOUT` costs a large slice of decode time;
- **`mincore()` is asked at every access** whether the expert really was resident, so the hit
  rate reported is the kernel's, not the engine's bookkeeping (`--expert-no-residency` to skip).

Nothing is approximated. The weights are never copied, re-quantized or reordered: a streamed
expert is a window of the model file, read whole and dropped when cold. The output is the same
tokens and the same logits, bit for bit, as a run without streaming — the whole point is that
this is a memory-management change, not a numerical one.

## Usage

```bash
# stream a 35B MoE through 3 GB of RAM for its experts
llama-server -m Qwen3.5-35B-A3B-UD-IQ2_XXS.gguf --expert-budget 3000

# measure your own hot set, then warm-start the next run from it
llama-cli -m MODEL -f story.txt -n 256 --expert-budget 4000 --expert-heat-out heat.txt
llama-cli -m MODEL -f other.txt  -n 256 --expert-budget 4000 --expert-heat-in  heat.txt
```

| Flag | |
|---|---|
| `--expert-budget MIB` | RAM the routed experts may occupy; 0 (default) turns streaming off |
| `--expert-io-threads N` | whole experts read in parallel (default 8) |
| `--expert-policy lru\|admit` | what to keep when the budget is full (default `admit`) |
| `--expert-half-life N` | how many decode steps a used expert stays interesting for (default 256) |
| `--expert-heat-in FILE` / `--expert-heat-out FILE` | preload the hot set / record usage counts |
| `--expert-warm` | do not drop what the page cache already holds before the first token |
| `--expert-no-prefetch` | keep the budget but let the kernel page each expert in (measures the prefetch) |
| `--expert-pageout` | reclaim with `MADV_PAGEOUT` instead of `MADV_DONTNEED` + fadvise |
| `--expert-no-residency` | stop asking the kernel whether an expert really is resident |

Streaming needs the weights to stay where the file has them, so it turns off weight repacking
(`use_extra_bufts`) and, when layers are offloaded to a GPU, pins the routed experts to host
memory — dense layers and the KV cache go to the GPU, experts stay in RAM and are streamed.
That is the same split as `--n-cpu-moe`, and the two are not meant to be combined.

At context teardown the engine logs one line with the budget, the resident set, the hit rate as
counted by the engine and as verified by `mincore()`, evictions, and the bytes it fetched.

## Measured

Qwen3.5-35B-A3B (IQ2_XXS, 9.9 GB file, 40 MoE layers × 256 experts, top-8, expert pool 8.98 GB)
on a Xeon E5-2697A v4, 16 threads, greedy decode, one prompt. Every number below produced
**identical tokens and bit-identical logits at every step**.

| configuration | decode tok/s | hit | disk MiB/token | major faults |
|---|---:|---:|---:|---:|
| everything resident, no streaming | 11.97 | – | 0 | 0 |
| budget 2000 MiB, kernel pages experts in | 1.75 | 32.1 % | 617 | 109 631 |
| budget 2000 MiB, whole experts in parallel | 2.59 | 32.1 % | 371 | 21 760 |
| budget 2000 MiB, `--expert-policy lru` | 2.51 | 31.0 % | 377 | 22 126 |
| budget 4000 MiB | 4.06 | 40.5 % | 325 | 19 066 |
| budget 6000 MiB (hot set fits) | 6.55 | 45.0 % | 301 | 17 647 |
| budget 2000 MiB, warm-started from `--expert-heat-in` | **7.76** | 64.2 % | – | – |

At an identical budget and an identical hit rate, reading experts whole and in parallel is
**+48 % decode and 1.7× prefill** over letting the kernel fault them in, with **5× fewer major
faults and 40 % less data read**; frequency-gated admission beats plain LRU on both hit rate and
evictions; warm-starting from the previous run's heat is worth **3× at the same budget**.

The machine this was measured on has 184 GB of RAM, so the model fits and streaming it is a
choice, not a necessity — the last row of the table is the same model at the same budget with a
head start, and the first row is what it costs to stream something that never needed streaming.
The interesting case, a model that does not fit, is the same code path with more of it missed.

## What it does not do

- It does not make the model smaller. It moves the same bytes better.
- One request at a time benefits most; the router choices are needed before the expert mat-muls,
  which splits the graph at every MoE layer.
- The budget is a budget for routed experts only. Dense weights, shared experts and the KV cache
  are left to the page cache.
- `--expert-budget` on a model whose expert pool is smaller than the budget does nothing useful
  and costs a little: use `--expert-warm` if the file already fits.
