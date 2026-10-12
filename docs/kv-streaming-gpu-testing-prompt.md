# KV Streaming GPU Testing Prompt

This prompt is for an AI or human with GPU access to test the KV streaming feature.

## Build

```bash
cd "/home/aphom/Documents/Unsloth Studio/Projects/llamacpp-3x-31eeaae7/sandbox/llama.cpp"
rm -rf build-cuda
mkdir build-cuda
cd build-cuda
cmake .. -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DBUILD_SHARED_LIBS=ON
make -j$(nproc)
```

## Test 1: Basic KV Streaming (Single GPU)

Run inference with KV streaming enabled and compare to baseline:

```bash
# Baseline (no KV streaming)
CUDA_VISIBLE_DEVICES=2 ./build-cuda/bin/llama-cli \
    -m ../model-q4kxl/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
    -m ../model-q4kxl/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf \
    -m ../model-q4kxl/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf \
    -m ../model-q4kxl/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf \
    -ngl 99 -lzm -v -n 128 --temp 0 --seed 42 \
    -p "Explain quantum computing in one sentence."

# With KV streaming (2048 resident entries)
CUDA_VISIBLE_DEVICES=2 ./build-cuda/bin/llama-cli \
    -m ../model-q4kxl/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
    -m ../model-q4kxl/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf \
    -m ../model-q4kxl/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf \
    -m ../model-q4kxl/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf \
    -ngl 99 -lzm -v -n 128 --temp 0 --seed 42 --kv-resident 2048 \
    -p "Explain quantum computing in one sentence."
```

**Expected:** Both runs should produce identical output (same seed, temp 0). KV streaming run should use less VRAM for KV cache.

## Test 2: VRAM Savings Measurement

Monitor VRAM usage with and without KV streaming:

```bash
# In one terminal, monitor VRAM
watch -n 0.5 nvidia-smi

# In another terminal, run with KV streaming
CUDA_VISIBLE_DEVICES=2 ./build-cuda/bin/llama-cli \
    -m ../model-q4kxl/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
    -m ../model-q4kxl/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf \
    -m ../model-q4kxl/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf \
    -m ../model-q4kxl/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf \
    -ngl 99 -lzm -v -n 128 --temp 0 --seed 42 --kv-resident 1024 \
    -p "Write a haiku about artificial intelligence."
```

**Expected:** With `--kv-resident 1024`, the KV cache in VRAM should be limited to ~1024 entries worth of memory, regardless of context length.

## Test 3: Long Context with KV Streaming

Test with a longer context to verify VRAM savings scale:

```bash
CUDA_VISIBLE_DEVICES=2 ./build-cuda/bin/llama-cli \
    -m ../model-q4kxl/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
    -m ../model-q4kxl/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf \
    -m ../model-q4kxl/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf \
    -m ../model-q4kxl/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf \
    -ngl 99 -lzm -v -n 256 --temp 0 --seed 42 --kv-resident 4096 \
    -c 16384 \
    -p "Summarize the following text in three bullet points: [paste long text here]"
```

**Expected:** Should run without OOM even with large context, as KV cache is streamed from CPU memory.

## Test 4: KV Streaming with Expert Cache

Test KV streaming combined with expert cache:

```bash
CUDA_VISIBLE_DEVICES=2 ./build-cuda/bin/llama-cli \
    -m ../model-q4kxl/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
    -m ../model-q4kxl/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf \
    -m ../model-q4kxl/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf \
    -m ../model-q4kxl/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf \
    -ngl 99 -lzm -v -n 128 --temp 0 --seed 42 \
    --kv-resident 2048 --expert-cache 2000 \
    -p "What are the key differences between TCP and UDP?"
```

**Expected:** Both features should work together. VRAM should be shared between KV streaming window and expert cache.

## Test 5: Losslessness Check

Verify that KV streaming produces byte-identical output to baseline (for short contexts where all KV fits in the window):

```bash
# Baseline
CUDA_VISIBLE_DEVICES=2 ./build-cuda/bin/llama-cli \
    -m ../model-q4kxl/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
    -m ../model-q4kxl/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf \
    -m ../model-q4kxl/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf \
    -m ../model-q4kxl/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf \
    -ngl 99 -lzm -v -n 64 --temp 0 --seed 123 \
    -p "Hello, world!" > /tmp/baseline_output.txt 2>&1

# With KV streaming (large window to include all KV)
CUDA_VISIBLE_DEVICES=2 ./build-cuda/bin/llama-cli \
    -m ../model-q4kxl/Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf \
    -m ../model-q4kxl/Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf \
    -m ../model-q4kxl/Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf \
    -m ../model-q4kxl/Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf \
    -ngl 99 -lzm -v -n 64 --temp 0 --seed 123 --kv-resident 8192 \
    -p "Hello, world!" > /tmp/kvstream_output.txt 2>&1

# Compare
diff /tmp/baseline_output.txt /tmp/kvstream_output.txt
```

**Expected:** No differences (byte-identical output).

## Debugging Tips

- Use `CUDA_LAUNCH_BLOCKING=1` to catch async CUDA errors
- Check logs for "kv streaming: initialized with N resident entries on GPU"
- If OOM, reduce `--kv-resident` value
- Monitor VRAM with `nvidia-smi` during inference
- If output differs from baseline, check that the KV window is large enough to include all needed entries
