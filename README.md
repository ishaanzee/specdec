# specdec

Speculative decoding for Qwen3.8-27B on an M3 Pro (36 GB), without changing what the model says.

Plain decoding of this model on this machine is stuck near 7 tok/s: every token reads all 17.7 GB of weights once, and memory bandwidth (about 150 GB/s) caps that around 8. Speculative decoding gets past the cap by checking several guessed tokens in one pass. This repo measures how far that goes, and fixes the two GPU kernels that were holding it back.

## Results

Generation speed in tok/s. Greedy decoding, thinking off, 256 new tokens, mean over the six prompts in `bench/prompts.jsonl`.

| Engine | Setup | tok/s | Speedup | Output |
|---|---|---|---|---|
| llama.cpp | Q4_K_M, no speculation | 6.64 | 1.00x | reference |
| llama.cpp | stock build, MTP (2 drafts) + n-gram | 11.39 | 1.71x | identical on 6/6 prompts |
| llama.cpp | patched Metal kernel, DFlash2 (7 drafts) + n-gram | 14.13 | 2.13x | 5/6 identical, one near-tie flip |
| MLX | same Q4_K_M weights, new kernel, MTP (7 drafts) | 18.30 | 2.76x | perplexity 6.131 vs 6.122 |
| MLX | mlx-community 4-bit, new kernel, MTP (7 drafts) | 28.31 | 4.26x | worse model: perplexity 6.275 |

The 18.3 row is the one that counts. The 28.3 row is faster only because the weights are a lower-quality quantization (it picks the same top token as the GGUF 90% of the time), so it isn't a fair comparison.

It depends a lot on the prompt. For the 18.3 configuration: 26.2 tok/s on a code edit, 22.8 on code generation, 21.3 on math, 16.6 on summarizing, 13.3 on explanation, 9.6 on prose. Prose is hard to guess, so few drafts are accepted.

## Why the stock kernels were the problem

Checking k drafted tokens costs one model pass over k+1 tokens. That is only a win if the pass costs about the same as a pass over one token, which requires the quantized matmuls to read the weights once no matter how many rows they multiply. Neither engine did that for the batch sizes speculative decoding uses.

In MLX, a pass over 1 to 4 tokens costs the same, then it climbs: 1.2x at 5 tokens, 1.5x at 6, 1.9x at 8, and about 2.9x from 11 up. For 2 to 12 rows the kernel splits the rows into tiles of at most 5, and each tile reads and dequantizes all the weights again. From 13 rows it always computes a full 32-row tile.

llama.cpp's Metal backend has the same shape: 149 / 207 / 298 / 416 / 573 ms for 1 / 3 / 4 / 5 / 8 tokens. Its 4-to-8-token kernel handles at most 5 tokens per pass, and the 9+ kernel pads to 32 columns.

Both get the same fix. Read and dequantize each weight once into 8x8 matrix tiles, then multiply by every row.

- **MLX** (`mlx_spec/qmm_kernel.py`): a custom Metal kernel loaded at runtime, so MLX doesn't need rebuilding. A full-model pass over 5 to 8 tokens now takes 129–131 ms instead of 145–219, and the best draft length moves from 3 to 7.
- **llama.cpp** (`scripts/metal-multirow-mv.patch`, against commit `b29c606e2`): 5 to 8 tokens now take 270–280 ms instead of 416–573.

Both are checked against the stock kernels: unit tests in `tests/test_qmm_kernel.py` for MLX, and llama.cpp's own `test-backend-ops` for the patch.

## Keeping the model the same

The usual MLX 4-bit conversion of this model is smaller and faster than the GGUF, and measurably worse: KL divergence 0.068 against the GGUF, 89.9% top-token agreement, perplexity 6.275 against 6.122.

`scripts/gguf_to_mlx.py` converts the GGUF's own weights into MLX's layout instead. Q4_K and Q8_0 blocks map over almost exactly; Q6_K goes to 8-bit. The result has KL 0.0024, 98% top-token agreement and perplexity 6.131. It costs disk and memory (20 GB instead of 17.7 GB, because MLX stores a scale and bias for every 32 weights), which is why plain decoding in MLX is slower than llama.cpp (5.63 vs 6.64 tok/s) before speculation.

## Rolling back a model that isn't a plain transformer

48 of this model's 64 layers are gated-delta layers that carry about 151 MB of recurrent state. When drafted tokens are rejected, a normal transformer just trims its KV cache. Recurrent state can't be trimmed, and mlx-lm's built-in speculative path refuses this model for that reason.

`mlx_spec/gdn.py` keeps the state from the start of the pass plus each token's recurrence inputs, and on rejection replays only the recurrence for the tokens that were kept. That takes 3–4 ms and is bitwise equal to running the model again. The server uses the same mechanism to cache prompts between chat turns.

## How exact is "the same output"

In fp32 the MLX loop reproduces greedy decoding exactly: 12 of 12 runs, including 6 to 8 drafts with the new kernel.

In bf16 it doesn't always. Verifying a batch sums in a different order than decoding one token at a time, and when the top two logits are within one bf16 step, the winner can flip. That happens on 2 to 4 of the six prompts per configuration, with or without the new kernel. Every flip was traced to a near-tie; none is a logic bug. The patched llama.cpp flips one prompt for the same reason.

## Drafters

- **MTP**, the model's own multi-token-prediction head, is the best drafter in MLX. Acceptance is 0.89 / 0.82 / 0.73 for 1 / 2 / 3 drafts. Limiting its output to the first 65,536 token ids halves its cost and can't affect correctness, since verification still uses the full vocabulary.
- **DFlash2** (`z-lab/Qwen3.8-27B-DFlash2`) wins in llama.cpp once the patched kernel makes 7 drafts affordable.
- **N-gram** drafting helps little on its own (1.03–1.17x) but costs nothing to add in llama.cpp.

## Running it

Needs an Apple Silicon Mac with 36 GB or more, plus `mlx` and `mlx-lm`. Versions aren't pinned, and the numbers above are from an M3 Pro only. `gguf_to_mlx.py` looks for one specific GGUF snapshot in the Hugging Face cache; pass `--gguf` if yours is elsewhere.

```sh
python -m venv .venv && .venv/bin/pip install mlx mlx-lm "huggingface_hub[cli]"
scripts/download_all.sh                                     # drafter and base models, tens of GB
.venv/bin/hf download ggml-org/Qwen3.8-27B-GGUF Qwen3.8-27B-Q4_K_M.gguf
.venv/bin/python scripts/gguf_to_mlx.py --check             # GGUF -> models/qwen38-27b-q4km-mlx
.venv/bin/python scripts/extract_mtp.py                     # MTP head -> models/mtp-mlx

# OpenAI-compatible server on port 8081
scripts/gpu.py .venv/bin/python -m mlx_spec serve \
  --model models/qwen38-27b-q4km-mlx --drafter mtp:k=7:vocab=65536 --qmm 5-16
```

`scripts/gpu.py` is a machine-wide lock, so two 27B models never load at once and timings don't disturb each other. More commands, including the benchmarks and tests, are in `mlx_spec/README.md`.

For llama.cpp, apply `scripts/metal-multirow-mv.patch`, build, and run `llama-server` with `GGML_METAL_MV_MULTIROW=1`. The exact flags are in `notes/llama-metal-kernel.md`.

The server streams, handles tool calls, and caches prompts across turns. It serves one request at a time.

## What isn't done

- The kernels are tuned and tested on one chip (M3 Pro) and on 4-bit and 8-bit weights. Other chips and bit widths are untested.
- The llama.cpp kernel reaches 70–75 GB/s of the machine's ~150.
- Draft length is fixed. Shortening it when acceptance drops would help prose.
- Reading the GGUF layouts directly in MLX, instead of transcoding, should keep the quality at 17.7 GB. Unmeasured.

## Layout

| Path | What |
|---|---|
| `mlx_spec/` | the MLX loop, drafters, rollback, kernel and server |
| `scripts/` | model conversion, the llama.cpp patch, the GPU lock |
| `bench/` | prompts and benchmark harnesses |
| `results/` | raw rows for every run, one JSON line per (method, config, prompt) |
| `notes/` | design notes and dead ends, one file per experiment |
| `tests/` | kernel, rollback, sampling and server tests |
| `REPORT.md` | the full write-up of both rounds |
