# specdec

Making Qwen3.8-27B generate faster on an M3 Pro (36 GB, ~150 GB/s) without changing its output, using speculative decoding.

Baseline: llama.cpp, ggml-org Q4_K_M, full GPU offload, flash attention. 6.75 tok/s generation, 98 tok/s prompt (llama-bench tg64 / pp512). The memory-bandwidth ceiling is about 8 tok/s.

## Tracks

**A. llama.cpp (practical).** Use llama-server's built-in speculative decoding and find the fastest setup.
1. n-gram drafting (`--spec-type ngram-*`), no extra model.
2. MTP, the model's own multi-token-prediction layer. ggml-org's GGUF drops it, so we convert our own from `Qwen/Qwen3.8-27B`.
3. A small model from an older Qwen release as the draft, if its tokenizer matches exactly.
4. DFlash, z-lab's block-diffusion drafter (`z-lab/Qwen3.8-27B-DFlash2`).

**B. From scratch (MLX).** Our own speculative decoding loop on top of mlx-lm's Qwen3.8 model code, with the same drafters. Qwen3.8 is a hybrid model (linear-attention layers with recurrent state, plus full-attention layers), so rejected draft tokens can't just be trimmed from a KV cache; the recurrent state has to be rolled back too.

## Rules for every experiment

- Anything that loads a model runs through `scripts/gpu.py <cmd>`. It holds a machine-wide lock so only one model is in memory and timings aren't disturbed.
- Prompts: `bench/prompts.jsonl`. Greedy (temperature 0), thinking off (`enable_thinking: false`), 256 new tokens unless noted.
- Results go in `results/<experiment>.jsonl`, one row per (method, config, prompt) with at least:
  `track, method, config, prompt_id, n_gen, gen_tok_s, prompt_tok_s, draft_n, draft_accepted, accept_rate, output_sha256, matches_baseline`.
- Notes, findings and dead ends go in `notes/<experiment>.md`.
- Spec decoding with greedy decoding should reproduce the baseline output. llama.cpp verifies in batches, so tiny float differences can occasionally flip a token; record the mismatch rather than hiding it.

## Layout

- `scripts/`: download, GPU lock, conversion helpers
- `bench/`: prompts and benchmark harnesses
- `mlx_spec/`: track B implementation
- `results/`, `notes/`, `logs/`
- `REPORT.md`: the morning summary
