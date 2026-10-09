# Speculative decoding for Qwen3.8-27B on an M3 Pro: results

All numbers are generation tok/s: greedy decoding, thinking off, 256 new tokens, mean over the 6 prompts in `bench/prompts.jsonl`. Each speedup is measured against the no-speculation run on the same engine and the same weights.

## Headline

| Engine | Setup | tok/s | Speedup | Output vs normal generation |
|---|---|---|---|---|
| llama.cpp | what you ran before (ggml-org Q4_K_M) | 6.64 | 1.00x | n/a |
| llama.cpp | **MTP draft, 2 tokens, + ngram-simple** | **11.39** | **1.71x** | 6/6 identical |
| llama.cpp | Qwen3.5-0.8B draft, n-max 16, p-min 0.7 | 10.78 | 1.61x | 5/6 (math: near-tie flip) |
| llama.cpp | DFlash2, 2 tokens, + ngram-simple | 10.88 | 1.64x | 6/6 identical |
| MLX (ours) | greedy (mlx-community 4-bit) | 8.57 | 1.00x | n/a |
| MLX (ours) | Qwen3.5-0.8B draft, k=3 | 19.56 | 2.28x | identical in fp32; bf16 flips at near-ties |
| MLX (ours) | **MTP head, k=3, draft vocab 65536** | **23.02** | **2.69x** | identical in fp32; bf16 flips at near-ties |

Per prompt for the best MLX config (code_edit / code_gen / summarize / prose / explain / math): 27.7 / 24.5 / 22.4 / 16.4 / 21.9 / 25.4 tok/s.

**Caveat on comparing the two engines.** About 1.3x of MLX's lead comes before any speculation, because mlx-community's 4-bit file is smaller (15 GB) than ggml-org's Q4_K_M (19 GB, which keeps attention and SSM weights at Q8_0). The two quants aren't guaranteed to be equal in quality. The fair comparison of the speculative decoding itself is the speedup column: **2.69x vs 1.71x**.

## What to run today

Drop-in, no custom code, output identical to before. The MTP file is the sidecar ggml-org publishes next to the main model; ours in `models/` is byte-identical to it.

```
llama-server -m <ggml-org Qwen3.8-27B-Q4_K_M.gguf> \
  -md ~/Documents/specdec/models/mtp-Qwen3.8-27B-q4_0.gguf \
  --spec-type draft-mtp,ngram-simple --spec-draft-n-max 2 \
  -ngl 999 -fa on -c 8192 --jinja
```

Separately, Qwen3.8's chat template defaults to thinking at `reasoning_effort: xhigh`. For everyday questions, passing `reasoning_effort: "low"` or `enable_thinking: false` cuts total answer time more than any of the above.

The MLX implementation is faster, but it's currently a research CLI (`python -m mlx_spec generate/bench`) with no server. To use it day to day it needs an OpenAI-compatible server wrapper (see next steps).

## What we learned

1. **The verification cost curve decides everything.** Checking k drafted tokens means one target pass over k+1 tokens. That pass is only "free" while the matmul kernels read the weights once.
   - **MLX:** flat for 1–4 tokens, then 1.2x at 5, 1.5x at 6, 1.9x at 8, and about 2.9x from 11 tokens up. `quantized_matmul` on its own shows the same staircase (`results/mlx_qmm_curve.jsonl`).
   - **llama.cpp Metal:** 152 / 161 / 212 / 304 / 423 / 583 ms for 1 / 2 / 3 / 4 / 5 / 8 tokens, then *drops* to 385 ms at 9. Batches of 4–8 tokens use `mul_mv_ext`, which re-reads K-quant weights; only batches over 8 use the mat-mat kernel. A local patch (`scripts/metal-mm-min-batch.patch`) that sends 5–8 to mat-mat lets DFlash at 7 tokens reach 14–16 tok/s on code, but it flips 3 of 6 outputs.
   - So the best draft length is 3 in MLX and 2 in llama.cpp. A small-batch (5–16 row) quantized matmul that reads the weights once is the biggest remaining lever in both engines. It's worth an upstream issue or PR to MLX (`quantized_matmul`) and to ggml-metal.
2. **Rolling back the hybrid model.** 48 of the 64 layers are gated-delta layers carrying about 151 MB of recurrent state, which can't be trimmed like a KV cache. mlx-lm's own speculative path refuses this model ("requires a trimmable prompt cache"). Ours keeps the start state plus the per-token recurrence inputs, and on rejection replays only the recurrence kernel for the kept tokens: 3–4 ms, bitwise equal (`mlx_spec/gdn.py`). Re-running the whole model instead costs 112–260 ms and drops the draft-model result from 19.4 to 14.1 tok/s. llama.cpp keeps per-token state snapshots only for MTP, DFlash, EAGLE3 and dspark, not for `draft-simple` or n-gram, which is why the classic draft model there needs `p-min`.
3. **MTP is the best drafter.** It has the highest per-position acceptance (0.86 / 0.79 / 0.81 vs 0.73 / 0.77 / 0.75 for the 0.8B draft) and is cheap to run. Our MLX acceptance matches llama.cpp's (0.89 / 0.82 / 0.73 vs 0.90 / 0.81 / 0.74 for 1 / 2 / 3 drafts), which is a good sign the head is wired correctly. Restricting the drafter's vocabulary to ids below 65536 halves drafting cost (lm_head was 70% of it) at a small acceptance cost. Correctness is unaffected because verification uses the full vocabulary.
4. **N-gram drafting barely helps here:** 1.03x in llama.cpp and 1.17x in MLX. It only helps when the answer copies the prompt verbatim, and the model paraphrases summaries. It's free when it finds no match, so `ngram-simple` costs nothing to add in llama.cpp.
5. **The small model from an older release works.** Qwen3.5-0.8B's tokenizer is identical apart from 7 audio/TTS tokens that never appear in text chat, and llama.cpp's strict vocab check passes. The 2B is worse (drafting is too slow).
6. **Output equivalence.** Greedy speculative decoding should reproduce normal output exactly. In fp32 our MLX loop does, 12/12 including MTP. In bf16/4-bit, batched verification sums in a different order, so outputs flip where the top-2 logits are within one bf16 step. `diagnose()` checked every divergence. llama.cpp shows the same thing once verify batches reach 4 tokens (the kernel switch); with 2 drafts it matches 6/6.

## Where things are

| | |
|---|---|
| `notes/llama-ngram-draft.md` | llama.cpp: harness, n-gram sweep, 0.8B/2B draft, tokenizer check |
| `notes/llama-mtp-dflash.md` | llama.cpp: MTP/DFlash conversion, sweeps, kernel analysis, patch |
| `notes/mlx-from-scratch.md` | MLX: design, rollback, verify-cost curve, MTP head, results |
| `mlx_spec/README.md` | MLX usage and code map |
| `bench/llama_bench.py` | llama-server harness (use `--paired`) |
| `results/` | raw rows; `.venv/bin/python -m mlx_spec summary` and `bench/llama_results_table.py` print tables |

## Next steps, roughly by value

1. An OpenAI-compatible server around `mlx_spec` (MTP k=3), so the 2.69x setup is usable from your chat app.
2. A cheaper 5–16-row `quantized_matmul` in MLX (and an issue for ggml-metal's `mul_mv_ext` range). This would unlock longer drafts and DFlash.
3. A matched-quality comparison: an MLX mixed quant with attention/SSM at 8-bit, like ggml-org's recipe, so the engines compare on equal footing and you aren't trading quality for speed.
4. Draft vocabulary from real token frequencies instead of id order, and a confidence-gated k for MTP.
5. A DFlash drafter in MLX (about 1000 lines to port). Only worth it after step 2.

## Housekeeping

- Disk: about 32 GB free. `models/` holds about 34 GB (DFlash bf16/q8/q4, MTP q8/q4, a 19 GB single-file Q4_K_M+MTP, MLX MTP weights). The full bf16 `Qwen/Qwen3.8-27B` download (54 GB in the HF cache) is no longer needed now that the MTP weights are extracted.
- `vendor/llama.cpp` has a patched build in `build-mm/`; the source tree was reverted.
- Nothing has been pushed anywhere.
