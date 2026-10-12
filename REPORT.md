# Speculative decoding for Qwen3.8-27B on an M3 Pro: results

All numbers are generation tok/s: greedy decoding, thinking off, 256 new tokens, mean over the 6 prompts in `bench/prompts.jsonl`, on AC power. Each speedup is measured against the no-speculation run on the same engine and the same weights. Round 1 ran 2026-10-09; round 2 (server, matched-quality MLX weights, kernels) ran 2026-10-10 to 10-12.

## Headline

| Engine | Setup | tok/s | vs 6.64 | Quality |
|---|---|---|---|---|
| llama.cpp | what you ran before (ggml-org Q4_K_M) | 6.64 | 1.00x | reference |
| llama.cpp | MTP n2 + ngram-simple, stock build | 11.39 | 1.71x | identical output 6/6 |
| llama.cpp | **patched kernel, DFlash2 n7 + ngram-simple** | **14.13** | **2.13x** | PPL same to 5e-4; 5/6 identical (one near-tie) |
| llama.cpp | patched kernel, MTP n7 (no DFlash file needed) | 13.07 | 1.97x | same |
| MLX (ours) | **Q4_K_M transcode + qmm kernel + MTP k=7** | **18.30** | **2.76x** | same weights as llama.cpp (PPL 6.131 vs 6.122) |
| MLX (ours) | mlx-community 4-bit + qmm kernel + MTP k=7 | 28.31 | 4.26x | measurably worse (PPL 6.275, 90% top-1 agreement) |

- **Per prompt, best MLX config on the transcode** (code_edit / code_gen / summarize / prose / explain / math): 26.2 / 22.8 / 16.6 / 9.6 / 13.3 / 21.3 tok/s.
- **Prose** is the weak spot, because acceptance is low; k=3 is slightly better there.
- **Speedups over each engine's own greedy:**
  - MLX MTP k=7 + kernel: 3.25x.
  - MLX MTP k=3 on MLX's own kernels: 2.71x.
  - llama.cpp patched DFlash: 2.06x.

## What to run

**Most speed at unchanged quality: the MLX server** (OpenAI-compatible, port 8081). Stop llama-server first, since two 27B models don't fit in 36 GB.

```
cd ~/Documents/specdec && scripts/gpu.py .venv/bin/python -m mlx_spec serve \
  --model models/qwen38-27b-q4km-mlx --drafter mtp:k=7:vocab=65536 --qmm 5-16
```

Point the chat app at `http://127.0.0.1:8081/v1`; any API key works.
- **Correctness:** text is identical to the CLI on 6/6 prompts.
- **Speed:** about 20 tok/s as the server reports it.
- **Features:**
  - streaming, with a reasoning/content split;
  - tool calls;
  - prompt caching across turns, so the next turn in a long chat starts in 0.4–2 s instead of 20–50 s.
- **Limits:**
  - one request at a time;
  - temperature > 0 is exact but slow with MTP's greedy drafts.
- Details are in `mlx_spec/README.md` (Server) and `notes/mlx-server.md`.

**Staying on llama.cpp:** use the patched build.

```
GGML_METAL_MV_MULTIROW=1 ~/Documents/specdec/vendor/llama.cpp/build-multirow/bin/llama-server \
  -m <ggml-org Qwen3.8-27B-Q4_K_M.gguf> \
  -md ~/Documents/specdec/models/Qwen3.8-27B-DFlash2-q4_0.gguf \
  --spec-type draft-dflash,ngram-simple --spec-draft-n-max 7 -ngl 999 -fa on -c 8192 --jinja
```

For byte-identical output with a stock build, use the round-1 command (MTP n2, 11.4 tok/s):

```
llama-server -m <Q4_K_M.gguf> -md ~/Documents/specdec/models/mtp-Qwen3.8-27B-q4_0.gguf \
  --spec-type draft-mtp,ngram-simple --spec-draft-n-max 2 -ngl 999 -fa on -c 8192 --jinja
```

Qwen3.8's chat template defaults to thinking at `reasoning_effort: xhigh`. For everyday questions, `reasoning_effort: "low"` or `enable_thinking: false` saves more total time than any of the above.

## What we learned

1. **The verify-cost curve decides everything.** Checking k drafts is one target pass over k+1 tokens, which is only cheap while the matmuls read the weights once.
   - **Stock MLX:** flat for 1–4 tokens, then 1.2x at 5, 1.5x at 6, 1.9x at 8, and ~2.9x from 11 up.
     - For 2–12 rows, `qmv_wide` splits the rows into tiles of at most 5, and every tile re-reads and re-dequantizes all the weights.
     - For 13+ rows, `qmm_t` always computes a full 32-row tile.
   - **Stock llama.cpp Metal:** 149 / 207 / 298 / 416 / 573 ms for 1 / 3 / 4 / 5 / 8 tokens, then 379 at 9.
     - `mul_mv_ext` (4–8 tokens) handles at most 5 tokens per pass and uses tiny byte loads (24–48 GB/s).
     - `mul_mm` (9+) pads to 32 columns.
2. **Both kernels are fixed the same way.** Read and dequantize each weight once into simdgroup 8x8 matrix tiles, then multiply by all the rows.
   - **MLX (`mlx_spec/qmm_kernel.py`, `mx.fast.metal_kernel`, no MLX rebuild):**
     - The full-model verify pass now costs 129–131 ms for 5–8 tokens, against MLX's 145–219.
     - The best draft length moves from 3 to 7.
   - **llama.cpp (`scripts/metal-multirow-mv.patch`):** 5–8 tokens now cost 270–280 ms instead of 416–573, which makes DFlash at 7 drafts worthwhile.
   - Both are tested against the stock kernels (MLX: 23 unit tests; llama.cpp: `test-backend-ops` 234/234, with deliberately corrupted kernels caught). Both look worth upstreaming; see "Upstream" below.
3. **Quant quality matters more than it looked.** mlx-community's 4-bit is smaller and faster, but measurably worse:
   - KL 0.068 vs the GGUF;
   - top-1 agreement 89.9%;
   - PPL 6.275 vs 6.122.

   So we transcoded the GGUF's own weights to MLX (`scripts/gguf_to_mlx.py`):
   - Q4_K → affine 4-bit g32, and Q8_0 → 8-bit g32, both nearly exact.
   - Q6_K → 8-bit.
   - Result: KL 0.0024, 98% top-1, PPL 6.131.

   The cost is size: 20 GB vs 17.7 GB, because MLX stores a bf16 scale and bias for every 32 weights. That's why its greedy speed (5.63) is below llama.cpp's (6.64).
4. **Rolling back the hybrid model.** 48 of the 64 layers are gated-delta layers with ~151 MB of recurrent state that can't be trimmed like a KV cache.
   - Ours keeps the start state plus the per-token recurrence inputs, and on rejection replays only the recurrence for the kept tokens: 3–4 ms, bitwise equal (`mlx_spec/gdn.py`).
   - mlx-lm's own speculative path refuses this model.
   - The server reuses the same machinery to cache prompts across chat turns.
5. **MTP is the best drafter in MLX; with the patched kernel, DFlash wins in llama.cpp.**
   - MTP acceptance matches between the engines (0.89 / 0.82 / 0.73 for 1 / 2 / 3 drafts).
   - Restricting MTP's draft vocabulary to ids below 65536 halves drafting cost, and correctness is unaffected because verification uses the full vocabulary.
   - N-gram drafting barely helps (1.03–1.17x) but is free to add in llama.cpp.
6. **Output equivalence.** In fp32 our MLX loop reproduces greedy exactly (12/12, including MTP k=6–8 with the kernel).
   - In bf16, batched verification sums in a different order, so outputs flip where the top-2 logits are within one bf16 step.
   - Every divergence was diagnosed, and all are near-ties. This affects 2–4 of 6 prompts per config, the same with or without our kernel.
   - The patched llama.cpp flips one prompt (`explain`, after char 572) for the same reason, and no longer shows two flips the stock build had at n≥3.

## Upstream

Both kernels could become upstream contributions. Porting notes are written for you, not as PR text (MLX bans AI-written commit and PR text).

- **MLX** (`notes/mlx-qmm-kernel.md` s.8):
  - Replace `qmv_wide` and the 32-row tile for 5 ≤ M ≤ 16 on gen ≥ 15 GPUs.
  - Still to cover: M1/M2/M4/M5, 3/5/6-bit, the gather (MoE) variants.
- **ggml-metal** (`notes/llama-metal-kernel.md`):
  - The matrix kernel and its dispatch ranges.
  - Drop the scalar variant and env knobs first.
  - Still to cover: more quant types, MUL_MAT_ID, other chips. Note in the PR that rounding to half precision changes at 4–8 tokens.

## Next steps, roughly by value

1. **Have MLX read GGUF's Q4_K/Q8_0/Q6_K layouts directly** with custom kernels (1-row and 5–16-row) instead of transcoding. That keeps the transcode's quality at 17.7 GB instead of 20 GB. The estimate is ~21–22 tok/s with MTP k=7, but that's unmeasured.
2. **Adaptive draft length:** shorter drafts when acceptance drops (it does on prose).
3. **Get the llama.cpp kernel to ≥110 GB/s.** It reaches 70–75 now; ideas are in the notes.
4. **A DFlash drafter in MLX.** It's now worth it, since long drafts are cheap.

## Where things are

| | |
|---|---|
| `mlx_spec/` | MLX loop, drafters, MTP head, `serve`, `qmm_kernel.py`; usage in `mlx_spec/README.md` |
| `notes/mlx-from-scratch.md` | MLX design, rollback, round-1 results |
| `notes/mlx-server.md` | server design, tests, smoke results |
| `notes/mlx-matched-quant.md` | transcode, PPL/KL comparison |
| `notes/mlx-qmm-kernel.md` | MLX kernel root cause, design, final headline numbers (s.9) |
| `notes/llama-ngram-draft.md`, `notes/llama-mtp-dflash.md` | llama.cpp round 1 |
| `notes/llama-metal-kernel.md` | llama.cpp kernel root cause, patch, end-to-end results |
| `results/` | raw rows (`mlx_final_session.jsonl`, `llama-multirow.jsonl`, `quality_ppl.jsonl`, ...) |

## Housekeeping

- **Models:**
  - `models/qwen38-27b-q4km-mlx`: 20 GB, the transcode, used by the server.
  - DFlash2 bf16/q8/q4, MTP q4/q8, a 19 GB single-file Q4_K_M+MTP, and the MLX MTP weights.
  - `models/wikitext/q4km-base.kld` (8.1 GB) was only needed for the KL comparison and can be deleted.
- **Disk:** ~67 GB free.
- **`vendor/llama.cpp`:**
  - `build-multirow/` is the patched build used above.
  - `build-mm/` is the round-1 experiment, superseded.
- **`scripts/gpu.py`** is the machine-wide GPU queue; it waits while your own llama-server or `mlx_spec serve` runs.
- Nothing has been pushed anywhere.
