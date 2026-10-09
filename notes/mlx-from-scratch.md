# Track B: speculative decoding from scratch in MLX

**Summary.** Our own MLX loop with **Qwen3.5-0.8B (4-bit) as the draft model, k = 3**
generates at **19.4 tok/s** mean over the six prompts, **2.27x** our MLX greedy
baseline (8.56 tok/s) and ~2.9x llama.cpp's plain decode (6.5-6.7 tok/s on the
same prompts, Track A). Prompt lookup (n-gram) alone gives 1.17x. Rollback of
the hybrid cache costs ~3.5 ms per partially accepted round (~3% of a verify
pass) with the `replay` strategy. Greedy outputs equal plain greedy
token-for-token in float32; in bf16, 13 of 24 prompt x config runs diverge at
one-ulp logit ties (diagnosed one by one), never with a clear margin.

Code: `mlx_spec/` (README there). mlx-lm 0.32.0 supplies the model definition
(`mlx_lm/models/qwen3_5.py`), weights, tokenizer and cache classes; the decode
loop, drafting, verification, acceptance and rollback are ours. Target:
`mlx-community/Qwen3.8-27B-4bit` (affine 4-bit, group 64, ~15 GB).

## The model, as far as speculative decoding cares

64 layers, `full_attention_interval = 4`: layers 3, 7, ..., 63 (16 of them)
are softmax attention (24 q heads, 4 kv heads, head_dim 256, output gate) with
a `KVCache`; the other 48 are gated delta nets (GDN, linear attention) with an
`ArraysCache(size=2)` holding

* `cache[0]`: causal-conv window, last 3 rows of `in_proj_qkv` output,
  `[1, 3, 10240]` bf16;
* `cache[1]`: recurrent state `S`, `[1, Hv=48, Dv=128, Dk=128]` float32 =
  3.1 MB per layer, **151 MB for the 48 layers**.

The GDN recurrence per token: `S <- g*S; S <- S + k (beta (v - S k))^T; y = S q`.
mlx-lm runs it in a Metal kernel (packed variant for Dk=128) that keeps `S` in
registers for the whole sequence and writes only the final state.

mlx-lm's own speculative path refuses this model: `stream_generate(...,
draft_model=...)` raises `ValueError: Speculative decoding requires a
trimmable prompt cache (got {'ArraysCache'})` (checked, `logs/mlx_mlxlm_compare.log`).
So there is no built-in MLX baseline to compare against for this model.

## Design

```
prefill(prompt) -> last = argmax
loop:
    drafts = drafter.draft(last, k)                 # <= k tokens, lazy mx.array
    out    = target.forward([last] + drafts)         # ONE pass, T = 1 + len(drafts)
    tgt    = argmax(out.logits)                      # eval drafts + tgt together, one sync
    n      = longest prefix with drafts[i] == tgt[i]
    emit drafts[:n] + [tgt[n]]                       # tgt[n] = correction or bonus
    target.commit(cache, n_keep = n + 1)             # keep [last, d1..dn], roll back the rest
    drafter.commit(out, n); last = tgt[n]
```

`SpecModel.forward` re-implements the 10-line layer loop of
`Qwen3_5TextModel.__call__` so it can return the pre-norm residual and the
post-norm hidden state of every position next to the logits
(`ForwardOutput`); MTP/EAGLE/DFlash drafters get them from the verify pass
for free.

### Rollback (`mlx_spec/gdn.py`, `SpecModel.commit`)

Attention layers: `KVCache.trim(T - n_keep)`. GDN layers can't be trimmed:
after the verify pass `S` has absorbed all `T` tokens. Each GDN layer is
wrapped in `RollbackGDN`, which (with our `LinearCache`) runs the same ops as
the stock layer but records what's needed to rewind. Three strategies:

| mode | how | extra work per round |
|---|---|---|
| `replay` (default) | keep `S_0` and the per-token recurrence inputs `q,k,v,g,beta` (~0.1 MB/layer/token); on partial accept rerun **only the recurrence kernel** for the `n_keep` kept tokens from `S_0` | one read + write of `S` (2 x 151 MB) only when something was rejected; queued lazily, runs at the start of the next pass |
| `capture` | our copy of mlx-lm's packed GDN kernel that writes `S` after every token to an extra `[1, T, 48, 128, 128]` output; commit indexes it | `T x 151 MB` of writes on every verify pass, rejected or not; memory `T x 151 MB` |
| `reforward` | snapshot `(conv, S)`, on partial accept restore, trim KV back to the pre-verify length and run the full model over the `n_keep` kept tokens | a whole extra target pass (~140 ms) per partially-accepted round |

The conv window is free in every mode: the layer keeps `[old window, qkv]`
(`3 + T` rows) and the window after `n_keep` tokens is rows
`n_keep : n_keep + 3`.

Exactness: the Metal kernel's per-token arithmetic doesn't depend on `T`,
so the replayed `S_{n_keep}` is bitwise the state the verify pass held in
registers after `n_keep` tokens, and the capture kernel's `states[:, t]` is
bitwise the stock kernel's final state for `t + 1` tokens
(`tests/test_tiny.py::test_capture_kernel_bitwise`). Rollback therefore adds
no numerical difference of its own; the only source of spec-vs-greedy
differences is that the verify pass computes the projections/attention for
`T` rows at once (different matmul/SDPA kernels than `T = 1`).

The draft model (Qwen3.5-0.8B, also a GDN hybrid) is rolled back differently:
it decodes one token per step, so snapshotting its cache after every step is
free (MLX arrays are immutable values; a snapshot is a list of references
plus KV offsets) and commit restores the right one (`drafters.snapshot/restore`).

## Baseline (`decode.greedy_generate`)

Prefill the prompt in one pass, then one token per step with mlx-lm's
pipelining pattern (queue step i+1 with `mx.async_eval` before blocking on
token i). Timing follows llama-server's `timings`: prompt phase = prefill +
first token; `gen_tok_s = n_gen / gen_time` with the first token counted, as
`predicted_per_second` does. The weights are wired
(`mx.set_wired_limit(max_recommended_working_set_size)`, as `mlx_lm.generate`
does).

* vs `mlx_lm.stream_generate` on the same prompts (`logs/mlx_mlxlm_compare.log`,
  measured while a 54 GB download was running, so absolute numbers are low):
  ours 7.26-7.47 tok/s, mlx-lm 6.99-7.53, mean 7.34 vs 7.31. Same tokens on 4/6
  prompts; the other 2 diverge late because mlx-lm prefills `prompt[:-1]` and
  then the last token separately (different batching, bf16 near-tie), while we
  prefill the whole prompt in one pass like llama.cpp.
* Clean run (`results/mlx_greedy.jsonl`): **8.56 tok/s** mean, 8.53-8.59 per
  prompt, prompt processing ~85-90 tok/s. That is 117 ms per token for 15 GB
  of weights, ~130 GB/s effective.
* Thinking is off: the template with `enable_thinking=False` renders
  `<|im_start|>assistant\n<think>\n\n</think>\n\n`, and every output starts
  directly with the answer (checked the decoded texts; no `<think>`).
* Comparison with llama.cpp (Track A, `results/llama-ngram-draft.jsonl`
  baseline rows): 6.3-6.7 tok/s for Q4_K_M. MLX's 4-bit file is smaller
  (15 GB vs 19 GB: ggml-org's Q4_K_M keeps attention/SSM tensors in Q8_0), which
  accounts for most of the gap.

## Verify cost vs k: the curve that explains everything

`python -m mlx_spec verify-cost` (`results/mlx_verify_cost.jsonl`): one target
pass over T = k + 1 tokens + argmax, cache primed with a 305-token prompt and
restored after every sample, median of 7. "verify" includes building the graph
(1.6 ms) and the sync.

| T | k | verify ms | x T=1 | capture verify | replay rollback ms | capture rollback ms | reforward rollback ms |
|---|---|---|---|---|---|---|---|
| 1 | 0 | 119.7 | 1.00 | 118.1 | - | - | - |
| 2 | 1 | 122.1 | 1.02 | 120.6 | 3.38 | 0.55 | 112.5 |
| 3 | 2 | 123.4 | 1.03 | 124.4 | 3.36 | 0.54 | 113.6 |
| 4 | 3 | 124.4 | 1.04 | 127.6 | 3.40 | 0.61 | 114.0 |
| 5 | 4 | 145.4 | 1.21 | 152.2 | 3.48 | 1.78 | 115.9 |
| 6 | 5 | 181.7 | 1.52 | 190.7 | 3.47 | 3.56 | 115.9 |
| 7 | 6 | 221.7 | 1.85 | 230.1 | 3.60 | 0.62 | 117.3 |
| 8 | 7 | 223.2 | 1.86 | 230.8 | 3.58 | 0.55 | 117.3 |
| 9 | 8 | 273.6 | 2.29 | 281.8 | 3.70 | 0.58 | 137.5 |
| 10 | 9 | 275.0 | 2.30 | 285.2 | 3.63 | 0.57 | 137.5 |
| 11 | 10 | 327.0 | 2.73 | 337.4 | 3.75 | 0.61 | 171.5 |
| 12 | 11 | 328.7 | 2.75 | 334.4 | 3.77 | 0.65 | 171.5 |
| 13 | 12 | 344.7 | 2.88 | 355.1 | 3.80 | 0.76 | 209.4 |
| 14-16 | 13-15 | 345 | 2.89 | 355-359 | 3.8-4.0 | 0.8 | 210 |
| 17 | 16 | 347.9 | 2.91 | 367.3 | 4.02 | 2.46 | 258.8 |

(rollback keeps `ceil(T/2)` tokens. The `replay` rows with chunked async
submission were measured while a CPU-heavy test of mine was running and are
left out; the "verify" column is the `replay/sync` variant. Plain stock
mlx-lm single-token step in the same conditions: ~118-120 ms.)

What it means:

* **T <= 4 is (almost) free**: verifying 3 drafts costs 4% more than decoding
  one token. Then the price climbs in steps: +21% at T=5, +52% at 6, 1.85x at
  7-8, 2.3x at 9-10, 2.75x at 11-12, and a plateau of ~2.9x from 13 on. The
  pairwise steps point at the 4-bit `quantized_matmul` dispatch: for few rows
  the matrix-vector kernel streams the weights once and the extra rows ride
  along; beyond 4 rows the cost grows with the row count until the tiled
  matmul takes over (`bench/mlx_qmm_curve.py` times `quantized_matmul` alone
  to check this; see below).
* So on this machine the useful draft length is **k = 3** (T = 4); k = 4 only
  pays off when acceptance is very high. llama.cpp's built-in defaults (k up
  to 16) are tuned for GPUs where the curve stays flat much longer.
* This is also why n-gram drafting is weak here: its value is in long free
  drafts, and long drafts aren't cheap on this GPU.

### Rollback cost

* `replay`: 3.4-4.0 ms per round with a rejection (read + write of the 151 MB
  of recurrent state plus 48 kernel launches), ~3% of a verify pass, and only
  paid on partial acceptance. Default.
* `capture`: rollback 0.5 ms, but the verify pass writes `T x 151 MB` more
  (+1 to +3 ms at T=2-4, +8 ms at T=8, +14 ms at T=16) on every round. At the
  useful T <= 4 it's a wash with replay; it gets worse with k and needs
  `T x 151 MB` of scratch memory (it OOM'd on a 305-token prefill before
  capture was limited to passes of <= 32 tokens).
* `reforward`: 112-260 ms (a whole extra target pass), i.e. a rejected round
  costs double. End-to-end with n-gram k=4 it is 8.47 tok/s vs 9.83 (replay)
  and 9.72 (capture); it is there as the reference implementation.

### Loop overhead

`spec-none` (the speculative loop with an empty drafter: one token per round,
blocking on every round, no `async_eval` pipelining) runs at 8.36 tok/s vs
8.56 for the pipelined greedy loop: 2.3% overhead. Graph construction is only
~1.6 ms per pass. I tried submitting the verify graph in chunks
(`mx.async_eval` every 16 layers, so the GPU starts while Python builds the
rest): 8.37 vs 8.36 tok/s, no difference, so it was removed again. (In
`results/mlx_verify_cost.jsonl`, `replay/sync` = replay without chunking,
`replay` = with chunking, measured under CPU contention; `capture` and
`reforward` rows used chunking.)

## Correctness

* `tests/test_tiny.py` (seconds, GPU): tiny random-weight `qwen3_5` model in
  float32 with real GDN + attention layers. Checks (a) the capture kernel's
  per-token states are bitwise equal to the stock kernel's final state for
  every prefix; (b) after a T-token pass and `commit(n_keep)` for every
  `n_keep` in 0..T, the conv window, recurrent state and KV offset match a
  cache that only ever saw `n_keep` tokens, in all three rollback modes, and
  the next pass gives the same logits; (c) speculative == greedy for every
  rollback mode with an empty drafter, n-gram, and a "noisy oracle" drafter
  that proposes the true continuation with 30% corruption (exercises every
  acceptance length), k = 1, 3, 8; (d) draft-model drafter (incl. adaptive k
  and the hybrid), with a draft identical to the target accepting 100%;
  (e) EOS / max-token edge cases. Mutation check: making `replay` skip the
  rollback fails (b) and (c).
* `tests/test_sampling.py` (CPU, 2 min): exact speculative sampling. Empirical
  marginals of tokens 2 and 3 over 3000 samples vs the exact marginals by
  enumeration on a 16-token-vocab model: TV < 0.06 for n-gram (one-hot
  proposal), a sampled draft model and a greedy draft model; a negative control
  that accepts every draft fails the same check.
* `tests/test_equivalence_27b.py` (real target, bf16, 256 tokens, 6 prompts x
  {n-gram k=4 replay / capture / reforward, draft k=4}): 11 of 24 runs are
  identical to greedy; the other 13 diverge, every time at a position where
  the two best logits are tied or one bf16 ulp apart (0.125 at |logit| ~ 20):
  `diagnose()` recomputes the logits one token at a time and with a batched
  pass; e.g. code_gen at token 147: single-step margin 0.125, batched margin
  0.0 (exact tie, argmax takes the lower id). replay and capture always
  diverge at the same token (they are bitwise identical by construction);
  reforward sometimes differs (it recomputes the kept tokens in another batch
  shape). The divergence points are the same for n-gram and draft-model runs
  (147, 147, 42, 34/8): wherever a T>1 pass covers that position, the tie
  breaks the other way. **Verdict: numerical, not a bug.** The test now
  tolerates mismatches that `diagnose` classifies as near-ties and fails on any
  other.
* `tests/test_fp32_27b.py`: same target with every bf16 parameter (incl.
  quantization scales/biases) upcast to float32, so the whole forward runs in
  fp32. On the four prompts that diverged in bf16, 160 tokens, n-gram replay,
  n-gram capture and draft k=4: **12/12 identical to greedy**, including past
  every bf16 divergence point. Rollback and acceptance are exact; the bf16
  differences come only from batched-vs-single-row kernels.
* Benchmark rows record `matches_baseline` and `diverge_at_token` against our
  greedy tokens (`results/mlx_baseline_tokens/`).

## Drafters

Interface (`mlx_spec/drafters.py` docstring, `mlx_spec/README.md`):
`start(prompt, prefill)`, `draft(last, k) -> mx.array`, `commit(out, n)`,
optional `draft_probs()` for sampling. `out` is the verify pass's
`ForwardOutput` with logits and pre/post-norm hidden states for every position.

* **N-gram** (`NgramDrafter`): most recent earlier occurrence of the last n
  tokens (n from `n_max` down to `n_min`) in prompt + output; propose what
  followed. Free on the CPU (~0.1 ms per round).
* **Draft model** (`DraftModelDrafter`): `mlx-community/Qwen3.5-0.8B-MLX-4bit`
  (and `Qwen3.5-2B-MLX-8bit`). Tokenizer check (mine, and Track A's in
  `notes/llama-ngram-draft.md`): same 248044-entry vocab and merges, same
  33 added tokens in the mlx-community conversions, vocab_size 248320; the
  only difference is the pre-tokenizer regex (`\p{M}` handling), which doesn't
  matter because the draft model is fed the target's token ids. Qwen3.5-0.8B
  is itself a GDN hybrid; it is rolled back with free per-step snapshots.
  ~6 ms per drafted token.
* **Hybrid** (`HybridDrafter`): n-gram when it finds a match of >= 3 tokens
  with >= 2 tokens to propose, else the draft model (which is told about the
  tokens it skipped).
* **Adaptive k** (`DraftModelDrafter(k_min=...)`): k += 1 after a fully
  accepted round, -1 after a fully rejected one.
* **MTP** (`MTPDrafter`): plan only, in the class docstring. The 4-bit
  mlx-community checkpoint has no `mtp.*` tensors (its index has 0 of them, and
  mlx-lm's `sanitize` drops them anyway); `Qwen/Qwen3.8-27B` (bf16, now in the
  HF cache) has 15: `mtp.fc`, `mtp.pre_fc_norm_{embedding,hidden}`,
  `mtp.layers.0.*` (one full-attention layer + MLP), `mtp.norm`. The verify
  pass already returns the target hidden states it needs.
