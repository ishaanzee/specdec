# Track B: speculative decoding from scratch in MLX

**Summary.** Best config: **Qwen3.8's own MTP layer as the drafter, k = 3,
drafting from the 65,536 lowest token ids**: **23.0 tok/s** mean over the six
prompts, **2.69x** our MLX greedy baseline (8.57 tok/s), 18% faster than the
Qwen3.5-0.8B draft model (19.6 tok/s, 2.28x) and ~3.4x llama.cpp's plain decode
(Track A, 6.65 tok/s); with the full vocabulary MTP k = 3 gives 21.4 (2.50x).
Prompt lookup (n-gram) alone gives 1.17x. Rollback of
the hybrid cache costs ~3.5 ms per partially accepted round (~3% of a verify
pass) with the `replay` strategy. Greedy outputs equal plain greedy
token-for-token in float32 (all drafters incl. MTP); in bf16 4 of 6 prompts
diverge at the same one-ulp logit ties for every drafter (diagnosed one by
one), never with a clear margin. MTP details: section "MTP drafter" below.

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
  matmul takes over. Confirmed below by timing `quantized_matmul` alone.
* So on this machine the useful draft length is **k = 3** (T = 4); k = 4 only
  pays off when acceptance is very high. llama.cpp's built-in defaults (k up
  to 16) are tuned for GPUs where the curve stays flat much longer.
* This is also why n-gram drafting is weak here: its value is in long free
  drafts, and long drafts aren't cheap on this GPU.

### The staircase is `quantized_matmul`

`bench/mlx_qmm_curve.py` (`results/mlx_qmm_curve.jsonl`) times
`mx.quantized_matmul` alone (4-bit, group 64, bf16 activations, ~1.5 GB of
weights per call) for M = 1..32 rows:

| M | MLP up 5120->17408 | x M=1 | eff. GB/s | lm_head 5120->248320 | x M=1 |
|---|---|---|---|---|---|
| 1 | 12.9 ms | 1.00 | 114 | 11.6 ms | 1.00 |
| 4 | 13.1 | 1.02 | 112 | 11.3 | 0.98 |
| 5 | 14.7 | 1.15 | 100 | 13.6 | 1.18 |
| 6 | 19.6 | 1.52 | 75 | | |
| 7-8 | 24.2-24.5 | 1.9 | 60 | 21.3 (M=8) | 1.84 |
| 9-10 | 31.0 | 2.4 | 47 | | |
| 11-32 | 36.3-36.9 | 2.84 | 40 | 31.4-31.6 (16, 32) | 2.73 |

Same shape as the full-model curve (1.00 / 1.21 / 1.52 / 1.86 / 2.3 / 2.75-2.9),
so the matmuls explain it entirely; attention, the GDN kernel and the
rollback are noise. Up to 4 rows the kernel streams the weights once at
~113 GB/s. From 11 to 32 rows it takes the same time whatever M is, i.e. it
is doing a full 32-row tile of work, at 40 GB/s effective. In between the
cost grows in 2-row steps. For a bandwidth-bound shape like this a kernel
that keeps the weights-once property up to 8-16 rows (or a qmm with smaller
row tiles for M = 5..16) would make k = 8-15 verification cost about what
k = 3 does now. That is the single biggest lever for speculative decoding
with MLX on this class of machine, and worth an MLX issue/PR.

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
* **MTP** (`mtp.MTPDrafter`): Qwen3.8's own MTP layer, see "MTP drafter" below.

## Results

All runs: greedy, thinking off, 256 new tokens (stop at EOS), the six prompts
of `bench/prompts.jsonl`, `HF_HUB_OFFLINE`, under the GPU lock, repeat 1 plus
one profiled run per prompt (greedy, ngram k=3 and draft k=3: median of 2,
re-run in a later session; the two repeats agree within 1% and with the
earlier single runs within 1-4%) (the profiled run adds a sync per phase and gives
the time split; its tok/s is within a few % of the timed run). Load average
(other jobs' CPU work) is recorded per row (`loadavg`, 2-4). Rows:
`results/mlx_{greedy,spec-none,ngram,draft,hybrid}.jsonl`; regenerate this
table with `.venv/bin/python -m mlx_spec summary`.

| method | config | code_edit | code_gen | summarize | prose | explain | math | mean tok/s | speedup | accept | tok/round |
|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy | (baseline) | 8.55 | 8.56 | 8.54 | 8.57 | 8.58 | 8.58 | **8.56** | 1.00x | - | - |
| spec-none | k=0 (loop overhead) | 8.49 | 8.34 | 8.48 | 8.29 | 8.35 | 8.21 | **8.36** | 0.98x | - | 1.00 |
| ngram | k=2, n=2-4 | 11.87 | 8.78 | 11.04 | 8.48 | 9.07 | 10.04 | **9.88** | 1.15x | 0.34 | 1.19 |
| ngram | k=3, n=2-4 | 12.14 | 8.63 | 11.63 | 8.49 | 8.90 | 10.05 | **9.97** | 1.16x | 0.27 | 1.22 |
| ngram | k=4, n=2-4 | 11.90 | 8.42 | 11.38 | 8.51 | 8.90 | 9.89 | **9.83** | 1.15x | 0.24 | 1.24 |
| ngram | k=4, n=1-4 | 13.21 | 8.53 | 11.44 | 8.18 | 8.74 | 10.33 | **10.07** | 1.18x | 0.14 | 1.35 |
| ngram | k=5, n=2-4 | 11.10 | 8.05 | 10.77 | 8.30 | 8.28 | 9.32 | **9.30** | 1.09x | 0.20 | 1.24 |
| ngram | k=8, n=2-4 | 10.10 | 7.67 | 9.87 | 8.04 | 8.01 | 8.14 | **8.64** | 1.01x | 0.15 | 1.26 |
| draft 0.8B-4bit | k=2 | 20.61 | 18.57 | 17.86 | 13.08 | 16.50 | 19.16 | **17.63** | 2.06x | 0.69 | 2.39 |
| draft 0.8B-4bit | **k=3** | 24.27 | 21.27 | 18.98 | 14.05 | 17.39 | 21.38 | **19.56** | **2.28x** | 0.60 | 2.80 |
| draft 0.8B-4bit | k=4 | 25.56 | 21.35 | 18.98 | 12.11 | 16.45 | 20.09 | **19.09** | 2.23x | 0.55 | 3.20 |
| draft 0.8B-4bit | k=2..4 adaptive | 24.89 | 20.72 | 18.77 | 13.25 | 16.98 | 20.15 | **19.13** | 2.23x | 0.58 | 3.02 |
| draft 2B-8bit | k=3 | 20.00 | 17.41 | 17.00 | 11.67 | 15.30 | 18.95 | **16.72** | 1.95x | 0.68 | 3.03 |
| hybrid (ngram n3-4 + 0.8B) | k=3 | 23.91 | 20.92 | 19.00 | 13.96 | 17.65 | 19.94 | **19.23** | 2.25x | 0.57 | 2.71 |
| hybrid | k=2..4 adaptive | 23.80 | 19.92 | 18.40 | 13.32 | 16.77 | 19.58 | **18.63** | 2.18x | 0.55 | 2.92 |

Rollback strategy, end to end (same drafts, so same acceptance):

| drafter | replay | capture | reforward |
|---|---|---|---|
| ngram k=4, n=2-4 | 9.83 | 9.72 | 8.47 |
| draft 0.8B k=3 | 19.42 (19.56 re-run) | 18.99 | 14.07 |

Time split, draft 0.8B k=3 (profiled runs): per round ~123 ms verify (T=4),
~19 ms drafting (3 x 6.3 ms), 2-4 ms rollback on rejecting rounds; verify is
~85% of the time. E.g. code_gen: 84 rounds, acceptance histogram
[17, 9, 12, 46] (rounds with 0/1/2/3 drafts accepted), draft 1.6 s,
verify 10.3 s, rollback 0.2 s of 12.1 s.

Reading it:

* **The draft model is the winner**: Qwen3.5-0.8B agrees with Qwen3.8-27B on
  55-88% of drafted tokens (prose lowest at 34-39%), and a 4-token verify is
  only 4% dearer than a decode step, so a round yields 2.8 tokens for ~1.2
  steps' worth of time. 2B-8bit accepts more (0.68 vs 0.60) but drafts at
  ~15 ms/token, a net loss.
* k = 3 is the sweet spot because of the verify curve (T = 4 is the last
  cheap width). k = 4 wins only on code_edit (88% acceptance); adaptive k
  doesn't beat fixed k = 3.
* N-gram alone: +15-18%, nearly all from code_edit / summarize / math, where
  the answer copies from the prompt; ~0 on prose / explain / code_gen. Short
  drafts (k = 2-3) beat long ones; k = 8 loses everything it gains to the
  wider verify. Same story as Track A's llama.cpp n-gram results.
* vs Track A (llama.cpp, same draft model, `notes/llama-ngram-draft.md`):
  their best is 10.78 tok/s (1.61x over 6.68). Ours is 19.6 tok/s (2.28x over
  8.56). Two factors: MLX's 4-bit target decodes 28% faster (smaller file),
  and our verify of T = 4 costs 1.04x a decode step with k = 3, where
  llama.cpp drafts up to 16 tokens (`n16 p0.7`) and has a different verify
  curve. Per prompt our lead is largest on prose/explain (14.1/17.4 vs
  7.4/7.2 tok/s), where llama.cpp's speculation barely helps.
* The hybrid (free n-gram drafts when the context has a long match) doesn't
  add anything over the draft model: on code_edit the 0.8B already predicts
  the copied spans, and drafting costs only ~13% of a round.
* Prompt processing is the same for all methods (35-305 token prompts,
  52-93 tok/s depending on length); the draft model's own prefill costs
  ~2% of it.


## MTP drafter (Qwen3.8's own multi-token-prediction layer)

Code: `mlx_spec/mtp.py` (`MTPHead`, `load_mtp_head`, `MTPDrafter`),
`scripts/extract_mtp.py`, `tests/test_mtp_tiny.py`, `bench/mlx_mtp_explore.py`,
`bench/mlx_mtp_session.py`. Rows: `results/mlx_mtp.jsonl` (diagnoses:
`results/mlx_mtp_diagnose.jsonl`). CLI: `--configs mtp:k=3:vocab=65536`,
options `bits=4|8|16`, `hidden=normed|hidden`, `vocab=N|ids.json`, and
`mtp-hybrid:k=3[:n=3-4]` (n-gram when it finds a >= 3-token match, else MTP).

### Weights

`mlx-community/Qwen3.8-27B-4bit` has no `mtp.*` tensors. `scripts/extract_mtp.py`
reads the bf16 checkpoint's safetensors index, finds the 15 `mtp.*` tensors (all
in shard 18) and reads only their byte ranges (0.85 GB, a few seconds, no 54 GB
load), adds 1 to every RMSNorm weight (stored zero-centred: `Qwen3_5RMSNorm`
scales by `1 + w`; mlx-lm's `sanitize` and llama.cpp's converter do the same,
the latter for `enorm`/`hnorm` too; raw means -0.46 / -0.16 for
`pre_fc_norm_{embedding,hidden}` confirm it), and writes
`models/mtp-mlx/mtp.safetensors` (+ `config.json`; gitignored). The head is
quantized at load time (`bits=4` default: affine, group 64, like the target),
so 4/8/16-bit variants need only the one file. Embedding and lm_head are
shared with the target (`mtp_use_dedicated_embeddings: false`), i.e. we use the
target's 4-bit ones.

### Architecture as implemented

Reference: llama.cpp `llama_model_qwen35::graph_mtp` (`src/models/qwen35.cpp`)
and `common_speculative_impl_draft_mtp` (`common/speculative.cpp`) at
b29c606e2, which agree with vLLM's `Qwen3NextMTP`.

```
e   = pre_fc_norm_embedding(embed_tokens(x[i+1]))   # shared 4-bit embedding
h   = pre_fc_norm_hidden(H[i])                       # H[i] = target hidden after x[i], AFTER the final norm
z   = fc(concat([e, h]))                              # 10240 -> 5120, order (embedding, hidden)
z   = DecoderLayer(z)                                 # one Qwen3.5 FULL-attention layer: gated q (q_proj -> 2x),
                                                      #   q/k RMSNorm, partial RoPE (64 of 256 dims), SwiGLU MLP 17408
g   = mtp.norm(z)                                     # "h_nextn"
p(x[i+2]) = lm_head(g)                                # shared 4-bit lm_head
```

* It is mlx-lm's own `qwen3_5.DecoderLayer` built with a full-attention
  `layer_idx`, so it has a plain `KVCache` (no recurrent state; nothing to
  replay).
* Hidden input: the post-final-norm state (`ForwardOutput.normed`), as
  llama.cpp (`t_h_nextn` is taken after `output_norm`) and vLLM do. The
  pre-norm residual (`hidden=hidden`) gives the same acceptance within noise
  (0.636 vs 0.636 at k = 3 on 3 prompts), presumably because `hnorm`
  renormalizes it anyway; we keep the reference.
* Chaining for k > 1: step j + 1 feeds `g` of step j as the hidden and the
  token drafted at step j, at the next position (llama.cpp's mode for a
  single-head MTP: `h_row` + `pos0 + i + 1`).
* MTP KV cache: one entry per committed pair `(H[i], x[i+1])`, RoPE position
  = cache offset (llama.cpp puts pair i at position i + 1; a constant shift is
  invisible to RoPE attention). The prompt's L - 1 pairs are filled in
  `start()` from the prefill's hidden states (`SpecModel.prefill` now returns
  the hidden states of every chunk). Per round: `draft(last, k)` runs ONE MTP
  pass over the pending committed pairs from the last round (n + 1 rows: the
  accepted drafts plus `(H[last-1], last)`), whose last row gives draft 1,
  then k - 1 chained single-row passes. `commit(out, n)` trims the k - 1
  chained entries (they were built from the MTP's own `g`, not the target's
  hidden, so they are wrong even when accepted) and queues `out.normed[:n+1]`
  with `out.tokens[1:n+1]` as the next pending pairs. Rounds drafted by
  someone else (k = 0, or the n-gram in `mtp-hybrid`) just accumulate
  pending rows. Everything is lazy: drafting and the verify pass are one
  graph, one sync per round.
* Cost: draft step = MTP layer (~0.24 GB at 4-bit) + lm_head (248320 x 5120,
  ~0.7 GB at 4-bit): 7.5 ms per drafted token, 22.7 ms for k = 3 (8-bit
  layer: 9.2 ms/token, bf16: 12.1). The lm_head is ~70% of it, hence:
* **Vocabulary subset** (`vocab=N`): draft argmax only over token ids < N plus
  the 276 special/added ids (byte-level BPE ids are in merge order, so low ids
  are the frequent ones; no corpus statistics used). The subset rows of the
  4-bit lm_head are gathered once per drafter. N = 65536 covers 97.8% of the
  greedy output tokens of the six prompts (32768: 93.7%, 100000: 99.9%) and
  cuts drafting to 3.6 ms/token (10.7 ms per k = 3 round). The verify pass
  still uses the full lm_head, so this only costs acceptance, never
  correctness.

### Results (6 prompts, 256 tokens, greedy, thinking off)

Session greedy 8.57 tok/s (re-run in the same session); `k3` and the
`vocab65536` k3/k4 rows are median of 2 repeats, the rest single runs. Per-round
time from the profiled run.

| config | code_edit | code_gen | summarize | prose | explain | math | mean tok/s | vs greedy | accept | tok/round | draft / verify ms per round |
|---|---|---|---|---|---|---|---|---|---|---|---|
| greedy | 8.56 | 8.57 | 8.55 | 8.58 | 8.58 | 8.58 | **8.57** | 1.00x | - | 1 | - |
| draft 0.8B-4bit k=3 (ref) | 24.16 | 21.63 | 18.02 | 14.29 | 17.51 | 21.75 | **19.56** | 2.28x | 0.60 | 2.72 | 17.1 / 122.9 |
| mtp k=1 | 15.12 | 15.07 | 14.44 | 13.11 | 13.96 | 15.07 | **14.46** | 1.69x | 0.89 | 1.89 | 7.8 / 121.8 |
| mtp k=2 | 20.69 | 20.06 | 19.26 | 15.37 | 18.56 | 19.80 | **18.96** | 2.21x | 0.82 | 2.64 | 15.2 / 123.2 |
| mtp k=3 | 25.35 | 23.26 | 21.26 | 14.60 | 20.57 | 23.39 | **21.41** | 2.50x | 0.73 | 3.19 | 22.9 / 125.8 |
| mtp k=4 | 25.45 | 21.24 | 21.10 | 13.15 | 18.76 | 22.00 | **20.28** | 2.37x | 0.65 | 3.59 | 30.3 / 145.1 |
| mtp k=2 vocab 65536 | 21.99 | 20.70 | 20.12 | 15.43 | 18.73 | 21.06 | **19.67** | 2.30x | 0.80 | 2.59 | 7.2 / 123.6 |
| **mtp k=3 vocab 65536** | 27.65 | 24.48 | 22.35 | 16.40 | 21.85 | 25.37 | **23.02** | **2.69x** | 0.71 | 3.14 | 10.7 / 124.5 |
| mtp k=3 vocab 32768 | 26.56 | 23.26 | 22.62 | 16.14 | 22.39 | 25.36 | **22.72** | 2.65x | 0.67 | 3.01 | 8.4 / 122.4 |
| mtp k=4 vocab 65536 | 26.99 | 22.84 | 21.77 | 14.33 | 20.38 | 24.14 | **21.74** | 2.54x | 0.63 | 3.50 | 14.2 / 145.3 |
| mtp-hybrid k=3 | 26.02 | 23.16 | 21.44 | 15.45 | 20.17 | 21.90 | **21.36** | 2.49x | 0.68 | 3.04 | 19.1 / 122.5 |
| mtp-hybrid k=3 vocab 65536 | 26.89 | 23.82 | 22.13 | 16.42 | 21.44 | 23.02 | **22.29** | 2.60x | 0.67 | 3.01 | 9.3 / 124.8 |

(`accept` = mean over prompts of accepted/drafted; rollback is 1-2 ms per
round on average.) MTP layer precision, k = 3, 3 prompts x 128 tokens
(`logs/mlx_mtp_explore.log`): 4-bit 0.636 accepted / 20.3 tok/s, 8-bit 0.643 /
19.7, bf16 0.638 / 18.6. Precision of the one layer doesn't move acceptance;
4-bit is fastest.

**Acceptance per draft position** (conditional: P(draft j accepted | 1..j-1
accepted), pooled over the 6 prompts):

| config | pos 1 | pos 2 | pos 3 | pos 4 |
|---|---|---|---|---|
| mtp k=1 | 0.885 | | | |
| mtp k=2 | 0.887 | 0.816 | | |
| mtp k=3 | 0.862 | 0.792 | 0.807 | |
| mtp k=4 | 0.859 | 0.796 | 0.734 | 0.769 |
| mtp k=3 vocab 65536 | 0.852 | 0.789 | 0.777 | |
| draft 0.8B k=3 | 0.732 | 0.769 | 0.747 | |

Per prompt, position 1 ranges from 0.98 (code_edit) to 0.69-0.73 (prose);
prose at k = 3 is 0.69 / 0.56 / 0.41. Chaining the head on its own output
costs less than expected (0.86 -> 0.79 -> 0.78-0.81 conditional), so k = 3
is still worth it; k = 4 loses because T = 5 costs 1.21x (verify 145 vs
125 ms), as the verify curve predicts.

Reading it:

* **MTP beats the 0.8B draft model** on every prompt: first-position
  acceptance 0.86 vs 0.73 (it sees the target's own last hidden state), and
  with the vocabulary subset it drafts 3 tokens in 10.7 ms vs 17 ms. +18%
  end to end (23.0 vs 19.6 tok/s). Without the subset, MTP drafting
  (7.5 ms/token, dominated by the 4-bit lm_head) is slower per token than the
  0.8B model (5.7 ms/token), and the gain is +9%.
* Verify is ~90% of a round; the remaining lever is acceptance (prose) or a
  cheaper wide verify (see the `quantized_matmul` section), not drafting.
* MTP + n-gram (`mtp-hybrid`) doesn't help: where the n-gram matches, MTP
  already predicts the copy (code_edit position 1: 0.98), and n-gram
  proposals are accepted less often than MTP's (hybrid accept 0.67 vs 0.71).
* vs Track A (llama.cpp MTP, `notes/llama-mtp-dflash.md`,
  `results/llama-mtp-dflash.jsonl`): their acceptance is the same as ours
  (n_max 1 / 2 / 3: 0.90 / 0.81 / 0.74 vs our k = 1 / 2 / 3: 0.89 / 0.82 /
  0.73), a good sign the head is wired the same way. Their best is 11.1
  tok/s (1.67x over 6.65, n_max 2, Q4 sidecar); ours 23.0 (2.69x over 8.57).
  The difference is the verify curve: llama.cpp's Q4_K_M verify costs 1.4x at
  3 tokens and 2x at 4, ours is flat to 4 tokens, so we can use k = 3; plus
  MLX's faster base decode.

### Correctness

* `tests/test_mtp_tiny.py` (tiny random fp32 qwen3_5 + random MTPHead;
  `SPECDEC_CPU=1` runs it on the CPU without the lock, 25 s): for
  rollback `replay` and `reforward`, hidden `normed` and `hidden`, k = 1-3,
  and three drivers (MTP alone; n-gram/MTP hybrid; a noisy oracle that lets
  MTP draft only every 3rd round, so pending rows accumulate over k = 0
  rounds): spec == greedy, and the first-step MTP output of **every** round
  equals (atol 2e-4) the output computed from scratch over the final committed
  sequence (target hiddens from one fresh pass, all pairs in one MTP pass).
  Mutation checks: skipping the trim of chained entries, or misordering the
  pending tokens, fails it. Also: chunked prefill returns all hidden rows;
  the vocab-subset argmax equals the argmax of the full logits restricted to
  the subset.
* 27B, bf16 (`bench/mlx_mtp_session.py`, every config above): code_edit and
  prose identical to greedy; code_gen, summarize, explain, math diverge at
  tokens 147, 147, 42, 8 for **every** drafter including the 0.8B draft
  model, the same points as before. `diagnose()` on each: one-bf16-ulp ties
  (single-step margin 0.125, batched margin 0.0 or 0.125) -> "numerical
  near-tie"; no case with a clear margin.
* 27B, float32 activations (`tests/test_fp32_27b.py` with
  `mtp:k=3`, `mtp-hybrid:k=3:vocab=65536`, `mtp:k=4`, now in its default
  config list): **12/12 identical to greedy** on those four prompts, past every
  bf16 divergence point (`logs/mlx_mtp_fp32.log`). `test_equivalence_27b.py`
  also lists the MTP configs now (not re-run as a whole; the session above
  covers the same check).

### Open items

* Prose acceptance (0.69 at position 1) is the weak spot; nothing in the
  drafter can fix it short of a better head.
* Smarter k: a confidence cutoff (llama.cpp's `p_min`) would only save
  drafting time (~3.6 ms/token with the subset), since T <= 4 verify is flat;
  an adaptive k = 4 when the head is confident might pay on code_edit-like
  text. Not tried.
* The vocabulary subset is an id-order heuristic; a frequency list from a real
  corpus (or a learned "hot vocab" as in FR-Spec) would cover more for the
  same size.
* DFlash (`z-lab/Qwen3.8-27B-DFlash2`) was not attempted. It needs the target's
  layers 5/19/33/47/61 outputs (`SpecModel.forward` would have to expose
  them), five 5120-wide layers with dynamic grouped convs and a selector
  lattice (llama.cpp `src/models/dflash.cpp`, 1000 lines) ~1.1 GB at 4-bit,
  and Track A measured the same acceptance as MTP at short blocks (q4 n2:
  0.82 vs MTP 0.81). With our verify curve capping useful blocks at 3 drafts,
  it is unlikely to beat MTP k = 3 here unless the wide verify gets cheaper.

## Dead ends and gotchas

* A run on a busy machine is worthless: the first benchmark pass (session 1)
  overlapped the end of the 54 GB bf16 download (greedy 7.3 tok/s instead of
  8.56) and then another job running torch; n-gram rounds suddenly cost
  210 ms instead of 125 ms. Those rows were discarded
  (`logs/contaminated/`), and rows now record `loadavg`.
* `mlx_lm.load` checks the Hub for updates; once a stalled TLS handshake hung a
  run for 10 minutes *while holding the GPU lock*. The CLI now defaults to
  `HF_HUB_OFFLINE=1`.
* `capture` during prefill tried to write 305 x 151 MB of states and hit
  Metal OOM; it now only captures passes of <= 32 tokens.
* `scripts/gpu.py` opens the lock file with `"w"`, so every waiting process
  truncates the holder's "pid cmd" line; the lock itself works.

## Next steps

1. ~~MTP drafter~~: done (section "MTP drafter"); open items there.
2. **A wider cheap verify**: the staircase above T = 4 is the main limit on
   every method. If `quantized_matmul` could stream the weights once for up to
   ~8-16 rows (the work is bandwidth-bound: 15 GB at ~130 GB/s ≈ 115 ms no
   matter the row count), k = 6-8 would cost about what k = 3 does now, and
   both n-gram and DFlash (block drafts of 16) would benefit. See the
   `quantized_matmul` section above.
3. **DFlash** (`z-lab/Qwen3.8-27B-DFlash2`, downloaded): block-diffusion
   drafts from target hidden states; same interface, but it wants large k,
   so it depends on (2).
4. Defer the replay into the next pass (run the kept tokens' recurrence as
   the first steps of the next verify's kernel call) to save the ~3.5 ms
   rollback read/write; small (~2%).
5. Fewer bf16 flips: computing lm_head with fp32 output (in both baseline
   and verify) would remove the exact ties (several divergences had batched
   margin 0.0, broken by token id), but not the ~0.15-logit batched-vs-single
   noise from the trunk, so some near-ties would still flip.
