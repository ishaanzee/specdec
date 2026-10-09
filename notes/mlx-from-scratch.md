# Track B: speculative decoding from scratch in MLX

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
