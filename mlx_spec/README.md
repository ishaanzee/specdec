# mlx_spec: speculative decoding from scratch in MLX (track B)

mlx-lm supplies the Qwen3.8 module tree (`mlx_lm.models.qwen3_5`), weight
loading, tokenizer and cache classes. Everything else is here: the greedy
baseline, the draft/verify/accept loop, cache rollback for the hybrid
architecture, and the drafters. Design notes, measurements and results are in
`notes/mlx-from-scratch.md`.

```
scripts/gpu.py .venv/bin/python -m mlx_spec generate --config ngram:k=8 --prompt-id code_edit --profile
scripts/gpu.py .venv/bin/python -m mlx_spec bench --configs greedy ngram:k=8:n=2-4 draft:k=4 --repeat 2
scripts/gpu.py .venv/bin/python -m mlx_spec verify-cost
scripts/gpu.py .venv/bin/python -m pytest tests/test_tiny.py -q                       # seconds
SPECDEC_27B=1 scripts/gpu.py .venv/bin/python -m pytest tests/test_equivalence_27b.py -q -s
```

| file | what |
|---|---|
| `gdn.py` | `RollbackGDN` (wraps each gated-delta layer), `LinearCache` (its rewindable cache), the per-token-state capture kernel |
| `model.py` | `SpecModel`: `forward(tokens, cache) -> ForwardOutput(logits, hidden, normed)`, `prefill`, `commit(cache, n_keep)` |
| `drafters.py` | `Drafter` protocol, `NgramDrafter`, `DraftModelDrafter`, `NoDrafter`, `MTPDrafter` plan |
| `decode.py` | `greedy_generate` (baseline), `speculative_generate` (the loop) |
| `bench.py` | results rows in the shared schema, verify-cost curve, mismatch diagnosis |

## Rollback in one paragraph

After verifying `T = 1 + k` tokens, `SpecModel.commit(cache, n_keep)` keeps
`n_keep = 1 + accepted`. Attention layers: `KVCache.trim(T - n_keep)`.
Gated-delta layers can't be trimmed, so the layer records the pass's
recurrence inputs and `LinearCache.commit` recomputes the state for the kept
tokens with the same Metal kernel (`replay`, default; bitwise equal to the
state the verify pass had at that position), or picks it from per-token
states written by a modified kernel (`capture`), or re-runs the model
(`reforward`, reference). The conv window is sliced from the recorded conv
input in all modes.

## Adding a drafter (MTP, DFlash, ...)

Implement three methods (see the docstring at the top of `drafters.py` for the
exact invariants):

```python
class MyDrafter:
    name = "mine"
    def start(self, prompt: list[int], prefill: ForwardOutput) -> None: ...
    def draft(self, last: int, k: int) -> mx.array: ...       # <= k tokens after `last`
    def commit(self, out: ForwardOutput, n_accepted: int) -> None: ...
```

`out.hidden` (pre-final-norm) and `out.normed` (post-norm) are the target's
hidden states for every verified position, computed in the verify pass at no
extra cost; rows `0..n_accepted` are committed. That's what MTP / EAGLE /
DFlash heads consume. Then register it in `bench.Runner.make_drafter` so the
CLI and benchmark can use it (`--configs mtp:k=2`). `MTPDrafter` in
`drafters.py` has the concrete plan for Qwen3.8's MTP layer.
