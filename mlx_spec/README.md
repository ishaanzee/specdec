# mlx_spec: speculative decoding from scratch in MLX (track B)

mlx-lm supplies the Qwen3.8 module tree (`mlx_lm.models.qwen3_5`), weight
loading, tokenizer and cache classes. Everything else is here: the greedy
baseline, the draft/verify/accept loop, cache rollback for the hybrid
architecture, and the drafters. Design notes, measurements and results are in
`notes/mlx-from-scratch.md`.

```
scripts/gpu.py .venv/bin/python -m mlx_spec generate --config draft:k=3 --prompt-id code_edit --profile
scripts/gpu.py .venv/bin/python -m mlx_spec bench --configs greedy ngram:k=3 draft:k=3 --repeat 2
scripts/gpu.py .venv/bin/python -m mlx_spec verify-cost
scripts/gpu.py .venv/bin/python -m pytest tests/test_tiny.py -q                       # seconds
.venv/bin/python scripts/extract_mtp.py                                              # once: MTP weights -> models/mtp-mlx/
SPECDEC_CPU=1 .venv/bin/python -m pytest tests/test_mtp_tiny.py -q                    # MTP cache sync, CPU
scripts/gpu.py .venv/bin/python -m mlx_spec bench --configs mtp:k=3:vocab=65536
.venv/bin/python -m pytest tests/test_sampling.py -q                                  # CPU only, 2 min
.venv/bin/python -m mlx_spec summary                                                  # results table
SPECDEC_27B=1 scripts/gpu.py .venv/bin/python -m pytest tests/test_equivalence_27b.py -q -s
```

| file | what |
|---|---|
| `gdn.py` | `RollbackGDN` (wraps each gated-delta layer), `LinearCache` (its rewindable cache), the per-token-state capture kernel |
| `model.py` | `SpecModel`: `forward(tokens, cache) -> ForwardOutput(logits, hidden, normed)`, `prefill`, `commit(cache, n_keep)` |
| `drafters.py` | `Drafter` protocol, `NgramDrafter`, `DraftModelDrafter` (optionally adaptive k), `HybridDrafter`, `NoDrafter` |
| `mtp.py` | Qwen3.8's MTP layer: `MTPHead`, `load_mtp_head` (weights from `scripts/extract_mtp.py`), `MTPDrafter` (best: `mtp:k=3:vocab=65536`, 2.69x) |
| `decode.py` | `greedy_generate` (baseline), `speculative_generate` (the loop) |
| `bench.py` | results rows in the shared schema, verify-cost curve, mismatch diagnosis |

Sampling: `speculative_generate(..., temperature=t)` does exact speculative
sampling (accept with `min(1, p/q)`, resample from `max(p - q, 0)`), with
`q` from `drafter.draft_probs()` or one-hot for deterministic drafters;
`tests/test_sampling.py` checks the output distribution. Greedy is the
benchmarked path.

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
CLI and benchmark can use it (`--configs mtp:k=2`). `mtp.MTPDrafter` is a
worked example (hidden-state conditioned, own KV cache trimmed at commit).
