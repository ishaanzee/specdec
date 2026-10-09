"""Target-model wrapper for speculative decoding of mlx-lm ``qwen3_5`` models.

mlx-lm provides the module tree, weights, tokenizer and cache classes; this
file adds the three things the speculative loop needs from the target:

* ``forward(tokens, cache)``: one pass over ``T`` tokens that returns logits
  for every position *and* the hidden states (pre- and post-final-norm), so
  hidden-state drafters (MTP, EAGLE, DFlash) get them at no extra cost.
* ``make_cache()``: ``KVCache`` for the 16 full-attention layers,
  ``LinearCache`` (rewindable) for the 48 gated-delta layers.
* ``commit(cache, n_keep)``: after a pass over ``T`` tokens, keep the first
  ``n_keep`` and roll back the rest, in every layer.
"""

from dataclasses import dataclass
from typing import List

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache

from .gdn import LinearCache, RollbackGDN


@dataclass
class ForwardOutput:
    """Result of one target pass over ``tokens`` (length ``T``).

    Row ``i`` of every array is the model's output *after* reading
    ``tokens[i]``, i.e. ``logits[i]`` predicts the token at position ``i + 1``.
    Arrays are lazy; nothing is evaluated unless the caller asks.
    """

    tokens: mx.array  # [T] uint32, the inputs of this pass
    logits: mx.array  # [T, vocab]
    hidden: mx.array  # [T, hidden] residual stream before the final RMSNorm
    normed: mx.array  # [T, hidden] after the final RMSNorm (what lm_head sees)


def text_model(model: nn.Module) -> nn.Module:
    """The ``qwen3_5.TextModel`` inside a (possibly multimodal) mlx-lm model."""
    return getattr(model, "language_model", model)


def is_hybrid(model: nn.Module) -> bool:
    return any(getattr(l, "is_linear", False) for l in text_model(model).layers)


class SpecModel:
    """A loaded mlx-lm Qwen3.5/3.8 model prepared for speculative verification.

    Installing the wrapper replaces every ``layer.linear_attn`` by a
    ``RollbackGDN`` around the original module. With stock caches
    (``model.make_cache()``) the model still behaves exactly like mlx-lm, so
    the same weights can serve the plain baseline.
    """

    def __init__(self, model: nn.Module, rollback: str = "replay", async_every: int = 16):
        self.model = model
        self.rollback = rollback
        self.async_every = async_every
        tm = text_model(model)
        self.lm = tm
        self.inner = tm.model  # Qwen3_5TextModel: embed_tokens, layers, norm
        for layer in self.inner.layers:
            if getattr(layer, "is_linear", False) and not isinstance(
                layer.linear_attn, RollbackGDN
            ):
                layer.linear_attn = RollbackGDN(layer.linear_attn)
        self.fa_idx = next(i for i, l in enumerate(self.inner.layers) if not l.is_linear)

    # ---------------------------------------------------------------- caches
    def make_cache(self, rollback: str | None = None) -> List:
        mode = rollback or self.rollback
        return [LinearCache(mode) if l.is_linear else KVCache() for l in self.inner.layers]

    # --------------------------------------------------------------- forward
    def forward(self, tokens: mx.array, cache: List) -> ForwardOutput:
        """Run the target over ``tokens`` ([T] int) appending to ``cache``.

        Mirrors ``Qwen3_5TextModel.__call__`` (single device, batch 1, no
        padding, so the SSM mask is None) but keeps the pre-norm residual.
        """
        tokens = tokens.astype(mx.uint32)
        h = self.inner.embed_tokens(tokens[None])
        fa_mask = create_attention_mask(h, cache[self.fa_idx])
        for i, (layer, c) in enumerate(zip(self.inner.layers, cache)):
            h = layer(h, mask=None if layer.is_linear else fa_mask, cache=c)
            # The loop has to block on every verify result, so unlike plain
            # decoding it can't hide graph construction behind the previous
            # step. Submitting the graph in pieces lets the GPU start on the
            # first layers while Python is still building the rest.
            if self.async_every and (i + 1) % self.async_every == 0:
                mx.async_eval(h)
        normed = self.inner.norm(h)
        if self.lm.args.tie_word_embeddings:
            logits = self.inner.embed_tokens.as_linear(normed)
        else:
            logits = self.lm.lm_head(normed)
        return ForwardOutput(tokens, logits[0], h[0], normed[0])

    def prefill(self, tokens: mx.array, cache: List, step: int = 2048) -> ForwardOutput:
        """Process a prompt in chunks; returns the output of the last chunk.

        Every chunk is committed in full and evaluated, so no rollback
        information is kept alive for the prompt.
        """
        out = None
        for s in range(0, tokens.size, step):
            out = self.forward(tokens[s : s + step], cache)
            self.commit(cache, out.tokens.size)
            mx.eval(out.logits, out.hidden, [c.state for c in cache])
        return out

    # -------------------------------------------------------------- rollback
    def commit(self, cache: List, n_keep: int, last: ForwardOutput | None = None):
        """Keep the first ``n_keep`` tokens of the last pass, drop the rest.

        Attention layers: trim ``T - n_keep`` entries off the KV cache (the
        rows stay allocated and are overwritten by the next pass).
        GDN layers: ``LinearCache.commit`` (see gdn.py).

        For ``rollback == "reforward"`` the GDN layers are restored to their
        pre-pass state, the KV caches trimmed back to the pre-pass length, and
        the kept tokens are run through the model again (``last`` must be the
        ``ForwardOutput`` of the pass being rolled back). Returns the
        ``ForwardOutput`` of that re-run, or None.
        """
        T = next(c.pending["T"] for c in cache if isinstance(c, LinearCache) and c.pending)
        reforward = (
            n_keep not in (0, T)
            and any(isinstance(c, LinearCache) and c.mode == "reforward" for c in cache)
        )
        for c in cache:
            if isinstance(c, LinearCache):
                c.commit(n_keep)
            else:
                c.trim(T - (0 if reforward else n_keep))
        if reforward:
            assert last is not None, "reforward rollback needs the rolled-back pass"
            out = self.forward(last.tokens[:n_keep], cache)
            self.commit(cache, n_keep)
            return out
        return None
