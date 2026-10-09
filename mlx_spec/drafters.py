"""Drafters: anything that proposes tokens for the target to verify.

The interface
-------------
A drafter sees the same token stream as the target. The speculative loop
(``decode.speculative_generate``) drives it with three calls::

    drafter.start(prompt, prefill)      # once, after the target prefilled the prompt
    loop:
        drafts = drafter.draft(last, k) # propose <= k tokens that follow `last`
        out = target.forward([last] + drafts)
        n = number of accepted drafts
        drafter.commit(out, n)          # out.tokens[:n + 1] are now committed

Invariants the drafter can rely on:

* ``last`` is always a committed token (the target's own choice), but the
  target has not consumed it yet. The committed context is therefore
  ``prompt + every out.tokens[:n + 1] seen in commit() + [last]``.
* ``draft`` may return fewer than ``k`` tokens (even zero: the round then
  degenerates to a normal one-token decode step). It returns a 1-D ``mx.array``
  and may leave it lazy; the loop evaluates it together with the verify pass.
* Optional, for sampling (temperature > 0): ``draft_probs()`` returns the
  ``[n_drafts, V]`` distributions the drafts were sampled from. Drafters
  without it are treated as deterministic proposals (one-hot q), which is
  also exact.
* ``commit(out, n)`` receives the target's ``ForwardOutput`` for the verified
  pass: ``out.tokens`` = ``[last] + drafts``, and ``out.hidden`` /
  ``out.normed`` / ``out.logits`` rows ``0..n`` belong to committed tokens
  (row ``i`` = target state after reading ``out.tokens[i]``). The token after
  the committed ones is ``argmax(out.logits[n])``, which arrives as the next
  ``last``. ``start`` gets the prompt's ``ForwardOutput`` (for prompts longer
  than the prefill chunk: of the last chunk only; ask for more if needed).

Adding MTP (Qwen3.8 ships one MTP layer) on top of this, see ``MTPDrafter``.
"""

from typing import List, Optional, Protocol

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import ArraysCache, KVCache, make_prompt_cache

from .model import ForwardOutput

EMPTY = mx.array([], dtype=mx.uint32)


class Drafter(Protocol):
    name: str

    def start(self, prompt: List[int], prefill: ForwardOutput) -> None: ...

    def draft(self, last: int, k: int) -> mx.array: ...

    def commit(self, out: ForwardOutput, n_accepted: int) -> None: ...


class NoDrafter:
    """Proposes nothing: the speculative loop becomes plain greedy decoding."""

    name = "none"

    def start(self, prompt, prefill):
        pass

    def draft(self, last, k):
        return EMPTY

    def commit(self, out, n_accepted):
        pass


class NgramDrafter:
    """Prompt-lookup decoding.

    Take the last ``n`` tokens of the context (``n`` from ``n_max`` down to
    ``n_min``), find their most recent earlier occurrence in the context
    (prompt + generated text), and propose the ``k`` tokens that followed it.
    No model, a few microseconds of CPU per round.
    """

    def __init__(self, n_max: int = 4, n_min: int = 2):
        self.n_max, self.n_min = n_max, n_min
        self.name = f"ngram_n{n_min}-{n_max}"

    def start(self, prompt, prefill):
        self.ctx = list(prompt)

    def lookup(self, ctx: np.ndarray, k: int) -> np.ndarray:
        L = ctx.size
        for n in range(min(self.n_max, L - 1), self.n_min - 1, -1):
            pattern = ctx[L - n :]
            # candidate starts i < L - n with ctx[i:i+n] == pattern
            hits = np.flatnonzero(ctx[: L - n] == pattern[0])
            for i in hits[::-1]:  # most recent first
                if np.array_equal(ctx[i : i + n], pattern):
                    cont = ctx[i + n : i + n + k]
                    if cont.size:
                        return cont
        return ctx[:0]

    def draft(self, last, k):
        if k <= 0:
            return EMPTY
        ctx = np.array(self.ctx + [last], dtype=np.uint32)
        return mx.array(self.lookup(ctx, k))

    def commit(self, out, n_accepted):
        self.ctx.extend(out.tokens[: n_accepted + 1].tolist())


# ---------------------------------------------------------------------------
# Small draft model


def snapshot(cache: List) -> list:
    """O(1) checkpoint of an mlx-lm cache list (KV offsets + array references).

    MLX arrays are immutable values, so keeping references to the current
    recurrent/conv arrays is a free snapshot; KV caches only need the offset
    because rolling back is a trim.
    """
    snap = []
    for c in cache:
        if isinstance(c, KVCache):
            snap.append(c.offset)
        elif isinstance(c, ArraysCache):
            snap.append(list(c.cache))
        else:
            raise TypeError(f"don't know how to snapshot {type(c).__name__}")
    return snap


def restore(cache: List, snap: list) -> None:
    for c, s in zip(cache, snap):
        if isinstance(c, KVCache):
            c.trim(c.offset - s)
        else:
            c.cache = list(s)


class DraftModelDrafter:
    """A separate small model (same tokenizer) drafting greedily, one token per step.

    The draft model keeps its own cache. Because it decodes one token per
    step, rewinding it is free: we snapshot the cache after every step
    (``snapshot``) and restore the one matching the number of accepted
    tokens. This works for hybrid draft models (e.g. Qwen3.5-0.8B, which has
    gated-delta layers itself) as well as pure-attention ones.

    Cache invariant: the draft cache has consumed the committed context minus
    ``self.pending + [last]``.
    """

    def __init__(self, model, name: str = "draft", prefill_step: int = 2048,
                 temperature: float = 0.0, k_min: Optional[int] = None):
        self.model = model
        self.name = name
        self.prefill_step = prefill_step
        # > 0: sample drafts from the draft model and expose their distributions
        # (draft_probs) for exact speculative sampling; 0: greedy drafts.
        self.temperature = temperature
        # Adaptive draft length in [k_min, k]: +1 after a fully accepted round,
        # -1 after a fully rejected one. None = always draft k.
        self.k_min = k_min
        self.k_cur = None

    def start(self, prompt, prefill):
        self.cache = make_prompt_cache(self.model)
        p = mx.array(prompt, mx.uint32)
        for s in range(0, p.size, self.prefill_step):
            self.model(p[None, s : s + self.prefill_step], cache=self.cache)
            mx.eval([c.state for c in self.cache])
        self.pending: List[int] = []

    def skip(self, tokens: List[int]):
        """Committed tokens the drafter didn't draft (e.g. a HybridDrafter round);
        they are fed to the draft model at the start of the next draft()."""
        self.pending.extend(tokens)
        self.snaps = None

    def draft(self, last, k):
        if self.k_min is not None:
            self.k_max = k
            self.k_cur = k if self.k_cur is None else min(self.k_cur, k)
            k = self.k_cur
        if k <= 0:
            self.pending.append(last)
            self.snaps, self.q = None, []
            return EMPTY
        x = mx.array(self.pending + [last], mx.uint32)
        self.pending = []
        self.snaps, out, self.q = [], [], []
        for _ in range(k):
            logits = self.model(x[None], cache=self.cache)
            self.snaps.append(snapshot(self.cache))  # consumed context + drafts so far
            if self.temperature > 0:
                lp = logits[0, -1:].astype(mx.float32) / self.temperature
                self.q.append(mx.softmax(lp, axis=-1))
                x = mx.random.categorical(lp).astype(mx.uint32)
            else:
                x = mx.argmax(logits[0, -1:], axis=-1).astype(mx.uint32)
            out.append(x)
        self.drafts = mx.concatenate(out)
        return self.drafts

    def draft_probs(self):
        """[k, V] distributions the last drafts were sampled from (None if greedy)."""
        return mx.concatenate(self.q) if self.temperature > 0 and self.q else None

    def commit(self, out, n_accepted):
        if self.snaps is None:
            return
        k = len(self.snaps)
        if self.k_min is not None:
            if n_accepted == k:
                self.k_cur = min(self.k_cur + 1, self.k_max)
            elif n_accepted == 0:
                self.k_cur = max(self.k_cur - 1, self.k_min)
        if n_accepted < k:
            # snaps[n] = state after [last, d_1 .. d_n]: exactly the committed tokens
            restore(self.cache, self.snaps[n_accepted])
        else:
            # all accepted: the cache saw [last, d_1 .. d_{k-1}], d_k still to feed
            self.pending = [int(self.drafts[-1].item())]
        self.snaps = None


class HybridDrafter:
    """Prompt lookup when it finds a long match, the draft model otherwise.

    N-gram drafts are free, so on text that copies from the context (code
    edits, summaries) this skips the draft model's ~6 ms per drafted token;
    elsewhere it is the draft model. The draft model is told about the
    tokens it didn't draft via ``skip`` so its cache stays in sync.
    """

    def __init__(self, ngram: NgramDrafter, model: DraftModelDrafter, min_len: int = 2):
        self.ngram, self.model, self.min_len = ngram, model, min_len
        self.name = f"hybrid({ngram.name}+{model.name})"

    def start(self, prompt, prefill):
        self.ngram.start(prompt, prefill)
        self.model.start(prompt, prefill)

    def draft(self, last, k):
        d = self.ngram.draft(last, k)
        self.used_ngram = d.size >= min(self.min_len, k)
        return d if self.used_ngram else self.model.draft(last, k)

    def draft_probs(self):
        return None if self.used_ngram else self.model.draft_probs()

    def commit(self, out, n_accepted):
        self.ngram.commit(out, n_accepted)
        if self.used_ngram:
            self.model.skip(out.tokens[: n_accepted + 1].tolist())
        else:
            self.model.commit(out, n_accepted)


# ---------------------------------------------------------------------------
# MTP (to be implemented on top of this interface)


class MTPDrafter:
    """Plan for Qwen3.8's multi-token-prediction head (not implemented yet).

    Weights: ``Qwen/Qwen3.8-27B`` has ``mtp_num_hidden_layers: 1`` with
    tensors ``mtp.fc.weight`` [5120, 10240], ``mtp.pre_fc_norm_embedding``,
    ``mtp.pre_fc_norm_hidden``, ``mtp.layers.0.*`` (one full-attention decoder
    layer: q/k/v/o_proj with output gate, q/k_norm, MLP, the two layernorms)
    and ``mtp.norm``. mlx-community/Qwen3.8-27B-4bit does NOT ship them
    (its index has no ``mtp.*`` keys, and ``qwen3_5.TextModel.sanitize`` drops
    them anyway), so load them separately from the bf16 checkpoint shards and
    quantize them (or keep bf16, it's ~0.9 GB).

    One MTP step (vLLM's Qwen3NextMTP)::

        e = pre_fc_norm_embedding(embed_tokens(tok_{i+1}))   # shared embedding
        h = pre_fc_norm_hidden(hidden_i)                     # target hidden at i
        x = fc(concat([e, h], -1))
        x = mtp_layer(x, cache=mtp_kv_cache)                 # position i + 1
        logits_{i+2} = lm_head(mtp.norm(x))                  # shared lm_head

    Check which hidden the reference implementation feeds (vLLM passes the
    target model's returned hidden states, i.e. post-final-norm, which is
    ``ForwardOutput.normed``; ``ForwardOutput.hidden`` is the pre-norm residual
    in case it is the other one). Both come for free from the verify pass.

    Hooking it into the interface:

    * ``start(prompt, prefill)``: run the MTP layer over the prompt to fill its
      KV cache: inputs ``(prefill.normed[i], prompt[i + 1])`` for each i
      (needs the hidden of the whole prompt; let ``SpecModel.prefill`` return
      all chunks if the prompt is longer than one chunk).
    * ``draft(last, k)``: first step uses the hidden of the last committed
      position (kept from the previous ``commit``) with token ``last``; for
      k > 1 feed the MTP layer's own output hidden back in (the head was
      trained for one step, so acceptance decays quickly; k = 1..3).
    * ``commit(out, n)``: trim the MTP KV cache like the target's (it is a
      plain ``KVCache``, no recurrent state), then append the entries for the
      newly committed positions using ``out.normed[:n + 1]`` and
      ``out.tokens[1:n + 1] + [next token]``, and remember ``out.normed[n]``.
    """

    name = "mtp"

    def __init__(self, *args, **kwargs):
        raise NotImplementedError("MTP drafter: see the class docstring for the plan")
