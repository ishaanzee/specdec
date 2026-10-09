"""Qwen3.8's own multi-token-prediction (MTP) layer as a drafter.

Architecture (as implemented; checked against llama.cpp
``llama_model_qwen35::graph_mtp`` + ``common_speculative_impl_draft_mtp`` at
vendor/llama.cpp b29c606e2, and vLLM's ``Qwen3NextMTP``)::

    e   = pre_fc_norm_embedding(embed_tokens(x_{i+1}))     # target's (shared) embedding
    h   = pre_fc_norm_hidden(H_i)                          # H_i = target hidden after x_i,
                                                           #   POST final norm (= ForwardOutput.normed)
    z   = fc(concat([e, h], -1))                           # [2*5120] -> 5120, order (e, h)
    z   = DecoderLayer(z)                                  # one Qwen3.5 *full-attention* layer
                                                           #   (gated q, q/k RMSNorm, partial RoPE,
                                                           #    SwiGLU MLP), own KV cache
    g_i = mtp.norm(z)                                      # "h_nextn"
    logits(x_{i+2}) = lm_head(g_i)                         # target's (shared) lm_head

All RMSNorm weights are stored zero-centred in the HF checkpoint (``1 + w``);
``scripts/extract_mtp.py`` adds the 1 (llama.cpp's converter does the same for
every ``*norm.weight`` incl. ``enorm``/``hnorm``).

Chaining (k > 1): step j + 1 feeds ``g`` of step j as the hidden and the token
drafted at step j (llama.cpp's "neither" mode for a single MTP head).

The MTP layer's KV cache holds one entry per committed pair ``(H_i, x_{i+1})``
in order; RoPE positions are the cache offsets (llama.cpp uses offset + 1;
a constant shift doesn't change attention). Chained draft entries are trimmed
at commit, because they were computed from the MTP's own ``g`` rather than the
target's ``H``; the committed pairs of the round are appended at the start of
the next ``draft`` (in the same MTP pass as the first draft step).
"""

import json
from pathlib import Path
from typing import List, Optional

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3_5 import DecoderLayer

from .model import ForwardOutput, SpecModel

EMPTY = mx.array([], dtype=mx.uint32)
MTP_DIR = Path(__file__).resolve().parent.parent / "models" / "mtp-mlx"


class MTPHead(nn.Module):
    def __init__(self, args):
        super().__init__()
        H = args.hidden_size
        self.fc = nn.Linear(2 * H, H, bias=False)
        self.pre_fc_norm_embedding = nn.RMSNorm(H, eps=args.rms_norm_eps)
        self.pre_fc_norm_hidden = nn.RMSNorm(H, eps=args.rms_norm_eps)
        # layer_idx = full_attention_interval - 1 makes DecoderLayer a full-attention layer
        self.layers = [DecoderLayer(args, layer_idx=args.full_attention_interval - 1)]
        assert not self.layers[0].is_linear
        self.norm = nn.RMSNorm(H, eps=args.rms_norm_eps)

    def __call__(self, emb: mx.array, hidden: mx.array, cache: KVCache) -> mx.array:
        """emb, hidden: [1, T, H] -> post-norm output g [1, T, H]; appends T KV entries."""
        x = mx.concatenate([self.pre_fc_norm_embedding(emb), self.pre_fc_norm_hidden(hidden)], axis=-1)
        x = self.fc(x)
        x = self.layers[0](x, mask=create_attention_mask(x, cache), cache=cache)
        return self.norm(x)


def load_mtp_head(args, path: Path = MTP_DIR, bits: int = 4, group_size: int = 64,
                  dtype=mx.bfloat16) -> MTPHead:
    """Build the head for the target's ``TextModelArgs`` and load the extracted weights.

    ``bits`` in (4, 8): affine-quantize every Linear (fc, q/k/v/o, MLP) like the
    target; 16: keep bf16. Norms are cast to ``dtype``.
    """
    head = MTPHead(args)
    w = mx.load(str(Path(path) / "mtp.safetensors"))
    w = {k[len("mtp."):]: v.astype(dtype) for k, v in w.items()}
    head.load_weights(list(w.items()), strict=True)
    if bits in (4, 8):
        nn.quantize(head, group_size=group_size, bits=bits,
                    class_predicate=lambda p, m: isinstance(m, nn.Linear))
    elif bits != 16:
        raise ValueError(f"bits must be 4, 8 or 16, got {bits}")
    mx.eval(head.parameters())
    return head


class MTPDrafter:
    """Drafts with the target's MTP layer, conditioned on the verify pass's hidden states.

    State between rounds (all relative to the drafter interface invariants):

    * ``self.cache``: MTP KV entries for committed pairs ``(H_i, x_{i+1})``
      up to some point;
    * ``self.pend_h`` [R, H] / ``self.pend_tok`` (R - 1 tokens): committed target
      hiddens whose pair isn't in the cache yet; the token of the last row is
      the ``last`` passed to the next ``draft``.

    ``draft(last, k)``: one MTP pass over the R pending pairs (the last one,
    ``(H_{m-1}, last)``, gives draft 1), then k - 1 chained single-row passes.
    Everything stays lazy, so drafting + verification are one graph with no
    CPU sync. ``commit``: trim the k - 1 chained entries, keep
    ``out.normed[:n + 1]`` / ``out.tokens[1:n + 1]`` as the new pending rows.
    ``k = 0`` (or a round drafted by someone else, see ``HybridDrafter``)
    just accumulates pending rows.

    ``hidden``: ``"normed"`` (post final norm, the reference) or ``"hidden"``
    (pre-norm residual, for comparison). ``chain``: hidden fed to chained
    steps, ``"normed"`` (g, the reference) or ``"prenorm"``.
    ``vocab``: optional [V'] array of token ids; draft argmax only over those
    rows of lm_head (cheaper drafting; drafts outside the subset are impossible).
    """

    def __init__(self, spec: SpecModel, head: MTPHead, name: str = "mtp", hidden: str = "normed",
                 vocab: Optional[mx.array] = None):
        self.spec, self.head, self.name = spec, head, name
        self.hidden = hidden
        self.embed = spec.inner.embed_tokens
        self.vocab = vocab
        if vocab is not None:
            # gather the subset rows of the (quantized) lm_head once
            lm = spec.lm.lm_head
            if isinstance(lm, nn.QuantizedLinear):
                sub = nn.QuantizedLinear(lm.weight.shape[1] * 32 // lm.bits, vocab.size, bias=False,
                                         group_size=lm.group_size, bits=lm.bits)
                sub.weight, sub.scales = lm.weight[vocab], lm.scales[vocab]
                sub.biases = lm.biases[vocab]
            else:
                sub = nn.Linear(lm.weight.shape[1], vocab.size, bias=False)
                sub.weight = lm.weight[vocab]
            mx.eval(sub.parameters())
            self.sub_head = sub
        self.trace = None  # set to a list to record first-step outputs (tests)

    def _h(self, out: ForwardOutput) -> mx.array:
        return out.normed if self.hidden == "normed" else out.hidden

    def _logits(self, g):
        if self.vocab is not None:
            return self.sub_head(g)
        if self.spec.lm.args.tie_word_embeddings:
            return self.embed.as_linear(g)
        return self.spec.lm.lm_head(g)

    def _argmax(self, logits):
        t = mx.argmax(logits, axis=-1).astype(mx.uint32)
        return self.vocab[t] if self.vocab is not None else t

    # ------------------------------------------------------------ interface
    def start(self, prompt: List[int], prefill: ForwardOutput):
        hs = self._h(prefill)
        assert hs.shape[0] == len(prompt), (
            "MTPDrafter needs the hidden states of the whole prompt "
            "(SpecModel.prefill returns them for every chunk)")
        self.cache = KVCache()
        self.pend_h, self.pend_tok = hs, list(prompt[1:])
        self.n_chain = 0
        self.consumed = False
        # the prompt's pairs go through the MTP layer now (they are known);
        # only the last row waits for `last`
        if len(prompt) > 1:
            self.head(self.embed(mx.array(prompt[1:], mx.uint32)[None]), hs[None, :-1], self.cache)
            mx.eval(self.cache.state)
            self.pend_h, self.pend_tok = hs[-1:], []

    def draft(self, last: int, k: int) -> mx.array:
        self.n_chain = 0
        if k <= 0:
            self.consumed = False
            return EMPTY
        self.consumed = True
        toks = mx.array(self.pend_tok + [last], mx.uint32)
        g = self.head(self.embed(toks[None]), self.pend_h[None], self.cache)[:, -1:]
        if self.trace is not None:
            self.trace.append(g)
        out = []
        for i in range(k):
            d = self._argmax(self._logits(g)[0])  # [1]
            out.append(d)
            if i + 1 < k:
                g = self.head(self.embed(d[None]), g, self.cache)
                self.n_chain += 1
        return mx.concatenate(out)

    def commit(self, out: ForwardOutput, n_accepted: int):
        n = n_accepted
        self.cache.trim(self.n_chain)
        self.n_chain = 0
        h = self._h(out)[: n + 1]
        toks = out.tokens[: n + 1].tolist()
        if self.consumed:
            self.pend_h, self.pend_tok = h, toks[1:]
        else:  # nothing was fed this round: out.tokens[0] (= last) pairs with the old last row
            self.pend_h = mx.concatenate([self.pend_h, h])
            self.pend_tok = self.pend_tok + toks
        self.consumed = False


def load_vocab_subset(spec: str, vocab_size: int = 248320, n_regular: int = 248044) -> mx.array:
    """``"first:N"``: the N lowest token ids (byte-level BPE ids are in merge
    order, so low ids are the frequent ones) plus every special/added token
    (ids >= ``n_regular``, e.g. <|im_end|>); otherwise a JSON list of ids."""
    if spec.startswith("first:"):
        n = int(spec.split(":", 1)[1])
        ids = list(range(min(n, n_regular))) + list(range(n_regular, vocab_size))
    else:
        ids = sorted(set(json.load(open(spec))))
    return mx.array(ids, mx.uint32)
