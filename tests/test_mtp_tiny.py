"""MTP drafter on the tiny random-weight qwen3_5 model (float32, seconds, GPU).

    scripts/gpu.py .venv/bin/python -m pytest tests/test_mtp_tiny.py -q
    SPECDEC_CPU=1 .venv/bin/python -m pytest tests/test_mtp_tiny.py -q     # CPU, no lock

Checks that the MTP layer's KV cache stays in sync with the committed
sequence through every acceptance length, k = 0 rounds (hybrid with n-gram)
and rollback mode: the first-step MTP output of every round equals the one
computed from scratch over the final committed sequence (target hiddens from
one fresh pass, all MTP pairs in one pass). And speculative == greedy.
"""

import os

import mlx.core as mx
import numpy as np
import pytest

from mlx_spec import HybridDrafter, NgramDrafter, SpecModel, greedy_generate, speculative_generate
from mlx_spec.mtp import MTPDrafter, MTPHead
from mlx_lm.models.cache import KVCache

from test_tiny import NoisyOracle, prompt, tiny_model


@pytest.fixture(scope="module", autouse=True)
def device():
    old = mx.default_device()
    if os.environ.get("SPECDEC_CPU") == "1":
        mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


@pytest.fixture(scope="module")
def setup(device):
    m = tiny_model()
    mx.random.seed(3)
    head = MTPHead(m.language_model.args)
    mx.eval(head.parameters())
    return m, head


class Traced:
    """Wraps a drafter; records (pair index, first-step MTP output) per MTP round."""

    def __init__(self, inner, mtp):
        self.inner, self.mtp, self.name = inner, mtp, "traced"
        self.log = []

    def start(self, prompt, prefill):
        self.pos = len(prompt)  # position of `last`
        self.mtp.trace = []
        self.inner.start(prompt, prefill)

    def draft(self, last, k):
        before = len(self.mtp.trace)
        d = self.inner.draft(last, k)
        if len(self.mtp.trace) > before:
            self.log.append((self.pos - 1, self.mtp.trace[-1]))
        return d

    def commit(self, out, n):
        self.inner.commit(out, n)
        self.pos += n + 1


class MixedOracle:
    """MTP drafts some rounds, a noisy oracle others (k = 0 for MTP), to
    exercise the pending-row accumulation."""

    def __init__(self, oracle, mtp, every=3):
        self.o, self.mtp, self.every, self.name = oracle, mtp, every, "mixed"

    def start(self, prompt, prefill):
        self.r = 0
        self.o.start(prompt, prefill)
        self.mtp.start(prompt, prefill)

    def draft(self, last, k):
        self.r += 1
        self.use_mtp = self.r % self.every == 0
        od = self.o.draft(last, k)
        if self.use_mtp:
            return self.mtp.draft(last, k)
        self.mtp.draft(last, 0)
        return od

    def commit(self, out, n):
        self.o.commit(out, n)
        self.mtp.commit(out, n)


def reference_g(spec, head, hidden, seq):
    """MTP outputs for every pair (H_i, x_{i+1}) of ``seq``, from scratch."""
    cache = spec.make_cache()
    out = spec.forward(mx.array(seq, mx.uint32), cache)
    H = out.normed if hidden == "normed" else out.hidden
    emb = spec.inner.embed_tokens(mx.array(seq[1:], mx.uint32)[None])
    return head(emb, H[None, :-1], KVCache())[0]


@pytest.mark.parametrize("mode", ["replay", "reforward"])
@pytest.mark.parametrize("hidden", ["normed", "hidden"])
def test_mtp_cache_sync(setup, mode, hidden):
    model, head = setup
    spec = SpecModel(model)
    for seed in range(2):
        ids = prompt(seed=seed)
        ref = greedy_generate(model, ids, 40).tokens
        for k in (1, 2, 3):
            for kind in ("mtp", "hybrid", "mixed"):
                mtp = MTPDrafter(spec, head, hidden=hidden)
                inner = {"mtp": mtp,
                         "hybrid": HybridDrafter(NgramDrafter(4, 1), mtp, min_len=1),
                         "mixed": MixedOracle(NoisyOracle(ref, 0.3, seed), mtp)}[kind]
                d = Traced(inner, mtp)
                r = speculative_generate(spec, d, ids, k=k, max_tokens=40, rollback=mode)
                assert r.tokens == ref, (kind, k, seed)
                assert d.log, kind
                g_ref = reference_g(spec, head, hidden, ids + ref)
                for idx, g in d.log:
                    assert np.allclose(g[0, 0], g_ref[idx], atol=2e-4), (kind, k, idx,
                                                                         mx.abs(g[0, 0] - g_ref[idx]).max().item())
                # after the run the cache has exactly one entry per pair fed so far
                assert mtp.n_chain == 0


def test_mtp_chunked_prefill(setup):
    model, head = setup
    spec = SpecModel(model)
    ids = prompt(n=24)
    a = spec.prefill(mx.array(ids), spec.make_cache(), step=7)
    b = spec.prefill(mx.array(ids), spec.make_cache())
    assert a.normed.shape[0] == len(ids) and a.tokens.tolist() == ids
    assert np.allclose(a.normed, b.normed, atol=1e-4)


def test_mtp_vocab_subset(setup):
    model, head = setup
    spec = SpecModel(model)
    ids = prompt()
    ref = greedy_generate(model, ids, 32).tokens
    vocab = mx.array(sorted(set(range(0, 512, 2)) | set(ref)), mx.uint32)
    full, sub = MTPDrafter(spec, head), MTPDrafter(spec, head, vocab=vocab)
    r = speculative_generate(spec, sub, ids, k=2, max_tokens=32)
    assert r.tokens == ref
    # the subset head drafts the argmax over the subset of the full logits
    g = mx.random.normal((1, 1, 256))
    lf = full._logits(g)[0]
    want = vocab[mx.argmax(lf[:, vocab], axis=-1)]
    assert mx.array_equal(sub._argmax(sub._logits(g)[0]), want)
