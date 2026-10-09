"""Fast correctness tests on a tiny random-weight qwen3_5 model (float32).

Run (takes seconds; uses the GPU, so take the lock):
    scripts/gpu.py .venv/bin/python -m pytest tests/test_tiny.py -q
"""

import random

import mlx.core as mx
import numpy as np
import pytest
from mlx_lm.models import qwen3_5
from mlx_lm.models.gated_delta import gated_delta_kernel

from mlx_spec import (DraftModelDrafter, NgramDrafter, NoDrafter, SpecModel,
                      greedy_generate, speculative_generate)
from mlx_spec.drafters import EMPTY
from mlx_spec.gdn import LinearCache, gated_delta_capture

MODES = ("replay", "capture", "reforward")


def tiny_model(seed=0, layers=8, vocab=512):
    mx.random.seed(seed)
    cfg = dict(
        model_type="qwen3_5", hidden_size=256, intermediate_size=512, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=64, vocab_size=vocab,
        linear_num_value_heads=4, linear_num_key_heads=2, linear_key_head_dim=128,
        linear_value_head_dim=128, linear_conv_kernel_dim=4, full_attention_interval=4,
        rms_norm_eps=1e-6, tie_word_embeddings=False,
        rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.25},
    )
    m = qwen3_5.Model(qwen3_5.ModelArgs.from_dict(cfg))
    # sharper logits so greedy is far from ties
    m.language_model.lm_head.weight = m.language_model.lm_head.weight * 8
    mx.eval(m.parameters())
    return m


@pytest.fixture(scope="module")
def model():
    return tiny_model()


def prompt(n=24, seed=1, vocab=512):
    rng = random.Random(seed)
    base = [rng.randrange(vocab) for _ in range(8)]
    return (base * 4)[:n]  # repetitive, so n-gram drafts sometimes hit


def test_capture_kernel_bitwise():
    mx.random.seed(0)
    B, T, Hk, Hv, D = 1, 9, 2, 4, 128
    q = mx.random.normal((B, T, Hk, D)) * 0.1
    k = mx.random.normal((B, T, Hk, D)) * 0.1
    v = mx.random.normal((B, T, Hv, D))
    g = mx.random.uniform(0.8, 1.0, (B, T, Hv))
    beta = mx.random.uniform(0, 1, (B, T, Hv))
    s0 = mx.random.normal((B, Hv, D, D))
    y, states = gated_delta_capture(q, k, v, g, beta, s0)
    y_ref, _ = gated_delta_kernel(q, k, v, g, beta, s0)
    assert mx.array_equal(y, y_ref)
    for t in range(T):
        _, s_ref = gated_delta_kernel(q[:, : t + 1], k[:, : t + 1], v[:, : t + 1],
                                      g[:, : t + 1], beta[:, : t + 1], s0)
        assert mx.array_equal(states[:, t], s_ref), t


@pytest.mark.parametrize("mode", MODES)
def test_rollback_matches_fresh_cache(model, mode):
    """After a T-token pass + commit(n_keep), the cache equals one that only
    ever saw the first n_keep tokens (up to float32 matmul-shape noise)."""
    spec = SpecModel(model)
    p = mx.array(prompt())
    extra = mx.array([5, 17, 99, 3, 250, 7, 8, 400], mx.uint32)
    for n_keep in range(0, extra.size + 1):
        cache = spec.make_cache(mode)
        spec.prefill(p, cache)
        out = spec.forward(extra, cache)
        spec.commit(cache, n_keep, out)

        ref = spec.make_cache(mode)
        spec.prefill(p, ref)
        if n_keep:
            spec.forward(extra[:n_keep], ref)
            spec.commit(ref, n_keep)
        for c, r in zip(cache, ref):
            if isinstance(c, LinearCache):
                assert c.pending is None
                assert np.allclose(c[0], r[0], atol=1e-5), (mode, n_keep, "conv")
                assert np.allclose(c[1], r[1], atol=1e-4), (mode, n_keep, "state")
            else:
                assert c.offset == r.offset
        # and the next pass produces the same logits
        nxt = mx.array([11, 12], mx.uint32)
        a = spec.forward(nxt, cache).logits
        b = spec.forward(nxt, ref).logits
        assert np.allclose(a, b, atol=1e-3), (mode, n_keep)


class NoisyOracle:
    """Proposes the true greedy continuation, corrupting each token with prob p,
    so every acceptance length (0..k) gets exercised."""

    name = "oracle"

    def __init__(self, reference, p=0.3, seed=0):
        self.ref, self.p, self.rng = reference, p, random.Random(seed)

    def start(self, prompt, prefill):
        self.pos = 1  # tokens committed so far, incl. the first `last`

    def draft(self, last, k):
        d = self.ref[self.pos : self.pos + k]
        d = [t if self.rng.random() > self.p else (t + 1) % 512 for t in d]
        return mx.array(d, mx.uint32) if d else EMPTY

    def commit(self, out, n):
        self.pos += n + 1


@pytest.mark.parametrize("mode", MODES)
def test_spec_equals_greedy(model, mode):
    spec = SpecModel(model, async_every=3)  # also exercise chunked submission
    for seed in range(3):
        ids = prompt(seed=seed)
        ref = greedy_generate(model, ids, 48).tokens
        drafters = [NoDrafter(), NgramDrafter(4, 1), NoisyOracle(ref, 0.3, seed), NoisyOracle(ref, 0.0)]
        for d in drafters:
            for k in (1, 3, 8):
                got = speculative_generate(spec, d, ids, k=k, max_tokens=48, rollback=mode).tokens
                assert got == ref, (mode, d.name, k, seed)


def test_draft_model_drafter(model):
    spec = SpecModel(model)
    draft = tiny_model(seed=5, layers=4)  # different weights, same vocab, also hybrid
    for seed in range(2):
        ids = prompt(seed=seed)
        ref = greedy_generate(model, ids, 40).tokens
        for k in (1, 4):
            r = speculative_generate(spec, DraftModelDrafter(draft), ids, k=k, max_tokens=40)
            assert r.tokens == ref
        # a draft model identical to the target must be accepted every time
        r = speculative_generate(spec, DraftModelDrafter(model), ids, k=4, max_tokens=40)
        assert r.tokens == ref
        assert r.draft_accepted == r.draft_n, (r.draft_accepted, r.draft_n)


def test_eos_and_budget(model):
    spec = SpecModel(model)
    ids = prompt()
    ref = greedy_generate(model, ids, 30).tokens
    eos = {ref[10]}
    want = ref[: ref.index(ref[10]) + 1]
    assert greedy_generate(model, ids, 30, eos=eos).tokens == want
    got = speculative_generate(spec, NoisyOracle(ref, 0.0), ids, k=8, max_tokens=30, eos=eos).tokens
    assert got == want
    for n in (1, 2, 7):
        got = speculative_generate(spec, NoisyOracle(ref, 0.0), ids, k=8, max_tokens=n).tokens
        assert got == ref[:n]
