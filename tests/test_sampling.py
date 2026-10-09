"""Exact speculative sampling (temperature > 0): distribution test.

Uses a 16-token-vocab random model on the CPU (no GPU, so no lock needed):
    .venv/bin/python -m pytest tests/test_sampling.py -q

Generates 3 tokens many times with speculative sampling and compares the
empirical marginals of tokens 2 and 3 (the ones that go through draft +
verify) with the exact marginals computed by enumerating the target model.
A negative control (accept every draft) must fail the same check, so the
test has the power to see a wrong acceptance rule.
"""

import mlx.core as mx
import numpy as np
import pytest

import mlx_spec.decode as decode
from mlx_spec import DraftModelDrafter, NgramDrafter, SpecModel, speculative_generate

from test_tiny import tiny_model

V, TEMP, N, PROMPT = 16, 1.0, 3000, [3, 7, 3, 7, 3, 9]


@pytest.fixture(scope="module", autouse=True)
def cpu():
    old = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(old)


def make(seed):
    m = tiny_model(seed=seed, layers=4, vocab=V)
    # undo tiny_model's x8 sharpening: we want spread-out distributions here
    m.language_model.lm_head.weight = m.language_model.lm_head.weight / 6
    return m


def probs(model, toks):
    logits = model(mx.array([toks]))[0, -1].astype(mx.float32)
    return np.array(mx.softmax(logits / TEMP))


def exact_marginals(model):
    p1 = probs(model, PROMPT)
    m2, m3 = np.zeros(V), np.zeros(V)
    for a in range(V):
        p2 = probs(model, PROMPT + [a])
        m2 += p1[a] * p2
        for b in range(V):
            m3 += p1[a] * p2[b] * probs(model, PROMPT + [a, b])
    return m2, m3


def empirical(spec, make_drafter, k):
    mx.random.seed(0)
    c2, c3 = np.zeros(V), np.zeros(V)
    for _ in range(N):
        t = speculative_generate(spec, make_drafter(), PROMPT, k=k, max_tokens=3,
                                 temperature=TEMP).tokens
        c2[t[1]] += 1
        c3[t[2]] += 1
    return c2 / N, c3 / N


def tv(a, b):
    return 0.5 * np.abs(a - b).sum()


@pytest.fixture(scope="module")
def setup():
    target, draft = make(0), make(7)
    return SpecModel(target), draft, exact_marginals(target)


TOL = 0.06  # sampling noise for N=3000, V=16 is ~0.03


@pytest.mark.parametrize("which", ["ngram", "draft-sampled", "draft-greedy"])
def test_speculative_sampling_is_exact(setup, which):
    spec, draft, (m2, m3) = setup
    make_drafter = {
        "ngram": lambda: NgramDrafter(n_max=2, n_min=1),
        "draft-sampled": lambda: DraftModelDrafter(draft, temperature=TEMP),
        "draft-greedy": lambda: DraftModelDrafter(draft),
    }[which]
    e2, e3 = empirical(spec, make_drafter, k=2)
    assert tv(e2, m2) < TOL and tv(e3, m3) < TOL, (tv(e2, m2), tv(e3, m3))


def test_negative_control_detects_wrong_rule(setup, monkeypatch):
    spec, draft, (m2, m3) = setup
    real = decode._sample_verify

    def accept_all(logits, drafts, q, temperature):
        accept, fix, bonus = real(logits, drafts, q, temperature)
        return (None if accept is None else mx.ones_like(accept)), fix, bonus

    monkeypatch.setattr(decode, "_sample_verify", accept_all)
    e2, e3 = empirical(spec, lambda: DraftModelDrafter(draft, temperature=TEMP), k=2)
    assert max(tv(e2, m2), tv(e3, m3)) > TOL
