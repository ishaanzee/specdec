"""Speculative greedy == plain greedy, token for token, on the real target.

Opt-in because it loads the 27B (about 10 minutes):
    SPECDEC_27B=1 scripts/gpu.py .venv/bin/python -m pytest tests/test_equivalence_27b.py -q -s

On a mismatch the test prints the logits diagnosis at the divergence point
(single-token vs batched pass): a near-tie that flips under a different
matmul shape is numerical and tolerated, a clear margin is a bug and fails.
"""

import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("SPECDEC_27B") != "1", reason="set SPECDEC_27B=1")

CONFIGS = [
    "ngram:k=4:n=2-4:rollback=replay",
    "ngram:k=4:n=2-4:rollback=capture",
    "ngram:k=4:n=2-4:rollback=reforward",
    "draft:k=4",
    "mtp:k=3",
    "mtp-hybrid:k=3:vocab=65536",
]
if os.environ.get("SPECDEC_CONFIGS"):
    CONFIGS = os.environ["SPECDEC_CONFIGS"].split()


@pytest.fixture(scope="module")
def runner():
    from mlx_spec.bench import Runner

    return Runner()


@pytest.mark.parametrize("prompt", [p["id"] for p in __import__("mlx_spec.bench", fromlist=["x"]).load_prompts()])
def test_prompt(runner, prompt):
    from mlx_spec.bench import diagnose, encode_chat, load_prompts, parse_config

    p = next(x for x in load_prompts() if x["id"] == prompt)
    ids = encode_chat(runner.tokenizer, p["prompt"])
    ref = runner.run(parse_config("greedy"), ids, 256).tokens
    failures = []
    for spec in CONFIGS:
        got = runner.run(parse_config(spec), ids, 256).tokens
        if got != ref:
            d = diagnose(runner, ids, ref, got)
            print(f"\n{prompt} {spec}: MISMATCH {d}")
            failures.append((spec, d))
        else:
            print(f"\n{prompt} {spec}: ok ({len(got)} tokens)")
    # bf16 logits are quantised to 1/8 at this magnitude, so exact or one-ulp
    # ties between the top two tokens happen every few hundred tokens and the
    # T-row verify pass can break them differently from the 1-row step. Those
    # are accepted; any mismatch with a clear margin is a bug.
    bugs = [(s, d) for s, d in failures if d.get("verdict") != "numerical near-tie"]
    assert not bugs, bugs
