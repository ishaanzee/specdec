"""Exact-match check on the real target with float32 activations.

In bf16 the speculative output can differ from greedy at one-ulp logit ties
(see test_equivalence_27b.py). Upcasting every non-quantized parameter (and
the quantization scales/biases) to float32 makes the whole forward float32,
which shrinks the batched-vs-single-token noise by ~1000x, so any remaining
mismatch would point at the algorithm rather than rounding. Run on the
prompts that diverged in bf16:

    SPECDEC_27B=1 scripts/gpu.py .venv/bin/python -m pytest tests/test_fp32_27b.py -q -s
"""

import os

import pytest

pytestmark = pytest.mark.skipif(os.environ.get("SPECDEC_27B") != "1", reason="set SPECDEC_27B=1")

PROMPTS = ["code_gen", "summarize", "explain", "math"]
CONFIGS = ["ngram:k=4:n=2-4:rollback=replay", "ngram:k=4:n=2-4:rollback=capture", "draft:k=4",
           "mtp:k=3", "mtp-hybrid:k=3:vocab=65536"]
if os.environ.get("SPECDEC_CONFIGS"):  # e.g. SPECDEC_CONFIGS="mtp:k=3 mtp:k=4"
    CONFIGS = os.environ["SPECDEC_CONFIGS"].split()
MAX_TOKENS = 160


@pytest.fixture(scope="module")
def runner():
    import mlx.core as mx
    from mlx.utils import tree_map

    from mlx_spec.bench import Runner

    r = Runner()
    to32 = lambda p: p.astype(mx.float32) if p.dtype == mx.bfloat16 else p
    r.model.update(tree_map(to32, r.model.parameters()))
    mx.eval(r.model.parameters())
    return r


@pytest.mark.parametrize("prompt", PROMPTS)
def test_prompt_fp32(runner, prompt):
    from mlx_spec.bench import diagnose, encode_chat, load_prompts, parse_config

    p = next(x for x in load_prompts() if x["id"] == prompt)
    ids = encode_chat(runner.tokenizer, p["prompt"])
    ref = runner.run(parse_config("greedy"), ids, MAX_TOKENS).tokens
    bad = []
    for spec in CONFIGS:
        got = runner.run(parse_config(spec), ids, MAX_TOKENS).tokens
        status = "ok" if got == ref else f"MISMATCH {diagnose(runner, ids, ref, got)}"
        print(f"\n[fp32] {prompt} {spec}: {status}")
        if got != ref:
            bad.append(spec)
    assert not bad
