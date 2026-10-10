"""Correctness of mlx_spec.qmm_kernel.qmm against mx.quantized_matmul.

Every supported (bits, group_size, M, dtype) on random weights of a few
shapes, plus real layer weights of the 27B checkpoints when they are on disk.
Both kernels are compared with an fp32 reference (x and the dequantized
weights in float32); ours must be about as close to it as MLX's own kernel
(they differ only in accumulation order and in where the output is rounded).

    .venv/bin/python -m pytest tests/test_qmm_kernel.py -q        # small GPU work, ~1 min
"""

import glob
import json
import os
import struct
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from mlx_spec import qmm_kernel as Q

EPS = {mx.bfloat16: 2.0**-8, mx.float16: 2.0**-11, mx.float32: 2.0**-24}
DTYPES = [mx.bfloat16, mx.float16, mx.float32]


def errors(x, w, s, b, gs, bits):
    ref = x.astype(mx.float32) @ mx.dequantize(w, s, b, group_size=gs, bits=bits).astype(mx.float32).T
    ours = Q.qmm(x, w, s, b, group_size=gs, bits=bits)
    theirs = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=gs, bits=bits)
    assert ours.shape == theirs.shape and ours.dtype == theirs.dtype
    scale = mx.abs(ref).max().item()
    e_ours = mx.abs(ours.astype(mx.float32) - ref).max().item()
    e_mlx = mx.abs(theirs.astype(mx.float32) - ref).max().item()
    return e_ours, e_mlx, scale


def check(x, w, s, b, gs, bits):
    e_ours, e_mlx, scale = errors(x, w, s, b, gs, bits)
    # output rounding of the dtype dominates for bf16/fp16; fp32 sees accumulation order
    eps = EPS[x.dtype] if x.dtype != mx.float32 else 2.0**-20
    assert e_ours <= max(1.5 * e_mlx, eps * scale), (e_ours, e_mlx, scale)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("gs", [32, 64, 128])
@pytest.mark.parametrize("dtype", DTYPES, ids=["bf16", "fp16", "fp32"])
def test_random(bits, gs, dtype):
    mx.random.seed(bits * 1000 + gs)
    for N, K in [(520, 1024), (52, 2048)]:  # N not a multiple of the 8/16-row tiles too
        w, s, b = mx.quantize((mx.random.normal((N, K)) * 0.02).astype(dtype), group_size=gs, bits=bits)
        for M in range(1, 17):
            x = mx.random.normal((M, K)).astype(dtype)
            check(x, w, s, b, gs, bits)


def test_leading_dims_and_configs():
    mx.random.seed(3)
    w, s, b = mx.quantize(mx.random.normal((256, 512)).astype(mx.bfloat16) * 0.02, group_size=64, bits=4)
    x = mx.random.normal((1, 7, 512)).astype(mx.bfloat16)
    ref = mx.quantized_matmul(x, w, s, b, transpose=True, group_size=64, bits=4)
    for cfg in [None, (1, 2, 1, 1), (2, 4, 1, 4), (4, 8, 2, 2), (1, 4, 2, 4)]:
        y = Q.qmm(x, w, s, b, group_size=64, bits=4, config=cfg)
        assert y.shape == (1, 7, 256)
        assert mx.allclose(y, ref, atol=2e-2, rtol=2e-2).item(), cfg


def test_hook_routes_by_rows():
    mx.random.seed(4)
    lin = nn.QuantizedLinear(512, 256, bias=True, group_size=64, bits=4)
    lin.update({"bias": mx.random.normal((256,))})
    mx.eval(lin.parameters())
    model = nn.Sequential(lin)
    xs = {M: mx.random.normal((1, M, 512)) for M in (1, 2, 9, 16, 17)}
    stock = {M: model(x) for M, x in xs.items()}
    assert Q.install(model, "2-16") == 1 and Q.installed_range(model) == (2, 16)
    assert isinstance(lin, Q.FastQuantizedLinear)
    for M, x in xs.items():
        y = model(x)
        if 2 <= M <= 16:
            want = Q.qmm(x, lin["weight"], lin["scales"], lin["biases"], 64, 4) + lin["bias"]
            assert mx.array_equal(y, want).item()
        else:
            assert mx.array_equal(y, stock[M]).item()  # untouched path
        assert mx.allclose(y, stock[M], atol=1e-4, rtol=1e-4).item()
    assert "_qmm_lo" not in lin and len(lin.parameters()) == 4  # no stray child entries
    assert Q.uninstall(model) == 1 and type(lin) is nn.QuantizedLinear
    assert Q.install(model, "2-16", min_n=512) == 0  # N = 256 < min_n stays on MLX
    assert Q.parse_range("off") is None and Q.parse_range("8") == (2, 8)


# ------------------------------------------------------------ real weights
def _read(shard: Path, names):
    """Raw tensors from a safetensors shard without loading the file."""
    dt = {"U32": (np.uint32, None), "BF16": (np.uint16, mx.bfloat16), "F16": (np.float16, None),
          "F32": (np.float32, None)}
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        out = {}
        for name in names:
            meta = header[name]
            np_t, view = dt[meta["dtype"]]
            a, b = meta["data_offsets"]
            f.seek(8 + n + a)
            arr = mx.array(np.frombuffer(f.read(b - a), dtype=np_t).reshape(meta["shape"]))
            out[name] = arr.view(view) if view is not None else arr
    return out


def _real_layers(snapshot: Path, prefixes, max_layers=8):
    idx = json.load(open(snapshot / "model.safetensors.index.json"))["weight_map"]
    cfg = json.load(open(snapshot / "config.json"))
    q = cfg.get("quantization", cfg.get("quantization_config", {}))
    found = []
    for p in prefixes:
        if p + ".scales" not in idx:
            continue
        t = _read(snapshot / idx[p + ".weight"], [p + ".weight"])
        t.update(_read(snapshot / idx[p + ".scales"], [p + ".scales", p + ".biases"]))
        per = q.get(p, q.get(p.replace("language_model.", ""), {}))
        gs = per.get("group_size", q.get("group_size", 64)) if isinstance(per, dict) else q.get("group_size", 64)
        bits = per.get("bits", q.get("bits", 4)) if isinstance(per, dict) else q.get("bits", 4)
        found.append((p, t[p + ".weight"], t[p + ".scales"], t[p + ".biases"], gs, bits))
        if len(found) >= max_layers:
            break
    return found


def _snapshots():
    out = []
    hf = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/*/config.json"))
    if hf:
        out.append(Path(hf[0]).parent)
    local = Path(__file__).resolve().parent.parent / "models" / "qwen38-27b-q4km-mlx"
    if (local / "model.safetensors.index.json").exists() and (local / "config.json").exists():
        out.append(local)
    return out


LAYER_NAMES = [
    "language_model.model.layers.{i}.mlp.down_proj",
    "language_model.model.layers.{i}.mlp.up_proj",
    "language_model.model.layers.{i}.linear_attn.in_proj_qkv",
    "language_model.model.layers.{i}.linear_attn.in_proj_a",
    "language_model.model.layers.{i}.linear_attn.out_proj",
    "language_model.model.layers.{j}.self_attn.q_proj",
    "language_model.model.layers.{j}.self_attn.k_proj",
    "language_model.model.layers.{j}.self_attn.o_proj",
]


@pytest.mark.parametrize("snapshot", _snapshots(), ids=lambda p: p.parent.parent.name if "snapshots" in str(p) else p.name)
def test_real_weights(snapshot):
    names = [n.format(i=0, j=3) for n in LAYER_NAMES]
    layers = _real_layers(snapshot, names)
    if not layers:
        pytest.skip(f"no affine-quantized layers found in {snapshot}")
    mx.random.seed(5)
    for name, w, s, b, gs, bits in layers:
        if not Q.supported(s.dtype, w, s, b, gs, bits):
            continue
        K = w.shape[1] * 32 // bits
        for M in (2, 4, 5, 8, 9, 12, 16):
            x = (mx.random.normal((M, K)) * 0.5).astype(s.dtype)
            e_ours, e_mlx, scale = errors(x, w, s, b, gs, bits)
            assert e_ours <= max(1.5 * e_mlx, EPS[s.dtype] * scale), (name, M, e_ours, e_mlx, scale)


# ------------------------------------------- end to end on a tiny quantized model
def test_spec_equals_greedy_tiny_quantized():
    """Tiny random qwen3_5, 4-bit (fp32 activations, scales and biases): with the
    kernel routing every 2..16-row pass, speculative == greedy for k = 1..15."""
    from test_tiny import NoisyOracle, prompt, tiny_model

    from mlx_spec import NgramDrafter, SpecModel, greedy_generate, speculative_generate

    m = tiny_model(seed=2)
    nn.quantize(m, group_size=64, bits=4)
    mx.eval(m.parameters())
    ids = prompt(24)
    ref = greedy_generate(m, ids, 40).tokens
    spec = SpecModel(m, qmm="2-16")
    assert Q.installed_range(m) == (2, 16)
    for k in (1, 3, 7, 15):
        for d in (NgramDrafter(4, 1), NoisyOracle(ref, 0.3, seed=k), NoisyOracle(ref, 0.0)):
            got = speculative_generate(spec, d, ids, k=k, max_tokens=40).tokens
            assert got == ref, (k, type(d).__name__)
    Q.uninstall(m)
