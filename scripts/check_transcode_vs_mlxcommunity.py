#!/usr/bin/env python3
"""Sanity check of the GGUF -> MLX name/layout mapping (CPU only, streaming).

Both checkpoints are quantizations of the same bf16 weights, so for every tensor
the transcoded Q4_K_M weights and mlx-community's 4-bit weights must agree up to
quantization noise (relative difference ~0.1), while any wrong permutation
(V-head order, q/gate interleave, conv layout, ...) gives ~1.4. Unquantized
tensors (norms, conv1d, dt_bias, A_log) must match to bf16 rounding.

    .venv/bin/python scripts/check_transcode_vs_mlxcommunity.py [--ours DIR] [--layers 0,1,2,3]
"""
import argparse
import glob
import json
import os
import re
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent


class ST:
    """Lazy reader for a sharded safetensors checkpoint."""

    def __init__(self, d):
        d = Path(d)
        idx = json.load(open(d / "model.safetensors.index.json"))["weight_map"]
        self.files, self.idx = {}, idx
        for fn in set(idx.values()):
            with open(d / fn, "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                self.files[fn] = (d / fn, json.loads(f.read(n)), 8 + n)

    def __contains__(self, k):
        return k in self.idx

    def get(self, k, rows=None):
        path, h, base = self.files[self.idx[k]]
        m = h[k]
        dt = {"BF16": np.uint16, "F32": np.float32, "U32": np.uint32, "F16": np.float16}[m["dtype"]]
        a = np.memmap(path, dtype=dt, mode="r", offset=base + m["data_offsets"][0], shape=tuple(m["shape"]))
        a = np.array(a[rows] if rows is not None else a)
        if m["dtype"] == "BF16":
            a = (a.astype(np.uint32) << 16).view(np.float32)
        return a


def dequant(st, name, rows, in_dim):
    w = st.get(name + ".weight", rows)
    s = st.get(name + ".scales", rows).astype(np.float64)
    b = st.get(name + ".biases", rows).astype(np.float64)
    bits = w.shape[-1] * 32 // in_dim
    gs = in_dim // s.shape[-1]
    u8 = np.ascontiguousarray(w).view(np.uint8)
    if bits == 4:
        q = np.stack([u8 & 15, u8 >> 4], -1).reshape(w.shape[0], in_dim)
    elif bits == 8:
        q = u8
    else:
        raise NotImplementedError(bits)
    q = q.reshape(w.shape[0], -1, gs).astype(np.float64)
    return (q * s[..., None] + b[..., None]).reshape(w.shape[0], in_dim)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ours", default=str(ROOT / "models" / "qwen38-27b-q4km-mlx"))
    ap.add_argument("--layers", default="all")
    ap.add_argument("--rows", type=int, default=2048, help="rows sampled per quantized tensor")
    a = ap.parse_args()
    ref = Path(glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/*"))[0])
    ours, theirs = ST(a.ours), ST(ref)
    cfg = json.load(open(Path(a.ours) / "config.json"))["text_config"]
    layers = range(cfg["num_hidden_layers"]) if a.layers == "all" else [int(x) for x in a.layers.split(",")]
    keys = sorted(k for k in ours.idx if not k.endswith((".scales", ".biases")))
    rng = np.random.default_rng(0)
    worst = {}
    missing = [k for k in theirs.idx if k.startswith("language_model") and k not in ours]
    extra = [k for k in ours.idx if k not in theirs]
    print(f"keys: ours {len(ours.idx)}, mlx-community language_model {sum(k.startswith('language_model') for k in theirs.idx)}, "
          f"missing {missing[:5]}, extra {extra[:5]}")
    for k in keys:
        m = re.search(r"layers\.(\d+)\.", k)
        if m and int(m.group(1)) not in layers:
            continue
        name = k[: -len(".weight")] if k.endswith(".weight") else k
        kind = re.sub(r"layers\.\d+\.", "", name)
        if name + ".scales" in ours:
            in_dim = ours.get(name + ".scales", slice(0, 1)).shape[-1] * 32  # our group size is 32
            R = ours.files[ours.idx[k]][1][k]["shape"][0]
            rows = np.sort(rng.choice(R, size=min(R, a.rows), replace=False))
            x, y = dequant(ours, name, rows, in_dim), dequant(theirs, name, rows, in_dim)
            rel = float(np.linalg.norm(x - y) / np.linalg.norm(y))
            corr = float(np.corrcoef(x.ravel(), y.ravel())[0, 1])
            stat = {"rel_diff": rel, "corr": corr}
        else:
            x, y = ours.get(k).astype(np.float64), theirs.get(k).astype(np.float64)
            assert x.shape == y.shape, (k, x.shape, y.shape)
            d = np.abs(x - y)
            ulp = np.abs(y) * 2.0 ** -8 + 1e-30  # one bf16 ulp is <= |y| * 2^-7, half-ulp 2^-8
            stat = {"max_abs": float(d.max()), "max_in_bf16_halfulps": float((d / ulp).max()),
                    "exact_frac": float((d == 0).mean())}
        w = worst.setdefault(kind, {"n": 0})
        w["n"] += 1
        for s, v in stat.items():
            if s == "corr":
                w[s] = min(w.get(s, 1.0), v)
            elif s == "exact_frac":
                w[s] = min(w.get(s, 1.0), v)
            else:
                w[s] = max(w.get(s, 0.0), v)
    for kind, w in sorted(worst.items()):
        print(f"{kind:55s} {json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in w.items()})}")


if __name__ == "__main__":
    main()
