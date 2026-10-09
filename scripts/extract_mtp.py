#!/usr/bin/env python3
"""Extract Qwen3.8-27B's MTP block (the 15 ``mtp.*`` tensors) from the bf16 HF
checkpoint into ``models/mtp-mlx/mtp.safetensors``, without loading the 54 GB.

Reads the safetensors index, then only the byte ranges of the ``mtp.*`` tensors
from their shard (all in model-00018). Applies the same conversion as mlx-lm's
``qwen3_5.sanitize`` / llama.cpp's converter: every ``*norm.weight`` is stored
zero-centred in the HF checkpoint (``Qwen3_5RMSNorm`` multiplies by ``1 + w``),
so we add 1 (in float32; norms are stored float32, matrices stay bf16).

    .venv/bin/python scripts/extract_mtp.py [--snapshot DIR] [--out models/mtp-mlx]
"""
import argparse
import glob
import json
import os
import struct
from pathlib import Path

import mlx.core as mx
import numpy as np

ROOT = Path(__file__).resolve().parent.parent


def default_snapshot():
    snaps = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*/model.safetensors.index.json"))
    assert snaps, "Qwen/Qwen3.8-27B not in the HF cache"
    return Path(snaps[0]).parent


def read_tensors(shard: Path, names):
    with open(shard, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name in names:
            meta = header[name]
            assert meta["dtype"] == "BF16", (name, meta["dtype"])
            a, b = meta["data_offsets"]
            f.seek(base + a)
            raw = np.frombuffer(f.read(b - a), dtype=np.uint16).reshape(meta["shape"])
            out[name] = mx.array(raw).view(mx.bfloat16)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "mtp-mlx")
    a = ap.parse_args()
    snap = a.snapshot or default_snapshot()
    wmap = json.load(open(snap / "model.safetensors.index.json"))["weight_map"]
    names = sorted(k for k in wmap if k.startswith("mtp."))
    print(f"{len(names)} mtp tensors in {sorted(set(wmap[k] for k in names))}")
    weights = {}
    for shard in sorted(set(wmap[k] for k in names)):
        weights.update(read_tensors(snap / shard, [k for k in names if wmap[k] == shard]))
    norms = {k for k in weights if k.endswith("norm.weight") or ".pre_fc_norm_" in k}
    for k in list(weights):
        v = weights[k]
        if k in norms:
            weights[k] = v.astype(mx.float32) + 1.0
        print(f"  {k:<50} {str(v.shape):<16} {weights[k].dtype}"
              + (f"  (raw mean {v.astype(mx.float32).mean().item():+.4f})" if k in norms else ""))
    a.out.mkdir(parents=True, exist_ok=True)
    mx.save_safetensors(str(a.out / "mtp.safetensors"), weights,
                        metadata={"source": str(snap), "note": "norm weights already +1 (float32)"})
    cfg = json.load(open(snap / "config.json"))
    json.dump(cfg["text_config"], open(a.out / "config.json", "w"), indent=1)
    print(f"wrote {a.out}/mtp.safetensors ({os.path.getsize(a.out / 'mtp.safetensors') / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
