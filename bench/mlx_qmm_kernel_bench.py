"""Microbenchmark: mlx_spec.qmm_kernel.qmm vs mx.quantized_matmul per row count
M = 1..16 on the real matmul shapes of the 27B models (read from the
checkpoints' safetensors headers; random weights of the same shape/quant).

    scripts/gpu.py .venv/bin/python bench/mlx_qmm_kernel_bench.py              # table -> results/mlx_qmm_kernel_bench.jsonl
    scripts/gpu.py .venv/bin/python bench/mlx_qmm_kernel_bench.py --tune       # sweep kernel configs per shape
    .venv/bin/python bench/mlx_qmm_kernel_bench.py --small                     # synthetic small shapes, no lock (preliminary)

Like bench/mlx_qmm_curve.py, each timed call runs the matmul over enough
distinct copies of the weight (~0.6 GB) that nothing is served from cache.
"""

import argparse
import glob
import json
import os
import statistics
import struct
import sys
import time
from collections import Counter
from pathlib import Path

import mlx.core as mx

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from mlx_spec import qmm_kernel as Q  # noqa: E402

TARGET_BYTES = 0.6e9


def model_shapes(snapshot: Path):
    """Counter of (N, K, bits, group_size) -> number of matmuls per forward pass."""
    cfg = json.load(open(snapshot / "config.json"))
    q = cfg.get("quantization", {})
    idx = json.load(open(snapshot / "model.safetensors.index.json"))["weight_map"]
    headers = {}
    for shard in set(idx.values()):
        with open(snapshot / shard, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            headers.update(json.loads(f.read(n)))
    shapes = Counter()
    names = {}
    for name in idx:
        if not name.endswith(".scales") or "vision" in name or "embed_tokens" in name:
            continue
        p = name[: -len(".scales")]
        per = q.get(p, {})
        bits = per.get("bits", q.get("bits")) if isinstance(per, dict) else q.get("bits")
        gs = per.get("group_size", q.get("group_size")) if isinstance(per, dict) else q.get("group_size")
        N, Kw = headers[p + ".weight"]["shape"]
        key = (N, Kw * 32 // bits, bits, gs)
        shapes[key] += 1
        names.setdefault(key, p.split("layers.")[-1].split(".", 1)[-1])
    return shapes, names


def snapshots():
    out = {}
    hf = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--mlx-community--Qwen3.8-27B-4bit/snapshots/*/config.json"))
    if hf:
        out["mlx-community 4bit g64"] = Path(hf[0]).parent
    local = ROOT / "models" / "qwen38-27b-q4km-mlx"
    if (local / "model.safetensors.index.json").exists():
        out["q4km transcode"] = local
    return out


def timed(f, reps):
    mx.eval(f())
    ts = []
    for _ in range(reps):
        mx.synchronize()
        t = time.perf_counter()
        mx.eval(f())
        ts.append(time.perf_counter() - t)
    return 1e3 * statistics.median(ts)


def make_weights(N, K, bits, gs, dtype=mx.bfloat16):
    nbytes = N * K * bits / 8 + N * (K // gs) * 4  # packed weights + bf16 scales and biases
    copies = max(1, int(TARGET_BYTES // nbytes))
    ws = [mx.quantize((mx.random.normal((N, K)) * 0.02).astype(dtype), group_size=gs, bits=bits)
          for _ in range(copies)]
    mx.eval(ws)
    gb = sum(w.nbytes + s.nbytes + b.nbytes for w, s, b in ws) / 1e9
    return ws, gb


def run_shape(N, K, bits, gs, Ms, reps, config=None):
    ws, gb = make_weights(N, K, bits, gs)
    rows = []
    for M in Ms:
        x = mx.random.normal((M, K)).astype(mx.bfloat16)
        mx.eval(x)
        t_mlx = timed(lambda: [mx.quantized_matmul(x, w, s, b, transpose=True, group_size=gs, bits=bits)
                               for w, s, b in ws], reps)
        t_ours = timed(lambda: [Q.qmm(x, w, s, b, group_size=gs, bits=bits, config=config)
                                for w, s, b in ws], reps)
        rows.append({"M": M, "mlx_ms": t_mlx / len(ws), "ours_ms": t_ours / len(ws), "gb": gb / len(ws)})
    del ws
    mx.clear_cache()
    return rows


def tune(shape, reps, Ms=(2, 4, 6, 8, 9, 12, 16)):
    N, K, bits, gs = shape
    ws, gb = make_weights(N, K, bits, gs)
    cfgs = [(SN, NSG, KS, KP) for SN in (1, 2, 4) for NSG in (2, 4, 8) for KS in (1, 2)
            for KP in (1, 2, 4, 8) if KP <= NSG and SN * (NSG // KP) <= 8 and gs >= 8 * KS
            and K % (32 * KS * KP) == 0]
    out = {}
    for M in Ms:
        x = mx.random.normal((M, K)).astype(mx.bfloat16)
        mx.eval(x)
        res = sorted((timed(lambda: [Q.qmm(x, w, s, b, group_size=gs, bits=bits, config=c)
                                     for w, s, b in ws], reps) / len(ws), c) for c in cfgs)
        t1 = timed(lambda: [mx.quantized_matmul(x[:1], w, s, b, transpose=True, group_size=gs, bits=bits)
                            for w, s, b in ws], reps) / len(ws)
        out[M] = res
        print(f"  tune N={N} K={K} b{bits} g{gs} M={M:<2} mlx M1 {1e3 * t1:7.1f} us | best "
              + ", ".join(f"{1e3 * t:6.1f} {c}" for t, c in res[:4])
              + f" | rule {1e3 * dict((c, t) for t, c in res).get(Q.pick_config(M, N, K, bits, gs), float('nan')):6.1f}",
              flush=True)
    del ws
    mx.clear_cache()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true")
    ap.add_argument("--small", action="store_true", help="synthetic small shapes (preliminary)")
    ap.add_argument("--reps", type=int, default=9)
    ap.add_argument("--out", default=str(ROOT / "results" / "mlx_qmm_kernel_bench.jsonl"))
    a = ap.parse_args()
    if a.small:
        global TARGET_BYTES
        TARGET_BYTES = 0.1e9
        models = {"synthetic": (Counter({(4096, 4096, 4, 64): 1, (1024, 4096, 8, 32): 1,
                                         (4096, 4096, 4, 32): 1}), {})}
    else:
        models = {tag: model_shapes(p) for tag, p in snapshots().items()}
    Ms = list(range(1, 17))
    rows = []
    tuned = {}
    for tag, (shapes, names) in models.items():
        print(f"== {tag}: {sum(shapes.values())} quantized matmuls per pass, {len(shapes)} distinct shapes"
              f" [power: {Q.power_source()}]")
        for shape, count in sorted(shapes.items(), key=lambda kv: -kv[0][0] * kv[0][1] * kv[1]):
            N, K, bits, gs = shape
            if a.tune:
                if N >= 256:
                    for M, res in tune(shape, a.reps).items():
                        tuned[f"{bits},{gs},{N},{K},{M}"] = list(res[0][1])
                continue
            res = run_shape(N, K, bits, gs, Ms, a.reps)
            base = res[0]["mlx_ms"]
            for r in res:
                row = {"experiment": "qmm_kernel_bench", "model": tag, "layer": names.get(shape, ""),
                       "N": N, "K": K, "bits": bits, "group_size": gs, "count_per_pass": count,
                       "M": r["M"], "mlx_ms": round(r["mlx_ms"], 4), "ours_ms": round(r["ours_ms"], 4),
                       "mlx_x_M1": round(r["mlx_ms"] / base, 3), "ours_x_M1": round(r["ours_ms"] / base, 3),
                       "mlx_GB_s": round(r["gb"] / r["mlx_ms"] * 1e3, 1),
                       "ours_GB_s": round(r["gb"] / r["ours_ms"] * 1e3, 1),
                       "config": list(Q.pick_config(r["M"], N, K, bits, gs)),
                       "preliminary": a.small, "power": Q.power_source(), "time": time.strftime("%F %T")}
                rows.append(row)
            print(f"{names.get(shape, ''):<28} N={N:<6} K={K:<6} b{bits} g{gs} x{count:<3} M=1 {1e3 * base:7.1f} us | "
                  "mlx/ours x M1: " + " ".join(f"{r['M']}:{r['mlx_ms'] / base:.2f}/{r['ours_ms'] / base:.2f}"
                                               for r in res), flush=True)
    if tuned and not a.small:
        with open(Q._TUNED_PATH, "w") as f:
            json.dump({"device": mx.device_info()["device_name"], "mlx": mx.__version__,
                       "power": Q.power_source(),
                       "time": time.strftime("%F %T"), "key": "bits,group_size,N,K,M -> [SN,NSG,KS,KP]",
                       "configs": tuned}, f, indent=0)
        print(f"wrote {len(tuned)} tuned configs to {Q._TUNED_PATH}")
    if rows:
        with open(a.out, "a") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        # whole-pass estimate: sum over layers of count x time
        for tag in models:
            rs = [r for r in rows if r["model"] == tag]
            print(f"\n{tag}: predicted matmul time per pass (ms), MLX vs ours (ours only for M>=2):")
            for M in Ms:
                tm = sum(r["mlx_ms"] * r["count_per_pass"] for r in rs if r["M"] == M)
                to = sum((r["ours_ms"] if M >= 2 else r["mlx_ms"]) * r["count_per_pass"] for r in rs if r["M"] == M)
                print(f"  M={M:<2} mlx {tm:7.1f}  ours {to:7.1f}")


if __name__ == "__main__":
    main()
