#!/usr/bin/env python3
"""Transcode ggml-org's Qwen3.8-27B Q4_K_M GGUF into an mlx-lm ``qwen3_5`` checkpoint
with the same weights, so MLX runs exactly what llama.cpp runs.

Streaming, CPU only, never holds more than one tensor (or a row chunk of a big one).

Per GGUF type (``notes/mlx-matched-quant.md`` has the derivations and measured errors):

* Q4_K (256-superblocks, 8 x 32 sub-blocks, 6-bit sc/m, fp16 d/dmin):
  w = d*sc*q - dmin*m  ->  MLX affine 4-bit, group 32, scale = d*sc, bias = -dmin*m.
  Same q, bit for bit.
* Q8_0 (32-blocks, int8 q, fp16 d): w = d*q -> MLX affine 8-bit, group 32,
  q' = q + 128, scale = d, bias = -128*d. Same q.
* Q6_K (16-sub-blocks): MLX has no group 16 (32/64/128 only), so these tensors
  (lm_head, full-attention o_proj) are re-quantized from the exact Q6_K values to
  8-bit group 32 (``--q6k q8``, default) or stored bf16 (``--q6k bf16``).
* F32 (norms, conv1d, dt_bias, ssm_a): stored bf16 (all of them are bf16-exact,
  checked), except A_log = log(-ssm_a) in float32 (mlx-lm keeps A_log uncast).

Scales/biases are bf16, like mlx-community's 4-bit: fp16 scales would make
``quantized_matmul`` promote bf16 activations to fp32. d*sc is not always
representable in bf16, so per group (s', b') are picked from the bf16 neighbours of
(s, b + (s - s')*mean(q)) to minimise the group's squared error; the residual is
reported per tensor (``--check``).

Undone llama.cpp conversion steps (vendor/llama.cpp/conversion/qwen.py,
``_LinearAttentionVReorderBase`` and ``Qwen3NextModel``): V-head reorder of
in_proj_qkv's V rows, in_proj_z / in_proj_a / in_proj_b rows, out_proj columns,
conv1d V channels, A_log and dt_bias (tiled -> grouped order); ssm_a = -exp(A_log);
conv1d squeeze; ``+1`` on every RMSNorm except linear_attn.norm (kept: mlx-lm's
sanitize only adds it for raw HF checkpoints, recognised by the conv1d layout).

    .venv/bin/python scripts/gguf_to_mlx.py [--gguf F] [--out DIR] [--q6k q8|bf16] [--check]
"""
import argparse
import glob
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "vendor" / "llama.cpp" / "gguf-py"))
from gguf import GGUFReader, GGMLQuantizationType as QT  # noqa: E402
from gguf.quants import Q6_K, Q4_K, dequantize  # noqa: E402

HUB = Path(os.path.expanduser("~/.cache/huggingface/hub"))
GGUF = HUB / "models--ggml-org--Qwen3.8-27B-GGUF/snapshots/efbb3b1f70a21d97fd4495240648405f7228554f/Qwen3.8-27B-Q4_K_M.gguf"
OUT = ROOT / "models" / "qwen38-27b-q4km-mlx"
SHARD_BYTES = 5 << 30
CHUNK = 1 << 24  # elements per row chunk

# linear-attention dims (Qwen3.8-27B)
NK, NV, HK, HV = 16, 48, 128, 128


def mlx_ref_snapshot():
    return Path(glob.glob(str(HUB / "models--mlx-community--Qwen3.8-27B-4bit/snapshots/*"))[0])


# --------------------------------------------------------------------- bf16 helpers
def bf16_bits_rne(x):
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def bf16_to_f32(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def bf16_neighbours(x):
    """The two bf16 values bracketing float64 ``x`` (toward zero, away from zero), as float64."""
    f = x.astype(np.float32)
    u = f.view(np.uint32)
    lo = (u & 0xFFFF0000).view(np.float32).astype(np.float64)
    hi = ((u & 0xFFFF0000) + 0x10000).view(np.float32).astype(np.float64)
    # float32 rounding of x may have crossed a bf16 boundary; widen if x is outside [lo, hi]
    return lo, hi


def fit_bf16(q, s, b):
    """q [..., G, 32] codes, s/b [..., G] exact float64 scale/bias.
    Returns bf16 bit patterns (s', b') minimising sum_i ((s q_i + b) - (s' q_i + b'))^2."""
    qf = q.astype(np.float64)
    n = q.shape[-1]
    sq, sq2 = qf.sum(-1), (qf * qf).sum(-1)
    best, bs, bb = None, None, None
    for sc in bf16_neighbours(s):
        ds = s - sc
        bstar = b + ds * sq / n
        for bc in bf16_neighbours(bstar):
            db = b - bc
            sse = ds * ds * sq2 + 2 * ds * db * sq + n * db * db
            if best is None:
                best, bs, bb = sse, sc.copy(), bc.copy()
            else:
                m = sse < best
                best = np.where(m, sse, best)
                bs = np.where(m, sc, bs)
                bb = np.where(m, bc, bb)
    return bf16_bits_rne(bs), bf16_bits_rne(bb)


# --------------------------------------------------------------- GGUF block decoders
def q4k_decode(blocks):
    """blocks [N, 144] uint8 -> q [N, 8, 32] uint8, s [N, 8], b [N, 8] (float64, exact)."""
    n = blocks.shape[0]
    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float64)
    dmin = blocks[:, 2:4].copy().view(np.float16).astype(np.float64)
    sc, m = Q4_K.get_scale_min(blocks[:, 4:16])
    qs = blocks[:, 16:144].reshape(n, 4, 1, 32) >> np.array([0, 4], np.uint8).reshape(1, 1, 2, 1)
    q = (qs & 0x0F).reshape(n, 8, 32)
    return q, d * sc.astype(np.float64), -dmin * m.astype(np.float64)


def q80_decode(blocks):
    """blocks [N, 34] -> q' [N, 1, 32] uint8 (q+128), s, b [N, 1]."""
    d = blocks[:, 0:2].copy().view(np.float16).astype(np.float64)
    q = blocks[:, 2:34].copy().view(np.int8).astype(np.int16) + 128
    return q.astype(np.uint8)[:, None, :], d, -128.0 * d


def requant8(w):
    """w [R, K] float32 (exact Q6_K values) -> 8-bit group-32 codes and bf16 (s, b) bits."""
    R, K = w.shape
    g = w.reshape(R, K // 32, 32).astype(np.float64)
    mn, mxv = g.min(-1), g.max(-1)
    # bias = bf16 at or below the minimum, scale = bf16 at or above (max - bias)/255:
    # every value then has a code in 0..255 and the error is <= scale/2 (no clipping).
    lo, hi = bf16_neighbours(mn)
    a, b = lo <= mn, hi <= mn
    bb = np.where(a & b, np.maximum(lo, hi), np.where(a, lo, hi))
    s = (mxv - bb) / 255.0
    lo, hi = bf16_neighbours(s)
    a, b = lo >= s, hi >= s
    sb = np.where(a & b, np.minimum(lo, hi), np.where(a, lo, hi))
    assert (sb >= s).all() and (bb <= mn).all()
    safe = np.where(sb > 0, sb, 1.0)
    q = np.clip(np.rint((g - bb[..., None]) / safe[..., None]), 0, 255).astype(np.uint8)
    return q.reshape(R, K), bf16_bits_rne(sb), bf16_bits_rne(bb)


def pack(q, bits):
    """q [R, K] uint8 codes -> MLX packed uint32 [R, K*bits/32] (element i at bits*(i % (32/bits)))."""
    if bits == 8:
        return np.ascontiguousarray(q).view(np.uint32)
    assert bits == 4
    lo, hi = q[:, 0::2], q[:, 1::2]
    return np.ascontiguousarray((lo | (hi << 4)).astype(np.uint8)).view(np.uint32)


# ------------------------------------------------------------------- name mapping
def v_tiled_to_grouped(n_heads_v, head_dim):
    """Index map: grouped[i] = tiled[perm[i]] (inverse of llama.cpp's _reorder_v_heads)."""
    per = n_heads_v // NK
    # tiled order index (vi, ki, d) ; grouped order index (ki, vi, d)
    ki, vi, d = np.meshgrid(np.arange(NK), np.arange(per), np.arange(head_dim), indexing="ij")
    return ((vi * NK + ki) * head_dim + d).reshape(-1)


def row_perm(gname, rows):
    if gname.endswith("attn_qkv.weight"):
        qk = 2 * NK * HK
        return np.concatenate([np.arange(qk), qk + v_tiled_to_grouped(NV, HV)])
    if gname.endswith("attn_gate.weight"):
        return v_tiled_to_grouped(NV, HV)
    if gname.endswith(("ssm_alpha.weight", "ssm_beta.weight")):
        return v_tiled_to_grouped(NV, 1)
    return None


def col_perm(gname):
    """Permutation of the input dim (in units of 32-element groups), or None."""
    if gname.endswith("ssm_out.weight"):
        p = v_tiled_to_grouped(NV, HV)  # element-level, head blocks of 128 -> group level
        g = p.reshape(-1, 32)[:, 0] // 32
        assert np.array_equal(p, (g[:, None] * 32 + np.arange(32)).reshape(-1))
        return g
    return None


LAYER_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_gate.weight": "mlp.gate_proj",
    "ffn_up.weight": "mlp.up_proj",
    "ffn_down.weight": "mlp.down_proj",
    "attn_qkv.weight": "linear_attn.in_proj_qkv",
    "attn_gate.weight": "linear_attn.in_proj_z",
    "ssm_alpha.weight": "linear_attn.in_proj_a",
    "ssm_beta.weight": "linear_attn.in_proj_b",
    "ssm_out.weight": "linear_attn.out_proj",
    "ssm_a": "linear_attn.A_log",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "attn_q.weight": "self_attn.q_proj",
    "attn_k.weight": "self_attn.k_proj",
    "attn_v.weight": "self_attn.v_proj",
    "attn_output.weight": "self_attn.o_proj",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
}


def mlx_name(gname):
    if gname == "token_embd.weight":
        return "language_model.model.embed_tokens"
    if gname == "output.weight":
        return "language_model.lm_head"
    if gname == "output_norm.weight":
        return "language_model.model.norm.weight"
    m = re.match(r"blk\.(\d+)\.(.+)$", gname)
    return f"language_model.model.layers.{m.group(1)}.{LAYER_MAP[m.group(2)]}"


# ---------------------------------------------------------------------- planning
class Item:
    """One GGUF tensor -> one or three MLX tensors."""

    def __init__(self, t, q6k):
        self.t, self.gname = t, t.name
        self.name = mlx_name(t.name)
        self.ty = t.tensor_type
        shape = [int(x) for x in reversed(t.shape)]  # numpy order [rows, K]
        self.shape = shape
        if self.ty in (QT.F32, QT.F16):
            self.bits = None
            if self.gname.endswith("ssm_conv1d.weight"):
                self.outs = [(self.name, "BF16", [shape[0], shape[1], 1])]
            elif self.gname.endswith("ssm_a"):
                self.outs = [(self.name, "F32", shape)]
            else:
                self.outs = [(self.name, "BF16", shape)]
            return
        R, K = shape
        if self.ty == QT.Q4_K:
            self.bits = 4
        elif self.ty == QT.Q8_0:
            self.bits = 8
        elif self.ty == QT.Q6_K:
            self.bits = 8 if q6k == "q8" else None
        else:
            raise NotImplementedError(f"{self.gname}: {self.ty.name}")
        if self.bits is None:
            self.outs = [(self.name + ".weight", "BF16", [R, K])]
        else:
            self.outs = [(self.name + ".weight", "U32", [R, K * self.bits // 32]),
                         (self.name + ".scales", "BF16", [R, K // 32]),
                         (self.name + ".biases", "BF16", [R, K // 32])]

    def nbytes(self):
        sz = {"BF16": 2, "F32": 4, "U32": 4}
        return sum(sz[dt] * int(np.prod(sh)) for _, dt, sh in self.outs)


# ----------------------------------------------------------------------- convert
def convert_dense(it):
    a = np.asarray(it.t.data).reshape(it.shape)
    a = a.astype(np.float32) if it.ty == QT.F16 else np.array(a, dtype=np.float32)
    g = it.gname
    stats = {"bf16_exact": bool(np.array_equal(bf16_to_f32(bf16_bits_rne(a)), a))}
    if g.endswith(("ssm_a", "ssm_dt.bias")):
        a = a[v_tiled_to_grouped(NV, 1)]
    if g.endswith("ssm_conv1d.weight"):
        qk = 2 * NK * HK
        a = np.concatenate([a[:qk], a[qk:][v_tiled_to_grouped(NV, HV)]], 0)[:, :, None]
    if g.endswith("ssm_a"):
        assert (a < 0).all()
        return {it.name: np.log(-a.astype(np.float64)).astype(np.float32)}, stats
    return {it.name: bf16_bits_rne(a)}, stats


def convert_quant_rows(it, src_rows):
    """Convert source rows (already permuted) -> (packed, sbits, bbits) for these rows."""
    R = src_rows.shape[0]
    K = it.shape[1]
    if it.ty == QT.Q6_K:
        w = dequantize(src_rows, QT.Q6_K).reshape(R, K)
        if it.bits is None:
            return (bf16_bits_rne(w),)
        q, sb, bb = requant8(w)
    else:
        bsz = 144 if it.ty == QT.Q4_K else 34
        blocks = src_rows.reshape(-1, bsz)
        q, s, b = (q4k_decode if it.ty == QT.Q4_K else q80_decode)(blocks)
        G = K // 32
        q = q.reshape(R, G, 32)
        s, b = s.reshape(R, G), b.reshape(R, G)
        cp = col_perm(it.gname)
        if cp is not None:
            q, s, b = q[:, cp], s[:, cp], b[:, cp]
        sb, bb = fit_bf16(q, s, b)
        q = q.reshape(R, K)
    return pack(q, it.bits), sb, bb


def check_rows(it, src_rows, outs, rperm_rows=None):
    """Dequantize our MLX tensor with mx.dequantize (CPU) and compare with gguf's
    dequantization of the same rows (after our permutations). Returns stats."""
    import mlx.core as mx

    R, K = src_rows.shape[0], it.shape[1]
    ref = dequantize(src_rows, it.ty).reshape(R, K).astype(np.float64)
    cp = col_perm(it.gname)
    if cp is not None:
        ref = ref.reshape(R, K // 32, 32)[:, cp].reshape(R, K)
    if len(outs) == 1:
        ours = bf16_to_f32(outs[0]).astype(np.float64)
        scale = np.abs(ref).max(-1, keepdims=True) + 1e-30
    else:
        w, sb, bb = outs
        ours = np.array(mx.dequantize(mx.array(w), mx.array(bf16_to_f32(sb)), mx.array(bf16_to_f32(bb)),
                                      group_size=32, bits=it.bits, stream=mx.cpu)).astype(np.float64)
        scale = np.repeat(np.abs(bf16_to_f32(sb)).astype(np.float64), 32, axis=-1) + 1e-30
    err = ours - ref
    return {"max_abs": float(np.abs(err).max()), "max_rel_step": float((np.abs(err) / scale).max()),
            "sse": float((err * err).sum()), "ss": float((ref * ref).sum()),
            "exact_frac": float((err == 0).mean()), "n": int(err.size)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", default=str(GGUF))
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--q6k", choices=["q8", "bf16"], default="q8")
    ap.add_argument("--check", action="store_true", help="compare every tensor with gguf's dequantization")
    ap.add_argument("--only", default=None, help="regex on GGUF names (debug; writes nothing)")
    a = ap.parse_args()

    r = GGUFReader(a.gguf)
    items = [Item(t, a.q6k) for t in r.tensors if not (a.only and not re.search(a.only, t.name))]
    # order: embed, layers by index, norm, lm_head
    def key(it):
        m = re.match(r"blk\.(\d+)\.", it.gname)
        return (0, -1) if it.gname == "token_embd.weight" else (1, int(m.group(1))) if m else (2, it.gname != "output_norm.weight")
    items.sort(key=key)
    out = Path(a.out)
    write = a.only is None
    total = sum(it.nbytes() for it in items)
    print(f"{len(items)} GGUF tensors -> {sum(len(it.outs) for it in items)} MLX tensors, {total / 1e9:.2f} GB", flush=True)
    if write:
        free = shutil.disk_usage(out.parent).free
        assert free > total + (10 << 30), f"only {free / 1e9:.1f} GB free"
        out.mkdir(parents=True, exist_ok=True)

    # shard plan
    shards, cur, cur_b = [], [], 0
    for it in items:
        if cur and cur_b + it.nbytes() > SHARD_BYTES:
            shards.append(cur)
            cur, cur_b = [], 0
        cur.append(it)
        cur_b += it.nbytes()
    shards.append(cur)
    weight_map, report = {}, []
    t0 = time.time()
    for si, shard in enumerate(shards):
        fname = f"model-{si + 1:05d}-of-{len(shards):05d}.safetensors"
        header, off = {"__metadata__": {"format": "mlx"}}, 0
        sz = {"BF16": 2, "F32": 4, "U32": 4}
        for it in shard:
            for n, dt, sh in it.outs:
                nb = sz[dt] * int(np.prod(sh))
                header[n] = {"dtype": dt, "shape": sh, "data_offsets": [off, off + nb]}
                weight_map[n] = fname
                off += nb
        hb = json.dumps(header, separators=(",", ":")).encode()
        hb += b" " * (-len(hb) % 8)
        base = 8 + len(hb)
        fd = None
        if write:
            fd = os.open(out / fname, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            os.pwrite(fd, len(hb).to_bytes(8, "little") + hb, 0)
            os.ftruncate(fd, base + off)

        def put(name, arr, row0=0):
            if fd is None:
                return
            h = header[name]
            rowbytes = (h["data_offsets"][1] - h["data_offsets"][0]) // h["shape"][0]
            data = np.ascontiguousarray(arr).tobytes()
            os.pwrite(fd, data, base + h["data_offsets"][0] + row0 * rowbytes)

        for it in shard:
            rec = {"gguf": it.gname, "type": it.ty.name, "mlx": it.name, "shape": it.shape,
                   "bits": it.bits}
            if it.ty in (QT.F32, QT.F16):
                res, st = convert_dense(it)
                rec.update(st)
                for n, v in res.items():
                    put(n, v)
            else:
                R, K = it.shape
                raw = it.t.data  # memmap [R, bytes per row]
                rp = row_perm(it.gname, R)
                step = max(1, CHUNK // K)
                agg = None
                for r0 in range(0, R, step):
                    idx = np.arange(r0, min(R, r0 + step))
                    src = np.asarray(raw[rp[idx] if rp is not None else slice(r0, idx[-1] + 1)])
                    outs = convert_quant_rows(it, src)
                    names = [n for n, _, _ in it.outs]
                    for n, v in zip(names, outs):
                        put(n, v, r0)
                    if a.check:
                        st = check_rows(it, src, outs)
                        if agg is None:
                            agg = st
                        else:
                            agg = {k: (max(agg[k], st[k]) if k.startswith("max") else
                                       agg[k] + st[k] if k in ("sse", "ss", "n") else
                                       (agg[k] * agg["n"] + st[k] * st["n"]) / (agg["n"] + st["n"]) if k == "exact_frac" else agg[k])
                                   for k in agg}
                if agg is not None:
                    agg["rel_rms"] = (agg["sse"] / agg["ss"]) ** 0.5
                    rec.update(agg)
            report.append(rec)
            if a.check or a.only:
                print(json.dumps(rec), flush=True)
        if fd is not None:
            os.close(fd)
        print(f"[{time.time() - t0:6.0f}s] wrote {fname} ({off / 1e9:.2f} GB)", flush=True)

    if not write:
        return
    with open(out / "model.safetensors.index.json", "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": dict(sorted(weight_map.items()))}, f, indent=2)
    with open(out / "convert_report.jsonl", "w") as f:
        for rec in report:
            f.write(json.dumps(rec) + "\n")

    # config: mlx-community's, with per-module quantization
    ref = mlx_ref_snapshot()
    cfg = json.load(open(ref / "config.json"))
    quant = {"group_size": 32, "bits": 4, "mode": "affine"}
    for it in items:
        if it.bits is not None:
            quant[it.name] = {"group_size": 32, "bits": it.bits, "mode": "affine"}
        elif it.ty == QT.Q6_K:
            quant[it.name] = False
    cfg["quantization"] = quant
    cfg["quantization_config"] = quant
    cfg["source"] = {"gguf": os.path.basename(a.gguf), "q6k": a.q6k,
                     "converter": "scripts/gguf_to_mlx.py"}
    with open(out / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)
    for fn in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "chat_template.jinja",
               "generation_config.json"]:
        if (ref / fn).exists():
            shutil.copyfile(ref / fn, out / fn)
    print(f"done in {time.time() - t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
