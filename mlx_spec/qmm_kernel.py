"""Small-batch quantized matmul that reads every weight once for M = 2..16 rows.

``qmm(x, w, scales, biases, group_size, bits)`` computes what
``mx.quantized_matmul(x, w, scales, biases, transpose=True, ...)`` computes
for affine quantization (bits 4 or 8; group size 32, 64 or 128; bf16, fp16 or
fp32 activations with scales/biases of the same dtype). It is meant for
the rows of a speculative verify pass (M = k + 1 <= 16). MLX 0.32.3 on an M3
Pro reads the weights once only up to 5 rows and then re-reads them per tile
of <= 5 rows, then switches to a 32-row matmul tile at 13 rows
(``notes/mlx-qmm-kernel.md``).

Kernel (``_source_mma``): every simdgroup owns ``8 * SN`` output features and
all M rows (padded to 8 or 16) and walks K in chunks of ``32 * KS``. Per chunk
each lane loads its contiguous slice of the weights (one uint32 = 8 4-bit
values per row), dequantizes it in fp32 (``scale * q + bias``, as MLX's qmv
kernels do) straight into its two elements of 8x8 ``simdgroup_matrix`` A
fragments, loads the matching x elements into B fragments, and accumulates
``C[n, m] += W[n, k] X[m, k]`` with ``simdgroup_multiply_accumulate`` in fp32.
The weights are read once whatever M is; the 8x8x8 MMAs keep the M-fold
arithmetic cheap enough that the kernel stays bandwidth-bound to ~8-10 rows.
``KP`` simdgroups of a threadgroup split K for the same rows and reduce
through threadgroup memory at the end (more simdgroups in flight for short N).

``_source_simd`` is the first version, a qmv_fast-style kernel with M x R fp32
accumulators per lane and ``simd_sum`` at the end. It is ALU-bound from M ~ 4
on (scalar FMAs, x reloaded per row) and kept only for the comparison in the
notes (``qmm_simd``).

Opt-in hook: ``install(model, (2, 16))`` swaps the class of every eligible
``nn.QuantizedLinear`` in ``model`` to ``FastQuantizedLinear``, which sends
calls with 2 <= rows <= 16 to ``qmm`` and everything else (decode steps,
prefill) to ``mx.quantized_matmul``; ``uninstall(model)`` reverts. The env var
``SPECDEC_QMM=2-16`` makes ``SpecModel`` install it, and the benchmark config
key ``qmm=2-16`` (e.g. ``mtp:k=6:vocab=65536:qmm=2-16``) toggles it per run.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Dict, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

_KERNELS: Dict[Tuple, object] = {}


def _h(src: str) -> str:
    # metal_kernel caches compiled libraries by name, so the name must change with the source
    return hashlib.sha1(src.encode()).hexdigest()[:10]


# --------------------------------------------------------------- MMA kernel
_MMA_HEADER = """
#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
// D += A * B for 8x8 fragments held as each lane's two elements (MLX steel's
// layout: row fm, columns fn and fn + 1). Keeping fragments in plain float2
// registers and building the simdgroup_matrix values only here (as steel's
// mma() does) matters: writing thread_elements() of matrix arrays directly
// made the kernel ~4x slower.
inline void mma8(thread float2& D, float2 A, float2 B) {
    simdgroup_float8x8 Am, Bm, Dm;
    reinterpret_cast<thread float2&>(Am.thread_elements()) = A;
    reinterpret_cast<thread float2&>(Bm.thread_elements()) = B;
    reinterpret_cast<thread float2&>(Dm.thread_elements()) = D;
    simdgroup_multiply_accumulate(Dm, Am, Bm, Dm);
    D = reinterpret_cast<thread float2&>(Dm.thread_elements());
}
"""


def _source_mma(M: int, bits: int, group_size: int, SN: int, NSG: int, KS: int, KP: int) -> str:
    """C[n, m] += W[n, k] X^T[k, m] in 8x8x8 simdgroup MMAs.

    The 8 k-columns of each A/B fragment are a permutation of k chosen so that
    each lane's W slice is contiguous: within a chunk of 32*KS k values,
    fragment f column c <-> k = (c/2)*8*KS + 2f + c%2. A lane (fn = 0,2,4,6)
    then holds k = (fn/2)*8*KS .. +8*KS of its row, i.e. KS uint32 words of
    4-bit weights, all in one quantization group (group_size >= 8*KS).
    """
    return f"""
    constexpr int M = {M};
    constexpr int SM = (M + 7) / 8;     // 8-row m tiles
    constexpr int SN = {SN};            // 8-row n tiles per simdgroup
    constexpr int NSG = {NSG};          // simdgroups per threadgroup
    constexpr int KP = {KP};            // simdgroups splitting K for the same rows
    constexpr int RG = NSG / KP;        // row groups per threadgroup
    constexpr int KS = {KS};            // 32-wide k steps per chunk
    constexpr int BITS = {bits};
    constexpr int GS = {group_size};
    constexpr int CH = 32 * KS;
    constexpr int VPT = 8 * KS;         // weights per lane per row per chunk
    constexpr int PER_WORD = 32 / BITS;
    constexpr int WPT = VPT / PER_WORD;
    constexpr uint MASK = (1u << BITS) - 1u;

    const int K = x_shape[1];
    const int N = w_shape[0];
    const int Kw = K / PER_WORD;
    const int Kg = K / GS;
    const ushort lane = thread_index_in_simdgroup;
    const ushort sg = simdgroup_index_in_threadgroup;
    const short qid = lane / 4;
    const short fm = (qid & 4) + ((lane / 2) % 4);
    const short fn = (qid & 2) * 2 + (lane % 2) * 2;
    const int rg = sg / KP;
    const int kp = sg % KP;
    const int nbase = (int(threadgroup_position_in_grid.y) * RG + rg) * SN * 8;

    float2 C[SN][SM];
#pragma unroll
    for (int t = 0; t < SN; t++)
#pragma unroll
        for (int mt = 0; mt < SM; mt++) C[t][mt] = float2(0.0f);

    const device uint32_t* wr[SN];
    const device T* sr[SN];
    const device T* br[SN];
#pragma unroll
    for (int t = 0; t < SN; t++) {{
        int row = min(nbase + t * 8 + fm, N - 1);
        wr[t] = w + row * Kw + (fn / 2) * WPT;
        sr[t] = scales + row * Kg;
        br[t] = biases + row * Kg;
    }}
    const device T* xr[SM][2];
#pragma unroll
    for (int mt = 0; mt < SM; mt++)
#pragma unroll
        for (int e = 0; e < 2; e++)
            xr[mt][e] = x + min(mt * 8 + fn + e, M - 1) * K + (fm / 2) * VPT + (fm % 2);

    for (int k0 = kp * CH; k0 < K; k0 += KP * CH) {{
        float2 B[SM][4 * KS];
#pragma unroll
        for (int mt = 0; mt < SM; mt++)
#pragma unroll
            for (int f = 0; f < 4 * KS; f++)
                B[mt][f] = float2(float(xr[mt][0][k0 + 2 * f]), float(xr[mt][1][k0 + 2 * f]));
        const int g = (k0 + (fn / 2) * VPT) / GS;
#pragma unroll
        for (int t = 0; t < SN; t++) {{
            float s = float(sr[t][g]);
            float b = float(br[t][g]);
            uint32_t q[WPT];
#pragma unroll
            for (int j = 0; j < WPT; j++) q[j] = wr[t][k0 / PER_WORD + j];
            float2 A[4 * KS];
#pragma unroll
            for (int f = 0; f < 4 * KS; f++) {{
                const int v0 = 2 * f, v1 = 2 * f + 1;
                A[f] = float2(
                    s * float((q[v0 / PER_WORD] >> (BITS * (v0 % PER_WORD))) & MASK) + b,
                    s * float((q[v1 / PER_WORD] >> (BITS * (v1 % PER_WORD))) & MASK) + b);
            }}
#pragma unroll
            for (int mt = 0; mt < SM; mt++)
#pragma unroll
                for (int f = 0; f < 4 * KS; f++) mma8(C[t][mt], A[f], B[mt][f]);
        }}
    }}

    if (KP > 1) {{
        threadgroup float2 red[RG][KP > 1 ? KP - 1 : 1][SN][SM][32];
        if (kp > 0) {{
#pragma unroll
            for (int t = 0; t < SN; t++)
#pragma unroll
                for (int mt = 0; mt < SM; mt++) red[rg][kp - 1][t][mt][lane] = C[t][mt];
        }}
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (kp > 0) return;
#pragma unroll
        for (int i = 0; i < KP - 1; i++)
#pragma unroll
            for (int t = 0; t < SN; t++)
#pragma unroll
                for (int mt = 0; mt < SM; mt++) C[t][mt] += red[rg][i][t][mt][lane];
    }}
#pragma unroll
    for (int t = 0; t < SN; t++) {{
        int n = nbase + t * 8 + fm;
#pragma unroll
        for (int mt = 0; mt < SM; mt++) {{
#pragma unroll
            for (int e = 0; e < 2; e++) {{
                int m = mt * 8 + fn + e;
                if (m < M && n < N) y[m * N + n] = static_cast<T>(C[t][mt][e]);
            }}
        }}
    }}
"""


def _kernel_mma(M, bits, group_size, SN, NSG, KS, KP):
    key = ("mma", M, bits, group_size, SN, NSG, KS, KP)
    k = _KERNELS.get(key)
    if k is None:
        src = _source_mma(M, bits, group_size, SN, NSG, KS, KP)
        k = mx.fast.metal_kernel(
            name=f"specdec_qmm_m{M}_b{bits}_g{group_size}_n{SN}_s{NSG}_k{KS}_p{KP}_{_h(src)}",
            input_names=["x", "w", "scales", "biases"],
            output_names=["y"],
            source=src,
            header=_MMA_HEADER,
        )
        _KERNELS[key] = k
    return k


_TUNED_PATH = os.path.join(os.path.dirname(__file__), "qmm_tuned.json")
_TUNED: Optional[Dict[str, list]] = None


def _tuned() -> Dict[str, list]:
    global _TUNED
    if _TUNED is None:
        try:
            with open(_TUNED_PATH) as f:
                _TUNED = json.load(f)["configs"]
        except (OSError, ValueError, KeyError):
            _TUNED = {}
    return _TUNED


def pick_config(M: int, N: int, K: int, bits: int, group_size: int = 64) -> Tuple[int, int, int, int]:
    """(SN, NSG, KS, KP) for a call.

    Exact (bits, group_size, N, K, M) entries come from ``qmm_tuned.json``,
    written by ``bench/mlx_qmm_kernel_bench.py --tune`` on the M3 Pro for the
    27B models' shapes; other shapes use a rule that was within a few % of the
    best on synthetic shapes (notes/mlx-qmm-kernel.md).
    """
    c = _tuned().get(f"{bits},{group_size},{N},{K},{M}")
    if c is not None and group_size >= 8 * c[2] and K % max(32 * c[2] * c[3], group_size) == 0:
        return tuple(c)
    if M <= 8:
        return (1, 4, 2, 4)
    return (2, 4, 1, 4)


def qmm(x: mx.array, w: mx.array, scales: mx.array, biases: mx.array, group_size: int = 64,
        bits: int = 4, config: Optional[Tuple[int, int, int, int]] = None) -> mx.array:
    """``x @ dequantize(w, scales, biases).T`` for affine ``w`` ([N, K*bits/32] uint32), M <= 16 rows."""
    K = x.shape[-1]
    lead = x.shape[:-1]
    x2 = x.reshape(-1, K)
    M, N = x2.shape[0], w.shape[0]
    assert 1 <= M <= 16, "qmm handles 1..16 rows"
    SN, NSG, KS, KP = config or pick_config(M, N, K, bits, group_size)
    assert K % max(32 * KS, group_size) == 0 and group_size >= 8 * KS
    rows_per_tg = 8 * SN * (NSG // KP)
    n_tg = (N + rows_per_tg - 1) // rows_per_tg
    (y,) = _kernel_mma(M, bits, group_size, SN, NSG, KS, KP)(
        inputs=[x2, w, scales, biases],
        template=[("T", x.dtype)],
        grid=(32, NSG * n_tg, 1),
        threadgroup=(32, NSG, 1),
        output_shapes=[(M, N)],
        output_dtypes=[x.dtype],
    )
    return y.reshape(*lead, N)


def supported(x_dtype, w: mx.array, scales: mx.array, biases, group_size: int, bits: int,
              mode: str = "affine") -> bool:
    """Whether ``qmm`` handles this weight (otherwise use mx.quantized_matmul)."""
    if mode != "affine" or biases is None or bits not in (4, 8) or group_size not in (32, 64, 128):
        return False
    if w.ndim != 2 or x_dtype not in (mx.bfloat16, mx.float16, mx.float32):
        return False
    if scales.dtype != x_dtype or biases.dtype != x_dtype:
        return False
    K = w.shape[-1] * 32 // bits
    return K % 64 == 0 and K % group_size == 0


# ------------------------------------------------- simd_sum kernel (kept for reference)
def _source_simd(M: int, bits: int, group_size: int, R: int, NSG: int, V: int) -> str:
    wpl = V * bits // 32
    return f"""
    constexpr int M = {M};
    constexpr int R = {R};
    constexpr int NSG = {NSG};
    constexpr int V = {V};
    constexpr int WPL = {wpl};
    constexpr int BITS = {bits};
    constexpr int PER_WORD = 32 / BITS;
    constexpr int GS = {group_size};
    constexpr int BLOCK = 32 * V;
    constexpr uint MASK = (1u << BITS) - 1u;

    const int K = x_shape[1];
    const int N = w_shape[0];
    const int Kw = K * BITS / 32;
    const int Kg = K / GS;
    const uint lane = thread_index_in_simdgroup;
    const uint sg = simdgroup_index_in_threadgroup;
    const int row0 = (int(threadgroup_position_in_grid.y) * NSG + int(sg)) * R;

    const device uint32_t* wp[R];
    const device T* sp[R];
    const device T* bp[R];
#pragma unroll
    for (int r = 0; r < R; r++) {{
        int row = min(row0 + r, N - 1);
        wp[r] = w + row * Kw + lane * WPL;
        sp[r] = scales + row * Kg + (int(lane) * V) / GS;
        bp[r] = biases + row * Kg + (int(lane) * V) / GS;
    }}
    const device T* xp = x + lane * V;

    float acc[M][R];
#pragma unroll
    for (int m = 0; m < M; m++)
#pragma unroll
        for (int r = 0; r < R; r++) acc[m][r] = 0.0f;

    for (int k = 0; k < K; k += BLOCK) {{
        float wd[R][V];
#pragma unroll
        for (int r = 0; r < R; r++) {{
            float s = float(sp[r][k / GS]);
            float b = float(bp[r][k / GS]);
#pragma unroll
            for (int j = 0; j < WPL; j++) {{
                uint32_t q = wp[r][k / PER_WORD + j];
#pragma unroll
                for (int i = 0; i < PER_WORD; i++)
                    wd[r][j * PER_WORD + i] = s * float((q >> (BITS * i)) & MASK) + b;
            }}
        }}
#pragma unroll
        for (int m = 0; m < M; m++) {{
            float xv[V];
#pragma unroll
            for (int i = 0; i < V; i++) xv[i] = float(xp[m * K + k + i]);
#pragma unroll
            for (int r = 0; r < R; r++) {{
                float d = 0.0f;
#pragma unroll
                for (int i = 0; i < V; i++) d = fma(xv[i], wd[r][i], d);
                acc[m][r] += d;
            }}
        }}
    }}

#pragma unroll
    for (int m = 0; m < M; m++) {{
#pragma unroll
        for (int r = 0; r < R; r++) {{
            float v = simd_sum(acc[m][r]);
            if (lane == 0 && row0 + r < N) y[m * N + row0 + r] = static_cast<T>(v);
        }}
    }}
"""


def qmm_simd(x, w, scales, biases, group_size=64, bits=4, R=4, NSG=2, V=8):
    """The qmv_fast-style variant (M x R scalar accumulators + simd_sum); slower, see notes."""
    K = x.shape[-1]
    lead = x.shape[:-1]
    x2 = x.reshape(-1, K)
    M, N = x2.shape[0], w.shape[0]
    assert K % (32 * V) == 0
    key = ("simd", M, bits, group_size, R, NSG, V)
    if key not in _KERNELS:
        src = _source_simd(M, bits, group_size, R, NSG, V)
        _KERNELS[key] = mx.fast.metal_kernel(
            name=f"specdec_qmm_simd_m{M}_b{bits}_g{group_size}_r{R}_s{NSG}_v{V}_{_h(src)}",
            input_names=["x", "w", "scales", "biases"], output_names=["y"], source=src,
            header="#include <metal_simdgroup>\n")
    n_tg = (N + R * NSG - 1) // (R * NSG)
    (y,) = _KERNELS[key](inputs=[x2, w, scales, biases], template=[("T", x.dtype)],
                         grid=(32, NSG * n_tg, 1), threadgroup=(32, NSG, 1),
                         output_shapes=[(M, N)], output_dtypes=[x.dtype])
    return y.reshape(*lead, N)


# ---------------------------------------------------------------- the hook
class FastQuantizedLinear(nn.QuantizedLinear):
    """``nn.QuantizedLinear`` that sends calls with ``m_min <= rows <= m_max`` to ``qmm``.

    Instances are made by ``install`` (class swap of existing modules), never
    constructed directly; ``_qmm_lo`` / ``_qmm_hi`` are set per instance (plain
    ints: nn.Module would store a tuple attribute as a child entry).
    """

    def __call__(self, x):
        lo, hi = self._qmm_lo, self._qmm_hi
        rows = x.size // x.shape[-1]
        if lo <= rows <= hi and x.dtype == self["scales"].dtype:
            y = qmm(x, self["weight"], self["scales"], self["biases"],
                    group_size=self.group_size, bits=self.bits)
            if "bias" in self:
                y = y + self["bias"]
            return y
        return super().__call__(x)


def parse_range(spec) -> Optional[Tuple[int, int]]:
    """'2-16' -> (2, 16); '8' -> (2, 8); '', '0', None, 'off' -> None."""
    if spec is None or isinstance(spec, tuple):
        return spec
    spec = str(spec).strip()
    if spec in ("", "0", "off", "none"):
        return None
    if "-" in spec:
        lo, hi = (int(v) for v in spec.split("-"))
    else:
        lo, hi = 2, int(spec)
    assert 1 <= lo <= hi <= 16, "qmm range must lie within 1..16"
    return lo, hi


def install(model: nn.Module, m_range=(2, 16)) -> int:
    """Route eligible QuantizedLinear layers of ``model`` through ``qmm`` for
    ``m_range`` rows (None uninstalls). Returns the number of layers routed."""
    m_range = parse_range(m_range)
    if m_range is None:
        return uninstall(model)
    n = 0
    for _, mod in model.named_modules():
        if isinstance(mod, nn.QuantizedLinear) and getattr(mod, "mode", "affine") == "affine" \
                and "biases" in mod and supported(mod["scales"].dtype, mod["weight"], mod["scales"],
                                                  mod["biases"], mod.group_size, mod.bits):
            mod.__class__ = FastQuantizedLinear
            mod._qmm_lo, mod._qmm_hi = m_range
            n += 1
    return n


def uninstall(model: nn.Module) -> int:
    n = 0
    for _, mod in model.named_modules():
        if isinstance(mod, FastQuantizedLinear):
            mod.__class__ = nn.QuantizedLinear
            n += 1
    return n


def installed_range(model: nn.Module) -> Optional[Tuple[int, int]]:
    for _, mod in model.named_modules():
        if isinstance(mod, FastQuantizedLinear):
            return mod._qmm_lo, mod._qmm_hi
    return None


def env_range() -> Optional[Tuple[int, int]]:
    return parse_range(os.environ.get("SPECDEC_QMM"))
