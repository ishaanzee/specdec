# A small-batch quantized matmul for MLX verify passes

**Summary.** _(filled in at the end)_

Code: `mlx_spec/qmm_kernel.py` (kernel + opt-in hook), `mlx_spec/qmm_tuned.json`
(per-shape launch configs), `tests/test_qmm_kernel.py`,
`bench/mlx_qmm_kernel_bench.py` (microbenchmark / tuner),
`bench/mlx_qmm_session.py` (verify-cost + end-to-end session),
`bench/mlx_qmm_pipeline.sh` (all of it under one GPU lock).

## 1. Why MLX 0.32.3 has a staircase above 4 rows

Source read: the `v0.32.3` tag of `~/Documents/oss/mlx` (the installed wheel is
0.32.3; the checkout itself is `v0.32.3-59-ga7e00f130` on a feature branch, and
none of the 59 commits change the M3 path: they touch the M1 row limit and the
`qmv_fast` partial-rows case). Files: `mlx/backend/metal/quantized.cpp`
(dispatch), `mlx/backend/metal/kernels/quantized.h` (kernels). The wheel's
`mlx.metallib` contains the `affine_qmv_wide_*_nv_{2..5}_kl_8` kernels, so this
is the code that runs.

The device reports `applegpu_g15s` (M3 Pro: generation 15, size class `s`).
`QuantizedMatmul::eval_gpu` (quantized.cpp:1868) with `transpose=True`
(every `nn.QuantizedLinear`) picks a kernel by the row count M:

| M | kernel | what it does | weight reads |
|---|---|---|---|
| 1 | `affine_qmv_fast` (`dispatch_qmv` -> `qmv`) | 2 simdgroups x 4 output rows per threadgroup; each lane holds 16 x values in registers and dots them with 4 rows' packed weights (`qdot`, the scale/bias folded in after the sum); one `simd_sum` per row | 1x |
| 2-12 | `affine_qmv_wide` (`dispatch_qmv` -> `qmv_wide`, quantized.cpp:548; kernel `qmv_wide_impl`, quantized.h:988) | M is cut into `n_tiles = ceil(M / 5)` tiles of `vecs_per_tg = ceil(M / n_tiles)` vectors; one threadgroup per (tile, 4 output rows). 8 lanes per output row, each lane walks whole groups, dequantizes 8 values to fp32 and, for every vector of its tile, loads those 8 x values from device memory and does 8 scalar FMAs | **once per tile**: 1x for M <= 5, 2x for 6-10, 3x for 11-12 |
| >= 13 | `affine_qmm_t_splitk` / `affine_qmm_t` (quantized.cpp:1152 / 1042) | steel GEMM with a BM = 32 row tile (BN = 32, BK = 32, 2x2 simdgroups), weights dequantized to bf16 in threadgroup memory, 8x8 simdgroup MMAs; split-K over 2 partitions plus a reduction kernel when N/32 tiles < 512 | 1x, but computes 32 rows whatever M is |

The switch to `qmm` is `M >= get_qmv_batch_limit(K, N)` (quantized.cpp:89),
which for generation >= 15, size != 'd' returns 13 when K or N > 4096 (all the
27B's matmuls). `qmv_wide` is used for affine weights only on generation >= 15
(`use_qmv_wide`, quantized.cpp:542); on M1/M2 the same M range runs plain
`qmv` once per row.

Why each step costs what it does (measured `results/mlx_qmm_curve.jsonl`, MLP
up 5120->17408, x M=1):

* **M = 2-4: 1.0x.** One `qmv_wide` tile; the kernel is still bandwidth-bound.
* **M = 5: 1.12-1.25x.** One tile of 5, but `qmv_wide` is now ALU-bound. Its
  inner loop is scalar: per weight, a dequantize (shift, mask, convert, FMA)
  plus, per vector, a load of the x value from device memory (8 lanes of a row
  each load their own x slice; x is reused across rows only through the L1)
  and an FMA. Work per weight ~ 3 + 2 * nv instructions.
* **M = 6-12: tiles.** Each tile re-reads and re-dequantizes every weight. A
  model of `cost ~ n_tiles * (3 + 2 * vecs_per_tg) / 12` (normalised to the M=1
  bandwidth time) gives 1.50 / 1.83 / 2.17 / 2.75 for M = 6 / 8 / 10 / 12;
  measured 1.46-1.54 / 1.82-1.92 / 2.34-2.48 / 2.73-2.83. So it is the tile
  count times an ALU-bound tile, not memory: the weights are re-read, but the
  re-dequantization and the per-vector scalar work are what take the time.
* **M >= 13: ~2.75-2.9x, flat to 32.** The 32-row MMA tile does 32 rows of
  arithmetic for any M; 2.8x the bandwidth time is ~70% of the M3 Pro's fp32
  peak for 32 rows (27 G weights x 32 rows = 0.86 TFMA per pass; peak ~3.2
  TFMA/s with 18 cores x 128 lanes x ~1.4 GHz).

Two things are therefore missing for 5-16 rows on this GPU: (1) one pass over
the weights for all M rows (`qmv_wide` caps tiles at 5), and (2) arithmetic
cheap enough that M = 8-16 rows per weight doesn't make the kernel ALU-bound.
The scalar-FMA route can't do (2): 16 rows x 27 G weights = 0.43 TFMA per pass,
already ~135 ms at full fp32 peak, before dequantization and x loads. The
8x8 simdgroup MMA units can: they reach ~70% of peak in steel, and an 8-row
(or 16-row) tile wastes nothing at M = 8 (16).

## 2. Kernel design (`mlx_spec/qmm_kernel.py`)

Written with `mx.fast.metal_kernel` (no MLX rebuild). One launch per matmul,
`y = x @ dequant(W).T`, x `[M, K]` with M <= 16, W affine-quantized
(`[N, K*bits/32]` uint32, bf16/fp16/fp32 scales and biases per group).

* **Work split.** A threadgroup has `NSG` simdgroups = `RG` row groups x `KP`
  K-splits. Each simdgroup owns `8*SN` output features (SN 8-row n-tiles) and
  all M rows (`SM = ceil(M/8)` 8-row m-tiles, padded rows clamp their x loads
  and are not written). It walks K in chunks of `32*KS`, starting at its K-split
  offset; the `KP` partial results are summed through threadgroup memory at
  the end (only `KP-1` x `SN*SM*32` float2 per row group).
* **Weights read once, dequantized once.** Per chunk and n-tile every lane
  loads its contiguous `KS` uint32 (4-bit: 8*KS values; 8-bit: 2*KS words)
  of one weight row, plus that row's scale and bias, and dequantizes in fp32
  (`scale * q + bias`, like MLX's `qmv`/`qmv_wide`) straight into its two
  elements of `4*KS` 8x8 A fragments. Those fragments are used for every
  m-tile, so the dequantization cost is shared by all M rows.
* **K permutation.** The 8 columns of fragment f are mapped to
  k = (c/2)*8*KS + 2f + c%2 inside the chunk. Under MLX steel's fragment layout
  (lane holds row fm, columns fn and fn+1, fn in {0,2,4,6}) this makes each
  lane's weight slice contiguous (one 32-bit load per 8 4-bit values) without
  any threadgroup-memory staging; x (B fragments, `X^T[k, m]`) is loaded with
  the same permutation: per lane two rows x 4*KS strided elements, from L1.
* **Arithmetic.** `simdgroup_multiply_accumulate` on float 8x8 fragments,
  fp32 accumulation, one rounding to the output dtype at the end. The
  fragments are kept in plain `float2` registers and wrapped into
  `simdgroup_float8x8` only around each MMA (as steel's `mma()` does);
  assigning to `thread_elements()` of arrays of matrices made the kernel 4x
  slower (spills).
* **Launch configs.** `(SN, NSG, KS, KP)` per (bits, group, N, K, M) from
  `mlx_spec/qmm_tuned.json` (written by `bench/mlx_qmm_kernel_bench.py --tune`
  for the 27B shapes); a fallback rule elsewhere.
* **Supported:** affine, bits 4 and 8, group 32/64/128, K a multiple of 64 and
  of the group size, any N, M 1-16, bf16/fp16/fp32 (scales of the same dtype).
  Everything else (and every call with M outside the routed range) goes to
  `mx.quantized_matmul`.

What did not work:

* **The qmv_fast-style kernel first asked for** (`qmm_simd`: lanes split K,
  R output rows per simdgroup, M x R fp32 accumulators, scalar FMAs, one
  `simd_sum` per output at the end). It reads the weights once, but it is
  ALU-bound from M ~ 4: preliminary, 4096x4096 4-bit x16 copies, best launch
  per M: 1.4 / 1.5 / 2.3 / 3.8 ms for M = 2 / 4 / 8 / 16 against MLX's
  1.6 / 1.35 / 2.46 / 3.77 (and 1.3-1.6 ms for M = 1). With 16 rows the lane
  needs 16 x 8 x values plus 64 accumulators in registers, and every weight
  costs M scalar FMAs. Same wall as `qmv_wide`.
* half-precision fragments (fp16 A/B, fp32 or fp16 accumulation): no faster
  on the M3 (1.80 vs 1.95 ms at M = 16, within noise), so fp32 stays.
* larger KS (64-128 k per chunk): no gain; more registers.

## Status of measurements

All numbers above are **preliminary**: synthetic shapes (4096-16384 wide), run
while the GPU lock was unavailable (the user's own llama-server was up, idle,
sharing the GPU; another agent's CPU-heavy transcode check was also running),
power source not recorded (the Mac was later found on battery, so treat these
as suspect). The real-shape microbenchmark, tuning, verify-cost curve and
end-to-end runs are queued as one GPU-lock job (`bench/mlx_qmm_pipeline.sh`)
and record the power source per stage.
