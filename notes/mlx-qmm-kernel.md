# A small-batch quantized matmul for MLX verify passes

**Headline (s.9):** MTP k=7 + this kernel: 18.3 tok/s on the Q4_K_M transcode (3.25x its greedy, +20% over k=3), 28.3 on mlx-community 4-bit (3.23x).

**Summary.** MLX 0.32.3's `quantized_matmul` on the M3 Pro runs 2-12 rows
with `qmv_wide`, which cuts M into tiles of <= 5 vectors that each re-read and
re-dequantize all weights with scalar FMAs, and 13+ rows with a 32-row MMA
tile; hence 1.0 / 1.2 / 1.5 / 1.9 / 2.3 / 2.9x for M = 4 / 5 / 6 / 8 / 10 / 13+.
`mlx_spec/qmm_kernel.py` is an `mx.fast.metal_kernel` that reads and
dequantizes each weight once and does the M-row arithmetic with 8x8 simdgroup
MMAs: at the real shapes it costs 1.02-1.05x MLX's 1-row time for M = 2-8 and
1.33-1.5x for 9-16 (4-bit; 8-bit stays ~1.05x to 16). Full-model verify: T = 8
in 131 ms vs 219 (T = 1: 118), T = 16 in 186 vs 340. End to end the best
draft length moves from 3 to 7: **MTP k = 7 (vocab 65536) 26.6 tok/s vs 21.0
for the best MLX-kernel config in the same session (+27%, 3.12x greedy)**;
0.8B draft k = 6 21.3 vs 17.5. Outputs: fp32 12/12 identical to greedy;
bf16 divergences all one-ulp near-ties. Opt-in: `SPECDEC_QMM=5-16` or the
`qmm=5-16` config key.

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

## 3. Correctness

`tests/test_qmm_kernel.py` (23 tests, ~25 s, small GPU work): every
bits {4, 8} x group {32, 64, 128} x dtype {bf16, fp16, fp32} x M = 1..16 on
random weights (N = 520 and N = 52, so partial 8/16-row tiles are covered),
leading batch dims and several launch configs, real layer weights of both
checkpoints (mlx-community 4-bit g64; the Q4_K_M transcode's 4-bit g32 MLP and
8-bit g32 attention/SSM projections, read straight from the safetensors), the
hook (routing by rows, untouched path bitwise equal to stock, N < min_n left
alone), and speculative == greedy on a tiny 4-bit qwen3_5 with the kernel on
for k = 1..15. The criterion: max error vs an fp32 reference (fp32 x times
fp32 dequantized weights) <= 1.5x MLX's own error, or one output ulp. In
practice ours equals MLX's error for M <= 12 (both round the fp32 sum once)
and is ~2x smaller at M >= 13, where MLX's qmm rounds the dequantized
weights to bf16 before the MMA; in fp32 it is ~1e-6 vs MLX's ~5e-7 (MMA
accumulation order).

## 4. Microbenchmark at the real shapes

`bench/mlx_qmm_kernel_bench.py`, 2026-10-10 10:22, **AC power**, GPU lock held,
random weights of each shape, ~0.6 GB per timed call, median of 9; rows in
`results/mlx_qmm_kernel_bench.jsonl`; launch configs tuned in the same session
(`mlx_spec/qmm_tuned.json`, `logs/qmm_tune.log`). Time relative to MLX at M = 1
(MLX / ours):

| layer (4-bit g64) | M=1 | 2 | 4 | 5 | 6 | 8 | 9 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| mlp.gate/up 17408x5120 (385 us) | 1.00/1.02 | 0.99/1.00 | 0.99/1.03 | 1.19/1.03 | 1.52/1.04 | 1.89/1.04 | 2.36/1.48 | 2.94/1.50 | 2.85/1.52 |
| mlp.down 5120x17408 | 1.00/1.05 | 0.99/1.04 | 1.04/1.03 | 1.28/1.05 | 1.55/1.03 | 1.91/1.05 | 2.35/1.47 | 2.92/1.47 | 2.78/1.48 |
| in_proj_qkv 10240x5120 | 1.00/1.04 | 0.96/0.98 | 0.96/0.98 | 1.16/0.99 | 1.47/1.01 | 1.83/1.02 | 2.28/1.45 | 2.72/1.46 | 2.75/1.46 |
| k_proj 1024x5120 (24 us) | 1.00/1.18 | 0.99/1.13 | 1.01/1.14 | 1.17/1.20 | 1.46/1.15 | 1.81/1.15 | 2.25/1.53 | 2.68/1.55 | 3.00/1.56 |
| in_proj_a 48x5120 (3.6 us) | 1.00/2.82 | | | | | 1.07/2.89 | | 1.22/3.10 | |

| layer (Q4_K_M transcode) | M=1 | 2 | 4 | 5 | 6 | 8 | 9 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|
| mlp.gate/up 4-bit g32 | 1.00/1.02 | 0.98/1.02 | 1.03/1.03 | 1.11/1.04 | 1.44/1.04 | 1.77/1.04 | 2.13/1.33 | 2.63/1.35 | 2.55/1.35 |
| mlp.down 4-bit g32 | 1.00/1.02 | 0.98/1.01 | 1.07/1.03 | 1.30/1.05 | 1.47/1.09 | 1.81/1.04 | 2.20/1.32 | 2.70/1.35 | 2.56/1.35 |
| in_proj_qkv 8-bit g32 | 1.00/1.03 | 0.99/1.04 | 1.00/1.03 | 1.02/1.08 | 1.07/1.02 | 1.11/1.04 | 1.31/1.05 | 1.69/1.06 | 1.49/1.05 |
| k_proj 8-bit g32 | 1.00/1.11 | 1.00/1.08 | 1.08/1.21 | 1.10/1.21 | 1.05/1.11 | 1.12/1.12 | 1.32/1.11 | 1.63/1.11 | 1.69/1.13 |

Summed over a forward pass (count x time per shape), matmuls only, ms:

| M | 1 | 2 | 4 | 5 | 6 | 7 | 8 | 9 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|---|---|
| 4-bit g64, MLX | 112.6 | 111.0 | 112.3 | 135.1 | 170.2 | 209.1 | 211.3 | 262.2 | 322.7 | 315.8 |
| 4-bit g64, ours | - | 114.1 | 115.5 | 116.6 | 116.3 | 118.7 | 118.3 | 165.9 | 167.7 | 168.4 |
| Q4_K_M transcode, MLX | 155.4 | 153.3 | 159.8 | 171.6 | 196.2 | 231.0 | 229.3 | 274.5 | 341.2 | 321.7 |
| Q4_K_M transcode, ours | - | 160.2 | 161.5 | 163.8 | 163.6 | 164.7 | 162.6 | 186.7 | 189.0 | 190.2 |

Reading it:

* 4-bit: flat at 1.02-1.05x of MLX's M = 1 time for M = 2..8 (~110 GB/s),
  then 1.33-1.5x for 9..16 (two 8-row m-tiles; MMA-bound, the 16-row case runs
  at ~70% of fp32 peak). MLX: 1.9x at 8, 2.8-2.9x at 11-16.
* 8-bit (twice the bytes per FMA): bandwidth-bound all the way to 16 rows,
  1.03-1.08x; MLX reaches 1.5-1.7x.
* MLX is 2-4% faster at M = 2..4 (its `qmv_wide` is still bandwidth-bound
  there and needs no padding), so the hook should start at 5.
* Narrow outputs lose: k/v_proj (N = 1024) pay 10-20% (only 64-128
  threadgroups), and in_proj_a/b (N = 48) 2.9x (3 threadgroups; 0.7 ms per
  pass over 96 calls). `install()` leaves layers with N < 256 on MLX.

## 5. Verify-cost curve, full model

`python -m mlx_spec verify-cost` logic (`bench/mlx_qmm_session.py`), same
session, AC power, mlx-community 4-bit, 305-token context, median of 7, ms
(rows in `results/mlx_verify_cost.jsonl` with `"qmm"`):

| T (=k+1) | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10-12 | 13-16 | 17 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MLX | 117.8 | 121.4 | 123.4 | 123.7 | 144.9 | 178.7 | 217.7 | 219.3 | 268.6 | ~270-330 | ~340 | 341.9 |
| kernel 2-16 | 118.8 | 124.4 | 125.6 | 127.5 | 129.3 | 128.7 | 131.1 | 130.5 | 185.0 | 188 | 186 | 341.9 (MLX) |
| x T=1 (kernel) | 1.0 | 1.05 | 1.06 | 1.08 | 1.09 | 1.09 | 1.11 | 1.11 | 1.57 | 1.59 | 1.58 | |

So verifying 7 drafts now costs what 3 did (130 vs 124 ms), and 15 drafts cost
1.58x a decode step instead of 2.9x. At T = 2-4 the kernel is ~3 ms slower than
MLX (in line with the microbenchmark), hence the final range 5-16.

(The kernel row is the 10:24 session with routing 2-16 and in_proj_a/b still
routed, before `min_n`; a re-run at 13:37 with the final 5-16 range was
disturbed by the user working on the machine (MLX's own T = 6 came out at
275 ms instead of 179) and is not used for the curve; its rows are in the same
file. The end-to-end per-round verify times below confirm the curve.)

## 6. End to end (`bench/mlx_qmm_session.py`)

Session 2026-10-10 13:37-15:00, **AC power** throughout (`results/mlx_qmm_session.jsonl`),
GPU lock held, but the user was using the Mac (WindowServer, Chrome, Music,
Ollama app open), so absolute numbers are ~5-9% below the overnight runs
(MTP k=3 21.0 vs 23.0 tok/s then) and run-to-run noise is ~3% (the k = 3 pairs
below are the same computation with and without the hook: 20.99 / 21.10 and
17.46 / 16.99). Configs interleaved (each k without, then with the kernel),
greedy, 256 tokens, the six prompts, kernel routing **5-16 rows, N >= 256**.
Rows: `results/mlx_{greedy,mtp,draft}.jsonl` (configs ending `_qmm5-16`).

Greedy baseline in the session: **8.54 tok/s**.

| config | MLX tok/s | kernel tok/s | gain | accept | tok/round | verify ms/round MLX -> kernel |
|---|---|---|---|---|---|---|
| mtp k=3 vocab 65536 | 20.99 | 21.10 | (same code path) | 0.71 | 3.14 | 140 -> 142 |
| mtp k=4 | 19.22 | 22.57 | +17% | 0.63 | 3.50 | 170 -> 139 |
| mtp k=5 | 16.09 | 23.62 | +47% | 0.56 | 3.8 | 224 -> 138 |
| mtp k=6 | 13.99 | 23.81 | +70% | 0.50 | 4.0 | 261 -> 144 |
| **mtp k=7** | 15.33 | **26.63** | +74% | 0.48 | 4.3 | 253 -> 134 |
| mtp k=8 | 12.07 | 16.90 | +40% | 0.43 | 4.4 | 345 -> 218 |
| draft 0.8B k=3 | 17.46 | 16.99 | (same code path) | 0.60 | 2.80 | 139 -> 149 |
| draft k=4 | 16.00 | 19.61 | +23% | 0.55 | 3.2 | 171 -> 137 |
| draft k=5 | 14.00 | 20.06 | +43% | 0.49 | 3.45 | 217 -> 139 |
| **draft k=6** | 12.10 | **21.34** | +76% | 0.44 | 3.65 | 274 -> 137 |
| draft k=7 | 12.81 | 21.00 | +64% | 0.41 | 3.8 | 258 -> 141 |
| draft k=8 | 11.17 | 16.30 | +46% | 0.40 | 4.2 | 323 -> 209 |

Per prompt, best new config (mtp k=7 + kernel), code_edit / code_gen /
summarize / prose / explain / math: 39.6 / 29.9 / 25.4 / 14.4 / 23.6 / 26.9.

* **Best: MTP, k = 7, vocab 65536, kernel 5-16: 26.6 tok/s = 3.12x greedy, +27%
  over the best MLX-kernel config in the same session (MTP k = 3, 21.0).**
  The 0.8B draft model gains more relative to itself (best k = 6: 21.3 vs
  17.5, +22%) but stays behind MTP.
* T = 5-8 now costs one decode step (verify 134-144 ms per round at k = 4-7),
  so the best k moves from 3 to 7 (T = 8, the last 8-row tile). k = 8 (T = 9)
  pays for the second m-tile (~210 ms) and loses.
* Per prompt the gain is largest where acceptance is high (code_edit 25.3 ->
  39.6) and smallest on prose (15.2 -> 14.4: acceptance there is too low for
  long drafts; an adaptive or confidence-gated k would take k = 3 there).

## 7. Output equivalence

* **fp32** (`tests/test_fp32_27b.py`, every bf16 parameter incl. scales upcast,
  the kernel running in fp32): `mtp:k=6`, `draft:k=6`, `mtp:k=8` with the
  kernel on the four prompts that flip in bf16, 160 tokens: **12/12 identical
  to greedy** (`logs/qmm_fp32.log`). So the kernel and the routing add no
  algorithmic error.
* **bf16**: every divergence of a kernel run was diagnosed
  (`results/mlx_qmm_diagnose.jsonl`, `diagnose()` with the kernel installed):
  code_gen @147 and @216, summarize @147, explain @42 and @153, math @8, all
  **one-bf16-ulp near-ties** (single-step margin 0.125, batched margin 0.125
  or 0.0). The known points (147/147/42/8) are the ones every drafter hits
  with MLX's kernels too; the new ones (code_gen @216, explain @153) are the
  same kind. Kernel runs match greedy on 2-5 of 6 prompts, as MLX's do.
* Side finding, not from this kernel: with MLX's own kernels at k >= 5, prose
  diverges at token 2 and draft k=7 explain at 130 with a margin of 0.25 (one
  ulp at |logit| >= 32; the batched margin is 0.125/0.25), i.e. MLX's
  multi-tile `qmv_wide` path flips a tie that the kernel path doesn't.

## 8. Is it worth upstreaming?

Yes, as a change to MLX's small-batch dispatch on gen >= 15 GPUs: replacing
`qmv_wide` (tiles of <= 5 vectors) and the 32-row `qmm` for 5 <= M <= 16 with
an MMA kernel that keeps all M rows in one pass. On this M3 Pro it turns the
2.9x verify plateau into 1.0x for T <= 8 and 1.5x for T <= 16, and it is
what makes k = 7 the best draft length. Notes for a port (no PR text, per
MLX's rules on AI-written PR text):

* The kernel is self-contained: `_source_mma` + `_MMA_HEADER` map 1:1 onto a
  `qmv_mma` template in `kernels/quantized.h` (templated on T, group_size,
  bits, M-tiles, SN, KS, KP) and a dispatch branch in `dispatch_qmv`
  (quantized.cpp:1825) for `5 <= M <= 16`, with `get_qmv_batch_limit`
  raised to 17 on gen >= 15. It uses only steel's fragment layout.
* Things to check in MLX's CI that I couldn't here: M1/M2 (gen 13/14; no
  `qmv_wide` there, so the gain would be larger), M4/M5 (the latter has NAX
  `qmm`), the other bit widths (2-bit fits the same uint32 unpacking; 3/5/6-bit
  need MLX's byte-group unpacking, 8 values per 3/5/6 bytes), fp/nvfp4 modes, the gather variants used by MoE models.
* MLX's `metal_kernel` API was sufficient: no launch-overhead problem (8 us
  per call fixed cost, same as `quantized_matmul`), threadgroup memory is
  declared inside the body; so no MLX rebuild was needed to prototype this
  (step 5 of the brief did not trigger).

## 9. Headline numbers: Q4_K_M transcode vs mlx-community 4-bit (2026-10-11)

One GPU-lock job per target, back to back, AC power, memory free 80-87% at
start (`results/mlx_final_session.jsonl`, `logs/final_session*.log`), greedy,
256 tokens, six prompts, `--repeat 2` (median; every repeat pair agreed within
~2%) plus one profiled run. The first attempt loaded both models in one process:
the transcode's memory was not released and the mlx-community runs from k = 4
on swapped (2 tok/s); those rows were dropped and mlx-community re-run in its
own process 30 min later (`bench/mlx_final_pipeline2.sh`). Kernel routing
5-16 rows, N >= 256.

| config | Q4_K_M transcode (21 GB) | mlx-community 4-bit g64 (15 GB) |
|---|---|---|
| greedy | 5.63 | 8.77 |
| MTP k=3 vocab 65536, MLX kernels | 15.25 (2.71x) | 23.69 (2.70x) |
| MTP k=4 + qmm | 16.08 | 24.56 |
| MTP k=5 + qmm | 16.62 | 26.12 |
| MTP k=6 + qmm | 17.19 | 26.74 |
| **MTP k=7 + qmm** | **18.30 (3.25x, +20% vs k=3)** | **28.31 (3.23x, +19.5% vs k=3)** |
| 0.8B draft k=6 + qmm | 15.27 | 22.46 |

Verify ms per round (profiled run): transcode 187 at k=3, 195-202 at k=4-7;
mlx-community 121 at k=3, 125-127 at k=4-7.

Per prompt, MTP k=7 + qmm (code_edit / code_gen / summarize / prose / explain / math):
transcode 26.2 / 22.8 / 16.6 / 9.6 / 13.3 / 21.3; mlx-community
41.9 / 31.1 / 27.6 / 15.1 / 24.8 / 29.4. Prose is the one prompt where k=3
is better (transcode 10.5 vs 9.6, mlx-community 16.8 vs 15.1).

Output equivalence (bf16, `results/mlx_qmm_diagnose.jsonl`): matches greedy on
2-4/6 prompts per config (transcode draft k=6: 1/6), the same counts MLX's own
kernels give at k=3. Every divergence was diagnosed: all one-bf16-ulp
near-ties (margins 0.125 / 0.0 / -0.125) except transcode prose @ token 2,
single-step margin 0.25 and batched margin 0.0, which happens identically with
MLX's own kernels at k=3 (no qmm), so it is not the kernel; 0.25 is one ulp
if that logit is >= 32 (not checked).

**Server** (`tests/smoke_server_27b.py`, `SPECDEC_QMM=5-16`, transcode,
`--drafter mtp:k=7:vocab=65536:qmm=5-16`; results
`results/mlx_server_smoke_q4km_qmm_rerun.json`, `logs/final_smoke2.log`): text
identical to the CLI on 6/6 prompts; 20.2 tok/s mean as the server reports it
(CLI 18.3; the server's tok/s excludes prompt/first-token time). Streaming ==
non-streaming, thinking split, seeded sampling all OK. The smoke's prompt-cache
check differs at char 153 of turn 2 between cached and uncached prefill: that
path appends 28 tokens (MLX kernels, > 16 rows) vs a 2089-token prefill, a
batching near-tie, not the kernel. (A first smoke run right after the swap
episode was slow, 5.2 tok/s, from paging; the last prompt already ran at
CLI speed; same 6/6 text.)

Recommended:

```
SPECDEC_QMM=5-16 scripts/gpu.py .venv/bin/python -m mlx_spec serve \
  --model models/qwen38-27b-q4km-mlx --drafter mtp:k=7:vocab=65536
```
(18.3 tok/s CLI / ~20 tok/s server-reported on llama.cpp-equal quality; the
mlx-community 4-bit file does 28.3 with the same flags but is a lower-quality
quant.)

## Reproduce

```
.venv/bin/python -m pytest tests/test_qmm_kernel.py -q
scripts/gpu.py sh bench/mlx_qmm_pipeline.sh              # tune, microbench, tests, session, fp32
QMM_SKIP_TUNE=1 QMM_RANGE=5-16 scripts/gpu.py sh bench/mlx_qmm_pipeline.sh
SPECDEC_QMM=5-16 scripts/gpu.py .venv/bin/python -m mlx_spec bench --configs mtp:k=7:vocab=65536
scripts/gpu.py .venv/bin/python -m mlx_spec bench --configs mtp:k=7:vocab=65536:qmm=5-16
```

## Status of measurements (preliminary sections)

Sections 1-2's synthetic-shape numbers are **preliminary**: synthetic shapes (4096-16384 wide), run
while the GPU lock was unavailable (the user's own llama-server was up, idle,
sharing the GPU; another agent's CPU-heavy transcode check was also running),
power source not recorded (the Mac was later found on battery, so treat these
as suspect). The real-shape microbenchmark, tuning, verify-cost curve and
end-to-end runs are queued as one GPU-lock job (`bench/mlx_qmm_pipeline.sh`)
and record the power source per stage.

## State at hand-off (2026-10-10)

Done: root cause (s.1), kernel + hook (s.2), unit tests 23/23 pass (re-run
after the final edits), tuning + real-shape microbench (s.4, AC power),
verify-cost curve (s.5), end-to-end k = 3..8 for MTP vocab 65536 and the 0.8B
draft with/without the kernel, all complete (s.6, AC power, user active on the
machine, ~3% noise), every bf16 divergence diagnosed as a one-ulp near-tie,
fp32 12/12 identical (s.7). Best config: `mtp:k=7:vocab=65536:qmm=5-16`,
26.6 tok/s (3.12x greedy, +27% over MTP k=3 with MLX's kernels in the same
session). No gpu.py job of this track is queued or running.

Left (optional): a clean re-run of the verify-cost curve with the final
5-16 / N >= 256 routing on an idle machine; repeat=2 of the best configs on
an idle machine to pin absolute tok/s; the Q4_K_M transcode end to end with
the kernel (microbench says its verify pass drops from 229 to 163 ms at T=8);
an adaptive k (k=3 on low-acceptance text such as prose).
