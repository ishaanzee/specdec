# llama.cpp Metal: a multi-row mat-vec kernel for 2-16 token verify batches (M3 Pro)

**TL;DR.** Yes, the patched build is faster end to end. A simdgroup-MMA small-batch kernel brings the 5..8-token
verify forward from 416-573 ms down to 270-280 ms (1.8-1.9x a 1-token forward instead of 2.8-3.8x). Long drafts
then pay off. **Best: DFlash2 (Q4_0) n_max 7 + ngram-simple = 14.13 tok/s, 2.06x**, against 11.61 tok/s (1.70x)
for the previous best (MTP n_max 2 + ngram-simple) measured on the stock path of the same binary in the same
session. That is +22%. MTP peaks at n_max 7 (13.07). The Qwen3.5-0.8B draft goes from 1.61x to 1.83x.
Perplexity matches to 5e-4 relative. Output: 5/6 prompts are byte-identical to the baseline. `explain` flips at a
near-tie at char 572 in every patched config, including n_max 2; the cause is diagnosed below. The stock n >= 3
configs flip 2/6.

Source: `vendor/llama.cpp` at b29c606e2, branch `multirow-mv`; diff = `scripts/metal-multirow-mv.patch`.
Build: `vendor/llama.cpp/build-multirow` (Release, Metal, embedded library; llama-server, llama-bench,
llama-perplexity, test-backend-ops). Switch: `GGML_METAL_MV_MULTIROW=1` (unset = stock code path, same binary).

## What the model actually multiplies (verified per tensor with gguf-py)

| type | bytes | tensors (per layer x layers) |
|---|---:|---|
| Q4_K | 10.34 GB | ffn_gate/up 5120->17408, ffn_down 17408->5120 (x64), token_embd (get_rows, not a matmul) |
| Q8_0 | 7.16 GB | attn_qkv 5120->10240, attn_gate 5120->6144, ssm_out 6144->5120, ssm_alpha/beta (x48); attn_q/k/v (x16) |
| Q6_K | 1.46 GB | output 5120->248320, attn_output 6144->5120 (x16) |

So Q8_0 is 38% of the matmul bytes, not a side issue: it goes through the same small-batch kernels.
At ~150 GB/s the 19 GB of weights take >= ~127 ms per forward; the stock 1-token forward is 152 ms.

## Root cause: why B = 4..8 cost 2-3.8x and B = 9..16 cost 2.5x a 1-token forward

Dispatch in `ggml_metal_op_mul_mat` (ggml-metal-ops.cpp), src1 = f32, ne00 % 128 == 0:

| src1 rows (B) | Q4_K / Q6_K | Q8_0 |
|---|---|---|
| 1 | `kernel_mul_mv_<t>_f32` | `kernel_mul_mv_q8_0_f32` |
| 2-3 | `kernel_mul_mv_<t>_f32`, grid.y = B (one full weight pass per src1 row) | `mul_mv_ext` r1ptg 2/3 |
| 4-8 | `kernel_mul_mv_ext_<t>_f32_r1_{4,5,3,4,4}` | `mul_mv_ext` |
| >= 9 | `kernel_mul_mm_<t>_f32` (simdgroup mat-mat; tensor API is off on pre-M5) | same |

**mul_mv (B = 2, 3).** The single-row kernels are launched with grid.y = ne11, i.e. one complete pass over
the weights per token. B=2 costs only +6% (161 vs 152 ms) because the two passes over a tensor run
concurrently and the second mostly hits the SLC/L2; at B=3 that stops working (212 ms).

**mul_mv_ext (B = 4..8).** `kernel_mul_mv_ext_q4x4_f32_impl`: 8 lanes along a row, each lane owns one src0 row
and one 16-weight chunk per iteration (`chpt = 1`), dequantized with the generic `dequantize_q4_K` (16 single-byte
device loads + 2-4 scale-byte loads + d/dmin per 16 weights, i.e. ~22 load instructions for 9 bytes), then
r1ptg dot products against float4x4 src1 chunks. Problems:
1. r1ptg is at most 5, and the switch picks 4 for B=7/8 and 3 for B=6: B=6,7,8 run **two full passes** over the
   weights (grid.y = 2), each re-reading and re-dequantizing them (B=5 is one pass with r1ptg=5, B=4 one pass).
2. Even one pass is far from bandwidth-bound (B=4: 304 ms, B=5: 423 ms, single pass): the inner loop is
   load-instruction bound (byte loads) and has very little memory-level parallelism per thread (9 bytes of weights
   in flight per lane per iteration), so per-pass throughput is ~2-3x below the 1-row `mul_mv_q4_K`, whose lanes
   load 2x8 bytes of packed quants + scales per block with uint16 loads and keep 2 rows x 4 blocks in flight.
3. Q8_0 uses the same ext scheme (float4 chunks, `dequantize_q8_0_t4`) for B = 2..8.

**mul_mm (B >= 9).** `kernel_mul_mm` (non-tensor path): threadgroup = 4 simdgroups, output tile 64 src0 rows x
**32 src1 rows**, K step 32. For any B <= 32 there is one column tile, so the work is the same for B = 9 and B = 32
(measured: pp9 385, pp16 389, pp32 402 ms — flat). Per K step each of the 128 threads dequantizes 16 weights
with the byte-load `dequantize_q4_K`, does 16 scalar half stores into threadgroup memory, two threadgroup
barriers, then 4 x 8 = 32 simdgroup 8x8x8 MMAs per simdgroup, 75% of which multiply zero padding at B = 8..16.
Padded work = 2 x 25.6e9 weights x 32 = 1.64 TFLOP per forward; at ~380 ms that's ~4.3 TFLOP/s, close to the
M3 Pro 18-core GPU's FP32 peak (~6.4 TFLOP/s nominal): **at small B the mat-mat path is compute-bound on padding
plus dequant/threadgroup traffic, not bandwidth-bound** (~50 GB/s effective). Confirmed with test-backend-ops
perf on the ffn shape: n = 9, 16, 32 all cost ~1.25 ms, n = 64 costs 2.35 ms and n = 128 costs 4.54 ms. Cost is per
32-column tile, so for n = 9..16 most of the work multiplies zero padding.

## Kernel design

The patch has two kernels. Only the second one is used by default (see "Final dispatch" below).

### 1. Scalar multi-row mat-vec (`kernel_mul_mv_mr_{q4_K,q6_K,q8_0}_f32_r<R>_nr0_<NR0>`): the design first asked for

* Same lane layout and block stride as the 1-row kernel of each type (`mul_mv_q4_K`: 4 blocks per step, lane =
  (block, 64-half, 8-slice), 32 weights per lane per block; `mul_mv_q6_K`: 2 blocks per step, 16 weights per lane;
  `mul_mv_q8_0`: 8 blocks per step, 8 weights per lane), so the weight stream looks exactly like the 1-row case.
* Each simdgroup owns NR0 src0 rows. Per block it loads the packed quants once with vector loads
  (uchar4 / packed_uchar4 / packed_char4), dequantizes them once into float4 registers (scale and min folded
  in: `w = d*sc*q - dmin*m`), then loops over the R src1 rows: load the matching src1 float4s, `R x NR0`
  accumulators per thread. One `simd_sum` per (r, row) at the end.
* One dispatch covers up to R = 16 src1 rows (grid.y = ceil(ne11/R); ne11 > R is split into equal tiles).
  Rows past ne11 in the last tile are clamped for reads and never written; src0 rows past ne01 are clamped too
  (the stock kernels read past the end there).
* Templates: R = 2..16, NR0 = 1/2/4 for each of the 3 types; NSG (simdgroups per threadgroup) is a function constant.
* Dispatch (`ggml_metal_op_mul_mat_use_mv_mr`, ggml-metal-common.cpp): only with `GGML_METAL_MV_MULTIROW=1`,
  src0 in {Q4_K, Q6_K, Q8_0}, src1/dst f32, 2 <= ne11 <= `GGML_METAL_MV_MULTIROW_MAX` (16), src1 rows 16-byte
  aligned. Tuning knobs: `GGML_METAL_MV_MULTIROW_{R,NR0,NSG}`. Checked before the ext/mm branches; everything
  else (and the whole path with the env var unset) is unchanged. MUL_MAT_ID is not touched.
* Numerics: f32 weights x f32 activations, f32 accumulation (the mm path it replaces for 9..16 uses half for
  both operands), but a different summation order than any stock kernel, so near-tie argmax flips vs the stock
  baseline are possible.

## Correctness

`test-backend-ops test -b MTL0 -o MUL_MAT -p "type_a=(q4_K|q6_K|q8_0),type_b=f32"` with
`GGML_METAL_MV_MULTIROW=1`: **234/234 pass** (added cases: n = 2..17 for m = 1023 / k = 1280 and
m = 67 / k = 4352, plus src1 broadcast over dims 2/3 at n = 3, 8, 13, for all three types). Also passes with
NR0 = 1, NR0 = 4, R = 5, R = 3 + NSG = 4, R = 8 + NSG = 1. Sanity check that the tests really hit the new
kernels: adding +1 to every output made 185 of them fail with the env var on and 0 with it off.

Perplexity: see the section below (matches to 5e-4 relative).

## Kernel timings (test-backend-ops perf, model shapes) — interim, 2026-10-10

Logs: `logs/metal_mv_perf_run1.out` (battery; stock + scalar NR0=1/2), `logs/metal_mv_perf_run2.log` (AC; stock,
MMA v1 = shared src1 tile + 4 simdgroups, scalar NR0=4), `logs/metal_mv_perf_run3.log` (AC; stock, MMA v2 =
per-simdgroup src1 tile, NSG 1/2/4). Table: `python3 bench/metal_mv_perf_table.py <log>`. Cells: us/run.
n=1 rows are the same code in every column, so their spread (368-614 us) is the session noise level.

ffn_up/gate Q4_K 17408x5120 (50 MB; 1-row stock ~370 us = 135 GB/s):

| n | stock | scalar NR0=2 | scalar NR0=4 | MMA v1 (NSG 4) | MMA v2 (NSG 1) |
|---|---:|---:|---:|---:|---:|
| 2 | 400-540 | 410 | 486 | 668 | 744 |
| 3 | 600-770 | 565 | 653 | 662 | 679 |
| 4 | 1100 | 785 | 928 | 680 | 634 |
| 5 | 1475-1690 | 988 | 1185 | 685 | 684 |
| 6 | 1500-1600 | 1389 | 1415 | 666 | 652 |
| 8 | 2056-2137 | 2106 | 2425 | 717 | 689 |
| 9 | 1241-1267 | 3011 | 3048 | 896 | 1258 |
| 16 | 1247-1386 | 6386 | 10207 | 911 | 1365 |
| 32 / 64 / 128 (mm) | 1257 / 2350 / 4538 | | | | |

Findings so far:
* **mm root cause confirmed:** stock mm costs the same for n = 9..32 and doubles per extra 32-column tile
  (1257 -> 2350 -> 4538 us for n = 32/64/128): cost is per 64x32 output tile, i.e. compute/dequant-bound on a
  tile that is 2-3.5x padded at n = 9..16, at ~40 GB/s.
* **Scalar multi-row kernel (the design asked for) only wins at n = 2..4.** Each src1 float4 it loads feeds only
  NR0 dot products, so at n >= 5 it is bound by src1 re-reads from L1 (~0.5-0.7 TB/s of src1 traffic), not by
  the weight stream; NR0=1 is worst, NR0=4 runs out of registers. Q8_0/Q6_K at n = 2..3 with NR0=4 are at or
  slightly better than stock.
* **MMA variant (simdgroup 8x8 matrices, 16 rows x 8/16 src1 rows per simdgroup) is flat ~650-720 us for
  n = 3..8 on Q4_K (2.3-3x faster than stock at n = 5..8), ~900 us for n = 9..16 with v1 (30% faster than mm)**;
  Q6_K: 330-370 us vs 530-1070 stock at n = 4..8; LM head 248320x5120 Q6_K: ~10-11 ms flat for n = 3..8 vs
  22-44 ms stock. Q8_0: ~600-670 us at n = 4..8 vs 570-1190 stock, but **worse than mm at n = 9..16** (780-900 vs
  705-770). It reaches only ~75 GB/s: latency-bound (each simdgroup loads ~1.1 KB of weights per K step and waits).
* v2 (per-simdgroup src1 staging, no threadgroup barriers) = same as v1 at n <= 8, worse at 9..16 (redundant
  src1 staging). v1 is the better base.

### 2. Simdgroup-MMA small-batch kernel (`kernel_mul_mv_mma_{q4_K,q6_K,q8_0}_f32_nc_{1,2}`), the one that ships

* The threadgroup has NSG = 4 simdgroups and covers 64 src0 rows by 8 (nc=1, B <= 8) or 16 (nc=2, B <= 16) src1
  rows. The K step is 128.
* Per K step, the threadgroup stages the src1 tile once in threadgroup memory, as half and zero-padded beyond ne11.
* Each lane converts 64 weights of one src0 row (2 lanes per row) into its simdgroup's 16 x 128 half tile.
* Each simdgroup then does 16 x 2 x nc `simdgroup_multiply_accumulate` 8x8x8 with f32 accumulators.
* The output is stored transposed directly to dst. A partial tile goes through threadgroup memory.
* Weight loads are split into a raw load and a convert step. The raw quants for step k+1 (uint4 / packed loads
  into registers) are issued before the barrier and MMAs of step k.
* Unlike stock `kernel_mul_mm`, the src1 tile is 8/16 rows instead of 32, so there is less padding work. Weights
  are read with wide loads (2 x uint4 per lane for Q4_K) instead of 16 byte loads per 16 weights.
* Achieved bandwidth is ~70-75 GB/s on Q4_K/Q8_0 and ~105-110 GB/s on the Q6_K LM head. That is roughly flat
  from B = 2 to 8.
* The prefetch did not raise it further. Per-simdgroup src1 staging (no threadgroup barriers) was tried and was
  worse at B = 9..16. NSG = 2 / 8 were also tried; NSG = 4 is best overall.
* So the kernel is still latency-bound, not bandwidth-bound. A 1-token forward is ~135 GB/s, so there is room left
  (see the upstream notes).

### Final dispatch (`ggml_metal_mv_mr_kernel`, ggml-metal-common.cpp), with `GGML_METAL_MV_MULTIROW=1`

Per-type table from the kernel A/B (run4, AC power). The MMA kernel is used where it beats the stock kernels:
* Q4_K and Q6_K at ne11 = 4..16.
* Q8_0 at ne11 = 5..8. Stock `mul_mv_ext` wins at 2..4 and stock `mul_mm` wins at 9..16.

Everything else stays on the stock path, and so does everything when the variable is unset. The scalar kernel is
reachable only through the tuning override `GGML_METAL_MV_MULTIROW_MMA=N` (MMA for ne11 >= N, scalar below).
Other knobs: `GGML_METAL_MV_MMA_NSG`, `GGML_METAL_MV_MULTIROW_{MAX,R,NR0,NSG}`.

### Kernel A/B, final kernel (run4: `logs/metal_mv_perf_run4.log`, 2026-10-11, AC power), us per matmul

| shape (type) | n=1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 12 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| ffn_up 17408x5120 (Q4_K) stock | 370 | 398 | 599 | 1042 | 1478 | 1505 | 1876 | 2061 | 1256 | 1245 | 1253 |
| same, MMA (NSG 4) | 366 | 689 | 692 | 694 | 704 | 713 | 718 | 719 | 947 | 960 | 997 |
| ffn_down 5120x17408 (Q4_K) stock | 381 | 408 | 636 | 1062 | 1458 | 1581 | 1972 | 2133 | 1449 | 1451 | 1459 |
| same, MMA | 374 | 915 | 930 | 941 | 968 | 983 | 1003 | 1023 | 1274 | 1327 | 1410 |
| attn_qkv 10240x5120 (Q8_0) stock | 406 | 411 | 429 | 515 | 751 | 770 | 963 | 1050 | 708 | 714 | 710 |
| same, MMA | 403 | 644 | 642 | 649 | 651 | 654 | 659 | 667 | 803 | 815 | 834 |
| output 248320x5120 (Q6_K) stock | 7695 | 8645 | 12613 | 21247 | 27451 | 26924 | 38025 | 42627 | 16202 | 16240 | 16317 |
| same, MMA | 7587 | 9221 | 9291 | 9531 | 9553 | 9638 | 9975 | 10107 | 12459 | 12982 | 13708 |

(n=1 is the same stock kernel in both rows, which shows the noise level.)

## Forward cost, llama-bench -p B -n 0 (-fa 1 -ngl 999), B = 1..17

`scripts/gpu.py bench/llama_multirow_sweep.sh kbench`, 2026-10-11 17:18, AC power, same binary, stock / multirow
alternating twice (rounds agree within 1%; log `logs/llama_multirow_kbench_ppl.log`). ms per forward = 1000*B/(tok/s),
mean of the two rounds. Default per-type table (MMA kernel: Q4_K/Q6_K B >= 4, Q8_0 B = 5..8; stock path elsewhere).

| B | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 12 | 14 | 16 | 17 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| stock ms | 149 | 158 | 207 | 298 | 416 | 429 | 528 | 573 | 379 | 380 | 382 | 384 | 382 | 383 |
| multirow ms | 148 | 157 | 207 | 243 | 270 | 277 | 277 | 280 | 328 | 333 | 335 | 342 | 347 | 383 |
| change | 0 | 0 | 0 | -18% | -35% | -35% | -48% | -51% | -13% | -12% | -12% | -11% | -9% | 0 |

A 5..8-token verify forward now costs 1.8-1.9x a 1-token forward instead of 2.8-3.8x. (llama-bench pp only computes
logits for the last token, so the LM head runs at B=1 here; in speculative verification it runs at B, where the MMA
kernel helps even more: 248320x5120 Q6_K at B=8 is 10 ms vs 43 ms stock.)

## Perplexity (wikitext-2 test, -c 128 -b 128, 24 chunks)

| ubatch | stock | multirow | diff |
|---|---:|---:|---:|
| 16 | 9.5565 | 9.5523 | -0.0042 (-4.4e-4 rel.) |
| 5 | 9.5574 | 9.5523 | -0.0051 (-5.3e-4 rel.) |

(+/- 0.66 statistical error.) Within the ~1e-3 relative target. Part of the difference is that the MMA kernel rounds
weights and activations to half before multiplying (as the stock mat-mat kernel does for B >= 9; the stock mat-vec
kernels at B <= 8 are all-f32). Note stock ub=16 and ub=5 differ from each other by 0.0009 as well.


## End-to-end (bench/llama_bench.py --paired via bench/llama_bench_server.py)

`scripts/gpu.py bench/llama_multirow_sweep.sh e2e-stock|e2e-mtp|e2e-dflash|e2e-draft|e2e-mtp2`, 2026-10-11 17:40-22:30.
All runs were on AC power, with the same binary (`build-multirow`). `stock-*` rows = `GGML_METAL_MV_MULTIROW` unset
(stock kernels), `mr-*` rows = set. Every run is paired with a fresh no-spec run of the same prompt; `ref tok/s`
stayed at 6.79-6.88 all evening, so the sessions are comparable. Greedy, thinking off, 256 tokens, 1 run per config.
Raw rows: `results/llama-multirow.jsonl`. Table: `python3 bench/summarize_multirow.py`. Drafters:
`models/mtp-Qwen3.8-27B-q4_0.gguf`, `models/Qwen3.8-27B-DFlash2-q4_0.gguf`, ggml-org Qwen3.5-0.8B Q8_0.

| config | code_edit | code_gen | summarize | prose | explain | math | mean tok/s | ref tok/s | speedup | accepted/drafted | match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| stock-mtp-n2+ngs | 14.44 | 12.56 | 11.77 | 9.32 | 10.44 | 11.36 | **11.65** | 6.87 | 1.695x | 932/1319 (71%) | 6/6 |
| stock-mtp-n3 | 10.80 | 10.91 | 9.04 | 6.54 | 8.00 | 9.83 | **9.19** | 6.87 | 1.337x | 1014/1473 (69%) | 4/6 (diff: explain, math) |
| stock-dflash-n7+ngs | 10.85 | 9.30 | 6.58 | 3.83 | 5.18 | 8.15 | **7.32** | 6.87 | 1.065x | 1170/2368 (49%) | 4/6 (diff: explain, math) |
| mr-mtp-n2 | 12.61 | 12.38 | 11.64 | 9.20 | 9.95 | 11.97 | **11.29** | 6.87 | 1.643x | 915/1174 (78%) | 5/6 (diff: explain) |
| mr-mtp-n2+ngs | 14.47 | 12.54 | 11.76 | 9.31 | 10.00 | 11.24 | **11.55** | 6.79 | 1.700x | 928/1326 (70%) | 5/6 (diff: explain) |
| mr-mtp-n3 | 13.42 | 13.82 | 11.43 | 8.09 | 9.38 | 12.69 | **11.47** | 6.87 | 1.669x | 1012/1482 (68%) | 5/6 (diff: explain) |
| mr-mtp-n3+ngs | 14.34 | 13.84 | 11.45 | 8.09 | 9.40 | 12.07 | **11.53** | 6.86 | 1.681x | 1019/1622 (63%) | 5/6 (diff: explain) |
| mr-mtp-n4 | 14.32 | 13.82 | 11.49 | 7.42 | 8.97 | 12.93 | **11.49** | 6.87 | 1.673x | 1065/1774 (60%) | 5/6 (diff: explain) |
| mr-mtp-n4+ngs | 15.24 | 13.80 | 11.48 | 7.41 | 8.95 | 12.49 | **11.56** | 6.88 | 1.681x | 1071/1904 (56%) | 5/6 (diff: explain) |
| mr-mtp-n5 | 16.43 | 15.44 | 12.00 | 7.21 | 9.04 | 13.39 | **12.25** | 6.87 | 1.783x | 1095/2052 (53%) | 5/6 (diff: explain) |
| mr-mtp-n5+ngs | 17.82 | 15.44 | 11.99 | 7.21 | 9.04 | 12.91 | **12.40** | 6.87 | 1.806x | 1100/2146 (51%) | 5/6 (diff: explain) |
| mr-mtp-n6 | 17.55 | 16.33 | 11.88 | 6.89 | 8.85 | 14.34 | **12.64** | 6.87 | 1.839x | 1113/2341 (48%) | 5/6 (diff: explain) |
| mr-mtp-n6+ngs | 17.80 | 16.32 | 11.89 | 6.90 | 8.85 | 13.60 | **12.56** | 6.88 | 1.826x | 1114/2475 (45%) | 5/6 (diff: explain) |
| mr-dflash-n3 | 13.07 | 13.55 | 11.31 | 8.82 | 10.04 | 12.72 | **11.59** | 6.87 | 1.686x | 1034/1410 (73%) | 5/6 (diff: explain) |
| mr-dflash-n3+ngs | 14.76 | 13.58 | 11.33 | 8.84 | 10.06 | 12.28 | **11.81** | 6.87 | 1.719x | 1044/1509 (69%) | 5/6 (diff: explain) |
| mr-dflash-n4 | 13.77 | 13.79 | 11.58 | 8.00 | 9.39 | 13.33 | **11.64** | 6.84 | 1.703x | 1085/1678 (65%) | 5/6 (diff: explain) |
| mr-dflash-n4+ngs | 14.42 | 13.81 | 11.61 | 8.00 | 9.51 | 13.07 | **11.74** | 6.87 | 1.708x | 1090/1812 (60%) | 5/6 (diff: explain) |
| mr-dflash-n5 | 15.87 | 16.85 | 12.94 | 8.35 | 9.55 | 15.52 | **13.18** | 6.86 | 1.921x | 1125/1896 (59%) | 5/6 (diff: explain) |
| mr-dflash-n5+ngs | 17.11 | 16.87 | 12.97 | 8.36 | 9.55 | 14.64 | **13.25** | 6.87 | 1.928x | 1129/1995 (57%) | 5/6 (diff: explain) |
| mr-dflash-n6 | 17.12 | 18.19 | 13.06 | 7.95 | 9.55 | 16.05 | **13.65** | 6.87 | 1.987x | 1147/2135 (54%) | 5/6 (diff: explain) |
| mr-dflash-n6+ngs | 18.00 | 18.18 | 13.06 | 7.95 | 9.54 | 16.00 | **13.79** | 6.87 | 2.006x | 1151/2189 (53%) | 5/6 (diff: explain) |
| mr-dflash-n7 | 18.96 | 18.20 | 13.24 | 7.69 | 9.00 | 15.75 | **13.81** | 6.87 | 2.009x | 1151/2459 (47%) | 5/6 (diff: explain) |
| mr-dflash-n7+ngs | 20.88 | 18.20 | 13.26 | 7.69 | 8.99 | 15.78 | **14.13** | 6.87 | 2.057x | 1156/2465 (47%) | 5/6 (diff: explain) |
| mr-q35-0.8b-n16-p0.7 | 17.88 | 15.46 | 11.88 | 7.82 | 7.90 | 14.45 | **12.56** | 6.88 | 1.828x | 931/1135 (82%) | 5/6 (diff: explain) |
| mr-mtp-n7 | 18.63 | 16.98 | 12.32 | 6.65 | 8.43 | 15.43 | **13.07** | 6.87 | 1.904x | 1132/2618 (43%) | 5/6 (diff: explain) |
| mr-mtp-n8 | 16.52 | 14.89 | 11.25 | 5.74 | 7.35 | 13.64 | **11.56** | 6.87 | 1.683x | 1135/2938 (39%) | 5/6 (diff: explain) |
| mr-mtp-n10 | 17.96 | 14.78 | 10.60 | 5.42 | 6.94 | 14.13 | **11.64** | 6.87 | 1.694x | 1150/3549 (32%) | 5/6 (diff: explain) |
| mr-mtp-n8+ngs | 18.52 | 14.91 | 11.25 | 5.74 | 7.36 | 13.65 | **11.91** | 6.87 | 1.732x | 1140/2938 (39%) | 5/6 (diff: explain) |
| mr-mtp-n8-pmin0.3 | 16.52 | 15.09 | 11.88 | 6.76 | 7.67 | 13.74 | **11.94** | 6.87 | 1.738x | 1134/2578 (44%) | 5/6 (diff: explain) |

Before/after for the configs that matter:

| config | stock kernels | multirow | change |
|---|---:|---:|---:|
| MTP n_max 2 + ngram-simple (previous best) | 11.61 | 11.55 | same (B=3 verify does not touch the new kernel) |
| MTP n_max 3 | 9.19 | 11.47 | +25% |
| MTP n_max 6 / 7 | (n=4 was 6.90 in the old notes) | 12.64 / **13.07** | |
| DFlash2 n_max 7 + ngram-simple | 7.30 | **14.13** | +94% |
| DFlash2 n_max 6 + ngram-simple | (n=5 was 8.31 in the old notes) | 13.79 | |
| Qwen3.5-0.8B draft, n-max 16, p-min 0.7 | 10.78 (old notes, 1.61x) | 12.56 (1.83x) | +16% by speedup |

Findings:
* With cheap 5..8-token verification, the optimum moves to the longest drafts the drafter can produce.
  * DFlash2 is capped at 7 by its block size. It improves monotonically: n=3 11.81, n=4 11.74, n=5 13.25,
    n=6 13.79, n=7 14.13 (all + ngram-simple). On code it reaches 18-21 tok/s (code_edit 20.9, code_gen 18.2);
    prose 7.7, explain 9.0.
  * MTP peaks at n_max 7 (13.07) and drops at 8 (11.56) and 10 (11.64). Its chained acceptance falls (43% -> 39%),
    and at B >= 9 the verify cost steps up again (280 -> 330 ms).
* DFlash2 beats MTP at long drafts. It drafts all 7 positions in one forward, while MTP runs 7 sequential steps
  through the 248k-vocab LM head.
* ngram-simple adds +0.1-0.3 tok/s (mostly code_edit), as before.

### Output match (diagnosed)

* Every patched config reproduces the baseline on 5/6 prompts. `explain` diverges at char 572, with the same text
  (sha 72664c05) in all 26 patched configs, n_max 2 included:
  * baseline: `[A]  [ ]  [B]->[C]  [ ]  [D]`
  * patched:  `[B]->[D]  [C]`
  * Both are part of an ASCII bucket diagram; the text stays fluent.
* Cause, found by logging which matmuls take the new path (`GGML_METAL_MV_MULTIROW_LOG` instrumentation, since
  removed): the server evaluates the **last 11 tokens of the prompt as their own ubatch**, after the
  context-checkpoint split for the recurrent state. Those prompt-tail matmuls (ffn_* Q4_K, attn_output Q6_K,
  ne11 = 11) run through the MMA kernel. That kernel rounds activations and weights to half, like stock mul_mm
  does for B >= 9, but unlike the all-f32 stock mat-vec kernels. A near-tie later in `explain` flips. This is a
  float-rounding near-tie, not a logic bug: test-backend-ops passes and perplexity agrees to 5e-4.
* The patched n >= 3 configs do **not** show the stock build's `explain`@156 / `math`@55 flips. Those came from
  `mul_mv_ext`'s accumulation order, which the MMA kernel now replaces.
* The 0.8B-draft config also no longer flips `math`.
* For bit-identical output with the stock build on all prompts, use `GGML_METAL_MV_MULTIROW` unset with
  MTP n_max 2 (11.6 tok/s). The patch trades one near-tie for +22%.


## Best command (patched build)

```bash
GGML_METAL_MV_MULTIROW=1 vendor/llama.cpp/build-multirow/bin/llama-server \
  -m ~/.cache/huggingface/hub/models--ggml-org--Qwen3.8-27B-GGUF/snapshots/efbb3b1f70a21d97fd4495240648405f7228554f/Qwen3.8-27B-Q4_K_M.gguf \
  -md models/Qwen3.8-27B-DFlash2-q4_0.gguf \
  --spec-type draft-dflash,ngram-simple --spec-draft-n-max 7 \
  -ngl 999 -fa on -c 8192 --jinja
```
Measured: 14.13 tok/s mean (2.06x), code 18-21 tok/s.

Rebuild from scratch:
```bash
cd vendor/llama.cpp && git apply ../../scripts/metal-multirow-mv.patch
cmake -B build-multirow -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON -DGGML_METAL_EMBED_LIBRARY=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build build-multirow -j 10 -t llama-server llama-bench llama-perplexity test-backend-ops
```

MTP alternative (no DFlash sidecar needed): the same command with `-md models/mtp-Qwen3.8-27B-q4_0.gguf
--spec-type draft-mtp --spec-draft-n-max 7`, 13.07 tok/s.

## Upstream notes

* **What's upstreamable:** the MMA small-batch kernel plus its dispatch rule. The scalar multi-row kernel is not
  worth upstreaming: it loses from B >= 5, and its only wins at B = 2-4 are small. For upstreaming, drop it and the
  env knobs, and make the MMA path the default for the measured (type, B) ranges. Gate it on the
  `has_simdgroup_mm && !has_tensor` path (pre-M5); M5's tensor-API mm has different trade-offs and was not measured.
* **The upstream argument, in numbers:** stock Metal on an M3 Pro makes a 5-8-token forward 2.8-3.8x a 1-token
  forward. That is because `mul_mv_ext` does two passes for B = 6..8 and is load-bound, and `mul_mm` pads to
  32 columns.
  * With the MMA kernel the ratio is 1.8-1.9x. It also fixes the non-monotonic stock curve, where B = 9 is cheaper
    than B = 5.
  * Every speculative-decoding user on Apple Silicon (MTP, DFlash, draft models) hits this range.
  * The kernel A/B table and the llama-bench table above are the evidence. test-backend-ops perf cases for the
    shapes are in the patch. Reviewers will want the same numbers on M1/M2/M4 Max; the crossovers may move.
* **Before a PR:**
  1. Cover more types. This patch has only Q4_K / Q6_K / Q8_0. Q4_0/Q5_K/IQ4 etc. need a `mv_mma_load/store` pair
     each (same 64-weights-per-lane layout). MUL_MAT_ID (MoE) is untouched.
  2. K must be a multiple of 128 (checked in dispatch). Add a tail path or keep the check.
  3. The src1 16-byte alignment check reads `src1->data`. Upstream may prefer `ggml_is_contiguous_rows` plus an
     offset check in `supports_op` style.
  4. The kernel is still latency-bound (~70 GB/s vs ~135 GB/s for the 1-row kernel). Ideas:
     - Keep 2 K steps of raw weights in flight.
     - Use more rows per lane (32 rows x 64 per lane).
     - Skip the threadgroup-memory round trip for A by building the 8x8 half fragments in registers
       (simdgroup_matrix thread_elements()).
     Reaching ~110 GB/s would cut another ~70 ms off each 5-8-token forward.
  5. Numerics: half rounding of activations at B = 4..8 is new for those sizes (the stock mat-vec kernels are f32).
     It is the same precision as stock mm at B >= 9, and perplexity is unchanged at the 5e-4 level, but say so in
     the PR.
* **Tests added** (in the patch): test-backend-ops MUL_MAT eval cases for n = 2..17 on Q4_K/Q6_K/Q8_0, with odd
  row counts and broadcast; perf cases with Qwen3.8-27B shapes.


## Gotcha found on the way

`scripts/gpu.py` detects an already-running model server with `pgrep -f "llama-server|mlx_spec serve"`. Other queued jobs' wait
loops of the form `until ! pgrep -f "llama-server -m" ...` contain that string in their own command line, so
they (a) never terminate and (b) keep gpu.py waiting even after the real server exits. If the queue
stays blocked with no real `llama-server` binary running, check `pgrep -fl llama-server` for zsh wrappers.
