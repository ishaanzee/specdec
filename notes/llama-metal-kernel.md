# llama.cpp Metal: a multi-row mat-vec kernel for 2-16 token verify batches (M3 Pro)

Status: **kernel implemented and correct (test-backend-ops); all GPU timings still pending** — the GPU lock
(`scripts/gpu.py`) has been waiting on the user's own llama-server (pid 70535, idle, up since ~18:00) all night.
Queued: `bench/metal_mv_perf.sh` (kernel A/B), then `bench/llama_multirow_sweep.sh kbench|ppl|e2e-*`.
Tables marked TODO get filled from those runs.

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
plus dequant/threadgroup traffic, not bandwidth-bound** (~50 GB/s effective). TODO: confirm with the perf cases
`m=17408,k=5120,n=32/64/128/512` (if n=64 ~ 2x n=32 the per-tile cost is compute).

## Kernel design (`kernel_mul_mv_mr_{q4_K,q6_K,q8_0}_f32_r<R>_nr0_<NR0>`, kernels/mul_mv.metal)

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

Perplexity (wikitext-2 test, -c 128, ubatch 16 and 5, 24 chunks), stock vs multirow: TODO.

## Kernel timings (test-backend-ops perf, model shapes) — TODO

## Forward cost, llama-bench -p B -n 0 (-fa 1 -ngl 999), B = 1..17 — TODO

## End-to-end (bench/llama_bench.py --paired via bench/llama_bench_server.py) — TODO

## Upstream notes — TODO

## Gotcha found on the way

`scripts/gpu.py` detects "the user's server" with `pgrep -f "llama-server|mlx_spec serve"`. Other agents' wait
loops of the form `until ! pgrep -f "llama-server -m" ...` contain that string in their own command line, so
they (a) never terminate and (b) keep gpu.py waiting even after the user's real server exits. If the queue
stays blocked with no real `llama-server` binary running, check `pgrep -fl llama-server` for zsh wrappers.
