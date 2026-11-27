# MLX with the same weights as llama.cpp's Q4_K_M

**Summary.** Plan A works. ggml-org's Q4_K_M GGUF transcodes to an MLX
checkpoint with the same 4-/8-bit codes (`models/qwen38-27b-q4km-mlx`, 21.1 GB).
On 16 x 2048 wikitext-2 tokens it scores **PPL 6.131** against llama.cpp's
**6.122** for the GGUF, with KL 0.0024 to llama.cpp's own distribution and the
same top-1 token 98.0 % of the time. mlx-community's 4-bit scores **6.275**
(KL 0.068, top-1 agreement 89.9 %), so it really was the dumber model. Greedy
outputs match llama.cpp's text exactly on 4 of 6 prompts, and the other 2
diverge at exact or one-ulp ties. The price is speed: the file is 31 % bigger,
so greedy drops from 8.42 to 5.71 tok/s and MTP k=3 from 18.45 to 15.76 tok/s
(same session, on AC, but with other jobs' CPU load). That is still 1.4x
llama.cpp's best (11.4 tok/s) on the same weights.

## Why

Track B's MLX numbers used `mlx-community/Qwen3.8-27B-4bit` (affine 4-bit,
group 64, every matrix incl. attention/SSM and lm_head; 16.1 GB), while llama.cpp
runs ggml-org's `Qwen3.8-27B-Q4_K_M.gguf` (19.0 GB), which keeps all
attention/SSM matrices at 8 bits. MLX was faster partly because its model is
smaller and possibly worse. Here the GGUF is transcoded to MLX so both engines
run the same weights.

## Plan A worked: exact transcode (`scripts/gguf_to_mlx.py`)

The per-tensor types in the GGUF (from the file itself, not the recipe):

| GGUF tensor | count | GGUF type | mlx-lm parameter | MLX format |
|---|---|---|---|---|
| `blk.N.ffn_{gate,up,down}` | 64 x 3 | Q4_K | `mlp.{gate,up,down}_proj` | affine 4-bit, group 32 |
| `token_embd` | 1 | Q4_K | `model.embed_tokens` | affine 4-bit, group 32 |
| `blk.N.attn_qkv` / `attn_gate` / `ssm_alpha` / `ssm_beta` / `ssm_out` | 48 each | Q8_0 | `linear_attn.in_proj_{qkv,z,a,b}`, `out_proj` | affine 8-bit, group 32 |
| `blk.N.attn_{q,k,v}` | 16 each | Q8_0 | `self_attn.{q,k,v}_proj` | affine 8-bit, group 32 |
| `blk.N.attn_output` | 16 | **Q6_K** | `self_attn.o_proj` | affine 8-bit, group 32 (re-quantized) |
| `output` | 1 | **Q6_K** | `lm_head` | affine 8-bit, group 32 (re-quantized) |
| norms, `ssm_conv1d`, `ssm_dt.bias`, `ssm_norm` | | F32 | `*norm.weight`, `conv1d.weight`, `dt_bias` | bf16 |
| `blk.N.ssm_a` | 48 | F32 | `linear_attn.A_log` | float32, `log(-ssm_a)` |

So the recipe is not quite "`--pure` Q4_K with attn/ssm at Q8_0 and output at
Q6_K": the full-attention output projections are Q6_K too, and the
linear-attention output (`ssm_out`) is Q8_0.

### Format mapping

MLX affine quantization: `w = scale * q + bias` per group, `q` unsigned,
packed little-endian into uint32 (element i at bits `bits*(i % (32/bits))`),
group sizes 32/64/128 only, scales/biases in the activation dtype.

* **Q4_K** (super-block of 256: fp16 `d`, `dmin`; 8 sub-blocks of 32 with 6-bit
  `sc`, `m`; 4-bit `q`): `w = d*sc*q - dmin*m` is affine 4-bit group 32 with
  `scale = d*sc`, `bias = -dmin*m` and the **same 4-bit codes**. The nibble
  order differs (Q4_K stores sub-blocks 2j and 2j+1 as low/high nibbles of the
  same 32 bytes), so codes are unpacked and repacked.
* **Q8_0** (32-blocks: fp16 `d`, int8 `q`): `w = d*q` is affine 8-bit group
  32 with codes `q + 128`, `scale = d`, `bias = -128*d`. Same codes.
* **Q6_K** (16-element sub-blocks with int8 scales): MLX has no group 16
  (`mx.quantize(..., group_size=16)` -> "supported group sizes are 32, 64,
  and 128"), and two Q6_K sub-blocks with different scales can't share one
  affine 8-bit group exactly. These 17 tensors (o_proj x 16, lm_head) are
  re-quantized from the exact Q6_K values to 8-bit group 32, with the bias
  rounded down and the scale up to bf16 so that every value has a code and the
  error is at most half an 8-bit step (no clipping). `--q6k bf16` stores them
  as bf16 instead (+1.6 GB).
* **F32**: kept as bf16. conv1d, dt_bias and ssm_norm are exactly bf16
  values; the RMSNorm weights are `1 + w` computed in f32 by llama.cpp's
  converter and are not, so they round like mlx-community's (bit-identical to
  mlx-community's, checked). `ssm_a = -exp(A_log)` -> `A_log = log(-ssm_a)`
  in float32 (mlx-lm leaves A_log uncast), which reproduces llama.cpp's
  `ssm_a` to float32 rounding (mlx-community's bf16 A_log does not).

**Scale dtype.** Products `d*sc` (fp16 x 6-bit) and Q8_0's fp16 `d` aren't
always representable in bf16. fp16 scales would be exact for Q8_0 and closer
for Q4_K, but `quantized_matmul` with bf16 activations and fp16 scales
promotes to float32 (checked: output dtype float32), i.e. converts the scales
on every call and pushes the whole model to fp32 activations. So scales and
biases are bf16, like mlx-community's; per group the converter picks
`(scale', bias')` among the bf16 neighbours of `(scale, bias + (scale -
scale')*mean(q))` to minimise the group's squared error (the codes stay the
GGUF's). This is the only approximation besides Q6_K, and it is at the level
of bf16 activation rounding (2^-9 relative), which every MLX bf16 model has
anyway.

**Layout changes undone** (llama.cpp `conversion/qwen.py`,
`_LinearAttentionVReorderBase` + `Qwen3NextModel`): the 48 value heads of the
linear-attention layers are stored "tiled" in the GGUF (`[v0 of k0..k15, v1 of
k0..k15, v2 ...]`) instead of grouped by key head (`[k0: v0 v1 v2, k1: ...]`);
undone for in_proj_qkv's V rows, in_proj_z / in_proj_a / in_proj_b rows,
out_proj's input columns (head blocks of 128 = 4 whole quant groups, so the
codes just move), conv1d's V channels, A_log and dt_bias. `ssm_a = -exp(A_log)`
inverted, conv1d `[10240, 4]` -> `[10240, 4, 1]`, `+1` on norms kept (mlx-lm's
`sanitize` only adds it to raw HF checkpoints, recognised by the conv1d
layout). Full attention q_proj (q and gate interleaved per head) is stored in
HF order in the GGUF; nothing to undo. Vision tower and MTP aren't in this GGUF.

Streaming: the converter reads the GGUF mmap'd one tensor (or 16M-element row
chunk) at a time and `pwrite`s into pre-sized safetensors shards; 4.5 min on
the CPU including the checks below. Output: 21.1 GB in 4 shards.

### Exactness checks

**(1) Against gguf-py's dequantization** (`--check`, every element of every
tensor; `max err` in units of that group's MLX scale; `rel rms` = ||ours -
gguf|| / ||gguf||):

| GGUF type -> MLX | tensors | max err (MLX steps) | rel rms err | codes |
|---|---|---|---|---|
| Q4_K -> 4-bit g32 | 193 | 0.105 | 0.36 % | identical |
| Q8_0 -> 8-bit g32 | 288 | 0.496 | 0.16 % | identical |
| Q6_K -> 8-bit g32 | 17 | 0.500 | 0.48 % | re-quantized |
| (Q6_K -> bf16, option) | | | 0.16 % | |

For scale: the quantization noise of the formats themselves, relative to the
bf16 originals, is ~7 % (Q4_K) and ~0.7 % (Q8_0) of the weight norm (estimated
in (2)), a few % for Q6_K. The transcode adds its 0.36 / 0.16 / 0.48 % in
quadrature: about +0.3 % of the existing noise *variance* for Q4_K, +5 % for
Q8_0 and a few % for the Q6_K tensors. In other words, the Q8_0 tensors become
"Q8_0 with 2.5 % more rms noise", the rest is indistinguishable.

**(2) Against mlx-community's 4-bit** (`scripts/check_transcode_vs_mlxcommunity.py`,
all 1847 tensors, 2048 random rows per matrix): both are quantizations of the
same bf16 weights, so a correct mapping gives a small difference and any wrong
permutation ~1.4. Result: correlation 0.993-0.996 for every matrix (relative
difference 0.094-0.121); norms, conv1d, dt_bias bit-identical; A_log within
float32 rounding of mlx-community's bf16 value. Same key set (1847 language
model tensors), and `mlx_lm.load` loads it strictly with the per-module
quantization from `config.json`.

The relative differences also measure mlx-community's 4-bit noise: 0.094 on
the tensors that are Q8_0 in the GGUF (where the GGUF side adds only ~0.7 %),
0.117-0.121 on the Q4_K MLP (so Q4_K itself is at ~sqrt(0.119^2 - 0.094^2)
~ 7 %). mlx-community's 4-bit g64 is ~30 % noisier than Q4_K on the MLP and
~13x noisier than Q8_0 on attention/SSM.

## Measurements (2026-10-10, on AC power, GPU lock)

### Perplexity and KL vs llama.cpp

wikitext-2-raw test, `llama-perplexity -c 2048 --chunks 16` convention: 16
non-overlapping chunks of 2048 tokens, no BOS, second half of each chunk scored
(16,368 tokens). The MLX side (`bench/mlx_ppl.py`) reads the token ids and
llama.cpp's log-probs from the `--kl-divergence-base` file, so all three score
identical tokens. HF tokenization of the file was checked to match llama.cpp's
token for token. KL is KL(llama.cpp Q4_K_M || model), computed as in
`llama-perplexity --kl-divergence`. There is no bf16 reference (that would need
the 54 GB download), so KL measures fidelity to the GGUF, not to the original model.

| model | size | PPL | KL vs llama.cpp mean / median / p99 | same top-1 | RMS Δp |
|---|---|---|---|---|---|
| llama.cpp, ggml-org Q4_K_M (f32 activations) | 19.0 GB | **6.1223 ± 0.116** | (reference) | | |
| MLX, Q4_K_M transcode (ours) | 21.1 GB | **6.1308 ± 0.116** | 0.0024 / 0.0004 / 0.0074 | 98.0 % | 1.6 % |
| MLX, ours, lm_head in fp32 | | 6.1294 | 0.0022 / 0.0002 / 0.0071 | 99.0 % | 1.6 % |
| MLX, mlx-community 4-bit g64 | 16.1 GB | **6.2746 ± 0.119** | 0.068 / 0.026 / 0.52 | 89.9 % | 7.0 % |
| MLX, mlx-community, fp32 lm_head | | 6.2730 | 0.068 / 0.026 / 0.53 | 90.1 % | 7.0 % |

(The uint16-quantized base log-probs give 6.1207 for llama.cpp itself; the
difference from 6.1223 is that quantization.) Ours is within 0.14 % of
llama.cpp's perplexity. The residual KL (0.0024) comes from bf16 activations,
not the weights: the per-tensor transcode error is far smaller than bf16
activation rounding, and half of the top-1 disagreements go away just by
computing lm_head in fp32. mlx-community's 4-bit is +2.5 % perplexity and 28x
the KL. That is a real quality loss.

### Greedy outputs vs llama.cpp (`results/mlx_matched_quant_vs_llama.jsonl`)

Compared with llama.cpp's baseline texts (`results/baseline_outputs/Qwen3.8-27B-Q4_K_M`,
256 tokens, same chat template, which is byte-identical in both). `diagnose()`
gives our model's single-step logit margin at the first differing token:

| prompt | ours (Q4_K_M transcode) | mlx-community 4-bit |
|---|---|---|
| code_edit | **identical** | diverges at token 200 (margin 0.375) |
| code_gen | **identical** | token 33 (0.375) |
| summarize | **identical** | token 63 (0.375) |
| prose | token 46, exact tie (margin 0.0) | token 23 (0.25) |
| explain | **identical** | token 0 (4.75: different first word) |
| math | token 13, one bf16 ulp (0.125) | token 8 (0.125) |

### Speed (`bench/mlx_matched_quant.py`, `results/mlx_matched_quant.jsonl`)

`Runner.bench`, 6 prompts, 256 tokens, median of 2 runs, both models in one
session on AC power. Other CPU jobs were running (load average 3.5-7),
so absolute numbers are below the clean Track B session: mlx-community's
MTP k=3 gives 18.45 here vs 23.02 before, and its verify round is 170 ms vs
125 ms. The ratio between the two models is the useful number.

| model | greedy tok/s | MTP k=3 vocab 65536 tok/s | MTP accept (pos 1/2/3) | verify ms/round |
|---|---|---|---|---|
| mlx-community 4-bit | 8.42 | 18.45 (2.19x) | 0.71 (0.87/0.79/0.76) | 170 |
| **ours, Q4_K_M transcode** | **5.71** | **15.76 (2.76x)** | 0.70 (0.86/0.77/0.78) | 181 |
| llama.cpp Q4_K_M (REPORT.md) | 6.64 | 11.39 (MTP 2 + ngram) | | |

Per prompt, ours MTP: 19.4 / 18.4 / 16.3 / 11.1 / 13.2 / 16.2 (code_edit,
code_gen, summarize, prose, explain, math). MTP acceptance with the bf16-extracted
MTP head is the same as with mlx-community (0.70 vs 0.71). Greedy is 0.68x
mlx-community because 21.1 GB vs 16.1 GB is read per token (0.76x expected from bytes alone). Speculation hides part of it
(0.85x), because verify cost scales less than linearly with weight bytes at T=4.
The MTP outputs of ours diverge from our own greedy on 3/6 prompts at the
usual bf16 batched-vs-single near-ties (Track B), as they do for mlx-community.
A clean rerun (quiet machine, AC) should be done before quoting absolute tok/s.

## Recommendation

* Use `models/qwen38-27b-q4km-mlx` as the MLX target when quality matters. It is
  the llama.cpp model, at llama.cpp's perplexity to within 0.14 %, and with
  MTP k=3 it still runs ~1.4x faster than llama.cpp's best config on the same
  weights (15.8 vs 11.4 tok/s, this session's conditions).
* mlx-community's 4-bit is measurably worse (+2.5 % PPL, KL 0.068). Its speed
  advantage comes from dropping attention/SSM from 8 to 4 bits.
* For the kernel work, the target now has three matmul shapes: 4-bit g32 (MLP,
  embedding), 8-bit g32 (attention, GDN projections, o_proj, lm_head), bf16
  scales/biases throughout. The 8-bit g32 matvec path matters now. It carries
  ~7.5 GB of the ~20 GB read per token.
* Options not taken: `--q6k bf16` (Q6_K tensors exactly to bf16 rounding,
  +1.6 GB, slower; the PPL/KL above leave nothing for it to fix); fp32
  activations (would remove the residual KL but cost speed); Plan B (bf16
  download) is unnecessary.

## Reproduce

```
.venv/bin/python scripts/gguf_to_mlx.py --check                  # CPU, 4.5 min, 21.1 GB
.venv/bin/python scripts/check_transcode_vs_mlxcommunity.py
scripts/gpu.py sh bench/llama_ppl.sh                              # writes models/wikitext/q4km-base.kld (8.1 GB)
scripts/gpu.py sh bench/mlx_matched_quant_session.sh ppl
scripts/gpu.py sh bench/mlx_matched_quant_session.sh speed
scripts/gpu.py .venv/bin/python -m mlx_spec bench --target models/qwen38-27b-q4km-mlx --configs mtp:k=3:vocab=65536
```
