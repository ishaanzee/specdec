# MLX with the same weights as llama.cpp's Q4_K_M

**Summary.** Plan A works: ggml-org's Q4_K_M GGUF transcodes to an MLX
checkpoint with the same 4-/8-bit codes (`models/qwen38-27b-q4km-mlx`, 21.1 GB).
The only differences are bf16 rounding of the scales/biases and 8-bit
re-quantization of the 17 Q6_K tensors. Perplexity, KL vs llama.cpp and speed:
see "Measurements" (they run under the GPU lock).

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

## Measurements

Queued under `scripts/gpu.py` (they wait while the user's own llama-server
runs): `bench/llama_ppl.sh` (llama.cpp PPL on 16 x 2048 wikitext-2 tokens + base
log-probs), `bench/mlx_matched_quant_session.sh ppl` (MLX PPL and KL vs
llama.cpp, same tokens, both MLX models, bf16 and fp32 lm_head) and `... speed`
(greedy and `mtp:k=3:vocab=65536`, both models, same session, plus output
comparison against llama.cpp's baseline texts with `diagnose()`). Rows go to
`results/quality_ppl.jsonl`, `results/mlx_matched_quant.jsonl`,
`results/mlx_matched_quant_vs_llama.jsonl`, each with the power source
(timings on battery are suspect).

PENDING

