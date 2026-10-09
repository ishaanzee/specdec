# llama.cpp: MTP and DFlash drafters for Qwen3.8-27B (Track A, part 2)

Source read: `vendor/llama.cpp` at `b29c606e2` (tag v0.4.1, same commit as the Homebrew build 10964).

## How this build expects the drafters

### Sidecar convention (what ggml-org ships)

llama.cpp's convention is a *main GGUF without MTP* plus separate sidecar GGUFs used as the draft model (`-md`).
`common/download.cpp` even resolves `mtp-*.gguf` / `dflash-*.gguf` sidecars automatically for `-hf` repos.
The ggml-org repo `ggml-org/Qwen3.8-27B-GGUF` actually contains them (we only had the main file cached):

| file | size |
|---|---|
| Qwen3.8-27B-Q4_K_M.gguf (the one we run) | 18.97 GB |
| mtp-Qwen3.8-27B-{BF16,Q8_0,Q4_0}.gguf | 5.95 / 3.16 / 1.68 GB |
| dflash-Qwen3.8-27B-{BF16,Q8_0,Q4_0}.gguf | 3.86 / 2.06 / 1.09 GB |

Their `convert.log` shows the exact recipe (useful for reproducing the main file too):

```
convert_hf_to_gguf.py PRIMARY --outtype bf16 --no-mtp           --outfile Qwen3.8-27B-BF16.gguf
convert_hf_to_gguf.py PRIMARY --outtype bf16 --mtp              --outfile mtp-Qwen3.8-27B-BF16.gguf
convert_hf_to_gguf.py DFLASH  --outtype bf16 --target-model PRIMARY --outfile dflash-Qwen3.8-27B-BF16.gguf
llama-quantize --pure --tensor-type output.weight=q6_k --tensor-type shexp=q8_0 --tensor-type latent=q8_0 \
    --tensor-type attn_=q8_0 --tensor-type ssm_=q8_0  BF16.gguf Q4_K_M.gguf Q4_K_M
llama-quantize mtp-BF16.gguf mtp-Q8_0.gguf Q8_0 ;  llama-quantize --pure mtp-BF16.gguf mtp-Q4_0.gguf Q4_0
llama-quantize dflash-BF16.gguf dflash-Q8_0.gguf Q8_0 ; llama-quantize --pure dflash-BF16.gguf dflash-Q4_0.gguf Q4_0
```

So ggml-org's "Q4_K_M" is really: all matrices Q4_K (`--pure`), every `attn_*`/`ssm_*` tensor Q8_0, `output` Q6_K.
That explains why it is 19.0 GB rather than ~16.5 GB.

### MTP (`--spec-type draft-mtp`)

* Converter (`conversion/qwen.py`, `_QwenMtpMixin`, used by `Qwen3_5ForConditionalGeneration`):
  MTP is **kept by default**. `--no-mtp`/`--no-nextn` strips it (that is what ggml-org did for the main file),
  `--mtp` exports *only* the MTP block + `token_embd`, `output_norm`, `output` as `mtp-<name>.gguf`.
  HF `mtp.layers.0.*` -> `blk.64.*` (appended after the 64 trunk layers, `block_count` = 65,
  `qwen35.nextn_predict_layers = 1`); `mtp.fc` -> `blk.64.nextn.eh_proj`, `mtp.pre_fc_norm_embedding` -> `nextn.enorm`,
  `mtp.pre_fc_norm_hidden` -> `nextn.hnorm`, `mtp.norm` -> `nextn.shared_head_norm`.
* Runtime (`common/speculative.cpp`, `common_speculative_init_result`): with `--spec-type draft-mtp`
  * **with `-md mtp-*.gguf`**: the sidecar is loaded as a separate model (qwen35 arch; `src/models/qwen35.cpp`
    detects "MTP-only" when `blk.0.attn_norm` is missing and makes the trunk tensors optional), context type
    `LLAMA_CONTEXT_TYPE_MTP`, `ctx_other = target ctx`.
  * **without `-md`**: a second context of type MTP is created on the *target model itself*; the target must
    then contain `blk.64.*` (`mparams.load_mtp` is set only when draft-mtp is requested, otherwise the MTP
    tensors are skipped at load time, so a combined GGUF costs no RAM for plain decoding).
* Qwen3.5/3.8 MTP graph (`llama_model_qwen35::graph_mtp`): only 1 MTP block supported (asserts `n_layer_nextn == 1`).
  Multi-step drafting reuses the same head autoregressively (the "neither" mode in `common_speculative_impl_draft_mtp`):
  step k feeds the MTP's own output hidden (`h_nextn` = post-`shared_head_norm`) and the token it just drafted.
  `--spec-draft-n-max` = number of chained MTP steps. Drafting uses a top-k=10 sampler and takes the argmax;
  `--spec-draft-p-min` stops drafting below that probability.

### DFlash (`--spec-type draft-dflash`)

* Converter: `DFlashModel` in `conversion/qwen.py` registers both `DFlashDraftModel` and **`DFlash2DraftModel`**,
  so z-lab/Qwen3.8-27B-DFlash2 converts directly. Requires `--target-model-dir <dir with target config.json +
  tokenizer>` (only the tokenizer/config are read, not target weights). Writes arch `dflash`, `dflash.block_size`,
  conv kernel/group size, selector rank/top-k, `target_layers` (= HF `target_layer_ids` + 1), SWA pattern, M-RoPE
  degenerate sections, mask token id.
* Runtime: separate GGUF via `-md`. The draft has **no token embedding / LM head of its own** (full vocab); it
  shares the target's `token_embd` and `output` through `ctx_other`.
* `--spec-draft-n-max` is clamped to the trained block size (8 for DFlash2 -> at most 7 draft tokens + anchor).

## What we built (reproduce)

All models in `models/` (gitignored). `vendor/llama.cpp` = source at b29c606e2 (for the converter + gguf-py).
Python: `.venv` + `uv pip install --python .venv/bin/python torch` (transformers 5.19 from the venv works with this
converter; the converter puts `vendor/llama.cpp/gguf-py` on sys.path itself).

### DFlash2 drafter

The converter only needs the target's `config.json` + tokenizer. Before the 54 GB download finished we used
`models/qwen38-target-meta/` = Qwen/Qwen3.8-27B `config.json` + tokenizer files from mlx-community/Qwen3.8-27B-4bit
(same tokenizer); `--target-model-dir ~/.cache/.../Qwen--Qwen3.8-27B/snapshots/<rev>` works the same now.

```bash
D=~/.cache/huggingface/hub/models--z-lab--Qwen3.8-27B-DFlash2/snapshots/50307d4c4cde6860d4eee73e2547cd786fe8e8a4
scripts/gpu.py .venv/bin/python vendor/llama.cpp/convert_hf_to_gguf.py $D \
    --target-model-dir models/qwen38-target-meta --outtype bf16 --outfile models/Qwen3.8-27B-DFlash2-bf16.gguf
llama-quantize        models/Qwen3.8-27B-DFlash2-bf16.gguf models/Qwen3.8-27B-DFlash2-q8_0.gguf Q8_0   # 2.06 GB
llama-quantize --pure models/Qwen3.8-27B-DFlash2-bf16.gguf models/Qwen3.8-27B-DFlash2-q4_0.gguf Q4_0   # 1.09 GB
```
(81 tensors, 3.86 GB bf16; byte sizes identical to ggml-org's dflash-* sidecars; only `general.name` differs.)
`bench/llama_mtp_dflash_sweep.sh dflash-quant` does the two quantizations.

### MTP sidecar (our own, from the official weights)

```bash
Q=~/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/<rev>
scripts/gpu.py .venv/bin/python vendor/llama.cpp/convert_hf_to_gguf.py $Q --mtp --outtype bf16 \
    --model-name Qwen3.8-27B --outfile models/mtp-Qwen3.8-27B-bf16.gguf          # 5.95 GB, 18 tensors
llama-quantize        models/mtp-Qwen3.8-27B-bf16.gguf models/mtp-Qwen3.8-27B-q8_0.gguf Q8_0   # 3.16 GB
llama-quantize --pure models/mtp-Qwen3.8-27B-bf16.gguf models/mtp-Qwen3.8-27B-q4_0.gguf Q4_0   # 1.68 GB
rm models/mtp-Qwen3.8-27B-bf16.gguf
```
(`bench/llama_mtp_dflash_sweep.sh mtp-own-convert`.) Only shards 3 (embed_tokens), 16 (final norm) and 18
(lm_head + all `mtp.*`) are read. **Our Q8_0 sidecar is tensor-for-tensor byte-identical to
`ggml-org/Qwen3.8-27B-GGUF/mtp-Qwen3.8-27B-Q8_0.gguf`** (all 18 tensors, same types; only `general.finetune`
metadata differs). ggml-org's `.src_sha` says PRIMARY=1d4bf0f2..., the same revision we downloaded.

### Single-file Q4_K_M with the MTP layer

Disk was the constraint: after the downloads there were ~47 GB free, and the "convert to q8_0 (28.6 GB) ->
llama-quantize to Q4_K_M (19 GB)" route needs ~48 GB at the peak (plus it would requantize from q8_0, which is
lossier than ggml-org's bf16 -> Q4_K_M). Instead we exploited the fact that ggml-org's main file was produced
from the same source revision with `--no-mtp`: we splice the MTP block into it.

```bash
.venv/bin/python scripts/splice_mtp_gguf.py \
    ~/.cache/.../ggml-org--Qwen3.8-27B-GGUF/snapshots/<rev>/Qwen3.8-27B-Q4_K_M.gguf \
    models/mtp-Qwen3.8-27B-q8_0.gguf  models/Qwen3.8-27B-Q4_K_M-mtp.gguf
```
`scripts/splice_mtp_gguf.py` copies the 851 trunk tensors + all metadata byte-for-byte, appends the 15 `blk.64.*`
tensors (Q8_0; the sidecar's own `token_embd`/`output`/`output_norm` are dropped since the trunk's are shared), and
sets `qwen35.block_count=65`, `qwen35.nextn_predict_layers=1`, `qwen35.attention.recurrent_layers` (65 entries).
This is exactly what `convert_hf_to_gguf.py` (without `--no-mtp`) + ggml-org's quantize recipe would give for the
trunk, with the MTP block at Q8_0 instead of Q4_K (their recipe would make `ffn_*`/`eh_proj` Q4_K and `attn_*` Q8_0).
If you have the disk, the from-scratch equivalent is:

```bash
convert_hf_to_gguf.py $Q --outtype bf16 --outfile Qwen3.8-27B-mtp-BF16.gguf          # 54 GB, MTP kept by default
llama-quantize --pure --tensor-type output.weight=q6_k --tensor-type attn_=q8_0 --tensor-type ssm_=q8_0 \
    --tensor-type 'blk\.64\.'=q8_0  Qwen3.8-27B-mtp-BF16.gguf Qwen3.8-27B-Q4_K_M-mtp.gguf Q4_K_M
```
(untested here; the `blk\.64\.` pattern relies on llama-quantize's regex `--tensor-type` matching).

## Why short drafts win here: verification cost on the M3 Pro

`llama-bench -p B -n 0` on the ggml-org Q4_K_M (stock Homebrew build) = cost of one target forward over B tokens,
i.e. of verifying B-1 draft tokens:

| B (tokens/forward) | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 16 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| tok/s | 6.57 | 12.41 | 14.12 | 13.15 | 11.81 | 13.75 | 13.71 | 41.11 |
| ms per forward | 152 | 161 | 212 | 304 | 423 | 436 | 583 | 389 |

A 2-token forward costs about the same as a 1-token one (memory bound), 3 tokens cost 1.4x, but 4-8 tokens cost
2-3.8x, and a 16-token forward is *cheaper* than an 8-token one. The cause is Metal kernel selection in
`ggml/src/ggml-metal/ggml-metal-ops.cpp` (`ggml_metal_op_mul_mat`): for K-quant weights, batches 4..8 go to the
small-batch `mul_mv_ext` kernels (re-reading the weights for every 4-5 src1 rows), batches 1..3 to plain `mul_mv`,
and only batches > 8 to the simdgroup mat-mat kernel (`ggml_metal_op_mul_mat_use_mm`: `ne11 > 8`). Since every
verification of n drafted tokens is a forward of B = n+1 tokens, n = 1..2 is nearly free, n = 3 is OK, and n >= 3
quickly eats the acceptance gains. This is the main reason the optimum is `--spec-draft-n-max 2` for both MTP and
DFlash, while the H200 numbers in the DFlash2 README use 7.

## Output match vs baseline (and why n >= 3 flips two prompts)

Every config with `--spec-draft-n-max <= 2` reproduces the baseline text exactly on all 6 prompts. Every config
whose verify batches reach 4 tokens (n_max >= 3, any drafter) changes `explain` (first difference at char 156) and
`math` (char 55), always at the same places. That is the kernel switch above: 4..8-token batches use
`mul_mv_ext` for the Q4_K/Q6_K/Q8_0 matmuls, whose different accumulation order changes logits by float rounding,
and those two prompts have a near-tie at that point. It is not a drafter bug (MTP and DFlash diverge identically,
and the text stays fluent); it is the "tiny float differences" case the README anticipates. `--spec-draft-p-min`
configs with n_max 3 only sometimes reach 4-token batches, so they may or may not match.

## Results (stock Homebrew llama-server, ggml-org Q4_K_M target)

Harness `bench/llama_bench.py` (greedy, thinking off, 256 tokens, fresh server per prompt, fixed args
`-ngl 999 -fa on -c 8192 -t 6 -tb 6 -np 1`), results in `results/llama-mtp-dflash.jsonl`, table produced by
`bench/summarize_mtp_dflash.py`. Every lock session starts with its own `--spec-type none` reference (6.64-6.67 tok/s
in all clean sessions). Gen tok/s is the server's `predicted_per_second`. "accepted/drafted" is summed over the
6 prompts.

### Best configurations (same session, median of 2 runs each)

| config | code_edit | code_gen | summarize | prose | explain | math | **mean** | speedup | match |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| no speculation | 6.50 | 6.67 | 6.67 | 6.68 | 6.68 | 6.61 | **6.64** | 1.00x | 6/6 |
| MTP sidecar Q4_0, n=2, + ngram-simple | 14.10 | 12.27 | 11.54 | 9.11 | 10.21 | 11.09 | **11.39** | **1.71x** | 6/6 |
| single-file Q4_K_M+MTP, n=2, + ngram-simple | 13.70 | 11.86 | 11.00 | 9.26 | 9.93 | 11.16 | **11.15** | 1.68x | 6/6 |
| DFlash2 Q4_0, n=2, + ngram-simple | 13.25 | 11.87 | 10.54 | 9.12 | 9.92 | 10.60 | **10.88** | 1.64x | 6/6 |

### Everything (one run each unless noted)

Note: the first `mtp-ggml` block (ggml-sidecar-n1..n4) ran at 02:00-02:30 while the 54 GB HF download was
decompressing on the CPU and has no in-session reference; it is ~15% low compared with the later clean sessions
(e.g. Q8_0 sidecar n=2: 9.18 then vs 10.58 in session `mtp3`). Compare within blocks that share a `ref-*`/`none` row.

| method | config | code_edit | code_gen | summarize | prose | explain | math | mean tok/s | accepted/drafted | match |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| draft-mtp | ggml-sidecar-n1 | 8.64 | 8.88 | 8.50 | 8.08 | 8.58 | 8.54 | **8.54** | 710/794 (89%) | 6/6 |
| draft-mtp | ggml-sidecar-n2 | 10.30 | 9.97 | 9.24 | 7.65 | 8.35 | 9.57 | **9.18** | 924/1157 (80%) | 6/6 |
| draft-mtp | ggml-sidecar-n3 | 9.33 | 9.38 | 8.08 | 5.55 | 7.39 | 8.62 | **8.06** | 1024/1444 (71%) | 4/6 (diff: explain, math) |
| draft-mtp | ggml-sidecar-n4 | 8.36 | 8.38 | 6.76 | 4.45 | 5.77 | 7.67 | **6.90** | 1081/1706 (63%) | 4/6 (diff: explain, math) |
| none | ref-dflash1 | 6.66 | 6.67 | 6.68 | 6.68 | 6.66 | 6.68 | **6.67** | - | 6/6 |
| draft-dflash | q8-n3 | 9.71 | 10.47 | 8.64 | 6.88 | 7.78 | 9.28 | **8.79** | 1032/1414 (73%) | 4/6 (diff: explain, math) |
| draft-dflash | q8-n5 | 10.08 | 10.35 | 8.14 | 5.26 | 6.54 | 9.49 | **8.31** | 1129/1873 (60%) | 4/6 (diff: explain, math) |
| draft-dflash | q8-n7 | 8.96 | 8.94 | 6.38 | 3.88 | 4.92 | 7.91 | **6.83** | 1166/2354 (50%) | 4/6 (diff: explain, math) |
| draft-dflash | bf16-n7 | 9.00 | 8.73 | 6.40 | 3.89 | 5.00 | 7.94 | **6.83** | 1166/2354 (50%) | 4/6 (diff: explain, math) |
| draft-dflash | q4-n7 | 9.20 | 9.02 | 6.39 | 3.71 | 5.03 | 7.90 | **6.87** | 1165/2362 (49%) | 4/6 (diff: explain, math) |
| none | ref-mtp2 | 6.53 | 6.66 | 6.65 | 6.67 | 6.68 | 6.69 | **6.65** | - | 6/6 |
| draft-mtp | ggml-sidecar-q4req-n2 | 12.34 | 12.12 | 11.39 | 8.98 | 10.07 | 11.72 | **11.11** | 919/1167 (79%) | 6/6 |
| draft-mtp | ggml-sidecar-q4req-n3 | 10.42 | 10.78 | 8.99 | 6.36 | 7.61 | 9.65 | **8.97** | 1010/1484 (68%) | 4/6 (diff: explain, math) |
| draft-mtp | ggml-sidecar-n3-pmin0.6 | 10.14 | 10.22 | 9.44 | 7.82 | 8.59 | 9.72 | **9.32** | 980/1161 (84%) | 6/6 |
| draft-mtp | ggml-sidecar-n2+ngram-simple | 13.46 | 11.65 | 10.81 | 8.93 | 9.78 | 10.95 | **10.93** | 938/1263 (74%) | 6/6 |
| none | single-nospec | 6.67 | 6.68 | 6.68 | 6.67 | 6.68 | 6.68 | **6.67** | - | 6/6 |
| draft-mtp | single-n2 | 12.05 | 11.72 | 10.88 | 9.15 | 9.84 | 11.33 | **10.83** | 926/1153 (80%) | 6/6 |
| none | ref-dflash2 | 6.67 | 6.68 | 6.67 | 6.66 | 6.68 | 6.68 | **6.67** | - | 6/6 |
| draft-dflash | q4-n2 | 11.69 | 11.75 | 10.44 | 9.05 | 9.84 | 11.21 | **10.66** | 929/1149 (81%) | 6/6 |
| draft-dflash | q4-n3 | 10.03 | 10.38 | 8.64 | 6.82 | 7.71 | 9.17 | **8.79** | 1030/1418 (73%) | 4/6 (diff: explain, math) |
| draft-dflash | q4-n4 | 8.70 | 8.72 | 7.29 | 5.15 | 6.39 | 8.43 | **7.45** | 1091/1650 (66%) | 4/6 (diff: explain, math) |
| draft-dflash | q4-n7-pmin0.5 | 9.58 | 9.25 | 7.71 | 5.66 | 6.36 | 8.74 | **7.88** | 1154/1719 (67%) | 4/6 (diff: explain, math) |
| draft-dflash | q4-n7-pmin0.8 | 9.80 | 10.14 | 8.56 | 6.52 | 7.36 | 9.37 | **8.63** | 1065/1176 (91%) | 4/6 (diff: explain, math) |
| draft-dflash | q4-n3+ngram-simple | 11.66 | 10.51 | 8.77 | 6.91 | 7.81 | 9.28 | **9.15** | 1041/1472 (71%) | 4/6 (diff: explain, math) |
| none | ref-mtp3 | 6.64 | 6.67 | 6.65 | 6.66 | 6.68 | 6.66 | **6.66** | - | 6/6 |
| draft-mtp | own-q8-n2 | 11.73 | 11.50 | 10.66 | 8.83 | 9.65 | 11.12 | **10.58** | 924/1157 (80%) | 6/6 |
| draft-mtp | own-q4-n1 | 11.46 | 11.30 | 11.03 | 9.93 | 10.65 | 10.66 | **10.84** | 706/798 (88%) | 6/6 |
| draft-mtp | own-q4-n2 | 12.25 | 12.14 | 11.09 | 8.97 | 10.09 | 11.71 | **11.04** | 919/1167 (79%) | 6/6 |
| draft-mtp | own-q4-n3 | 10.61 | 10.71 | 8.87 | 6.42 | 7.86 | 9.65 | **9.02** | 1014/1473 (69%) | 4/6 (diff: explain, math) |
| draft-mtp | own-q4-n3-pmin0.6 | 10.70 | 10.86 | 9.95 | 8.18 | 8.75 | 10.41 | **9.81** | 975/1169 (83%) | 6/6 |
| draft-mtp | own-q4-n4-pmin0.6 | 9.48 | 9.37 | 8.91 | 7.55 | 7.69 | 9.39 | **8.73** | 1028/1292 (80%) | 5/6 (diff: math) |
| draft-mtp | own-q4-n2+ngram-simple | 14.12 | 12.29 | 11.54 | 9.12 | 10.21 | 11.12 | **11.40** | 932/1319 (71%) | 6/6 |
| none | final-nospec | 6.50 | 6.67 | 6.67 | 6.68 | 6.68 | 6.61 | **6.64** | - | 6/6 |
| draft-mtp | final-own-q4-n2+ngram-simple | 14.10 | 12.27 | 11.54 | 9.11 | 10.21 | 11.09 | **11.39** | 932.0/1319.0 (71%) | 6/6 |
| draft-dflash | final-q4-n2+ngram-simple | 13.25 | 11.87 | 10.54 | 9.12 | 9.92 | 10.60 | **10.88** | 941.0/1303.0 (72%) | 6/6 |
| draft-mtp | final-single-n2+ngram-simple | 13.70 | 11.86 | 11.00 | 9.26 | 9.93 | 11.16 | **11.15** | 940.0/1259.0 (75%) | 6/6 |

### Findings

* **Draft length 2 is the sweet spot for every drafter** with the stock build (see verification cost above).
  MTP: n=1 10.84, n=2 11.04, n=3 9.02, (n=4 6.9 in the slow session). DFlash2: n=2 10.66, n=3 8.79, n=4 7.45,
  n=5 8.31, n=7 6.83-6.87. The MTP head chains fine (acceptance per drafted token 88% at n=1, 79% at n=2, 69% at
  n=3, ~63% at n=4) but the extra verify cost of 4+-token batches dominates.
* `--spec-draft-p-min` (confidence cutoff) helps long drafts but does not beat n=2: MTP n=3 p-min 0.6 = 9.81
  (and it keeps all outputs identical), n=4 p-min 0.6 = 8.73; DFlash n=7 p-min 0.8 = 8.63, p-min 0.5 = 7.88.
* Drafter precision barely matters for DFlash (bf16 / Q8_0 / Q4_0 at n=7: 6.83 / 6.83 / 6.87, identical acceptance).
  For MTP the Q4_0 sidecar is a bit faster than Q8_0 (n=2: 11.04 vs 10.58, same acceptance) because each MTP step
  runs the full 248k-vocab LM head (Q8_0: 1.3 GB, Q4_0: 0.7 GB per drafted token).
* MTP vs DFlash2 on this machine: equal acceptance at n=2 (~80%), MTP slightly faster (its draft step is
  one decoder layer + LM head; DFlash runs 5 layers over an 8-token block plus the target's LM head on all 8
  positions). DFlash2's advantage is long blocks (README: 7 tokens/step on H200), which the M3 Pro can't verify
  cheaply with the stock kernels.
* Comma lists work: `--spec-type draft-mtp,ngram-simple` tries the n-gram drafter first (only fires on 12-token
  verbatim repeats, drafting up to 48 tokens) and falls back to MTP. +2 tok/s on `code_edit` (repeats the
  input), neutral elsewhere: mean 11.04 -> 11.40.
* The MTP sidecar (`-md mtp-*.gguf`) and the single spliced GGUF behave the same (10.83 vs 11.04 at n=2;
  the single file uses the trunk's Q6_K LM head for drafting, the Q4_0 sidecar its own Q4_0 head); the
  sidecar is more flexible (drafter quant independent of the target) and costs no extra disk for the main model.
* The single-file GGUF without speculation reproduces the ggml-org outputs exactly and at the same speed
  (6.67 tok/s, 6/6 match): MTP tensors are skipped at load unless `--spec-type draft-mtp` is given.
* Prompt processing drops from ~85/45/77 tok/s to ~83/44/76 with a drafter (it also processes the prompt);
  negligible.

## Best llama-server command (stock Homebrew build)

```bash
llama-server -m ~/.cache/huggingface/hub/models--ggml-org--Qwen3.8-27B-GGUF/snapshots/efbb3b1f70a21d97fd4495240648405f7228554f/Qwen3.8-27B-Q4_K_M.gguf \
    -md ~/Documents/specdec/models/mtp-Qwen3.8-27B-q4_0.gguf \
    --spec-type draft-mtp,ngram-simple --spec-draft-n-max 2 \
    -ngl 999 -fa on -c 8192 --jinja
```
(Our Q4_0 sidecar uses the same recipe as ggml-org's `mtp-Qwen3.8-27B-Q4_0.gguf`, same byte size; their file
should work identically. `-hf ggml-org/Qwen3.8-27B-GGUF --spec-type draft-mtp` also auto-downloads an `mtp-*`
sidecar, but we did not check which quant it picks.) Single-file alternative:
`llama-server -m models/Qwen3.8-27B-Q4_K_M-mtp.gguf --spec-type draft-mtp,ngram-simple --spec-draft-n-max 2 ...`.
