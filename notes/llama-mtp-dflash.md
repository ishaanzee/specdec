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
