# llama.cpp: n-gram drafting and a small-Qwen draft model (Track A, part 1)

Target: `ggml-org/Qwen3.8-27B-GGUF` Q4_K_M, Homebrew llama.cpp 0.4.1 (build 10964, `b29c606e2`), M3 Pro 36 GB.
Harness: `bench/llama_bench.py` (usage at the top of the file). Sweeps: `bench/llama_ngram_sweep.sh <phase>`.
Raw rows: `results/llama-ngram-draft.jsonl`. Tables: `.venv/bin/python bench/llama_results_table.py results/llama-ngram-draft.jsonl`.

## TL;DR

__TLDR__

## Method

* Every run: `llama-server -ngl 999 -fa on -c 8192 -t 6 -tb 6 -np 1 --no-webui`, plus the config's flags.
  Requests: `/v1/chat/completions`, temperature 0 (top_k 1), max_tokens 256, `cache_prompt: false`,
  `chat_template_kwargs: {"enable_thinking": false}`. Checked: with that kwarg the template emits the empty
  `<think>\n\n</think>` block, the response has no `reasoning_content` and no `<think>` in content, and the
  "Reasoning effort is set to xhigh" system text is not injected (it is only added when thinking is enabled).
* Timings come from the server's `timings` block (`predicted_per_second`, `prompt_per_second`, `draft_n`,
  `draft_n_accepted`). Acceptance = `draft_n_accepted / draft_n`.
* A fresh server per (round, prompt): ngram-mod's hash pool is shared across requests and n-gram caches
  learn from earlier output, so re-running a prompt on the same server would "remember" the answer.
  Load is ~20 s from page cache.
* Output check: sha256 of `message.content` vs `results/baseline_outputs/Qwen3.8-27B-Q4_K_M/<prompt>.txt`
  (saved from the `--spec-type none` run, identical across its 3 repeats). Differing outputs are saved
  under `results/mismatch_outputs/`.
* **The machine was shared all night** (other agents queued on the GPU lock, plus unrelated
  ffmpeg / OCR / Python jobs of the user's, 8-10 GB of swap in use). The same no-spec config measured
  6.5 tok/s at 00:30, 5.0 tok/s at 02:00 and 3.6 tok/s at 03:30, then 6.6-6.7 again from ~04:00.
  So the phase-1 screening numbers (unpaired) are only comparable within a few minutes of each other.
  From phase 2 on, every run is **paired** (`--paired`): each candidate run of a prompt sits right next
  to a fresh `--spec-type none` run of the same prompt, and the table reports `speedup` = candidate / reference.
  That's the number to trust; absolute tok/s are only meaningful when the reference column is ~6.6.

## Why speculation is expensive on this model (read this before tuning)

Qwen3.8 is hybrid: 48 of 64 layers are Gated-DeltaNet linear attention with a recurrent state
(48 layers x 48 heads x 128 x 128 x f32 = ~150 MB per sequence), 16 are full attention.
A recurrent state can't be trimmed like a KV cache. In this build (`tools/server/server-context.cpp`):

* Before verifying a draft, the server snapshots the target's sequence state (`spec_ckpt.update_tgt`) unless
  the context keeps `n_rs_seq` per-token recurrent snapshots that cover the draft length.
* `n_rs_seq` is only enabled for `draft-mtp`, `draft-eagle3`, `draft-dflash`, `draft-dspark`
  (`common_params_speculative::need_n_rs_seq()` in `common/common.h`). For `draft-simple` and every
  `ngram-*` type it is 0.
* So for n-gram and classic draft-model speculation, **any partial rejection restores the snapshot and
  re-decodes the accepted tokens** (`spec_is_replay`): one extra target forward pass. A fully accepted draft
  costs one pass; a draft rejected at token 0 costs a snapshot save + restore + nothing gained.
* Consequence: long, speculative drafts hurt much more than on a pure-transformer model. What helps is drafting
  only when confident (`--spec-draft-p-min`) and short n-gram drafts. When nothing is drafted, there is no
  overhead (ngram-simple on prose/explain ran at exactly the reference speed, 6.67-6.68 vs 6.67).

(MTP and DFlash, track A part 2, do get the cheap rollback path. That's a structural advantage for them here.)

## Tokenizer compatibility (option 3: older small Qwen as draft)

Candidates on the Hub (HF API, author Qwen): Qwen3.5-{0.8B, 2B, 4B, 9B, 27B, 35B-A3B, 122B-A10B, 397B-A17B}
(+ `-Base`), Qwen3.6-{27B, 35B-A3B}. No Qwen3.6/3.7 model smaller than 27B exists; the small ones are Qwen3.5.
All have `model_type qwen3_5`, `vocab_size 248320`, eos 248044 in config.

Compared tokenizer files of Qwen3.5-0.8B / 2B / 4B and Qwen3.6-27B against Qwen/Qwen3.8-27B
(script in my scratchpad; files: tokenizer.json, tokenizer_config.json, vocab.json, merges.txt, chat_template.jinja):

| item | result |
|---|---|
| BPE vocab (248,044 entries) | identical |
| BPE merges (247,587) | identical; vocab.json and merges.txt byte-identical |
| normalizer / pre_tokenizer / post_processor / decoder | identical |
| `added_tokens_decoder` in tokenizer_config (33 entries, ids 248044-248076) | identical |
| `added_tokens` in tokenizer.json | **differs**: Qwen3.8 has 33, Qwen3.5/3.6 have 26. Missing in the old ones: 248070 `<\|audio_start\|>`, 248071 `<\|audio_end\|>`, 248072 `<tts_pad>`, 248073 `<tts_text_bos>`, 248074 `<tts_text_eod>`, 248075 `<tts_text_bos_single>`, 248076 `<\|audio_pad\|>` |
| special tokens used by the chat template (`<\|im_start\|>`, `<\|im_end\|>`, `<think>`, `</think>`, `<tool_call>`, vision tokens...) | same ids, same strings |
| chat template | differs in logic only: Qwen3.8 adds the `reasoning_effort` system text (default xhigh) and defaults to thinking on; tool-arg rendering tweaks. Irrelevant for drafting (the target's template tokenizes the prompt; the draft only sees token ids). |

Encoding check with `tokenizers`: all 6 prompts and all 6 baseline outputs tokenize to identical ids with
the 3.8 and 3.5-0.8B/2B/4B tokenizers. Only text containing the new audio/TTS special strings differs
(the old tokenizer.json splits `<|audio_start|>` into pieces). Those never occur in text chat.

GGUF level (what llama.cpp actually compares): `ggml-org/Qwen3.5-0.8B-GGUF` Q8_0 and `unsloth/Qwen3.5-2B-GGUF`
Q8_0 vs our target GGUF: 248,320 tokens each, **0 token-text differences, 0 token-type differences**, identical
merges, same `pre=qwen35`, eos 248046, add_bos false. (The converter used `added_tokens_decoder`, so the audio
tokens are present in the old GGUFs too.) Metadata differences: bos_token_id / padding id only, which
llama.cpp ignores when add_bos is false. llama.cpp's own check (`common_speculative_are_compatible`: vocab type,
bos/eos, size diff <= 128, token text for every id >= 5) passed silently: no warning in the server log, the
draft loaded and drafted.

**Verdict: Qwen3.5-0.8B / 2B (and 4B) are token-for-token compatible with Qwen3.8-27B for text.** The only
difference is 7 audio/TTS special tokens missing from the old `tokenizer.json` added-token list, which doesn't
affect GGUF vocabularies or any text prompt. No token translation is needed or happening.

## Results

Cells are gen tok/s for each prompt (256 new tokens, except summarize which stops at 231), `n` = repeats (median).
`accept` = accepted/drafted tokens summed over the 6 prompts. `match` = output identical to the no-spec baseline.

### Phase 1: unpaired screening (machine load drifting, see Method)

The `ref-tune1-b` row is the no-spec reference measured right after this batch: 5.0 tok/s, not 6.5.
Compare rows within this table only loosely; the paired table below is the reliable one.

| method | config | n | code_edit | code_gen | summarize | prose | explain | math | mean | accept | match |
|---|---|---|---|---|---|---|---|---|---|---|---|
| none | baseline | 3 | 6.46 | 6.61 | 6.71 | 6.54 | 6.34 | 6.40 | 6.51 | - | all |
| ngram-simple | defaults | 1 | 7.24 | 6.24 | 6.36 | 6.43 | 6.27 | 5.52 | 6.34 | 55/230 (24%) | all |
| ngram-map-k | defaults | 1 | 6.19 | 4.99 | 4.92 | 4.96 | 5.35 | 5.32 | 5.29 | 52/96 (54%) | all |
| ngram-map-k4v | defaults | 1 | 6.40 | 5.35 | 5.41 | 5.56 | 5.77 | 5.43 | 5.65 | 52/96 (54%) | all |
| ngram-mod | defaults | 1 | 6.44 | 5.63 | 5.50 | 5.50 | 5.43 | 5.26 | 5.63 | 41/64 (64%) | all |
| ngram-cache | defaults | 1 | 5.78 | 4.99 | 5.49 | 4.76 | 4.78 | 4.94 | 5.12 | 138/712 (19%) | all |
| ngram-simple | n4-m16 | 1 | 6.85 | 5.22 | 5.60 | 5.21 | 5.49 | 5.55 | 5.66 | 185/836 (22%) | all |
| ngram-simple | n6-m16 | 1 | 7.12 | 5.23 | 5.76 | 5.44 | 5.03 | 5.33 | 5.65 | 129/417 (31%) | all |
| ngram-simple | n8-m32 | 1 | 6.12 | 4.92 | 4.81 | 5.13 | 5.07 | 5.07 | 5.19 | 92/559 (16%) | all |
| ngram-simple | n4-m8 | 1 | 6.46 | 5.04 | 5.32 | 4.97 | 5.00 | 5.00 | 5.30 | 178/489 (36%) | all |
| ngram-mod | m16-min4-max16 | 1 | 5.92 | 4.64 | 4.90 | 4.95 | 4.55 | 5.01 | 5.00 | 47/48 (98%) | all |
| ngram-mod | m24-min8-max32 | 1 | 5.55 | 5.09 | 5.04 | 5.03 | 5.06 | 5.14 | 5.15 | 40/64 (62%) | all |
| ngram-map-k | n4-m16 | 1 | 6.74 | 5.17 | 5.58 | 5.36 | 5.30 | 5.28 | 5.57 | 177/759 (23%) | all |
| none | ref-tune1-b | 1 | 5.27 | 5.40 | 5.24 | 4.71 | 4.22 | 5.22 | 5.01 | - | all |

### Phase 2: paired runs (`speedup` vs a no-spec run of the same prompt right next to it)

The first two paired rows (`q35-0.8b-n3`, `ngram-simple n6-m16`) ran while the machine was heavily loaded
(reference ~3.6 tok/s); the rest ran on a quiet machine (reference 6.6-6.7 tok/s).

| method | config | n | code_edit | code_gen | summarize | prose | explain | math | mean | accept | match |
|---|---|---|---|---|---|---|---|---|---|---|---|
| draft-simple | q35-0.8b-n3 | 1 | 5.73 (1.60x) | 5.84 (1.59x) | 4.58 (1.24x) | 3.56 (0.96x) | 3.49 (0.96x) | 5.42 (1.44x) | 4.77 (1.30x) | 836/1338 (62%) | all |
| ngram-simple | n6-m16 | 1 | 4.71 (1.21x) | 3.66 (1.01x) | 3.63 (1.00x) | 3.72 (1.04x) | 3.70 (1.03x) | 3.78 (1.11x) | 3.87 (1.07x) | 129/417 (31%) | all |
| ngram-mod | defaults | 1 | 4.07 (1.18x) | 3.67 (1.07x) | 3.41 (0.98x) | 5.43 (0.84x) | 6.67 (1.01x) | 6.68 (1.00x) | 4.99 (1.01x) | 41/64 (64%) | all |
| ngram-simple | defaults | 1 | 7.83 (1.18x) | 6.67 (1.01x) | 6.67 (1.00x) | 6.68 (1.00x) | 6.68 (1.00x) | 6.46 (0.97x) | 6.83 (1.03x) | 55/230 (24%) | all |
| draft-simple | q35-0.8b-n5 | 1 | 8.97 (1.35x) | 8.69 (1.30x) | 6.19 (0.93x) | 4.52 (0.68x) | 4.45 (0.67x) | 7.29 (1.09x) | 6.69 (1.00x) | 921/1790 (51%) | math differ |
| draft-simple | q35-0.8b-n8-p0.5 | 1 | 12.46 (1.91x) | 11.23 (1.69x) | 8.85 (1.32x) | 7.46 (1.12x) | 6.17 (0.94x) | 9.28 (1.39x) | 9.24 (1.39x) | 959/1385 (69%) | math differ |
| draft-simple | q35-2b-n4 | 1 | 7.78 (1.17x) | 7.13 (1.07x) | 6.22 (0.94x) | 4.21 (0.63x) | 4.94 (0.74x) | 6.71 (1.01x) | 6.16 (0.93x) | 946/1491 (63%) | math differ |
| draft-simple,ngram-simple | q35-0.8b-n3+ngs-n6-m16 | 1 | 10.74 (1.62x) | 9.21 (1.38x) | 6.84 (1.03x) | 5.83 (0.87x) | 5.70 (0.85x) | 8.06 (1.32x) | 7.73 (1.18x) | 839/1580 (53%) | all |


## What worked / dead ends

__FINDINGS__

## Best command line

__BEST__
