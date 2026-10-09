#!/bin/bash
# n-gram / draft-model sweeps for notes/llama-ngram-draft.md (Track A part 1).
# Run one phase per GPU-lock session:  scripts/gpu.py bench/llama_ngram_sweep.sh <phase>
# (Don't edit this file while a phase is running: bash reads scripts lazily. Run a copy.)
set -u
cd /Users/ishaanmehta/Documents/specdec
M=/Users/ishaanmehta/.cache/huggingface/hub/models--ggml-org--Qwen3.8-27B-GGUF/snapshots/efbb3b1f70a21d97fd4495240648405f7228554f/Qwen3.8-27B-Q4_K_M.gguf
D08=/Users/ishaanmehta/.cache/huggingface/hub/models--ggml-org--Qwen3.5-0.8B-GGUF/snapshots/8fea620810c4afa23dd6443f999a48574c1611a3/Qwen3.5-0.8B-Q8_0.gguf
D2=$(ls /Users/ishaanmehta/.cache/huggingface/hub/models--unsloth--Qwen3.5-2B-GGUF/snapshots/*/Qwen3.5-2B-Q8_0.gguf 2>/dev/null | head -1)
EXP=llama-ngram-draft
B() { # B <method> <config> <repeat> [harness opts] -- server args...
  local method=$1 cfg=$2 rep=$3; shift 3
  .venv/bin/python bench/llama_bench.py --experiment $EXP --method "$method" --config-name "$cfg" \
    --model $M --repeat "$rep" "$@" 2>&1 | grep -v '^\['
}

case "${1:-}" in
  # phase 1 (unpaired screening, 1 run each); results rows have no `speedup`
  defaults)
    B ngram-simple  defaults 1 -- --spec-type ngram-simple
    B ngram-map-k   defaults 1 -- --spec-type ngram-map-k
    B ngram-map-k4v defaults 1 -- --spec-type ngram-map-k4v
    B ngram-mod     defaults 1 -- --spec-type ngram-mod
    B ngram-cache   defaults 1 -- --spec-type ngram-cache
    ;;
  tune1)
    B ngram-simple  n4-m16 1 -- --spec-type ngram-simple --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 16
    B ngram-simple  n6-m16 1 -- --spec-type ngram-simple --spec-ngram-simple-size-n 6 --spec-ngram-simple-size-m 16
    B ngram-simple  n8-m32 1 -- --spec-type ngram-simple --spec-ngram-simple-size-n 8 --spec-ngram-simple-size-m 32
    B ngram-simple  n4-m8  1 -- --spec-type ngram-simple --spec-ngram-simple-size-n 4 --spec-ngram-simple-size-m 8
    B ngram-mod     m16-min4-max16 1 -- --spec-type ngram-mod --spec-ngram-mod-n-match 16 --spec-ngram-mod-n-min 4 --spec-ngram-mod-n-max 16
    B ngram-mod     m24-min8-max32 1 -- --spec-type ngram-mod --spec-ngram-mod-n-match 24 --spec-ngram-mod-n-min 8 --spec-ngram-mod-n-max 32
    B ngram-map-k   n4-m16 1 -- --spec-type ngram-map-k --spec-ngram-map-k-size-n 4 --spec-ngram-map-k-size-m 16
    B none          ref-tune1-b 1 -- --spec-type none
    ;;
  # phase 2: paired (each candidate run sits next to a no-spec reference run)
  paired1)
    B draft-simple  q35-0.8b-n3  1 --paired -- --spec-type draft-simple -md $D08 -ngld all --spec-draft-n-max 3
    B ngram-simple  n6-m16       1 --paired -- --spec-type ngram-simple --spec-ngram-simple-size-n 6 --spec-ngram-simple-size-m 16
    B ngram-mod     defaults     1 --paired -- --spec-type ngram-mod
    B ngram-simple  defaults     1 --paired -- --spec-type ngram-simple
    ;;
  draft1)
    B draft-simple  q35-0.8b-n5  1 --paired -- --spec-type draft-simple -md $D08 -ngld all --spec-draft-n-max 5
    B draft-simple  q35-0.8b-n8-p0.5 1 --paired -- --spec-type draft-simple -md $D08 -ngld all --spec-draft-n-max 8 --spec-draft-p-min 0.5
    B draft-simple  q35-2b-n4    1 --paired -- --spec-type draft-simple -md $D2 -ngld all --spec-draft-n-max 4
    B draft-simple,ngram-simple q35-0.8b-n3+ngs-n6-m16 1 --paired -- --spec-type draft-simple,ngram-simple -md $D08 -ngld all --spec-draft-n-max 3 --spec-ngram-simple-size-n 6 --spec-ngram-simple-size-m 16
    ;;
  *) echo "unknown phase"; exit 1;;
esac
