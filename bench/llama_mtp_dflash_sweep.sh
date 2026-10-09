#!/bin/bash
# MTP / DFlash sweeps for notes/llama-mtp-dflash.md (Track A part 2).
# Run one phase per GPU-lock session:  scripts/gpu.py bench/llama_mtp_dflash_sweep.sh <phase>
set -u
cd "$(dirname "$0")/.."
M=/Users/ishaanmehta/.cache/huggingface/hub/models--ggml-org--Qwen3.8-27B-GGUF/snapshots/efbb3b1f70a21d97fd4495240648405f7228554f/Qwen3.8-27B-Q4_K_M.gguf
DF_BF16=models/Qwen3.8-27B-DFlash2-bf16.gguf
DF_Q8=models/Qwen3.8-27B-DFlash2-q8_0.gguf
DF_Q4=models/Qwen3.8-27B-DFlash2-q4_0.gguf
MTP_GGML=models/ggml-org/mtp-Qwen3.8-27B-Q8_0.gguf     # ggml-org's published MTP sidecar
MTP_OWN_BF16=models/mtp-Qwen3.8-27B-bf16.gguf           # ours, converted from Qwen/Qwen3.8-27B
MTP_OWN=models/mtp-Qwen3.8-27B-q8_0.gguf
MTP_OWN_Q4=models/mtp-Qwen3.8-27B-q4_0.gguf
MTP_GGML_Q4=models/mtp-Qwen3.8-27B-Q4_0-requant-from-ggml-q8.gguf
M_MTP=models/Qwen3.8-27B-Q4_K_M-mtp.gguf                # single-file Q4_K_M with the MTP layer
EXP=${EXP:-llama-mtp-dflash}
REP=${REP:-1}
B() { # B <method> <config> <model> -- server args...
  local method=$1 cfg=$2 model=$3; shift 3
  .venv/bin/python bench/llama_bench.py --experiment $EXP --method "$method" --config-name "$cfg" \
    --model "$model" --model-tag Qwen3.8-27B-Q4_K_M --repeat "$REP" "$@" 2>&1 | grep -v '^\['
}

case "${1:-}" in
  dflash-quant)
    llama-quantize $DF_BF16 $DF_Q8 Q8_0
    llama-quantize --pure $DF_BF16 $DF_Q4 Q4_0
    ;;
  dflash1)
    B none ref-dflash1 $M -- --spec-type none
    for n in 3 5 7; do
      B draft-dflash q8-n$n $M -- -md $DF_Q8 --spec-type draft-dflash --spec-draft-n-max $n
    done
    B draft-dflash bf16-n7 $M -- -md $DF_BF16 --spec-type draft-dflash --spec-draft-n-max 7
    B draft-dflash q4-n7   $M -- -md $DF_Q4   --spec-type draft-dflash --spec-draft-n-max 7
    ;;
  mtp-ggml)
    B none ref-mtp-ggml $M -- --spec-type none
    for n in 1 2 3 4; do
      B draft-mtp ggml-sidecar-n$n $M -- -md $MTP_GGML --spec-type draft-mtp --spec-draft-n-max $n
    done
    ;;
  mtp2)
    [ -f $MTP_GGML_Q4 ] || llama-quantize --allow-requantize --pure $MTP_GGML $MTP_GGML_Q4 Q4_0
    B none ref-mtp2 $M -- --spec-type none
    B draft-mtp ggml-sidecar-q4req-n2 $M -- -md $MTP_GGML_Q4 --spec-type draft-mtp --spec-draft-n-max 2
    B draft-mtp ggml-sidecar-q4req-n3 $M -- -md $MTP_GGML_Q4 --spec-type draft-mtp --spec-draft-n-max 3
    B draft-mtp ggml-sidecar-n3-pmin0.6 $M -- -md $MTP_GGML --spec-type draft-mtp --spec-draft-n-max 3 --spec-draft-p-min 0.6
    B draft-mtp ggml-sidecar-n2+ngram-simple $M -- -md $MTP_GGML --spec-type draft-mtp,ngram-simple --spec-draft-n-max 2
    ;;
  mtp-own-convert)
    # our own MTP sidecar from the official HF weights (only shards 3, 16, 18 are actually read)
    Q=$(echo /Users/ishaanmehta/.cache/huggingface/hub/models--Qwen--Qwen3.8-27B/snapshots/*)
    .venv/bin/python vendor/llama.cpp/convert_hf_to_gguf.py $Q --mtp --outtype bf16 \
        --model-name Qwen3.8-27B --outfile $MTP_OWN_BF16
    llama-quantize $MTP_OWN_BF16 $MTP_OWN Q8_0
    llama-quantize --pure $MTP_OWN_BF16 $MTP_OWN_Q4 Q4_0
    ;;
  verify-cost)
    # cost of one target forward over a batch of B tokens (= verifying B-1 draft tokens)
    llama-bench -m $M -ngl 999 -fa 1 -t 6 -p 1,2,3,4,5,6,8,16 -n 0 -r 3 -o md
    ;;
  dflash2)
    B none ref-dflash2 $M -- --spec-type none
    B draft-dflash q4-n2 $M -- -md $DF_Q4 --spec-type draft-dflash --spec-draft-n-max 2
    B draft-dflash q4-n3 $M -- -md $DF_Q4 --spec-type draft-dflash --spec-draft-n-max 3
    B draft-dflash q4-n4 $M -- -md $DF_Q4 --spec-type draft-dflash --spec-draft-n-max 4
    B draft-dflash q4-n7-pmin0.5 $M -- -md $DF_Q4 --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-p-min 0.5
    B draft-dflash q4-n7-pmin0.8 $M -- -md $DF_Q4 --spec-type draft-dflash --spec-draft-n-max 7 --spec-draft-p-min 0.8
    B draft-dflash q4-n3+ngram-simple $M -- -md $DF_Q4 --spec-type draft-dflash,ngram-simple --spec-draft-n-max 3
    ;;
  splice)
    .venv/bin/python scripts/splice_mtp_gguf.py $M $MTP_GGML $M_MTP
    ;;
  mtp-single)
    # no-spec baseline on the single-file GGUF (must equal ggml-org's output), then MTP from inside it
    B none single-nospec $M_MTP -- --spec-type none
    for n in ${NS:-2 3}; do
      B draft-mtp single-n$n $M_MTP -- --spec-type draft-mtp --spec-draft-n-max $n
    done
    ;;
  many)
    # several phases in one GPU-lock session: many <phase>...
    shift; for ph in "$@"; do echo "=== $(date '+%F %T') phase $ph"; NS=${NS:-2} "$0" $ph; done
    ;;
  *)
    if [ "${1:-}" = "custom" ]; then shift; B "$@"; else echo "unknown phase ${1:-}"; exit 1; fi
    ;;
esac
