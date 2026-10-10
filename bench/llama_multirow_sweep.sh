#!/bin/bash
# Metal multi-row mat-vec (scripts/metal-multirow-mv.patch, build vendor/llama.cpp/build-multirow) sweeps
# for notes/llama-metal-kernel.md. One phase per GPU-lock session:
#   scripts/gpu.py bench/llama_multirow_sweep.sh <phase>
# GGML_METAL_MV_MULTIROW=1 turns the new kernels on; unset = the stock code path of the same binary.
set -u
cd "$(dirname "$0")/.."
M=/Users/ishaanmehta/.cache/huggingface/hub/models--ggml-org--Qwen3.8-27B-GGUF/snapshots/efbb3b1f70a21d97fd4495240648405f7228554f/Qwen3.8-27B-Q4_K_M.gguf
DF_Q4=models/Qwen3.8-27B-DFlash2-q4_0.gguf
MTP_Q4=models/mtp-Qwen3.8-27B-q4_0.gguf
D08=/Users/ishaanmehta/.cache/huggingface/hub/models--ggml-org--Qwen3.5-0.8B-GGUF/snapshots/8fea620810c4afa23dd6443f999a48574c1611a3/Qwen3.5-0.8B-Q8_0.gguf
BIN=vendor/llama.cpp/build-multirow/bin
EXP=${EXP:-llama-multirow}
REP=${REP:-1}
TXT=models/wikitext/wikitext-2-raw/wiki.test.raw

# BS <MR 0|1> <method> <config> -- server args...   (paired: every run next to a fresh no-spec run)
BS() {
  local mr=$1 method=$2 cfg=$3; shift 3
  if [ "$mr" = 1 ]; then export GGML_METAL_MV_MULTIROW=1; else unset GGML_METAL_MV_MULTIROW; fi
  LLAMA_SERVER=$BIN/llama-server .venv/bin/python bench/llama_bench_server.py --experiment $EXP \
    --method "$method" --config-name "$cfg" --model $M --model-tag Qwen3.8-27B-Q4_K_M --repeat "$REP" \
    --paired "$@" 2>&1 | grep -v '^\['
  unset GGML_METAL_MV_MULTIROW
}

echo "[sweep] $(date) phase=${1:-} power: $(pmset -g batt | head -1)"
case "${1:-}" in
  kbench)
    # cost of one target forward over B tokens, stock path vs multi-row, same session (alternating)
    P=$(seq -s, 1 17)
    for i in 1 2; do
      echo "== stock (round $i)";    $BIN/llama-bench -m $M -ngl 999 -fa 1 -t 6 -p $P -n 0 -r 3 -o md
      echo "== multirow (round $i)"; GGML_METAL_MV_MULTIROW=1 $BIN/llama-bench -m $M -ngl 999 -fa 1 -t 6 -p $P -n 0 -r 3 -o md
    done
    ;;
  ppl)
    # perplexity, stock vs multi-row; small ubatches so the verify-size kernels are used for every matmul
    for ub in 16 5; do
      for mr in 0 1; do
        echo "== ub=$ub multirow=$mr"
        if [ $mr = 1 ]; then export GGML_METAL_MV_MULTIROW=1; else unset GGML_METAL_MV_MULTIROW; fi
        $BIN/llama-perplexity -m $M -f $TXT -ngl 999 -fa on -t 6 -c 128 -b 128 -ub $ub --chunks 24 2>&1 \
          | grep -E "^\[|Final estimate|ubatch"
        unset GGML_METAL_MV_MULTIROW
      done
    done
    ;;
  e2e-mtp)
    BS 1 draft-mtp mr-mtp-n2 -- -md $MTP_Q4 --spec-type draft-mtp --spec-draft-n-max 2
    BS 1 draft-mtp,ngram-simple mr-mtp-n2+ngs -- -md $MTP_Q4 --spec-type draft-mtp,ngram-simple --spec-draft-n-max 2
    for n in 3 4 5 6; do
      BS 1 draft-mtp mr-mtp-n$n -- -md $MTP_Q4 --spec-type draft-mtp --spec-draft-n-max $n
      BS 1 draft-mtp,ngram-simple mr-mtp-n$n+ngs -- -md $MTP_Q4 --spec-type draft-mtp,ngram-simple --spec-draft-n-max $n
    done
    ;;
  e2e-dflash)
    for n in 3 4 5 6 7; do
      BS 1 draft-dflash mr-dflash-n$n -- -md $DF_Q4 --spec-type draft-dflash --spec-draft-n-max $n
      BS 1 draft-dflash,ngram-simple mr-dflash-n$n+ngs -- -md $DF_Q4 --spec-type draft-dflash,ngram-simple --spec-draft-n-max $n
    done
    ;;
  e2e-draft)
    BS 1 draft-simple mr-q35-0.8b-n16-p0.7 -- --spec-type draft-simple -md $D08 -ngld all --spec-draft-n-max 16 --spec-draft-p-min 0.7
    ;;
  e2e-stock)
    # same binary, multi-row off: anchors for the comparison
    BS 0 draft-mtp,ngram-simple stock-mtp-n2+ngs -- -md $MTP_Q4 --spec-type draft-mtp,ngram-simple --spec-draft-n-max 2
    BS 0 draft-mtp stock-mtp-n3 -- -md $MTP_Q4 --spec-type draft-mtp --spec-draft-n-max 3
    BS 0 draft-dflash,ngram-simple stock-dflash-n7+ngs -- -md $DF_Q4 --spec-type draft-dflash,ngram-simple --spec-draft-n-max 7
    ;;
  *)
    echo "usage: $0 kbench|ppl|e2e-mtp|e2e-dflash|e2e-draft|e2e-stock"; exit 1 ;;
esac
