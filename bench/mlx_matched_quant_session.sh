#!/bin/sh
# notes/mlx-matched-quant.md sessions. Run each under the GPU lock:
#   scripts/gpu.py sh bench/llama_ppl.sh                      # llama.cpp PPL + base log-probs
#   scripts/gpu.py sh bench/mlx_matched_quant_session.sh ppl  # MLX PPL / KL vs llama.cpp
#   scripts/gpu.py sh bench/mlx_matched_quant_session.sh speed
set -e
cd "$(dirname "$0")/.."
PY=.venv/bin/python
T=models/qwen38-27b-q4km-mlx
C=mlx-community/Qwen3.8-27B-4bit
BASE=models/wikitext/q4km-base.kld
W=models/wikitext/wikitext-2-raw/wiki.test.raw
case "$1" in
ppl)
  test -s $BASE
  $PY bench/mlx_ppl.py --model $T --base $BASE --text $W --label "Q4_K_M transcode (ours)"
  $PY bench/mlx_ppl.py --model $C --base $BASE --label "mlx-community 4-bit"
  $PY bench/mlx_ppl.py --model $T --base $BASE --fp32-head --label "Q4_K_M transcode (ours), fp32 lm_head"
  $PY bench/mlx_ppl.py --model $C --base $BASE --fp32-head --label "mlx-community 4-bit, fp32 lm_head"
  ;;
speed)
  $PY bench/mlx_matched_quant.py --target $C --configs greedy mtp:k=3:vocab=65536 --repeat 2
  $PY bench/mlx_matched_quant.py --target $T --configs greedy mtp:k=3:vocab=65536 --repeat 2
  ;;
*) echo "usage: $0 ppl|speed"; exit 2 ;;
esac
