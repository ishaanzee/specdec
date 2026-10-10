#!/bin/sh
# llama.cpp perplexity of ggml-org's Q4_K_M on wikitext-2 test (16 x 2048 tokens),
# also writing the base log-probs that bench/mlx_ppl.py scores the MLX models against.
#   scripts/gpu.py sh bench/llama_ppl.sh
set -e
cd "$(dirname "$0")/.."
GGUF=$HOME/.cache/huggingface/hub/models--ggml-org--Qwen3.8-27B-GGUF/snapshots/efbb3b1f70a21d97fd4495240648405f7228554f/Qwen3.8-27B-Q4_K_M.gguf
llama-perplexity -m "$GGUF" -f models/wikitext/wikitext-2-raw/wiki.test.raw -c 2048 --chunks 16 \
  -ngl 999 -fa on --kl-divergence-base models/wikitext/q4km-base.kld
