#!/bin/bash
# Track B, session 4 (one lock): quantized_matmul cost vs rows, and a second
# repeat of the headline configs.
#   scripts/gpu.py bench/mlx_session4.sh
set -u
export HF_HUB_OFFLINE=1
cd "$(dirname "$0")/.."
PY=.venv/bin/python
L=logs
$PY bench/mlx_qmm_curve.py > $L/mlx_qmm_curve.log 2>&1
$PY -m mlx_spec bench --repeat 2 --configs greedy draft:k=3 ngram:k=3:n=2-4 \
    > $L/mlx_bench_session4.log 2>&1
echo "session4 done"
