#!/bin/bash
# Track B, session 1 (run the whole script under ONE lock):
#   scripts/gpu.py bench/mlx_session1.sh
# mlx-lm comparison, verify-cost curve, greedy baseline, n-gram sweep.
set -u
export HF_HUB_OFFLINE=1  # never block on the network while holding the GPU lock
cd "$(dirname "$0")/.."
PY=.venv/bin/python
L=logs
$PY -m mlx_spec mlxlm                      > $L/mlx_mlxlm_compare.log 2>&1
$PY -m mlx_spec verify-cost                > $L/mlx_verify_cost.log 2>&1
$PY -m mlx_spec bench --repeat 2 --configs \
    greedy spec-none:k=0 \
    ngram:k=4:n=2-4 ngram:k=8:n=2-4 ngram:k=12:n=2-4 ngram:k=8:n=1-4 \
                                           > $L/mlx_bench_session1.log 2>&1
echo "session1 done"
