#!/bin/bash
# Track B, session 2 (run the whole script under ONE lock):
#   scripts/gpu.py bench/mlx_session2.sh
# verify-cost curve (all rollback modes), greedy baseline, n-gram k sweep,
# draft models, rollback-mode comparison.
set -u
export HF_HUB_OFFLINE=1  # never block on the network while holding the GPU lock
cd "$(dirname "$0")/.."
PY=.venv/bin/python
L=logs
$PY -m pytest tests/test_tiny.py -q > $L/mlx_tests_tiny.log 2>&1 || { echo "tiny tests failed"; exit 1; }
$PY -m mlx_spec verify-cost > $L/mlx_verify_cost.log 2>&1
$PY -m mlx_spec bench --repeat 1 --configs \
    greedy spec-none:k=0 spec-none:k=0:async=0 \
    ngram:k=2:n=2-4 ngram:k=3:n=2-4 ngram:k=4:n=2-4 ngram:k=5:n=2-4 ngram:k=8:n=2-4 ngram:k=4:n=1-4 \
    draft:k=2 draft:k=3 draft:k=4 \
    draft:k=3:model=mlx-community/Qwen3.5-2B-MLX-8bit \
    ngram:k=4:n=2-4:rollback=capture ngram:k=4:n=2-4:rollback=reforward \
    > $L/mlx_bench_session2.log 2>&1
echo "session2 done"
