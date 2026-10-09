#!/bin/bash
# Track B, session 3 (one lock): draft-model refinements and rollback modes
# with frequent partial acceptance.
#   scripts/gpu.py bench/mlx_session3.sh
set -u
export HF_HUB_OFFLINE=1
cd "$(dirname "$0")/.."
PY=.venv/bin/python
L=logs
$PY -m pytest tests/test_tiny.py -q > $L/mlx_tests_tiny.log 2>&1 || { echo "tiny tests failed"; exit 1; }
$PY -m mlx_spec bench --repeat 1 --configs \
    draft:k=4:kmin=2 hybrid:k=3 hybrid:k=4:kmin=2 \
    draft:k=3:rollback=capture draft:k=3:rollback=reforward \
    > $L/mlx_bench_session3.log 2>&1
echo "session3 done"
