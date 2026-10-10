#!/bin/sh
# Everything for the small-batch qmm kernel in one GPU-lock session:
#   scripts/gpu.py sh bench/mlx_qmm_pipeline.sh
# Logs in logs/qmm_*.log. See notes/mlx-qmm-kernel.md.
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
export HF_HUB_OFFLINE=1
echo "== tune $(date)";      $PY bench/mlx_qmm_kernel_bench.py --tune --reps 5 > logs/qmm_tune.log 2>&1
echo "== microbench $(date)"; $PY bench/mlx_qmm_kernel_bench.py --reps 9 > logs/qmm_bench.log 2>&1
echo "== unit tests $(date)"; $PY -m pytest tests/test_qmm_kernel.py -q > logs/qmm_tests.log 2>&1
echo "== session $(date)";    $PY bench/mlx_qmm_session.py --range "${QMM_RANGE:-2-16}" > logs/qmm_session.log 2>&1
echo "== fp32 $(date)"
SPECDEC_27B=1 SPECDEC_CONFIGS="mtp:k=6:vocab=65536:qmm=${QMM_RANGE:-2-16} draft:k=6:qmm=${QMM_RANGE:-2-16} mtp:k=8:vocab=65536:qmm=${QMM_RANGE:-2-16}" \
  $PY -m pytest tests/test_fp32_27b.py -q -s > logs/qmm_fp32.log 2>&1
echo "== done $(date)"
