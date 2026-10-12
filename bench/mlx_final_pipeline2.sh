#!/bin/sh
# Second half of the headline session (each target in its own process: in one
# process the first model's memory was not released and the second swapped).
#   scripts/gpu.py sh bench/mlx_final_pipeline2.sh
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
export HF_HUB_OFFLINE=1
pwr() { pmset -g batt | head -1; }
echo "== mlx-community session $(date) [$(pwr)] $(memory_pressure | tail -1)"
$PY bench/mlx_final_session.py --repeat "${REPEAT:-2}" --targets mlx-community/Qwen3.8-27B-4bit > logs/final_session2.log 2>&1
echo "== smoke rerun $(date) [$(pwr)] $(memory_pressure | tail -1)"
cp results/mlx_server_smoke.json /tmp/specdec_smoke_backup.json 2>/dev/null
SPECDEC_QMM=5-16 $PY tests/smoke_server_27b.py --model models/qwen38-27b-q4km-mlx \
  --drafter "mtp:k=7:vocab=65536:qmm=5-16" > logs/final_smoke2.log 2>&1
mv results/mlx_server_smoke.json results/mlx_server_smoke_q4km_qmm_rerun.json
cp /tmp/specdec_smoke_backup.json results/mlx_server_smoke.json 2>/dev/null
echo "== done $(date) [$(pwr)]"
