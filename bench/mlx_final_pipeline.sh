#!/bin/sh
# Headline session: both targets x configs (bench/mlx_final_session.py), then the
# server smoke on the Q4_K_M transcode with the best MTP k and the qmm kernel.
#   scripts/gpu.py sh bench/mlx_final_pipeline.sh
cd "$(dirname "$0")/.." || exit 1
PY=.venv/bin/python
export HF_HUB_OFFLINE=1
pwr() { pmset -g batt | head -1; }
echo "== session $(date) [$(pwr)]"
$PY bench/mlx_final_session.py --repeat "${REPEAT:-2}" > logs/final_session.log 2>&1
K=$($PY - <<'PY'
import json, statistics as st
rows = [json.loads(l) for l in open("results/mlx_mtp.jsonl")]
start = [json.loads(l) for l in open("results/mlx_final_session.jsonl")][-1]["time"][:10]
by = {}
for r in rows:
    if r.get("model") == "models/qwen38-27b-q4km-mlx" and r["config"].endswith("_qmm5-16") and r["time"][:10] >= start:
        by.setdefault(r["config"], {})[r["prompt_id"]] = r["gen_tok_s"]
cfg = max(by, key=lambda c: st.mean(by[c].values())) if by else "k7_vocab65536_qmm5-16"
print(cfg.split("_")[0][1:])
PY
)
echo "== smoke k=$K $(date) [$(pwr)]"
cp results/mlx_server_smoke.json /tmp/specdec_smoke_backup.json 2>/dev/null
SPECDEC_QMM=5-16 $PY tests/smoke_server_27b.py --model models/qwen38-27b-q4km-mlx \
  --drafter "mtp:k=$K:vocab=65536:qmm=5-16" > logs/final_smoke.log 2>&1
mv results/mlx_server_smoke.json results/mlx_server_smoke_q4km_qmm.json
cp /tmp/specdec_smoke_backup.json results/mlx_server_smoke.json 2>/dev/null
echo "== done $(date) [$(pwr)]"
