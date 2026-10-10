#!/bin/bash
# Kernel-level A/B of the Metal multi-row mat-vec (GGML_METAL_MV_MULTIROW) on Qwen3.8-27B shapes.
# Run under the GPU lock: scripts/gpu.py bench/metal_mv_perf.sh [config ...]
# A config is a space-separated list of KEY=VAL env settings; "stock" = multirow off.
cd "$(dirname "$0")/.." || exit 1
BIN=vendor/llama.cpp/build-multirow/bin/test-backend-ops
FILTER=${FILTER:-'type_b=f32,m=(17408|5120|10240|248320),n=[0-9]+,k=(5120|17408|6144),bs=\[1,1\],nr=\[1,1\]'}
OUT=${OUT:-logs/metal_mv_perf_$(date +%Y%m%d_%H%M%S).log}
[ $# -eq 0 ] && set -- stock "GGML_METAL_MV_MULTIROW=1"
echo "power: $(pmset -g batt | head -1)" | tee -a "$OUT"
for cfg in "$@"; do
  echo "=== config: $cfg" | tee -a "$OUT"
  if [ "$cfg" = stock ]; then envs=(); else read -r -a envs <<< "$cfg"; fi
  env "${envs[@]}" $BIN perf -b MTL0 -o MUL_MAT -p "$FILTER" 2>&1 | grep -E "MUL_MAT" | tee -a "$OUT"
done
echo "wrote $OUT"
