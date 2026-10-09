#!/bin/bash
# Fetch every model the experiments need, smallest first. Logs to logs/download.log.
set -u
cd "$(dirname "$0")/.."
export HF_HUB_ENABLE_HF_TRANSFER=1
HF=.venv/bin/hf
for repo in z-lab/Qwen3.8-27B-DFlash2 mlx-community/Qwen3.8-27B-4bit Qwen/Qwen3.8-27B; do
  echo "$(date '+%F %T') start $repo"
  $HF download "$repo" >/dev/null 2>&1 && echo "$(date '+%F %T') done $repo" || echo "$(date '+%F %T') FAILED $repo"
done
echo "$(date '+%F %T') all downloads finished"
