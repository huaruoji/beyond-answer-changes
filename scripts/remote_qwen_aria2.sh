#!/usr/bin/env bash
# Fallback/resume downloader for Qwen's large safetensor shards.
set -euo pipefail

REPO=${REPO:-$HOME/projects/audio-causal-grounding}
MODEL_DIR="$REPO/models/Qwen2.5-Omni-3B"
REVISION=f75b40e3da2003cdd6e1829b1f420ca70797c34e
BASE_URL="https://huggingface.co/Qwen/Qwen2.5-Omni-3B/resolve/$REVISION"
mkdir -p "$REPO/logs" "$MODEL_DIR"
LOG_SUFFIX=${LOG_SUFFIX:-main}
exec > >(tee -a "$REPO/logs/acg-qwen-aria2-${LOG_SUFFIX}.log") 2>&1
echo "START $(date -Is) host=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "ABORT: a compute process appeared during the preflight probe"
  exit 2
fi

files=(model-00001-of-00003.safetensors model-00002-of-00003.safetensors model-00003-of-00003.safetensors)
if [[ -n "${ONLY_FILE:-}" ]]; then
  files=("$ONLY_FILE")
fi
for file in "${files[@]}"; do
  if [[ -f "$MODEL_DIR/$file" && ! -f "$MODEL_DIR/$file.aria2" ]]; then
    echo "already complete: $file"
    continue
  fi
  aria2c --continue=true --file-allocation=none --allow-overwrite=true \
    --auto-file-renaming=false --max-connection-per-server=8 --split=8 \
    --min-split-size=1M --dir="$MODEL_DIR" --out="$file" \
    "$BASE_URL/$file?download=true"
done

du -sh "$MODEL_DIR"
echo "__DONE__ $(date -Is)"
