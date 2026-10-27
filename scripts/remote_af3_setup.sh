#!/usr/bin/env bash
# Run on the GPU host inside the dedicated dl-tasks tmux window.
set -euo pipefail

REPO=${REPO:-$HOME/projects/audio-causal-grounding}
MODEL_DIR="$REPO/models/audio-flamingo-3-hf"
ENV_DIR="$REPO/.venv-model"
HF_HOME=${HF_HOME:-$HOME/hf-cache}
IMM_HF=${IMM_HF:-$HOME/miniconda3/envs/imm/bin/hf}
mkdir -p "$REPO/logs" "$REPO/models" "$HF_HOME"
exec > >(tee -a "$REPO/logs/acg-af3-setup.log") 2>&1

echo "START $(date -Is) host=$(hostname)"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "ABORT: a compute process appeared during the preflight probe"
  exit 2
fi

if [[ ! -f "$MODEL_DIR/model-00004-of-00004.safetensors" ]]; then
  mkdir -p "$MODEL_DIR"
  HF_HOME="$HF_HOME" "$IMM_HF" download nvidia/audio-flamingo-3-hf \
    --revision 7d4bae64ee29878af6504ae6f6bb3e40492838ad \
    --local-dir "$MODEL_DIR" --exclude model.safetensors
fi

du -sh "$MODEL_DIR" "$HF_HOME" "$ENV_DIR"
echo "__DONE__ $(date -Is)"
