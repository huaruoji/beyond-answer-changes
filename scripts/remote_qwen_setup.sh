#!/usr/bin/env bash
# Run on the GPU host inside the dedicated dl-tasks tmux window.
set -euo pipefail

REPO=${REPO:-$HOME/projects/audio-causal-grounding}
MODEL_DIR="$REPO/models/Qwen2.5-Omni-3B"
ENV_DIR="$REPO/.venv-model"
HF_HOME=${HF_HOME:-$HOME/hf-cache}
IMM_PYTHON=${IMM_PYTHON:-$HOME/miniconda3/envs/imm/bin/python}
IMM_HF=${IMM_HF:-$HOME/miniconda3/envs/imm/bin/hf}
mkdir -p "$REPO/logs" "$REPO/models" "$HF_HOME"
exec > >(tee -a "$REPO/logs/acg-qwen-setup.log") 2>&1

echo "START $(date -Is) host=$(hostname)"
echo "REPO=$REPO HF_HOME=$HF_HOME"
nvidia-smi --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
if nvidia-smi --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "ABORT: a compute process appeared during the preflight probe"
  exit 2
fi

if [[ ! -x "$ENV_DIR/bin/python" ]]; then
  "$IMM_PYTHON" -m venv --system-site-packages "$ENV_DIR"
fi
"$ENV_DIR/bin/python" -m pip install --disable-pip-version-check --no-cache-dir \
  'qwen-omni-utils>=0.0.8' 'soundfile<1'
# qwen-omni-utils imports its vision helper at package import time even for an
# audio-only request.  Reuse the system CUDA torch and install only the matching
# torchvision wheel (no second torch download).
"$ENV_DIR/bin/python" -m pip install --disable-pip-version-check --no-cache-dir --no-deps \
  'torchvision==0.21.0'

if [[ ! -f "$MODEL_DIR/model-00003-of-00003.safetensors" ]]; then
  mkdir -p "$MODEL_DIR"
  HF_HOME="$HF_HOME" "$IMM_HF" download Qwen/Qwen2.5-Omni-3B \
    --revision f75b40e3da2003cdd6e1829b1f420ca70797c34e \
    --local-dir "$MODEL_DIR"
fi

du -sh "$MODEL_DIR" "$HF_HOME" "$ENV_DIR"
echo "__DONE__ $(date -Is)"
