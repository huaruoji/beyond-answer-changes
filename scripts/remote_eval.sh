#!/usr/bin/env bash
# Generic remote evaluator; invoke from the repository in a fresh dl-tasks window.
set -euo pipefail

REPO=${REPO:-$HOME/projects/audio-causal-grounding}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH}
OUTPUT=${OUTPUT:?set OUTPUT}
GPU=${GPU:?set GPU}
MANIFEST=${MANIFEST:-data/processed/mvp_manifest.jsonl}
PAIRS=${PAIRS:-20}
STYLE=${STYLE:-direct}
MODEL_FAMILY=${MODEL_FAMILY:-qwen25-omni-3b}
SHARD=${SHARD:-0}
SHARDS=${SHARDS:-1}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
CONDITIONS=${CONDITIONS:-}
ENV_DIR=${ENV_DIR:-$REPO/.venv-model}
cd "$REPO"
mkdir -p logs outputs
COND_ARGS=()
if [[ -n "$CONDITIONS" ]]; then
  read -r -a COND_ARRAY <<< "$CONDITIONS"
  COND_ARGS=(--conditions "${COND_ARRAY[@]}")
fi
LOG="$REPO/logs/$(basename "$OUTPUT").log"
exec > >(tee -a "$LOG") 2>&1
echo "START $(date -Is) host=$(hostname) gpu=$GPU model=$MODEL_PATH output=$OUTPUT"
nvidia-smi --id="$GPU" --query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu --format=csv,noheader
if nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "ABORT: a compute process already occupies the selected GPU $GPU"
  exit 2
fi
test -f "$MODEL_PATH/config.json"
set +e
CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 PYTHONPATH=. "$ENV_DIR/bin/python" \
  scripts/evaluate_manifest.py "$MANIFEST" \
  --output "$OUTPUT" --model-path "$MODEL_PATH" --limit-pairs "$PAIRS" \
  --model-family "$MODEL_FAMILY" \
  --pair-shard "$SHARD" --pair-shards "$SHARDS" --prompt-style "$STYLE" \
  --max-new-tokens "$MAX_NEW_TOKENS" "${COND_ARGS[@]}"
rc=$?
set -e
echo "PIPE_RC=$rc"
if [[ "$rc" -ne 0 ]]; then exit "$rc"; fi
nvidia-smi --id="$GPU" --query-gpu=index,memory.used,memory.free,utilization.gpu --format=csv,noheader
echo "__DONE__ $(date -Is)"
