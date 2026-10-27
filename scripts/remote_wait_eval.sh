#!/usr/bin/env bash
# Wait for a model download window, then run remote_eval with its own GPU probe.
set -euo pipefail

REPO=${REPO:-$HOME/projects/audio-causal-grounding}
READY_LOG=${READY_LOG:?set READY_LOG}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH}
MODEL_FAMILY=${MODEL_FAMILY:?set MODEL_FAMILY}
OUTPUT=${OUTPUT:?set OUTPUT}
GPU=${GPU:?set GPU}
PAIRS=${PAIRS:-8}
STYLE=${STYLE:-direct}
cd "$REPO"
while ! grep -q '__DONE__' "$READY_LOG" 2>/dev/null; do
  echo "waiting for $READY_LOG ($(date -Is))"
  sleep 60
done
MODEL_PATH="$MODEL_PATH" MODEL_FAMILY="$MODEL_FAMILY" OUTPUT="$OUTPUT" GPU="$GPU" \
  PAIRS="$PAIRS" STYLE="$STYLE" "$REPO/scripts/remote_eval.sh"
