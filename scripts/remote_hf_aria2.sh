#!/usr/bin/env bash
# Generalized multi-connection downloader for a Hugging Face model repo.
#
# Usage:
#   MODEL_ID=Qwen/Qwen2.5-Omni-7B LOCAL_NAME=Qwen2.5-Omni-7B REVISION=<sha> \
#     ONLY_FILE=model-00003-of-00005.safetensors LOG_SUFFIX=shard3 \
#     scripts/remote_hf_aria2.sh
#
# The Hugging Face Xet path was measured at ~0.1 MB/s per connection from this
# host, while aria2 range requests reach multiple MB/s, so large shards are
# fetched here and every file is resumable.
set -euo pipefail

REPO=${REPO:-$HOME/projects/audio-causal-grounding}
MODEL_ID=${MODEL_ID:?set MODEL_ID, e.g. Qwen/Qwen2.5-Omni-7B}
LOCAL_NAME=${LOCAL_NAME:-${MODEL_ID#*/}}
REVISION=${REVISION:?set REVISION (commit sha)}
MODEL_DIR="$REPO/models/$LOCAL_NAME"
BASE_URL="https://huggingface.co/$MODEL_ID/resolve/$REVISION"
LOG_SUFFIX=${LOG_SUFFIX:-main}

mkdir -p "$REPO/logs" "$MODEL_DIR"
exec > >(tee -a "$REPO/logs/acg-dl-${LOCAL_NAME}-${LOG_SUFFIX}.log") 2>&1
echo "START $(date -Is) host=$(hostname) model=$MODEL_ID revision=$REVISION"

small_files=(
  added_tokens.json chat_template.json config.json generation_config.json
  merges.txt model.safetensors.index.json preprocessor_config.json
  special_tokens_map.json tokenizer.json tokenizer_config.json vocab.json
)
shards=(
  model-00001-of-00005.safetensors model-00002-of-00005.safetensors
  model-00003-of-00005.safetensors model-00004-of-00005.safetensors
  model-00005-of-00005.safetensors
)
if [[ -n "${ONLY_FILE:-}" ]]; then
  shards=("$ONLY_FILE")
fi

fetch() {
  local file="$1"
  if [[ -f "$MODEL_DIR/$file" && ! -f "$MODEL_DIR/$file.aria2" ]]; then
    echo "already complete: $file"
    return 0
  fi
  echo "fetching $file $(date -Is)"
  aria2c --continue=true --file-allocation=none --allow-overwrite=true \
    --auto-file-renaming=false --max-connection-per-server=8 --split=8 \
    --min-split-size=1M --dir="$MODEL_DIR" --out="$file" \
    "$BASE_URL/$file?download=true"
}

if [[ -z "${ONLY_FILE:-}" ]]; then
  for file in "${small_files[@]}"; do
    curl -fsSL --retry 3 -o "$MODEL_DIR/$file" "$BASE_URL/$file?download=true" || echo "skip $file"
  done
fi
for file in "${shards[@]}"; do
  fetch "$file"
done

du -sh "$MODEL_DIR"
echo "__DONE__ $(date -Is)"
