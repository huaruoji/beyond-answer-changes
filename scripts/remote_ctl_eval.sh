#!/usr/bin/env bash
# Run one CTL pilot shard on the GPU host, from a fresh dl-tasks window.
#
# Why this exists next to remote_eval.sh: the evaluator resolves a relative audio
# path as either "$PWD/<path>" or "manifest.parents[2]/<path>". For a manifest at
# <run>/manifest.remote.jsonl the second candidate lands on <run>/../audio, which is
# not where the audio is, so the run would fail on the first file. Launching from the
# run directory makes the first candidate correct: $PWD/audio/<file>.
#
# It also refuses to start on a busy GPU and fails the run if any item errored —
# an error rate above zero makes the MIG uninterpretable, so it is a gate, not a
# warning.
#
# Required: MODEL_PATH, OUTPUT. Optional: GPU (default: first free), MANIFEST,
# RUN_DIR, MODEL_FAMILY, PAIRS, SHARD, SHARDS, STYLE, CONDITIONS, PREFLIGHT.
set -euo pipefail

REPO=${REPO:-$HOME/projects/audio-causal-grounding}
RUN_DIR=${RUN_DIR:?set RUN_DIR (the directory holding audio/ and the manifest)}
MANIFEST=${MANIFEST:-manifest.remote.jsonl}
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH}
OUTPUT=${OUTPUT:?set OUTPUT}
MODEL_FAMILY=${MODEL_FAMILY:-qwen25-omni-3b}
PAIRS=${PAIRS:-10000}
SHARD=${SHARD:-0}
SHARDS=${SHARDS:-1}
STYLE=${STYLE:-direct}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
CONDITIONS=${CONDITIONS:-}
ENV_DIR=${ENV_DIR:-$REPO/.venv-model}
PREFLIGHT=${PREFLIGHT:-0}

cd "$REPO"
[[ -f "$MODEL_PATH/config.json" ]] || { echo "ABORT: no config.json at $MODEL_PATH"; exit 2; }
[[ -f "$RUN_DIR/$MANIFEST" ]] || { echo "ABORT: no manifest at $RUN_DIR/$MANIFEST"; exit 2; }
[[ -d "$RUN_DIR/audio" ]] || { echo "ABORT: no audio/ under $RUN_DIR"; exit 2; }

# Verify the audio on this host is the audio that was measured before spending GPU
# time on it. A path or sync problem otherwise shows up as plausible-looking answers
# about the wrong recordings, which no later stage can detect.
if [[ -f "$RUN_DIR/audio_digests.json" ]]; then
  echo "verifying $(python3 -c "import json;print(len(json.load(open('$RUN_DIR/audio_digests.json'))))") audio digests before launch"
  python3 - "$RUN_DIR" <<'DIGESTS'
import hashlib, json, sys
from pathlib import Path
root = Path(sys.argv[1])
recorded = json.loads((root / "audio_digests.json").read_text())
bad, missing = [], []
for name, want in recorded.items():
    path = root / "audio" / name
    if not path.exists():
        missing.append(name)
        continue
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != want:
        bad.append(name)
print(f"digest check: {len(recorded)} recorded, {len(missing)} missing, {len(bad)} mismatched")
if missing or bad:
    print("FAIL: audio on this host is not the audio that was measured", file=sys.stderr)
    sys.exit(1)
DIGESTS
  [[ $? -eq 0 ]] || { echo "ABORT: audio digest mismatch"; exit 4; }
else
  echo "WARN: no audio_digests.json in $RUN_DIR, skipping provenance check"
fi

if [[ -z "${GPU:-}" ]]; then
  GPU=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
        | awk -F', *' '$2 < 200 {print $1; exit}')
  [[ -n "$GPU" ]] || { echo "ABORT: no idle GPU (all above 200 MiB used)"; exit 2; }
fi
if nvidia-smi --id="$GPU" --query-compute-apps=pid --format=csv,noheader | grep -q '[0-9]'; then
  echo "ABORT: GPU $GPU already has a compute process"; exit 2
fi

mkdir -p "$REPO/logs" "$(dirname "$REPO/$OUTPUT")"
LOG="$REPO/logs/$(basename "$OUTPUT").log"
exec > >(tee -a "$LOG") 2>&1
echo "START $(date -Is) host=$(hostname) gpu=$GPU family=$MODEL_FAMILY shard=$SHARD/$SHARDS preflight=$PREFLIGHT"
nvidia-smi --id="$GPU" --query-gpu=index,name,memory.total,memory.used --format=csv,noheader

COND_ARGS=()
if [[ -n "$CONDITIONS" ]]; then
  read -r -a COND_ARRAY <<< "$CONDITIONS"
  COND_ARGS=(--conditions "${COND_ARRAY[@]}")
fi
LIMIT_ARGS=()
[[ "$PREFLIGHT" == "1" ]] && LIMIT_ARGS=(--limit-pairs 2)

# run from the audio directory so the evaluator's first path candidate resolves
set +e
( cd "$RUN_DIR" && CUDA_VISIBLE_DEVICES="$GPU" PYTHONUNBUFFERED=1 PYTHONPATH="$REPO" \
    "$ENV_DIR/bin/python" "$REPO/scripts/evaluate_manifest.py" "$MANIFEST" \
    --output "$REPO/$OUTPUT" --model-path "$MODEL_PATH" \
    --model-family "$MODEL_FAMILY" --pair-shard "$SHARD" --pair-shards "$SHARDS" \
    --prompt-style "$STYLE" --max-new-tokens "$MAX_NEW_TOKENS" \
    "${LIMIT_ARGS[@]}" "${COND_ARGS[@]}" )
rc=$?
set -e
echo "PIPE_RC=$rc"
[[ "$rc" -eq 0 ]] || exit "$rc"

python3 - "$REPO/$OUTPUT" <<'PY'
import json, sys
path = sys.argv[1]
total = errors = 0
for line in open(path):
    if not line.strip():
        continue
    rec = json.loads(line)
    total += 1
    if rec.get("error") or rec.get("parsed_choice") is None:
        errors += 1
print(f"GATE records={total} errors_or_unparsed={errors}")
if errors:
    sys.exit(3)
PY
gate=$?
echo "GATE_RC=$gate"
[[ "$gate" -eq 0 ]] || { echo "FAIL: error or unparsed answers; MIG would be uninterpretable"; exit "$gate"; }

nvidia-smi --id="$GPU" --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
echo "__DONE__ $(date -Is)"
