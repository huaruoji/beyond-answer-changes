#!/usr/bin/env bash
# Launch the CTL pilot as soon as a GPU frees up, with only the two required probes.
#
# Run this on the remote in a dl-tasks window when the target cards are busy. It does
# not kill anything and does not queue work onto a shared card: it polls until a card
# is genuinely idle (memory low AND no compute process), re-checks immediately before
# launching, and then runs one shard per card.
#
# Usage (remote, inside tmux):
#   RUN_DIR=$PWD/runs/ctl-sample60 bash scripts/ctl_launch_when_free.sh
#
# Optional: MODELS="3b 7b"  MAX_WAIT_H=12  POLL_S=120
set -uo pipefail

REPO=${REPO:-$PWD}
RUN_DIR=${RUN_DIR:-$REPO/runs/ctl-sample60}
MODELS=${MODELS:-"3b 7b"}
POLL_S=${POLL_S:-120}
MAX_WAIT_H=${MAX_WAIT_H:-12}
IDLE_MIB=${IDLE_MIB:-200}

free_gpu() {
  # a card is free only if it holds little memory AND has no compute process
  local idx mem procs
  while read -r idx mem; do
    mem=${mem// /}
    [[ "${mem%MiB}" -lt "$IDLE_MIB" ]] || continue
    procs=$(nvidia-smi --id="$idx" --query-compute-apps=pid --format=csv,noheader | grep -c '[0-9]' || true)
    [[ "$procs" -eq 0 ]] && { echo "$idx"; return 0; }
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | tr ',' ' ')
  return 1
}

declare -A WANT=([3b]="Qwen2.5-Omni-3B" [7b]="Qwen2.5-Omni-7B")
declare -A DONE=()   # set -u trips on an empty associative array expansion
deadline=$(( $(date +%s) + MAX_WAIT_H * 3600 ))
echo "START $(date -Is) waiting for a free GPU (idle < ${IDLE_MIB} MiB, max ${MAX_WAIT_H}h)"

while (( $(date +%s) < deadline )); do
  for tag in $MODELS; do
    [[ -n "${DONE[$tag]:-}" ]] && continue
    gpu=$(free_gpu) || { echo "$(date -Is) no free GPU"; break; }
    # re-check right before launch: someone may have taken it during the loop
    sleep 2
    gpu2=$(free_gpu) || { echo "$(date -Is) card vanished"; continue; }
    [[ "$gpu" == "$gpu2" ]] || continue
    out="outputs/ctl/pred_${tag}.jsonl"
    echo "$(date -Is) launching $tag on GPU $gpu -> $out"
    RUN_DIR="$RUN_DIR" MODEL_PATH="$REPO/models/${WANT[$tag]}" OUTPUT="$out" GPU="$gpu" \
      MODEL_FAMILY=$([[ $tag == 3b ]] && echo qwen25-omni-3b || echo qwen25-omni-3b) \
      bash "$REPO/scripts/remote_ctl_eval.sh" && DONE[$tag]=1
    echo "$(date -Is) $tag finished rc=$?"
  done
  wanted=$(echo $MODELS | wc -w)
  [[ ${#DONE[@]} -ge $wanted ]] && break
  sleep "$POLL_S"
done

echo "RESULT completed: ${!DONE[*]:-none}"
echo "__DONE__ $(date -Is)"
