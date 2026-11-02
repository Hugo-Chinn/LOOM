#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU_LIST="${GPU_LIST:-2,3,4,5,6,7}"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
PYTHON_BIN="$ROOT/.venv-unirank/bin/python"
RUN_DIR="$ROOT/runs_local/hidden_analysis_training"
mkdir -p "$RUN_DIR"

"$PYTHON_BIN" scripts/build_hidden_analysis_config.py

EXPERIMENTS=(
  Hidden_RankMixer_Baseline
  Hidden_RankMixer_Loop
  Hidden_OneTrans_Baseline
  Hidden_OneTrans_Loop
  Hidden_EST_Baseline
  Hidden_EST_Loop
  Hidden_HeMix_Baseline
  Hidden_HeMix_Loop
  Hidden_UltraHSTU_Baseline
  Hidden_UltraHSTU_Loop
)

run_queue() {
  local gpu="$1"
  shift
  local expid
  for expid in "$@"; do
    echo "[$(date '+%F %T')] GPU ${gpu}: ${expid}"
    CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" run_expid.py \
      --config ./config_hidden_analysis --expid "$expid" --gpu 0 \
      > "$RUN_DIR/${expid}.stdout.log" 2>&1 || \
      echo "[$(date '+%F %T')] FAILED: ${expid}"
  done
}

declare -a PIDS=()
for ((i=0; i<${#GPUS[@]}; i++)); do
  queue=()
  for ((j=i; j<${#EXPERIMENTS[@]}; j+=${#GPUS[@]})); do
    queue+=("${EXPERIMENTS[$j]}")
  done
  if ((${#queue[@]})); then
    run_queue "${GPUS[$i]}" "${queue[@]}" &
    PIDS+=("$!")
  fi
done
for pid in "${PIDS[@]}"; do wait "$pid"; done
