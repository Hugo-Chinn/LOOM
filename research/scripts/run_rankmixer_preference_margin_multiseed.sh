#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv-unirank/bin/python"
CONFIG="./config_rankmixer_preference_margin"
RUN_DIR="$ROOT/runs_local/rankmixer_preference_margin"
mkdir -p "$RUN_DIR"
"$PY" scripts/build_rankmixer_preference_margin_config.py

IFS=',' read -r -a GPUS <<< "${GPU_LIST:-0}"
SEEDS=(20262027 20262028 20262029 20262030)
VARIANTS=(Baseline FixedMargin ProgressiveMargin ReverseProgressiveMargin)
EXPIDS=()
for seed in "${SEEDS[@]}"; do
  for variant in "${VARIANTS[@]}"; do
    EXPIDS+=("RankMixer_Branch_${variant}_Seed${seed}")
  done
done

run_one() {
  local gpu="$1"
  local expid="$2"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" run_expid.py \
    --config "$CONFIG" --expid "$expid" --gpu 0 \
    > "$RUN_DIR/${expid}.stdout.log" 2>&1
}

active_pids=()
for index in "${!EXPIDS[@]}"; do
  gpu="${GPUS[$((index % ${#GPUS[@]}))]}"
  run_one "$gpu" "${EXPIDS[$index]}" &
  active_pids+=("$!")
  if (( ${#active_pids[@]} == ${#GPUS[@]} )); then
    for pid in "${active_pids[@]}"; do wait "$pid"; done
    active_pids=()
  fi
done
for pid in "${active_pids[@]}"; do wait "$pid"; done
