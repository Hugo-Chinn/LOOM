#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv-unirank/bin/python"
IFS=',' read -r -a GPUS <<< "${GPU_LIST:-0,1,2,3,4,5,6,7}"
"$PY" scripts/build_rankmixer_preference_margin_phase3_config.py
mkdir -p runs_local/rankmixer_preference_margin_phase3
pids=()
for gpu in "${GPUS[@]}"; do
  "$PY" scripts/run_rankmixer_preference_margin_phase3_worker.py --gpu "$gpu" \
    > "runs_local/rankmixer_preference_margin_phase3/worker_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
