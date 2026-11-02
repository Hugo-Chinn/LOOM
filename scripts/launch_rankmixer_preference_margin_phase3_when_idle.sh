#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv-unirank/bin/python"
mkdir -p runs_local/rankmixer_preference_margin_phase3
"$PY" scripts/build_rankmixer_preference_margin_phase3_config.py

pids=()
for gpu in 0 1 2 3 4 5 6 7; do
  "$PY" scripts/wait_rankmixer_preference_margin_phase3_gpu.py --gpu "$gpu" \
    > "runs_local/rankmixer_preference_margin_phase3/wait_gpu${gpu}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
