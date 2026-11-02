#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="$ROOT/.venv-unirank/bin/python"
RUN_DIR="$ROOT/runs_local/onetrans_residual_multiseed"
mkdir -p "$RUN_DIR"
"$PYTHON_BIN" scripts/build_onetrans_residual_multiseed_config.py

GPUS=(2 3 4)
SEEDS=(20262028 20262029 20262030)
pids=()
for i in "${!SEEDS[@]}"; do
  expid="OneTrans_ResidualLogit_Seed${SEEDS[$i]}"
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PYTHON_BIN" run_expid.py \
    --config ./config_onetrans_residual_multiseed --expid "$expid" --gpu 0 \
    > "$RUN_DIR/${expid}.stdout.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
