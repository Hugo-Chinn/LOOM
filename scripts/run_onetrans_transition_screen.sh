#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="$ROOT/.venv-unirank/bin/python"
RUN_DIR="$ROOT/runs_local/onetrans_transition_screen"
mkdir -p "$RUN_DIR"
"$PYTHON_BIN" scripts/build_onetrans_transition_screen_config.py

GPUS=(2 3 4 5)
EXPIDS=(
  OneTrans_ResidualLogit_Seed20262027
  OneTrans_ResidualLogit_Direction01_Seed20262027
  OneTrans_ResidualLogit_Risk01_Seed20262027
  OneTrans_ResidualLogit_Direction05Risk05_Seed20262027
)

pids=()
for i in "${!EXPIDS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PYTHON_BIN" run_expid.py \
    --config ./config_onetrans_transition_screen --expid "${EXPIDS[$i]}" --gpu 0 \
    > "$RUN_DIR/${EXPIDS[$i]}.stdout.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
