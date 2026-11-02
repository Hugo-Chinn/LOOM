#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="$ROOT/.venv-unirank/bin/python"
RUN_DIR="$ROOT/runs_local/onetrans_warmstart"
mkdir -p "$RUN_DIR"
"$PYTHON_BIN" scripts/build_onetrans_warmstart_config.py

GPUS=(2 3 4)
EXPIDS=(
  OneTrans_QVLoRA_Conservative_LR1e4_Seed20262027
  OneTrans_QVLoRA_Conservative_LR3e5_Seed20262027
  OneTrans_QVLoRA_Conservative_LR1e5_Seed20262027
)

pids=()
for i in "${!EXPIDS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PYTHON_BIN" run_expid.py \
    --config ./config_onetrans_warmstart --expid "${EXPIDS[$i]}" --gpu 0 \
    > "$RUN_DIR/${EXPIDS[$i]}.stdout.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
