#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="$ROOT/.venv-unirank/bin/python"
RUN_DIR="$ROOT/runs_local/onetrans_peft_multiseed"
mkdir -p "$RUN_DIR"
"$PYTHON_BIN" scripts/build_onetrans_peft_multiseed_config.py

GPUS=(2 3 4 5 6 7)
EXPIDS=(
  OneTrans_Peft_FiLM_Seed20262028
  OneTrans_Peft_QVLoRA_R4_Seed20262028
  OneTrans_Peft_FiLM_Seed20262029
  OneTrans_Peft_QVLoRA_R4_Seed20262029
  OneTrans_Peft_FiLM_Seed20262030
  OneTrans_Peft_QVLoRA_R4_Seed20262030
)

pids=()
for i in "${!EXPIDS[@]}"; do
  expid="${EXPIDS[$i]}"
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PYTHON_BIN" run_expid.py \
    --config ./config_onetrans_peft_multiseed --expid "$expid" --gpu 0 \
    > "$RUN_DIR/${expid}.stdout.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
