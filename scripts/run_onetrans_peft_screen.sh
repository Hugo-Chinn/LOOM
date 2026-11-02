#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="$ROOT/.venv-unirank/bin/python"
RUN_DIR="$ROOT/runs_local/onetrans_peft_screen"
mkdir -p "$RUN_DIR"
"$PYTHON_BIN" scripts/build_onetrans_peft_screen_config.py

GPU_LIST="${GPU_LIST:-2,3,4,5}"
IFS=',' read -r GPU_FILM GPU_ADAPTER GPU_QLORA GPU_QVLORA <<< "$GPU_LIST"

run_one() {
  local gpu="$1"
  local expid="$2"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" run_expid.py \
    --config ./config_onetrans_peft_screen --expid "$expid" --gpu 0 \
    > "$RUN_DIR/${expid}.stdout.log" 2>&1 &
}

run_one "$GPU_FILM" OneTrans_Peft_FiLM_Seed20262027
p1=$!
run_one "$GPU_ADAPTER" OneTrans_Peft_Adapter16_Seed20262027
p2=$!
run_one "$GPU_QLORA" OneTrans_Peft_QLoRA_R4_Seed20262027
p3=$!
run_one "$GPU_QVLORA" OneTrans_Peft_QVLoRA_R4_Seed20262027
p4=$!

wait "$p1"
wait "$p2"
wait "$p3"
wait "$p4"
