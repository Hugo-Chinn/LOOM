#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="$ROOT/.venv-unirank/bin/python"
RUN_DIR="$ROOT/runs_local/onetrans_baseline_multiseed"
mkdir -p "$RUN_DIR"
"$PYTHON_BIN" scripts/build_onetrans_refiner_multiseed_config.py

GPU_LIST="${GPU_LIST:-2,3,4}"
IFS=',' read -r GPU_28 GPU_29 GPU_30 <<< "$GPU_LIST"

run_one() {
  local gpu="$1"
  local expid="$2"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" run_expid.py \
    --config ./config_onetrans_refiner_multiseed --expid "$expid" --gpu 0 \
    > "$RUN_DIR/${expid}.stdout.log" 2>&1 &
}

run_one "$GPU_28" OneTrans_Parallel_Seed20262028
p1=$!
run_one "$GPU_29" OneTrans_Parallel_Seed20262029
p2=$!
run_one "$GPU_30" OneTrans_Parallel_Seed20262030
p3=$!

wait "$p1"
wait "$p2"
wait "$p3"
