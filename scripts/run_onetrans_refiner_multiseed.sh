#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="$ROOT/.venv-unirank/bin/python"
RUN_DIR="$ROOT/runs_local/onetrans_refiner_multiseed"
mkdir -p "$RUN_DIR"
"$PYTHON_BIN" scripts/build_onetrans_refiner_multiseed_config.py

GPU_LIST="${GPU_LIST:-2,3,4,5,6,7}"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
EXPIDS=(
  OneTrans_BaseRefiner_Branch_Seed20262027
  OneTrans_BaseRefiner_Linear_Seed20262028
  OneTrans_BaseRefiner_Branch_Seed20262028
  OneTrans_BaseRefiner_Linear_Seed20262029
  OneTrans_BaseRefiner_Branch_Seed20262029
  OneTrans_BaseRefiner_Linear_Seed20262030
  OneTrans_BaseRefiner_Branch_Seed20262030
)

if (( ${#GPUS[@]} == 0 )); then
  echo "GPU_LIST must contain at least one GPU" >&2
  exit 2
fi

run_one() {
  local gpu="$1"
  local expid="$2"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" run_expid.py \
    --config ./config_onetrans_refiner_multiseed --expid "$expid" --gpu 0 \
    > "$RUN_DIR/${expid}.stdout.log" 2>&1
}

pids=()
for i in "${!EXPIDS[@]}"; do
  gpu="${GPUS[$((i % ${#GPUS[@]}))]}"
  expid="${EXPIDS[$i]}"
  run_one "$gpu" "$expid" &
  pids+=("$!")
  if (( ${#pids[@]} == ${#GPUS[@]} )); then
    wait "${pids[0]}"
    pids=("${pids[@]:1}")
  fi
done
for pid in "${pids[@]}"; do wait "$pid"; done
