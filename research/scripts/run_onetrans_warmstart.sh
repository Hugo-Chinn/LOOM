#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="$ROOT/.venv-unirank/bin/python"
RUN_DIR="$ROOT/runs_local/onetrans_warmstart"
mkdir -p "$RUN_DIR"
"$PYTHON_BIN" scripts/build_onetrans_warmstart_config.py

GPUS=(2 3 4 5)
SEEDS=(20262027 20262028 20262029 20262030)

run_one() {
  local gpu="$1"
  local expid="$2"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" run_expid.py \
    --config ./config_onetrans_warmstart --expid "$expid" --gpu 0 \
    > "$RUN_DIR/${expid}.stdout.log" 2>&1 &
}

# Stage 1: train and retain one baseline checkpoint per seed.
pids=()
for i in "${!SEEDS[@]}"; do
  expid="OneTrans_WarmSource_Seed${SEEDS[$i]}"
  run_one "${GPUS[$i]}" "$expid"
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

# Stage 2a: frozen-base warm start.
pids=()
for i in "${!SEEDS[@]}"; do
  expid="OneTrans_QVLoRA_FrozenBase_Seed${SEEDS[$i]}"
  run_one "${GPUS[$i]}" "$expid"
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done

# Stage 2b: low-LR base warm start.
pids=()
for i in "${!SEEDS[@]}"; do
  expid="OneTrans_QVLoRA_LowLRBase_Seed${SEEDS[$i]}"
  run_one "${GPUS[$i]}" "$expid"
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
