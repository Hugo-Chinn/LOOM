#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="$ROOT/.venv-unirank/bin/python"
RUN_DIR="$ROOT/runs_local/rankmixer_taac_loop"
mkdir -p "$RUN_DIR"
"$PYTHON_BIN" scripts/build_rankmixer_taac_loop_config.py

GPUS=(2 3 4 5 6 7)
EXPIDS=(
  RankMixer_TAAC_Parallel
  RankMixer_TAAC_Sequential
  RankMixer_TAAC_Loop_Progressive
  RankMixer_TAAC_Loop_Deep
  RankMixer_TAAC_Loop_GatedDeep
  RankMixer_TAAC_Loop_GatedDeep_Pre2
)
pids=()
for i in "${!EXPIDS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PYTHON_BIN" run_expid.py \
    --config ./config_rankmixer_taac_loop --expid "${EXPIDS[$i]}" --gpu 0 \
    > "$RUN_DIR/${EXPIDS[$i]}.stdout.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
