#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="$ROOT/.venv-unirank/bin/python"
RUN_DIR="$ROOT/runs_local/rankmixer_multiseed"
mkdir -p "$RUN_DIR"
"$PYTHON_BIN" scripts/build_rankmixer_multiseed_config.py

GPUS=(2 3 4 5 6 7)
EXPIDS=(
  RankMixer_LinearDeep_Seed20262028
  RankMixer_BranchGated_Seed20262028
  RankMixer_LinearDeep_Seed20262029
  RankMixer_BranchGated_Seed20262029
  RankMixer_LinearDeep_Seed20262030
  RankMixer_BranchGated_Seed20262030
)

pids=()
for i in "${!EXPIDS[@]}"; do
  expid="${EXPIDS[$i]}"
  gpu="${GPUS[$i]}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" run_expid.py \
    --config ./config_rankmixer_multiseed --expid "$expid" --gpu 0 \
    > "$RUN_DIR/${expid}.stdout.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
