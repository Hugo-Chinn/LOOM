#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="$ROOT/.venv-unirank/bin/python"
RUN_DIR="$ROOT/runs_local/rankmixer_branch_gate_ablation"
mkdir -p "$RUN_DIR"
"$PYTHON_BIN" scripts/build_rankmixer_branch_gate_ablation.py

GPUS=(2 3 4 5 6 7)
EXPIDS=(
  RankMixer_Branch_Seed20262028
  RankMixer_BranchGated_Seed20262028
  RankMixer_Branch_Seed20262029
  RankMixer_BranchGated_Seed20262029
  RankMixer_Branch_Seed20262030
  RankMixer_BranchGated_Seed20262030
)

pids=()
for i in "${!EXPIDS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PYTHON_BIN" run_expid.py \
    --config ./config_rankmixer_branch_gate_ablation \
    --expid "${EXPIDS[$i]}" --gpu 0 \
    > "$RUN_DIR/${EXPIDS[$i]}.stdout.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
