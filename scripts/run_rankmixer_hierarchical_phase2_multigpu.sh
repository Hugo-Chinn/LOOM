#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="$ROOT/.venv-unirank/bin/python"
RUN_DIR="$ROOT/runs_local/rankmixer_hierarchical_phase2"
mkdir -p "$RUN_DIR"
"$PYTHON_BIN" scripts/build_rankmixer_hierarchical_phase2_config.py

GPUS=(2 3 4 5 6 7)
EXPIDS=(
  RankMixer_MerRec_DAG_Gated_Softmax
  RankMixer_MerRec_DAG_Gated_Softmax_Pre2
  RankMixer_KuaiRand_DAG_Gated_Softmax
  RankMixer_KuaiRand_DAG_Gated_Softmax_Pre2
  RankMixer_KuaiRand_DAG_Gated_Mean_Pre2
  RankMixer_KuaiRand_DAG_Gated_Softmax_Progressive
)
pids=()
for i in "${!EXPIDS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PYTHON_BIN" run_expid.py \
    --config ./config_rankmixer_hierarchical_phase2 --expid "${EXPIDS[$i]}" --gpu 0 \
    > "$RUN_DIR/${EXPIDS[$i]}.stdout.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
