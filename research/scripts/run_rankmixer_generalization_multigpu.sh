#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="$ROOT/.venv-unirank/bin/python"
RUN_DIR="$ROOT/runs_local/rankmixer_generalization"
mkdir -p "$RUN_DIR"
"$PYTHON_BIN" scripts/build_rankmixer_generalization_config.py

GPUS=(2 3 4 5 6 7)
EXPIDS=(
  RankMixer_MerRec_Baseline
  RankMixer_MerRec_LoopDeep
  RankMixer_QKVideo_Baseline
  RankMixer_QKVideo_LoopDeep
  RankMixer_KuaiRand_Baseline
  RankMixer_KuaiRand_LoopDeep
)

pids=()
for i in "${!EXPIDS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PYTHON_BIN" run_expid.py \
    --config ./config_rankmixer_generalization \
    --expid "${EXPIDS[$i]}" --gpu 0 \
    > "$RUN_DIR/${EXPIDS[$i]}.stdout.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
