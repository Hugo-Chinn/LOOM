#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="$ROOT/.venv-unirank/bin/python"
RUN_DIR="$ROOT/runs_local/onetrans_peft_pair_multiseed"
mkdir -p "$RUN_DIR"
"$PYTHON_BIN" scripts/build_onetrans_peft_pair_multiseed_config.py

GPUS=(2 3 4 5 6 7)
EXPIDS=(
  OneTrans_PeftPair_Exact_W005_Seed20262028
  OneTrans_PeftPair_Score10_StageWeighted_Seed20262028
  OneTrans_PeftPair_Exact_W005_Seed20262029
  OneTrans_PeftPair_Score10_StageWeighted_Seed20262029
  OneTrans_PeftPair_Exact_W005_Seed20262030
  OneTrans_PeftPair_Score10_StageWeighted_Seed20262030
)

pids=()
for i in "${!EXPIDS[@]}"; do
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PYTHON_BIN" run_expid.py \
    --config ./config_onetrans_peft_pair_multiseed --expid "${EXPIDS[$i]}" --gpu 0 \
    > "$RUN_DIR/${EXPIDS[$i]}.stdout.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do wait "$pid"; done
