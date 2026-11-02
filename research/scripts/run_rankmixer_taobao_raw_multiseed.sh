#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv-unirank/bin/python"
CONFIG="./config_rankmixer_taobao_raw_multiseed"
RUN_DIR="$ROOT/runs_local/rankmixer_taobao_raw_multiseed"
mkdir -p "$RUN_DIR"
"$PY" scripts/build_rankmixer_taobao_raw_multiseed_config.py

GPUS=(2 3 4)
SEEDS=(20262028 20262029 20262030)
pids=()
for i in "${!SEEDS[@]}"; do
  seed="${SEEDS[$i]}"
  expid="RankMixer_Taobao_Raw_Seed${seed}"
  log="$RUN_DIR/${expid}.stdout.log"
  if [[ -f "$log" ]] && grep -q 'Training finished.' "$log" && grep -q 'Test evaluation' "$log"; then
    echo "SKIP completed $expid"
    continue
  fi
  echo "START gpu=${GPUS[$i]} $expid"
  CUDA_VISIBLE_DEVICES="${GPUS[$i]}" "$PY" run_expid.py \
    --config "$CONFIG" --expid "$expid" --gpu 0 > "$log" 2>&1 &
  pids+=("$!")
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
