#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU_LIST="${GPU_LIST:-2,3,4,5,6,7}"
IFS=',' read -r -a GPUS <<< "$GPU_LIST"
PYTHON_BIN="$ROOT/.venv-unirank/bin/python"
CONFIG_DIR="$ROOT/config_taobao_all_baselines"
RUN_DIR="$ROOT/runs_local/taobao_all_baselines"
STATUS_DIR="$RUN_DIR/status"
mkdir -p "$RUN_DIR" "$STATUS_DIR"

if ! "$PYTHON_BIN" - <<'PY'
import torch
assert torch.cuda.is_available(), "CUDA is unavailable"
print(f"CUDA devices visible: {torch.cuda.device_count()}")
PY
then
  echo "CUDA is unavailable; refusing to launch." >&2
  exit 2
fi

"$PYTHON_BIN" scripts/build_taobao_baseline_config.py

# RankMixer and OneTrans already completed under the same local protocol.
BASELINES=(
  UniMixer_Taobao_Action
  HyFormer_Taobao_Action
  MixFormer_Taobao_Action
  TokenMixer_Taobao_Action
  HiFormer_Taobao_Action
  INFNet_Taobao_Action
  EST_Taobao_Action
  LONGER_Taobao_Action
  Zenith_Taobao_Action
  HeMix_Taobao_Action
  TokenFormer_Taobao_Action
  UltraHSTU_Taobao_Action
  SSR_Taobao_Action
)

run_queue() {
  local gpu="$1"
  shift
  local expid logfile
  for expid in "$@"; do
    logfile="$RUN_DIR/${expid}.stdout.log"
    echo "[$(date '+%F %T')] GPU ${gpu}: start ${expid}" | tee "$STATUS_DIR/${expid}.running"
    if CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" run_expid.py \
      --config "$CONFIG_DIR" --expid "$expid" --gpu 0 \
      > "$logfile" 2>&1; then
      echo "[$(date '+%F %T')] GPU ${gpu}: complete ${expid}" | tee "$STATUS_DIR/${expid}.complete"
    else
      code="$?"
      echo "[$(date '+%F %T')] GPU ${gpu}: failed ${expid}, exit=${code}" | tee "$STATUS_DIR/${expid}.failed"
    fi
  done
}

declare -a PIDS=()
for ((i=0; i<${#GPUS[@]}; i++)); do
  queue=()
  for ((j=i; j<${#BASELINES[@]}; j+=${#GPUS[@]})); do
    queue+=("${BASELINES[$j]}")
  done
  if ((${#queue[@]} > 0)); then
    run_queue "${GPUS[$i]}" "${queue[@]}" &
    PIDS+=("$!")
  fi
done

failed=0
for pid in "${PIDS[@]}"; do
  wait "$pid" || failed=1
done
exit "$failed"
