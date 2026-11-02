#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GPU_LIST="${GPU_LIST:-0,1,2}"
IFS=',' read -r GPU_WHOLE GPU_L1 GPU_L3 <<< "$GPU_LIST"

PYTHON_BIN="$ROOT/.venv-unirank/bin/python"
RUN_DIR="$ROOT/runs_local/taobao_onetrans_p2_multigpu"
mkdir -p "$RUN_DIR"

if pgrep -f "run_expid.py.*OneTrans_Taobao_P2_" >/dev/null; then
  echo "A Taobao OneTrans phase-2 experiment is already running; refusing duplicate launch." >&2
  exit 3
fi

if ! "$PYTHON_BIN" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit(1)
print(torch.cuda.device_count())
PY
then
  echo "CUDA is unavailable; refusing to launch partial experiments." >&2
  exit 2
fi

run_one() {
  local gpu="$1"
  local expid="$2"
  local logfile="$RUN_DIR/${expid}.stdout.log"
  echo "Launching ${expid} on physical GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" run_expid.py \
    --config ./config_local \
    --expid "$expid" \
    --gpu 0 \
    > "$logfile" 2>&1 &
  echo "$!" > "$RUN_DIR/${expid}.pid"
}

run_one "$GPU_WHOLE" OneTrans_Taobao_P2_WholeEncoderLoop
run_one "$GPU_L1" OneTrans_Taobao_P2_BaseRefiner_L1
run_one "$GPU_L3" OneTrans_Taobao_P2_BaseRefiner_L3

wait || true
