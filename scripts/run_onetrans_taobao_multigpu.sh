#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# One physical GPU per baseline. Override if the scheduler assigns different
# free devices, e.g. GPU_LIST=2,4,6.
GPU_LIST="${GPU_LIST:-2,3,4}"
IFS=',' read -r GPU_PARALLEL GPU_LOOPED GPU_DEEP <<< "$GPU_LIST"

PYTHON_BIN="$ROOT/.venv-unirank/bin/python"
RUN_DIR="$ROOT/runs_local/taobao_onetrans_multigpu"
mkdir -p "$RUN_DIR"

if pgrep -f "run_expid.py.*OneTrans_Taobao_P1_" >/dev/null; then
  echo "A Taobao OneTrans phase-1 experiment is already running; refusing duplicate launch." >&2
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

run_one "$GPU_PARALLEL" OneTrans_Taobao_P1_Parallel
run_one "$GPU_LOOPED" OneTrans_Taobao_P1_Looped
run_one "$GPU_DEEP" OneTrans_Taobao_P1_Looped_DeepSupervision

wait || true
