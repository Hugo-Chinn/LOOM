#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PYTHON_BIN="$ROOT/.venv-unirank/bin/python"
RUN_DIR="$ROOT/runs_local/onetrans_warmstart"
mkdir -p "$RUN_DIR"
"$PYTHON_BIN" scripts/build_onetrans_warmstart_config.py

CUDA_VISIBLE_DEVICES=2 "$PYTHON_BIN" run_expid.py \
  --config ./config_onetrans_warmstart \
  --expid OneTrans_QVLoRA_ZeroInitFrozen_Seed20262027 --gpu 0 \
  > "$RUN_DIR/OneTrans_QVLoRA_ZeroInitFrozen_Seed20262027.stdout.log" 2>&1 &
p1=$!

CUDA_VISIBLE_DEVICES=3 "$PYTHON_BIN" run_expid.py \
  --config ./config_onetrans_warmstart \
  --expid OneTrans_QVLoRA_ZeroInitLowLR_Seed20262027 --gpu 0 \
  > "$RUN_DIR/OneTrans_QVLoRA_ZeroInitLowLR_Seed20262027.stdout.log" 2>&1 &
p2=$!

wait "$p1"
wait "$p2"
