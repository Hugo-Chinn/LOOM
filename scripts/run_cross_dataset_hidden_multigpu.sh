#!/usr/bin/env bash
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv-unirank/bin/python"
OUT="$ROOT/runs_local/cross_dataset_hidden_training"
mkdir -p "$OUT"
"$PY" scripts/build_cross_dataset_hidden_config.py

EXPS=(Hidden_MerRec_Raw Hidden_MerRec_DAG Hidden_QK_Raw Hidden_QK_DAG Hidden_Kuai_Raw Hidden_Kuai_DAG)
GPUS=(2 3 4 5 6 7)
pids=()
for i in "${!EXPS[@]}"; do
  exp="${EXPS[$i]}"; gpu="${GPUS[$i]}"
  (CUDA_VISIBLE_DEVICES="$gpu" "$PY" run_expid.py --config ./config_cross_dataset_hidden --expid "$exp" --gpu 0 > "$OUT/$exp.stdout.log" 2>&1) &
  pids+=("$!")
  echo "$! $gpu $exp"
done
failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
exit "$failed"
