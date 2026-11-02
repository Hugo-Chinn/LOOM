#!/usr/bin/env bash
# Run the matched iteration-depth ablations on KuaiRand and TAAC.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv-unirank/bin/python"
CONFIG="./config_rankmixer_iteration_depth"
RUN_DIR="$ROOT/runs_local/rankmixer_iteration_depth"
mkdir -p "$RUN_DIR"
"$PY" scripts/build_rankmixer_iteration_depth_config.py

run_one() {
  local gpu="$1" expid="$2" log="$RUN_DIR/$2.stdout.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" run_expid.py \
    --config "$CONFIG" --expid "$expid" --gpu 0 > "$log" 2>&1
}

run_one 2 RankMixer_Kuai_DAGMeanPre2_IterDepth &
kuai_pid=$!
run_one 3 RankMixer_TAAC_GatedDeepPre2_IterDepth &
taac_pid=$!
wait "$kuai_pid"
wait "$taac_pid"

"$PY" scripts/summarize_rankmixer_iteration_depth.py
