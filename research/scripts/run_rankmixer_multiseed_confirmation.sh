#!/usr/bin/env bash
# Run frozen paired confirmations in six-GPU waves. Completed logs are skipped.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
PY="$ROOT/.venv-unirank/bin/python"
CONFIG="./config_rankmixer_multiseed_confirmation"
RUN_DIR="$ROOT/runs_local/rankmixer_multiseed_confirmation"
mkdir -p "$RUN_DIR"
"$PY" scripts/build_rankmixer_multiseed_confirmation.py

SEEDS=(20262028 20262029 20262030)
TAAC=(Parallel Sequential Loop_Progressive Loop_Deep Loop_GatedDeep Loop_GatedDeep_Pre2)
CROSS=(
  MerRec_Parallel MerRec_LinearGated MerRec_DAGGated
  QK_Parallel QK_Linear QK_DAGGated
  Kuai_Parallel Kuai_Linear Kuai_DAGMeanPre2
)

run_wave() {
  local jobs=("$@") pids=() gpu=2 expid log
  for expid in "${jobs[@]}"; do
    log="$RUN_DIR/${expid}.stdout.log"
    if [[ -f "$log" ]] && grep -q 'Training finished.' "$log" && grep -q 'Test evaluation' "$log"; then
      echo "SKIP completed $expid"
      continue
    fi
    echo "START gpu=$gpu $expid"
    CUDA_VISIBLE_DEVICES="$gpu" "$PY" run_expid.py --config "$CONFIG" \
      --expid "$expid" --gpu 0 > "$log" 2>&1 &
    pids+=("$!")
    gpu=$((gpu + 1))
  done
  local failed=0
  for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
  return "$failed"
}

# TAAC: all six ablations for each independent seed (three exactly paired waves).
for seed in "${SEEDS[@]}"; do
  wave=()
  for name in "${TAAC[@]}"; do wave+=("RankMixer_TAAC_${name}_S${seed}"); done
  run_wave "${wave[@]}"
done

# Cross-dataset frozen finalists: nine jobs per seed, split into 6+3 waves.
for seed in "${SEEDS[@]}"; do
  wave=()
  for name in "${CROSS[@]}"; do wave+=("RankMixer_${name}_S${seed}"); done
  run_wave "${wave[@]:0:6}"
  run_wave "${wave[@]:6:3}"
done
