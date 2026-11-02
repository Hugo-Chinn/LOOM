#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

RAW_DIR="${1:-$ROOT/data/Taobao/raw_transition_source}"
OUTPUT_DIR="${2:-$ROOT/data/preprocessed/Taobao_Action_Transition}"
PAIR_DIR="${3:-$ROOT/data/preprocessed/Taobao_Action_Transition_Pairs}"
PYTHON_BIN="$ROOT/.venv-unirank/bin/python"

required=(raw_sample behavior_log ad_feature user_profile)
for stem in "${required[@]}"; do
  if ! find "$RAW_DIR" -maxdepth 1 -type f \
      \( -name "${stem}.csv" -o -name "${stem}.csv.tar.gz" -o -name "${stem}.tar.gz" \) \
      | grep -q .; then
    echo "Missing ${stem}.csv[.tar.gz] under $RAW_DIR" >&2
    exit 2
  fi
done

"$PYTHON_BIN" data/Taobao/preprocess_Taobao_seq_action.py \
  --data_dir "$RAW_DIR" \
  --output_dir "$OUTPUT_DIR" \
  --retain_transition_timestamps \
  --overwrite

"$PYTHON_BIN" data/Taobao/build_all_transition_pairs.py \
  --dataset_root "$OUTPUT_DIR" \
  --output_root "$PAIR_DIR" \
  --controls_per_positive 4 \
  --strata hour
