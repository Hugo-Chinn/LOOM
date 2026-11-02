#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec "$ROOT/.venv-unirank/bin/python" -u \
  data/TAAC2025/preprocess_TAAC2025_seq_action.py \
  --data_dir data/TAAC2025/TencentGR-10M-merged \
  --output_dir data/TAAC2025/TencentGR_10M_Action \
  --n_user_parts 32 --seq_batch_rows 1000000 --buffer_flush_size 500000 \
  --train_blocks 32 --valid_blocks 8 --test_blocks 8 "$@"
