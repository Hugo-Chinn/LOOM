#!/usr/bin/env bash
set -euo pipefail
if [[ $# -ne 2 ]]; then
  echo "usage: $0 <dataset-id> <log-tag>" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ID="$1"
export HTTPS_PROXY="${HTTPS_PROXY:-http://star-proxy.oa.com:3128}"
export HTTP_PROXY="${HTTP_PROXY:-http://star-proxy.oa.com:3128}"
export ALL_PROXY="${ALL_PROXY:-http://star-proxy.oa.com:3128}"
export HF_HOME="${HF_HOME:-$ROOT/data/hf_cache_preprocessed}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
cd "$ROOT"
exec "$ROOT/.venv-unirank/bin/python" -u scripts/download_preprocessed_datasets.py \
  --output-root data/preprocessed --parallel-datasets 1 --workers-per-dataset 2 \
  --datasets "$DATASET_ID"
