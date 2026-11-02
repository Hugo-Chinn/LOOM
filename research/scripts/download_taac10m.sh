#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export HF_HOME="${HF_HOME:-$ROOT/data/TAAC2025/hf_cache}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-300}"
export HTTPS_PROXY="${HTTPS_PROXY:-http://star-proxy.oa.com:3128}"
export HTTP_PROXY="${HTTP_PROXY:-http://star-proxy.oa.com:3128}"
export ALL_PROXY="${ALL_PROXY:-http://star-proxy.oa.com:3128}"

cd "$ROOT"
"$ROOT/.venv-unirank/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="TAAC2025/TencentGR-10M",
    repo_type="dataset",
    revision="3397e805398ff7ece012a4f8789da62836da70d6",
    allow_patterns=["seq/*.parquet", "user_feat/*.parquet", "item_feat/*.parquet"],
    local_dir="data/TAAC2025/TencentGR-10M-raw",
    max_workers=8,
)
PY
"$ROOT/.venv-unirank/bin/python" "$ROOT/scripts/merge_taac_parquet.py" \
  --download-root "$ROOT/data/TAAC2025/TencentGR-10M-raw" \
  --output-root "$ROOT/data/TAAC2025/TencentGR-10M-merged"
