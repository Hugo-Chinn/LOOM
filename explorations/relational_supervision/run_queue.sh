#!/usr/bin/env bash
# Sequential, bounded queue over a RelSup config directory. Run from repo root.
#   bash explorations/relational_supervision/run_queue.sh CONFIG_DIR GPU [PATTERN]
# Skips experiments whose log already has a Test block. PATTERN is a grep -E
# filter on experiment ids (for example '^(full|sib_margin)_').
# Set PYTHON_BIN to the intended environment's Python executable.
# GPU runs need a GPU-enabled execution context; preflight fails before launch
# when an agent sandbox hides NVIDIA devices.
set -euo pipefail
CONFIG=${1:?config directory}
GPU=${2:?gpu id}
PATTERN=${3:-.}
PYTHON_BIN=${PYTHON_BIN:-python}
CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m explorations.relational_supervision.gpu_preflight
ROOT=$("$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys, yaml, pathlib
c = pathlib.Path(sys.argv[1])
base = yaml.safe_load((c / "model_config.yaml").read_text())["Base"]
data = next(iter(yaml.safe_load((c / "dataset_config.yaml").read_text())))
print(pathlib.Path(base["model_root"]) / data)
PY
)
{ grep -E "$PATTERN" "$CONFIG/expids.txt" || true; } | while read -r EXPID; do
  if grep -q "Test evaluation" "$ROOT/$EXPID.log" 2>/dev/null; then
    echo "skip finished $EXPID"; continue
  fi
  echo "=== $EXPID on GPU $GPU ==="
  CUDA_VISIBLE_DEVICES="$GPU" "$PYTHON_BIN" -m explorations.relational_supervision.run_research \
    --config "$CONFIG" --expid "$EXPID" --gpu 0 < /dev/null
done
