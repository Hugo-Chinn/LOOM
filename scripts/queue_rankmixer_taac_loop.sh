#!/usr/bin/env bash
# Wait for the current hierarchy phase-2 grid, then use GPUs 2-7 for TAAC.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
LOG_DIR="$ROOT/runs_local/rankmixer_taac_loop"
mkdir -p "$LOG_DIR"
while pgrep -f 'run_expid.py.*config_rankmixer_hierarchical_phase2' >/dev/null; do
  sleep 30
done
exec "$ROOT/scripts/run_rankmixer_taac_loop_multigpu.sh"
