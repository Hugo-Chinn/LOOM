#!/usr/bin/env bash
# Usage: queue_rankmixer_multiseed_confirmation.sh [PID_TO_WAIT_FOR]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WAIT_PID="${1:-}"
if [[ -n "$WAIT_PID" ]]; then
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
fi
exec "$ROOT/scripts/run_rankmixer_multiseed_confirmation.sh"
