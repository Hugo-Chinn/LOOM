#!/usr/bin/env python3
"""Prioritize original four-seed replication before the new head studies."""
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
GPUS = "0,1,2,3,4,5,6,7"


def main():
    runner = ROOT / "scripts/run_rankmixer_mtl_aligned.py"
    phases = [
        ("rankmixer_taobao_original_four_seeds", [
            "--config", "config_rankmixer_mtl_aligned",
            "--study", "historical_replication", "--dataset", "Taobao_Action"]),
        ("rankmixer_all_datasets_original_four_seeds", [
            "--config", "config_rankmixer_mtl_aligned",
            "--study", "historical_replication"]),
        ("onetrans_all_datasets_original_four_seeds", [
            "--config", "config_onetrans_mtl_aligned",
            "--study", "historical_replication"]),
    ]
    state = ROOT / "logs/historical_four_seed_queue_state.json"
    for stage, args in phases:
        state.write_text(json.dumps({"stage": stage, "status": "running",
                                     "updated_at": time.time()}, indent=2) + "\n")
        print(f"START {stage}", flush=True)
        result = subprocess.run(
            [sys.executable, str(runner), *args, "--gpus", GPUS, "--wait-for-gpus"],
            cwd=ROOT)
        if result.returncode:
            state.write_text(json.dumps({"stage": stage, "status": "failed",
                                         "returncode": result.returncode}, indent=2) + "\n")
            raise SystemExit(result.returncode)
        print(f"COMPLETE {stage}", flush=True)
    state.write_text(json.dumps({"stage": "historical_replication", "status": "complete"},
                                indent=2) + "\n")
    # Original studies are now complete; head studies still require pilot gates.
    result = subprocess.run([
        sys.executable, "scripts/queue_rankmixer_mtl_aligned.py",
        "--pilot-gpus", GPUS, "--extra-config", "config_onetrans_mtl_aligned"],
        cwd=ROOT)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
