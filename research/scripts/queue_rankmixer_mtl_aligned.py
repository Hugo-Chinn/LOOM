#!/usr/bin/env python3
"""Gate the eight-GPU studies on a six-model real-data pilot."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PILOT = "ParallelIndependentTower,MMoE,PLE,STEM,LoopMT,UntiedDeep"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "config_rankmixer_mtl_aligned")
    parser.add_argument("--pilot-gpus", default="6,7")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--extra-config", type=Path, action="append", default=[])
    args = parser.parse_args()
    config = args.config.resolve()
    runner = ROOT / "scripts/run_rankmixer_mtl_aligned.py"
    command = [sys.executable, str(runner), "--config", str(config)]
    statefile = config / "queue_state.json"

    def state(stage, status, **extra):
        statefile.write_text(json.dumps({
            "stage": stage, "status": status, "updated_at": time.time(),
            **extra,
        }, indent=2) + "\n")

    configs = [config] + [p.resolve() for p in args.extra_config]
    phases = [(f"pilot_{p.name}", [sys.executable, str(runner), "--config", str(p),
        "--study", "fixed_backbone_head", "--dataset", "Taobao_Action",
        "--seed", "20262027", "--variants", PILOT, "--gpus", args.pilot_gpus,
        "--wait-for-gpus"]) for p in configs]
    for stage, cmd in phases:
        state(stage, "running")
        print(f"START {stage}", flush=True)
        result = subprocess.run(cmd, cwd=ROOT)
        if result.returncode:
            state(stage, "failed", returncode=result.returncode)
            raise SystemExit(result.returncode)
        state(stage, "complete")
        print(f"COMPLETE {stage}", flush=True)
    if not args.pilot_only:
        state("all_studies", "running", configs=[str(p) for p in configs])
        processes = [subprocess.Popen([
            sys.executable, str(runner), "--config", str(p),
            "--study", "all", "--gpus", args.gpus, "--wait-for-gpus"], cwd=ROOT)
            for p in configs]
        codes = [p.wait() for p in processes]
        if any(codes):
            state("all_studies", "failed", returncodes=codes)
            raise SystemExit(1)
    state("all", "complete")


if __name__ == "__main__":
    main()
