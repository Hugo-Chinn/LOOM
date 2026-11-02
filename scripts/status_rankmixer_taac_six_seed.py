#!/usr/bin/env python3
"""Report progress and current test metrics for the TAAC six-seed confirmation."""
from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs_local/rankmixer_taac_six_seed"
SEEDS = (20262031, 20262032, 20262033)
ARMS = ("Sequential", "Loop_Deep", "Loop_GatedDeep_Pre2")
METRIC = re.compile(
    r"\[Task: ([^]]+)\]\[Metrics\].*?"
    r"(?:AUC: ([0-9.]+) - logloss: ([0-9.]+)|logloss: ([0-9.]+) - AUC: ([0-9.]+))"
)


def main():
    done = 0
    for arm in ARMS:
        for seed in SEEDS:
            expid = f"RankMixer_TAAC_{arm}_S{seed}"
            log = RUN_DIR / f"{expid}.stdout.log"
            if not log.exists():
                print(f"{expid:38s} PENDING")
                continue
            text = log.read_text(errors="ignore")
            # The "Test evaluation" header is printed before inference starts, so a
            # run only counts as done once its per-task metrics are on disk.
            section = text.rsplit("Test evaluation", 1)[1] if "Test evaluation" in text else ""
            aucs = [float(m.group(5) or m.group(2)) for m in METRIC.finditer(section)]
            if aucs:
                macro = sum(aucs) / len(aucs)
                done += 1
                print(f"{expid:38s} DONE    macro={macro:.6f} tasks={[round(a, 6) for a in aucs]}")
                continue
            else:
                tail = [l for l in text.splitlines() if "Epoch" in l or "it/s" in l]
                print(f"{expid:38s} RUNNING {tail[-1][-90:] if tail else ''}")
    print(f"completed {done}/{len(ARMS) * len(SEEDS)}")


if __name__ == "__main__":
    main()
