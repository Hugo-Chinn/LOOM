"""Per-seed/task test AUC and paired deltas vs `full` from historical logs.

    python -m explorations.relational_supervision.collect_research \
        --logs /absolute/outputs/taobao-relsup/Taobao_Action \
        --output runs_local/relsup-taobao-summary

Reads only the final Test evaluation block of each <arm>_S<seed>.log. Missing
or incomplete runs are listed, never filled in.
"""
import argparse
import csv
import json
from pathlib import Path
import re
from statistics import mean, stdev

LOG = re.compile(r"^(?P<arm>.+)_S(?P<seed>\d+)\.log$")
TASK = re.compile(r"\[Task: ([^\]]+)\]\[Metrics\](.*)")
AUC = re.compile(r"(?<![A-Za-z])AUC: ([0-9.]+)")
RELSUP = re.compile(r"\[RelSup\] step=(\d+) (.*)")
MARKER = "******** Test evaluation ********"


def parse(path):
    text = path.read_text(errors="ignore")
    if MARKER not in text:
        return None, None
    values = {}
    for task, rest in TASK.findall(text.rsplit(MARKER, 1)[1]):
        found = AUC.search(rest)
        if found:
            values[task] = float(found.group(1))
    diagnostics = None
    lines = RELSUP.findall(text)
    if lines:
        step, body = lines[-1]
        diagnostics = {"step": int(step), **{
            k: float(v) for k, v in (item.split("=") for item in body.split())}}
    return values or None, diagnostics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", default="full")
    args = parser.parse_args()
    runs, diagnostics, missing = {}, {}, []
    for path in sorted(args.logs.glob("*.log")):
        match = LOG.match(path.name)
        if not match:
            continue
        values, diag = parse(path)
        if values is None:
            missing.append(path.name)
            continue
        runs[match["arm"], int(match["seed"])] = values
        diagnostics[f"{match['arm']}_S{match['seed']}"] = diag
    if not runs:
        raise SystemExit("no completed <arm>_S<seed>.log with a Test block found")
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "per_seed_task.csv").open("w") as handle:
        writer = csv.writer(handle)
        writer.writerow(["arm", "seed", "task", "test_auc"])
        for (arm, seed), values in sorted(runs.items()):
            for task, auc in values.items():
                writer.writerow([arm, seed, task, auc])
    rows = []
    for arm in sorted({a for a, _ in runs}):
        tasks = sorted({t for (a, _), v in runs.items() if a == arm for t in v})
        for task in tasks:
            own = {s: v[task] for (a, s), v in runs.items() if a == arm and task in v}
            common = sorted(s for s in own if task in runs.get((args.baseline, s), {}))
            deltas = [own[s] - runs[args.baseline, s][task] for s in common]
            rows.append({
                "arm": arm, "task": task, "n_seeds": len(own),
                "auc_mean": mean(own.values()),
                "auc_sample_std": stdev(own.values()) if len(own) > 1 else None,
                "n_paired": len(common),
                "paired_delta_mean": mean(deltas) if deltas else None,
                "paired_delta_sample_std": stdev(deltas) if len(deltas) > 1 else None,
                "paired_wins": sum(d > 0 for d in deltas),
            })
    with (args.output / "per_task_summary.csv").open("w") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "diagnostics_and_missing.json").write_text(json.dumps(
        {"last_relsup_log": diagnostics, "missing_or_incomplete": missing}, indent=2))
    print(f"{len(runs)} completed runs, {len(missing)} missing/incomplete -> {args.output}")


if __name__ == "__main__":
    main()
