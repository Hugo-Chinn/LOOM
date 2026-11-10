"""Validation-only decisions and gated test reporting for RelSup (PLAN.md protocol).

    # decide phase-1 survivors from validation metrics only
    python -m explorations.relational_supervision.collect_research --stage phase1 \
        --config configs/local/taobao-relsup-p1 --output runs_local/relsup-taobao/phase1
    # freeze weight/margin per survivor from the full pre-registered grid
    ... --stage phase2 --config configs/local/taobao-relsup-p1 configs/local/taobao-relsup-p2 \
        --output runs_local/relsup-taobao/phase2
    # only now read test blocks; needs the frozen phase-2 selection
    ... --stage test --config <every config dir> \
        --selection runs_local/relsup-taobao/phase2/selection.json \
        --output runs_local/relsup-taobao/test

Stages phase1/phase2 never parse a Test block. Every attempted arm and seed is
listed, including missing and incomplete runs; nothing is filled in.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from statistics import mean, stdev

from . import arms as A

TASK = re.compile(r"\[Task: ([^\]]+)\]\[Metrics\](.*)")
AUC = re.compile(r"(?<![A-Za-z])AUC: ([0-9.]+)")
LOGLOSS = re.compile(r"logloss: ([0-9.]+)")
RELSUP = re.compile(r"\[RelSup\] step=(\d+) (.*)")
PARAMETERS = re.compile(r"Total number of parameters: (\d+)")
TEST_MARKER = "******** Test evaluation ********"
EVAL_MARKER = "Evaluation @epoch"
SAVE_MARKER = "Save best model"


def _metrics(text):
    values = {}
    for task, rest in TASK.findall(text):
        auc, logloss = AUC.search(rest), LOGLOSS.search(rest)
        if auc and logloss:
            values[task] = {"auc": float(auc.group(1)), "logloss": float(logloss.group(1))}
    return values


def parse_log(path, tasks, read_test=False):
    """Metrics of the evaluation that produced the saved (best) checkpoint.

    The run counts as complete only if training reached the Test block. The
    Test block's numbers are parsed only when `read_test` is true.
    """
    if not path.exists():
        return {"status": "missing"}
    text = path.read_text(errors="ignore")
    if TEST_MARKER not in text:
        return {"status": "incomplete"}
    before, after = text.rsplit(TEST_MARKER, 1)
    saved = [segment for segment in before.split(EVAL_MARKER)[1:] if SAVE_MARKER in segment]
    validation = _metrics(saved[-1]) if saved else {}
    if set(validation) != set(tasks):
        return {"status": "incomplete"}
    result = {"status": "complete", "validation": validation}
    found = PARAMETERS.search(before)
    result["parameters"] = int(found.group(1)) if found else None
    lines = RELSUP.findall(before)
    if lines:
        step, body = lines[-1]
        result["last_relsup_log"] = {"step": int(step), **{
            k: float(v) for k, v in (item.split("=") for item in body.split())}}
    if read_test:
        test = _metrics(after)
        if set(test) != set(tasks):
            return {"status": "incomplete"}
        result["test"] = test
    return result


def load_configs(directories):
    constants = A.protocol_constants()
    arms, seeds, reference = {}, {}, None
    for directory in directories:
        record = json.loads((directory / "protocol.json").read_text())
        if record["protocol"] != constants:
            raise SystemExit(f"refused: {directory} was created under different "
                             "protocol constants than the current arms.py")
        key = (record["dataset_id"], record["model_root"], tuple(record["tasks"]))
        if reference not in (None, key):
            raise SystemExit("refused: config directories differ in dataset/model root/tasks")
        reference = key
        for arm, spec in record["arms"].items():
            if arm in arms and arms[arm] != spec:
                raise SystemExit(f"refused: arm {arm} defined inconsistently")
            arms[arm] = spec
            seeds.setdefault(arm, set()).update(record["seeds"])
    dataset_id, model_root, tasks = reference
    return arms, seeds, Path(model_root) / dataset_id, list(tasks)


def role(seed):
    if seed in A.DEV_SEEDS:
        return "development"
    return "confirmation" if seed in A.CONFIRM_SEEDS else "unregistered"


def summarize(runs, arms, tasks, split, seed_role):
    """Per arm/task statistics over one seed role, paired against the control."""
    rows = []
    for arm in arms:
        own = {seed: run[split] for (name, seed), run in runs.items()
               if name == arm and role(seed) == seed_role}
        if not own:
            continue
        base = {seed: runs[A.BASELINE, seed][split] for seed in own
                if (A.BASELINE, seed) in runs}
        common = sorted(base)
        for task in tasks:
            aucs = [own[s][task]["auc"] for s in sorted(own)]
            losses = [own[s][task]["logloss"] for s in sorted(own)]
            deltas = [own[s][task]["auc"] - base[s][task]["auc"] for s in common]
            relative = [own[s][task]["logloss"] / base[s][task]["logloss"] - 1 for s in common]
            rows.append({
                "arm": arm, "seed_role": seed_role, "split": split, "task": task,
                "is_target": task in arms[arm]["targets"],
                "n_seeds": len(own), "auc_mean": mean(aucs),
                "auc_sample_std": stdev(aucs) if len(aucs) > 1 else None,
                "logloss_mean": mean(losses),
                "logloss_sample_std": stdev(losses) if len(losses) > 1 else None,
                "common_seeds": ";".join(map(str, common)), "n_paired": len(common),
                "auc_paired_delta_mean": mean(deltas) if deltas else None,
                "auc_paired_delta_sample_std": stdev(deltas) if len(deltas) > 1 else None,
                "auc_positive_seeds": sum(d > 0 for d in deltas),
                "logloss_paired_rel_delta_mean": mean(relative) if relative else None,
            })
    return rows


def judge(arm, arms, rows, runs):
    """Pre-registered survival rule on development-seed validation metrics."""
    seeds = set(A.DEV_SEEDS)
    done = {s for (name, s) in runs if name == arm} & seeds
    control = {s for (name, s) in runs if name == A.BASELINE} & seeds
    verdict = {"arm": arm, "targets": arms[arm]["targets"], "passes": False}
    if done != seeds or control != seeds:
        verdict["reason"] = "arm or control incomplete on the development seeds"
        return verdict
    mine = [r for r in rows if r["arm"] == arm and r["seed_role"] == "development"]
    verdict["target_positive_seeds"] = {
        r["task"]: r["auc_positive_seeds"] for r in mine if r["is_target"]}
    verdict["logloss_rel_delta"] = {r["task"]: r["logloss_paired_rel_delta_mean"] for r in mine}
    targets_ok = bool(verdict["targets"]) and all(
        count >= A.MIN_POSITIVE_SEEDS for count in verdict["target_positive_seeds"].values())
    safe = all(value <= A.LOGLOSS_TOLERANCE_REL for value in verdict["logloss_rel_delta"].values())
    verdict["score"] = mean(r["auc_paired_delta_mean"] for r in mine if r["is_target"]) \
        if verdict["targets"] else None
    verdict["passes"] = targets_ok and safe
    verdict["reason"] = "passes" if verdict["passes"] else (
        "target tasks not positive in enough seeds" if not targets_ok
        else "validation logloss tolerance violated")
    return verdict


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=["phase1", "phase2", "test"])
    parser.add_argument("--config", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selection", type=Path, default=None)
    args = parser.parse_args()
    arms, seeds, logs, tasks = load_configs(args.config)
    if A.BASELINE not in arms:
        raise SystemExit("refused: include the phase-1 config directory (control arm)")
    read_test = args.stage == "test"
    if read_test:
        if args.selection is None:
            raise SystemExit("refused: test metrics need --selection (frozen phase-2 choice)")
        selection = json.loads(args.selection.read_text())
        if selection.get("stage") != "phase2" or selection.get("protocol") != A.protocol_constants():
            raise SystemExit("refused: selection is not a frozen phase2 selection "
                             "under the current protocol")
    if args.output.exists():
        raise SystemExit("refused: output exists; decisions and reports are never overwritten")
    runs, attempts = {}, {}
    for arm in arms:
        for seed in sorted(seeds[arm]):
            parsed = parse_log(logs / f"{arm}_S{seed}.log", tasks, read_test)
            environment = logs / f"{arm}_S{seed}.env.json"
            attempts[f"{arm}_S{seed}"] = {
                "status": parsed["status"], "seed_role": role(seed),
                "commit": json.loads(environment.read_text()).get("commit")
                if environment.exists() else None,
                "parameters": parsed.get("parameters"),
                "last_relsup_log": parsed.get("last_relsup_log")}
            if parsed["status"] == "complete":
                runs[arm, seed] = parsed
    roles = ("development", "confirmation")
    validation = [row for r in roles for row in summarize(runs, arms, tasks, "validation", r)]
    decisions = {arm: judge(arm, arms, validation, runs) for arm in arms if arm != A.BASELINE}
    args.output.mkdir(parents=True)
    write_csv(args.output / "validation_per_seed_task.csv", [
        {"arm": arm, "seed": seed, "seed_role": role(seed), "task": task,
         "validation_auc": run["validation"][task]["auc"],
         "validation_logloss": run["validation"][task]["logloss"]}
        for (arm, seed), run in sorted(runs.items()) for task in tasks])
    write_csv(args.output / "validation_summary.csv", validation)
    (args.output / "attempts.json").write_text(json.dumps(attempts, indent=2))
    (args.output / "decisions.json").write_text(json.dumps(decisions, indent=2))
    record = {"stage": args.stage, "protocol": A.protocol_constants(),
              "configs": [str(c) for c in args.config], "tasks": tasks,
              "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "split_used": "validation", "decisions": decisions}
    if args.stage == "phase1":
        record["survivors"] = [term for term in A.TERMS
                               if term in decisions and decisions[term]["passes"]]
        (args.output / "selection.json").write_text(json.dumps(record, indent=2))
        print(f"survivors: {record['survivors'] or 'none'}")
    elif args.stage == "phase2":
        frozen = {}
        for term in A.TERMS:
            if term not in decisions or not decisions[term]["passes"]:
                continue
            grid = {term: arms[term], **A.phase2([term])}
            absent = [name for name in grid if name not in arms or any(
                attempts.get(f"{name}_S{s}", {}).get("status") != "complete"
                for s in A.DEV_SEEDS)]
            if absent:
                raise SystemExit(f"refused: grid of {term} incomplete: {absent}; "
                                 f"partial outputs kept in {args.output}")
            eligible = [name for name in grid if decisions[name]["passes"]]
            best = max(eligible, key=lambda name: (
                decisions[name]["score"], -grid[name]["weight"][term],
                -(grid[name]["margin"][term] or 0.0)))
            frozen[term] = {"arm": best, "weight": grid[best]["weight"][term],
                            "margin": grid[best]["margin"][term],
                            "score": decisions[best]["score"],
                            "candidates": {n: decisions[n]["score"] for n in grid},
                            "eligible": eligible}
        record["frozen"] = frozen
        (args.output / "selection.json").write_text(json.dumps(record, indent=2))
        print("frozen: " + (json.dumps({t: f["arm"] for t, f in frozen.items()}) or "none"))
    else:
        test = [row for r in roles for row in summarize(runs, arms, tasks, "test", r)]
        labels = {"development": "exploratory: development seeds were reused after selection",
                  "confirmation": "confirmation: fresh pre-declared seeds, frozen configuration"}
        for row in test:
            row["result_label"] = labels[row["seed_role"]]
        write_csv(args.output / "test_per_seed_task.csv", [
            {"arm": arm, "seed": seed, "seed_role": role(seed), "task": task,
             "test_auc": run["test"][task]["auc"], "test_logloss": run["test"][task]["logloss"]}
            for (arm, seed), run in sorted(runs.items()) for task in tasks])
        write_csv(args.output / "test_summary.csv", test)
        record.update({"split_used": "test (reporting only)", "frozen": selection["frozen"],
                       "selection_file": str(args.selection)})
        (args.output / "unblinding.json").write_text(json.dumps(record, indent=2))
        print("test tables written for every attempted arm; no selection may follow")
    commits = {a["commit"] for a in attempts.values() if a["status"] == "complete"}
    if len(commits) > 1:
        print(f"WARNING: completed runs come from {len(commits)} different commits/"
              "unrecorded environments; see attempts.json before comparing arms")
    incomplete = [name for name, a in attempts.items() if a["status"] != "complete"]
    print(f"{len(runs)} complete, {len(incomplete)} missing/incomplete -> {args.output}")


if __name__ == "__main__":
    main()
