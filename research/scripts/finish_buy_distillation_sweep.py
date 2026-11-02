"""Select KD arm by validation only, then evaluate it and its matched label control."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from tune_buy_distillation import ARMS, OUT, ROOT


def main():
    deadline = time.monotonic() + 6*3600
    while not all((OUT / (a["name"]+".json")).exists() for a in ARMS):
        errors = [p.name for p in OUT.glob("arm*.log") if "Traceback" in p.read_text(errors="ignore")]
        if errors:
            raise RuntimeError(f"Sweep failed: {errors}")
        if time.monotonic() > deadline:
            raise RuntimeError("Sweep timed out")
        time.sleep(30)
    rows = [json.loads((OUT / (a["name"]+".json")).read_text()) for a in ARMS]
    candidates = [(i,r) for i,r in enumerate(rows) if r["arm"]["alpha"]>0]
    selected, _ = max(candidates, key=lambda x:max(v["student_auc"] for v in x[1]["validation"]))
    control = 7 if ARMS[selected].get("freeze_front") else 0
    (OUT / "selection.json").write_text(json.dumps({
        "selected_arm": ARMS[selected], "control_arm": ARMS[control],
        "rule": "highest validation AUC; no test-based arm selection",
    }, indent=2)+"\n")
    for gpu, index in enumerate((selected, control)):
        with (OUT / f"test_{index}.log").open("w") as output:
            subprocess.run([sys.executable, str(ROOT/"scripts/tune_buy_distillation.py"),
                            "--arm", str(index), "--evaluate-selected"],
                           env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu),
                                "OMP_NUM_THREADS":"2", "MKL_NUM_THREADS":"2"},
                           stdout=output, stderr=subprocess.STDOUT, check=True)
    lines = [
        "# One-Step Buy Distillation Tuning", "",
        "Taobao, one seed 20262028, warm-start from the retained historical Branch teacher.",
        "Eight prespecified arms; fixed 128/512 train batches; 128 validation batches.",
        "Select checkpoint and KD arm by validation Buy AUC only.",
        "Each student runs exactly one Mixer and one Buy tower.",
        "Frozen-front means embedding, tokenizer and history attention are not updated.",
        "Test prefix expanded to 256 batches; some test data was already inspected in the first probe.",
        "Not a new independent held-out confirmation, not a multi-seed significance claim.",
        "", "| Arm | Train batches | Validation Buy AUC | Validation Logloss | Teacher AUC |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        for v in row["validation"]:
            lines.append(
                f"| {row['arm']['name']} | {v['train_batches']} | {v['student_auc']:.6f} | "
                f"{v['student_logloss']:.6f} | {v['teacher_auc']:.6f} |")
    lines += ["", "## Selected Test Comparison", "",
              f"Selected KD: {ARMS[selected]['name']}; matched control: {ARMS[control]['name']}.",
              "", "| Arm | Buy AUC | Buy Logloss | Teacher Buy AUC | Test N | Buy positives |",
              "|---|---:|---:|---:|---:|---:|"]
    for index in (selected, control):
        v = json.loads((OUT/(ARMS[index]["name"]+"_heldout_test.json")).read_text())["metrics"]
        lines.append(f"| {ARMS[index]['name']} | {v['student_auc']:.6f} | "
                     f"{v['student_logloss']:.6f} | {v['teacher_auc']:.6f} | {v['n']} | {v['positives']} |")
    (OUT/"SUMMARY.md").write_text("\n".join(lines)+"\n")
    print("DONE validation selection and test evaluation", flush=True)


if __name__ == "__main__":
    main()
