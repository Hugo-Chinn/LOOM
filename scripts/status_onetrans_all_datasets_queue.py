#!/usr/bin/env python3
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs_local/onetrans_all_datasets_multiseed"
state = json.loads((RUN_DIR / "queue.json").read_text())
print(
    f"pending={len(state['pending'])} running={len(state['running'])} "
    f"completed={len(state['completed'])} failed={len(state['failed'])}")
for task, gpu in sorted(state["running"].items(), key=lambda row: int(row[1])):
    log_path = RUN_DIR / f"{task}.stdout.log"
    progress = ""
    if log_path.exists():
        text = log_path.read_text(errors="replace").replace("\r", "\n")
        matches = re.findall(r"Rank 0 \| Epoch 1:.*", text)
        if matches:
            progress = matches[-1][-180:]
        elif "Test evaluation" in text:
            progress = "test evaluation"
    print(f"gpu={gpu} {task} {progress}")
if state["failed"]:
    print("failed:")
    for task in state["failed"]:
        print(f"  {task}")
