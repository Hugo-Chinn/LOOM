#!/usr/bin/env python3
"""Show queue counts and recent progress for OneTrans NS/S experiments."""
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN_DIR = ROOT / "runs_local/onetrans_ns_s_state_multiseed"
state = json.loads((RUN_DIR / "queue.json").read_text())
print(
    f"pending={len(state['pending'])} running={len(state['running'])} "
    f"completed={len(state['completed'])} failed={len(state['failed'])}"
)
for task, gpu in sorted(state["running"].items(), key=lambda item: int(item[1])):
    path = RUN_DIR / f"{task}.stdout.log"
    text = path.read_text(errors="ignore") if path.exists() else ""
    lines = [line for line in re.split(r"[\r\n]+", text) if line.strip()]
    recent = lines[-1][-220:] if lines else ""
    print(f"gpu={gpu} {task} {recent}")
if state["failed"]:
    print("failed_tasks:")
    for task in state["failed"]:
        print(task)
