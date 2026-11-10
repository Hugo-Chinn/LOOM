"""Record the software/hardware environment of a run (protocol requirement)."""
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def _run(*command):
    try:
        return subprocess.run(command, capture_output=True, text=True, timeout=20,
                              cwd=ROOT, check=True).stdout.strip()
    except Exception as error:  # recorded, never fatal
        return f"unavailable: {type(error).__name__}"


def snapshot(extra=None):
    info = {
        "commit": _run("git", "rev-parse", "HEAD"),
        "dirty_files": _run("git", "status", "--porcelain", "--untracked-files=no"),
        "python": sys.version, "platform": platform.platform(),
        "argv": sys.argv, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "nvidia_smi": _run("nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total",
                           "--format=csv,noheader"),
    }
    try:
        import torch
        info.update({
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        })
    except Exception as error:
        info["torch"] = f"unavailable: {type(error).__name__}"
    info.update(extra or {})
    return info


def write(path, extra=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(snapshot(extra), indent=2, default=str))
    return path
