#!/usr/bin/env python3
import importlib.metadata as metadata
import json
import platform
import subprocess

import torch

packages = [
    "torch", "numpy", "pandas", "pyarrow", "polars", "scikit-learn",
    "heavyball", "datasets", "huggingface-hub", "h5py",
]
result = {
    "python": platform.python_version(),
    "packages": {name: metadata.version(name) for name in packages},
    "torch_runtime": torch.__version__,
    "cuda_build": torch.version.cuda,
    "cuda_available": torch.cuda.is_available(),
    "cuda_devices": torch.cuda.device_count(),
    "git_commit": subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip(),
}
if torch.cuda.is_available():
    x = torch.randn(64, 64, device="cuda", requires_grad=True)
    x.square().mean().backward()
    result["cuda_backward_finite"] = bool(torch.isfinite(x.grad).all())
print(json.dumps(result, indent=2))
