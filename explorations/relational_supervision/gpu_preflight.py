"""Bounded CUDA forward/backward check on one explicitly visible GPU.

Run in a GPU-enabled execution context, for example:
  CUDA_VISIBLE_DEVICES=0 /path/to/venv/bin/python -m \
      explorations.relational_supervision.gpu_preflight
No training data, checkpoints or global environment settings are changed.
"""
import json
import os
from pathlib import Path
import sys


def main():
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible or visible == "-1":
        raise SystemExit("Set CUDA_VISIBLE_DEVICES to exactly one GPU index or UUID.")
    report = {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "cuda_visible_devices": visible,
        "device_nodes_visible": {
            path: Path(path).exists()
            for path in ("/dev/nvidiactl", "/dev/nvidia-uvm")
        },
    }
    try:
        import torch

        report.update(torch=torch.__version__, cuda_runtime=torch.version.cuda)
        torch.cuda.init()
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expected exactly one visible CUDA device")
        report.update(gpu=torch.cuda.get_device_name(0),
                      capability=list(torch.cuda.get_device_capability(0)))
        # An analytic gradient check exercises allocation, cuBLAS and autograd.
        # Only 128 x 128 tensors are allocated; no training or performance claim.
        checks = {}
        dtypes = [torch.float32]
        if torch.cuda.is_bf16_supported():
            dtypes.append(torch.bfloat16)
        for dtype in dtypes:
            x = torch.ones((128, 128), device="cuda", dtype=dtype,
                           requires_grad=True)
            identity = torch.eye(128, device="cuda", dtype=dtype)
            output = x @ identity
            loss = output.float().square().mean()
            loss.backward()
            torch.cuda.synchronize()
            torch.testing.assert_close(output, x)
            torch.testing.assert_close(x.grad, torch.full_like(x, 2 / x.numel()))
            checks[str(dtype)] = "forward/backward OK"
        report.update(checks=checks,
                      peak_tensor_memory_bytes=torch.cuda.max_memory_allocated(),
                      status="ok")
    except Exception as exc:
        report.update(status="error", error=f"{type(exc).__name__}: {exc}")
        print(json.dumps(report, indent=2), flush=True)
        print(
            "CUDA preflight failed. If NVIDIA device nodes are hidden by the "
            "execution sandbox, request an approved GPU-enabled execution "
            "context and repeat this check. Verify nvidia-smi there before "
            "diagnosing a driver failure.", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
