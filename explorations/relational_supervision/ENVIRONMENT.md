# Server GPU verification (2026-11-10)

The earlier CUDA/NVML failures were reproduced as an execution-context issue.
The restricted execution sandbox did not expose `/dev/nvidiactl`,
`/dev/nvidia0` or `/dev/nvidia-uvm`. The same diagnostic, interpreter and
libraries succeeded outside that sandbox with approved device access:

| Check | Restricted context | Approved GPU context |
|---|---|---|
| NVIDIA device nodes | Missing (`ENOENT`) | Open successfully |
| CUDA driver `cuInit(0)` | 304 | 0 (success) |
| PyTorch CUDA available | False | True |
| PyTorch device count (unmasked) | 0 | 8 |

This corrects the earlier claim that the server driver was broken. No driver
installation, device reset, service restart or library change was needed.
GPU commands need an execution context that exposes the NVIDIA devices.
Under a restricted sandbox, use the supported approval mechanism when needed.
In an unrestricted session, run directly; do not pass unsupported escalation
options. Follow the current session policy rather than assuming the policy
from this audit still applies. The preflight does not grant permission or
change the sandbox; it detects failure before a queue starts training.

## Verified environments

| Python executable | Python | PyTorch | CUDA build | Compute checks |
|---|---|---|---|---|
| `/home/HugoChinn/LoopMT/UniRank/.venv-unirank/bin/python` | 3.12.16 | 2.6.0+cu124 | 12.4 | GPUs 0, 1, 2, 3 passed |
| `/home/HugoChinn/LoopMT/.venv/bin/python` | 3.10.19 | 2.5.1+cu121 | 12.1 | GPU 0 passed |

All five invocations tested 128 x 128 matrix multiplication and backward in
FP32 and BF16, checked outputs and gradients against analytic expectations,
and synchronized the device. Each reported 67,764,736 bytes of peak PyTorch
allocated memory (including library workspace; excluding driver context).
These are execution checks, not model-training or performance measurements.

After integrating protocol commits `b182b9e` and `f9a7b87`, the current
`tests` and `explorations/relational_supervision/tests` suites passed on CPU:
**51 passed** (13.57 seconds, Python 3.12.16). One CUDA initialization warning
occurred inside the restricted context, consistent with the diagnosis above.
The queue's shell syntax and `git diff --check` also passed. A queue invoked
inside the restricted context exited at preflight with status 1 before
reading config or launching training.

The driver reports version `575.57.08`, with a maximum supported CUDA version
of 12.9. That is distinct from the two PyTorch CUDA build versions above.
All eight NVIDIA H20 cards report 97,871 MiB total memory and zero volatile or
aggregate ECC errors; no pending or failed row remapping was reported.

## Resource snapshot

GPUs 0–3 reported 0 MiB and 0% utilization before the compute checks. GPUs
4–7 reported about 429 MiB each and sustained utilization. At the final
check, **all eight GPUs were held by `rsi_holder.py`**, each at about 429 MiB
and 58–66% utilization. The holder's read-only `status` command reported
`holding` on all eight cards; its `gpu_pids` matched every NVML compute PID.
It is configured to target 65% utilization. All diagnostic processes exited.

NVML process names were `[Not Found]`. The holder implementation documents
the host/container PID mismatch and tracks NVML PIDs separately; the PID
matches in its status, not the missing process name, establish ownership.
The holder was left running. Its implementation yields on detecting foreign
compute PIDs and exposes per-UUID `pause` / `resume` commands. Before formal
training, arrange a per-card pause and verify that the holder PID has left
that card; restore its prior state after the job. Do not rely on an earlier
idle sample or stop the whole daemon.

This snapshot does not reserve GPUs. Recheck utilization and process UUIDs
immediately before launch. Prefer a GPU UUID when stable identity matters;
the queue and preflight accept one index or UUID per process.

Read-only holder status (in an approved execution context):

```bash
/home/HugoChinn/LoopMT/UniRank/.venv-unirank/bin/python \
  /jizhi/jizhi2/worker/trainer/rsi_holder/rsi_holder.py status
```

## Launch procedure

Use `/home/LOOM-GOON` for this study's code, with an explicitly selected
interpreter. The UniRank worktree and the older LOOM source copy are separate
repositories; neither was modified by this diagnosis.

In an approved GPU-enabled shell, from the current repository root:

```bash
nvidia-smi --query-gpu=index,uuid,memory.used,utilization.gpu --format=csv
CUDA_VISIBLE_DEVICES=0 /home/HugoChinn/LoopMT/UniRank/.venv-unirank/bin/python \
  -m explorations.relational_supervision.gpu_preflight
```

Once the experiment protocol, configs and budget are finalized:

```bash
PYTHON_BIN=/home/HugoChinn/LoopMT/UniRank/.venv-unirank/bin/python \
  bash explorations/relational_supervision/run_queue.sh \
  configs/local/taobao-relsup-p1 0 '^full_S20262027$'
```

The queue now uses `PYTHON_BIN` for both configuration parsing and training,
and runs the CUDA preflight first. It exits before training if CUDA is
unavailable. It does not automatically reserve GPUs or terminate other jobs.
No real-data training or protocol selection was performed in this GPU audit.
