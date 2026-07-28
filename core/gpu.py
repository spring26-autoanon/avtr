"""
GPU device assignment for the front-end model and server_conditioner processes.

See specs/moshirag-evals-requirements.md's "GPU Sizing and Multi-GPU
Deployment" section: MIG was ruled out for isolating conditioner GPU
contention from the front-end (front-end's measured peak memory, 47.8GB,
exceeds the largest sub-full MIG partition available on an 80GB A100), so
isolation requires either a second physical GPU or accepting known-contended
single-GPU numbers. This module resolves which physical GPU each process
role should use, auto-detected from real hardware -- no VM-specific
branching anywhere, the same code and the same rsynced files run unchanged
on the current single-A100 box and a future dual-A100 one.
"""

import os
import subprocess
from dataclasses import dataclass

# Front-end's own main model always claims physical GPU 0; the conditioner
# claims GPU 1 only if a second one is actually visible. Fixed and
# role-based, not negotiated between processes -- the eval path's two
# processes are launched independently, from two separate terminals (see
# CLAUDE.md's "Required setup: server_conditioner process"), with no shared
# parent to coordinate a split at launch time. Each has to independently
# arrive at the same answer from the same rule.
_FRONTEND_INDEX = "0"


@dataclass(frozen=True)
class DeviceAssignment:
    frontend_cuda_visible_devices: str
    conditioner_cuda_visible_devices: str
    contended: bool  # True iff conditioner and front-end share one physical GPU
    # frontend_cuda_visible_devices is "0,1" (not just "0") when a second
    # physical GPU exists -- see resolve_devices()'s own docstring for why:
    # it's what lets the front-end PROCESS pin its own in-process STT model
    # duplicate to physical GPU 1 internally, while the main model stays on
    # physical GPU 0 (always addressed as "cuda:0" from inside that
    # process, regardless of how many GPUs are visible to it).


def device_count() -> int:
    """
    Real, physical CUDA device count on this machine, via `nvidia-smi -L`.

    Deliberately NOT `torch.cuda.device_count()`: the CUDA runtime binds a
    process's device visibility (per CUDA_VISIBLE_DEVICES) the moment any
    CUDA call is made, including a plain device-count query -- so using
    torch.cuda here would permanently lock in "all GPUs visible" for
    whatever process calls this, making it too late to then self-restrict
    that same process via CUDA_VISIBLE_DEVICES (see
    ensure_cuda_visible_devices() below). `nvidia-smi` is a separate
    process and a system-management tool, not a CUDA-runtime client -- it
    always reports every physical GPU regardless of the calling shell's own
    CUDA_VISIBLE_DEVICES, so querying it never touches this process's own
    (not-yet-initialized) CUDA context.

    Returns 0 if `nvidia-smi` isn't available (no GPU, or local dev without
    the `gpu` extra) rather than raising -- same graceful-degradation
    convention this codebase already uses elsewhere for GPU-less
    environments (see evals/runner.py's build_model()).
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "-L"],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return 0
    if result.returncode != 0:
        return 0
    return sum(1 for line in result.stdout.splitlines() if line.strip().startswith("GPU "))


def resolve_devices(count: int | None = None) -> DeviceAssignment:
    """
    Pure function of device count -> role assignment. `count` defaults to
    a real device_count() call; passing it explicitly (as tests do) keeps
    this fully unit-testable with no GPU, subprocess, or mocking involved.

    frontend_cuda_visible_devices includes physical GPU 1 too (`"0,1"`,
    not just `"0"`) whenever a second GPU exists — this is deliberate,
    not scope creep: the front-end process's own in-process STT model
    duplicate is pinned to a second physical GPU when one is visible
    (core/model_interface.py's _patch_stt_second_gpu()), to give it a
    genuinely separate CUDA context/stream from the front-end's own main
    model, eliminating a real cross-thread CUDA-graph-capture crash that
    locking around specific call sites couldn't fully close (see that
    function's own docstring for the full history). That only works if
    this process can actually see the second physical GPU at all — once
    CUDA_VISIBLE_DEVICES restricts a process to one device before its CUDA
    runtime initializes, no other physical GPU is ever reachable from
    inside it again, regardless of what index code asks for. The main
    model itself is unaffected: it's still always addressed as "cuda:0"
    (see MoshiRAGAdapter._load_models()'s own args.device), the first of
    whatever's now visible.
    """
    n = device_count() if count is None else count
    conditioner_index = "1" if n >= 2 else _FRONTEND_INDEX
    frontend_visible = "0,1" if n >= 2 else _FRONTEND_INDEX
    return DeviceAssignment(
        frontend_cuda_visible_devices=frontend_visible,
        conditioner_cuda_visible_devices=conditioner_index,
        contended=(conditioner_index == _FRONTEND_INDEX),
    )


def _value_for_role(assignment: DeviceAssignment, role: str) -> str:
    if role == "frontend":
        return assignment.frontend_cuda_visible_devices
    if role == "conditioner":
        return assignment.conditioner_cuda_visible_devices
    raise ValueError(f"role must be 'frontend' or 'conditioner', got {role!r}")


def ensure_cuda_visible_devices(role: str, count: int | None = None) -> DeviceAssignment:
    """
    Sets this process's own CUDA_VISIBLE_DEVICES for `role`
    ("frontend"/"conditioner"), unless the operator already set it -- an
    explicit CUDA_VISIBLE_DEVICES in the environment always wins over
    auto-detection. This preserves scripts/gpu_diag_solo.sh/
    gpu_diag_contended.sh's existing pattern of deliberately forcing both
    processes onto the same GPU for a controlled comparison, and any other
    manual override generally.

    Must be called before any CUDA runtime call happens in this process
    (e.g. before `import torch`) -- CUDA_VISIBLE_DEVICES has no effect once
    the CUDA driver has already initialized for a process. Safe to call
    from plain Python before `import torch`, since detection itself goes
    through `nvidia-smi`, a separate process, never the CUDA runtime -- see
    device_count().

    The returned DeviceAssignment's `contended` field always reflects what
    auto-detection concluded from real device_count(), even when an
    existing override left the actual env var untouched -- an operator
    override is a deliberate, informed action (e.g. the diagnostic scripts
    above) and doesn't need the warning gate this field feeds; that gate
    exists to catch the *unattended*, auto-detected single-GPU case.
    """
    if role not in ("frontend", "conditioner"):
        raise ValueError(f"role must be 'frontend' or 'conditioner', got {role!r}")
    assignment = resolve_devices(count)
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = _value_for_role(assignment, role)
    return assignment


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Print the resolved CUDA_VISIBLE_DEVICES value for a process role. "
            "Used by scripts/run_demo.sh to pin the conditioner and the main "
            "server to separate physical GPUs when two are present. Does not "
            "set anything in this process's own environment -- it just prints "
            "the value a caller (e.g. a shell script) should export for the "
            "child process it's about to launch."
        )
    )
    parser.add_argument("role", choices=["frontend", "conditioner"])
    args = parser.parse_args()

    existing = os.environ.get("CUDA_VISIBLE_DEVICES")
    print(existing if existing is not None else _value_for_role(resolve_devices(), args.role))
