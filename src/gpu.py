"""CUDA/MPS memory diagnostics and explicit GPU resource cleanup.

`torch.cuda.empty_cache()` only returns *unused* cached blocks to the driver.
It cannot free tensors that are still referenced by Python objects (a loaded
model, generate() outputs, KV caches, etc.). Those references must be dropped
and garbage-collected first; empty_cache is the last step, not the fix.
"""

from __future__ import annotations

import gc
from typing import Any


def configure_cuda_allocator() -> None:
    """Secondary fragmentation mitigation. Call before importing torch."""
    import os

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def log_gpu_memory(label: str) -> dict[str, Any]:
    """Print and return a memory snapshot. Safe when CUDA/MPS/torch are absent."""
    payload: dict[str, Any] = {"label": label}
    try:
        import torch
    except ImportError:
        print(f"[GPU] {label}: torch not installed")
        payload["torch"] = False
        return payload

    payload["torch"] = True
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(idx)
        allocated = torch.cuda.memory_allocated(idx)
        reserved = torch.cuda.memory_reserved(idx)
        free_b, total_b = torch.cuda.mem_get_info(idx)
        payload.update(
            {
                "backend": "cuda",
                "gpu": props.name,
                "total_gib": total_b / 1024**3,
                "allocated_gib": allocated / 1024**3,
                "reserved_gib": reserved / 1024**3,
                "free_gib": free_b / 1024**3,
            }
        )
        print(
            f"[GPU] {label}: name={props.name} "
            f"total={payload['total_gib']:.2f} GiB "
            f"allocated={payload['allocated_gib']:.2f} GiB "
            f"reserved={payload['reserved_gib']:.2f} GiB "
            f"free={payload['free_gib']:.2f} GiB"
        )
        return payload

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        allocated = 0
        driver = 0
        if hasattr(torch, "mps"):
            allocated = int(getattr(torch.mps, "current_allocated_memory", lambda: 0)())
            driver = int(getattr(torch.mps, "driver_allocated_memory", lambda: 0)())
        payload.update(
            {
                "backend": "mps",
                "gpu": "mps",
                "allocated_gib": allocated / 1024**3,
                "reserved_gib": driver / 1024**3,
            }
        )
        print(
            f"[GPU] {label}: name=mps "
            f"allocated={payload['allocated_gib']:.2f} GiB "
            f"driver={payload['reserved_gib']:.2f} GiB"
        )
        return payload

    print(f"[GPU] {label}: CUDA/MPS not available")
    payload["backend"] = "cpu"
    return payload


def cleanup_gpu_resources(*objects: Any) -> None:
    """Close GPU-owning objects, collect cycles, then return unused cache to the driver."""
    for obj in objects:
        if obj is None:
            continue
        closer = getattr(obj, "close", None)
        if callable(closer):
            closer()
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()
        return
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available() and hasattr(torch, "mps") and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()
