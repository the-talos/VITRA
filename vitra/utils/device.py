"""Device-agnostic helpers for the XPU, CUDA and CPU backends.

Every place that used to ask ``torch.cuda`` directly goes through this module, so
training and inference share one code path across accelerators.

Two rules this module encodes:

* A missing CUDA build must not be a crash. ``torch.cuda.synchronize()`` raises
  ``AssertionError: Torch not compiled with CUDA enabled`` on a wheel built without
  CUDA, which is what an XPU or CPU-only install has. Callers use
  :func:`synchronize` and :func:`empty_cache`, which are no-ops when the requested
  backend is unavailable.
* Detection order is XPU, then CUDA, then CPU, and an explicit device always wins
  over detection (it is never silently replaced by another backend).
"""

from __future__ import annotations

from typing import Any

import torch

DeviceLike = str | torch.device | None
_ACCELERATORS = ("xpu", "cuda")


def is_available(device_type: str) -> bool:
    """Whether ``device_type`` is usable in this build and on this host."""
    if device_type == "xpu":
        return hasattr(torch, "xpu") and torch.xpu.is_available()
    if device_type == "cuda":
        return torch.cuda.is_available()
    return device_type == "cpu"


def available_device_types() -> list[str]:
    """Accelerator types present on this host in detection order, ending with ``cpu``."""
    return [name for name in _ACCELERATORS if is_available(name)] + ["cpu"]


def default_device_type() -> str:
    """Best available backend: ``xpu``, then ``cuda``, then ``cpu``."""
    return available_device_types()[0]


def default_device(index: int | None = None) -> torch.device:
    """Device of the best available backend, optionally with an explicit index."""
    name = default_device_type()
    if index is None or name == "cpu":
        return torch.device(name)
    return torch.device(name, int(index))


def resolve_device(device: DeviceLike) -> torch.device:
    """Resolve ``None``, a string or a :class:`torch.device` to a concrete device.

    ``None`` auto-detects with :func:`default_device`. Explicit values are returned
    as given, so an unsupported device fails at the first kernel instead of being
    replaced by a different backend.
    """
    if isinstance(device, torch.device):
        return device
    if device is None:
        return default_device()
    return torch.device(device)


def device_type(device: Any) -> str:
    """Backend type (``xpu``, ``cuda`` or ``cpu``) of a device, string, tensor or module."""
    if isinstance(device, torch.Tensor):
        return device.device.type
    if isinstance(device, torch.nn.Module):
        return next(device.parameters()).device.type
    if isinstance(device, torch.device):
        return device.type
    return str(device).split(":", 1)[0]


def _resolved_type(device: DeviceLike) -> str:
    return device_type(device) if device is not None else default_device_type()


def device_count(backend: str | None = None) -> int:
    """Number of visible devices of a backend, 0 when the backend is unavailable."""
    name = backend or default_device_type()
    if name == "xpu" and is_available("xpu"):
        return torch.xpu.device_count()
    if name == "cuda" and is_available("cuda"):
        return torch.cuda.device_count()
    return 0


def set_device(index: int, backend: str | None = None) -> None:
    """Make ``index`` the current device of the backend; no-op on CPU."""
    name = backend or default_device_type()
    if name == "xpu" and is_available("xpu"):
        torch.xpu.set_device(index)
    elif name == "cuda" and is_available("cuda"):
        torch.cuda.set_device(index)


def current_device(backend: str | None = None) -> torch.device:
    """Current device of the backend, ``cpu`` when there is no accelerator."""
    name = backend or default_device_type()
    if name == "xpu" and is_available("xpu"):
        return torch.device("xpu", torch.xpu.current_device())
    if name == "cuda" and is_available("cuda"):
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def synchronize(device: DeviceLike = None) -> None:
    """Wait for queued work on ``device``; no-op on CPU or an absent backend."""
    name = _resolved_type(device)
    if name == "xpu" and is_available("xpu"):
        torch.xpu.synchronize()
    elif name == "cuda" and is_available("cuda"):
        torch.cuda.synchronize()


def empty_cache(device: DeviceLike = None) -> None:
    """Release cached allocator blocks of ``device``; no-op on CPU or an absent backend."""
    name = _resolved_type(device)
    if name == "xpu" and is_available("xpu"):
        torch.xpu.empty_cache()
    elif name == "cuda" and is_available("cuda"):
        torch.cuda.empty_cache()


def distributed_backend(device: DeviceLike = None) -> str:
    """Process group backend for ``device``: ``xccl`` on XPU, ``nccl`` on CUDA, else ``gloo``."""
    name = _resolved_type(device)
    if name == "xpu":
        return "xccl"
    if name == "cuda":
        return "nccl"
    return "gloo"
