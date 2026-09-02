"""Lazy NVML loader shared by the power sampler and the quiet-GPU gate.

`pynvml` (package `nvidia-ml-py`, optional extra `gpu`) is imported only when a caller
actually needs the GPU, so every other module, the CLI and the test-suite import cleanly on a
CPU-only machine. Failures are normalised into `NvmlUnavailableError` with a message that says
what to install or check.
"""

from __future__ import annotations

from types import ModuleType

INSTALL_HINT = (
    "NVML is unavailable. Install the GPU extra (`uv sync --extra gpu`, package nvidia-ml-py) "
    "on the WSL2 measurement host and make sure the NVIDIA driver exposes libnvidia-ml."
)


class NvmlUnavailableError(RuntimeError):
    """Raised when pynvml cannot be imported or NVML cannot be initialised."""


def load_pynvml() -> ModuleType:
    """Import pynvml lazily; raise NvmlUnavailableError with an actionable message if absent."""
    try:
        import pynvml
    except ImportError as exc:  # pragma: no cover - exercised via monkeypatch in tests
        raise NvmlUnavailableError(f"{INSTALL_HINT} (import failed: {exc})") from exc
    return pynvml


def init_nvml(pynvml: ModuleType) -> None:
    """Call nvmlInit and translate any failure into NvmlUnavailableError."""
    try:
        pynvml.nvmlInit()
    except Exception as exc:  # NVMLError has many subclasses; keep the boundary generic
        raise NvmlUnavailableError(f"{INSTALL_HINT} (nvmlInit failed: {exc})") from exc
