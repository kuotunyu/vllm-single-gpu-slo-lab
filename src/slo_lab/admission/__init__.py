"""Admission control in front of an OpenAI-compatible server (design spec §3.5).

`policies` holds the pure-asyncio decision logic; `shim` wraps it in an aiohttp reverse proxy.
"""

from slo_lab.admission.policies import (
    AdmissionPolicy,
    Admitted,
    BoundedQueue,
    HardCap,
    Passthrough,
    Rejected,
    policy_from_config,
)

__all__ = [
    "AdmissionPolicy",
    "Admitted",
    "BoundedQueue",
    "HardCap",
    "Passthrough",
    "Rejected",
    "policy_from_config",
]
