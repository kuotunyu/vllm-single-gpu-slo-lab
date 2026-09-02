"""Admission policies (design spec §3.5), pure asyncio so the semantics are unit-testable.

(i)   Passthrough   - never rejects; vLLM's own unbounded queue absorbs everything.
(ii)  HardCap       - at most `capacity` requests in flight; the rest get HTTP 429 + Retry-After
                      immediately, no queue (LIBG semantics borrowed, not imported).
(iii) BoundedQueue  - `capacity` in flight, up to `queue_limit` waiting FIFO for at most
                      `timeout_s`; queue-full and queue-timeout both become 429.
                      Spec proposal: Q = C and T = 1 s (= the TTFT SLO).

All counters are mutated only between awaits, so no extra locking is needed on a single loop.
"""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, runtime_checkable

RejectReason = Literal["cap_full", "queue_full", "queue_timeout"]


@dataclass(frozen=True)
class Admitted:
    wait_s: float = 0.0


@dataclass(frozen=True)
class Rejected:
    reason: RejectReason
    retry_after_s: float


Decision = Admitted | Rejected


@runtime_checkable
class AdmissionPolicy(Protocol):
    name: str

    async def acquire(self) -> Decision: ...

    async def release(self) -> None: ...

    def snapshot(self) -> dict[str, int]: ...


class Passthrough:
    name = "passthrough"

    def __init__(self) -> None:
        self.in_flight = 0

    async def acquire(self) -> Decision:
        self.in_flight += 1
        return Admitted()

    async def release(self) -> None:
        self.in_flight -= 1

    def snapshot(self) -> dict[str, int]:
        return {"in_flight": self.in_flight, "waiting": 0}


class HardCap:
    name = "hard_cap"

    def __init__(self, capacity: int, *, retry_after_s: float = 1.0) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self.capacity = capacity
        self.retry_after_s = retry_after_s
        self.in_flight = 0

    async def acquire(self) -> Decision:
        if self.in_flight >= self.capacity:
            return Rejected("cap_full", self.retry_after_s)
        self.in_flight += 1
        return Admitted()

    async def release(self) -> None:
        self.in_flight -= 1

    def snapshot(self) -> dict[str, int]:
        return {"in_flight": self.in_flight, "waiting": 0, "capacity": self.capacity}


class BoundedQueue:
    name = "bounded_queue"

    def __init__(
        self,
        capacity: int,
        queue_limit: int,
        timeout_s: float,
        *,
        retry_after_s: float = 1.0,
    ) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if queue_limit < 0 or timeout_s < 0:
            raise ValueError("queue_limit and timeout_s must be >= 0")
        self.capacity = capacity
        self.queue_limit = queue_limit
        self.timeout_s = timeout_s
        self.retry_after_s = retry_after_s
        self.in_flight = 0
        self.waiting = 0
        self._cond = asyncio.Condition()

    def _slot_free(self) -> bool:
        return self.in_flight < self.capacity

    async def acquire(self) -> Decision:
        async with self._cond:
            # FIFO: a newcomer may only jump straight in when nobody is already waiting.
            if self._slot_free() and self.waiting == 0:
                self.in_flight += 1
                return Admitted()
            if self.waiting >= self.queue_limit:
                return Rejected("queue_full", self.retry_after_s)
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            self.waiting += 1
            try:
                await asyncio.wait_for(self._cond.wait_for(self._slot_free), self.timeout_s)
            except TimeoutError:
                # A notify may have been consumed by the cancelled wait; pass it on.
                self._cond.notify(1)
                return Rejected("queue_timeout", self.retry_after_s)
            finally:
                self.waiting -= 1
            self.in_flight += 1
            return Admitted(wait_s=loop.time() - t0)

    async def release(self) -> None:
        async with self._cond:
            self.in_flight -= 1
            self._cond.notify(1)

    def snapshot(self) -> dict[str, int]:
        return {
            "in_flight": self.in_flight,
            "waiting": self.waiting,
            "capacity": self.capacity,
            "queue_limit": self.queue_limit,
        }


def policy_from_config(cfg: Mapping[str, Any]) -> AdmissionPolicy:
    """Build a policy from a `config/admission/*.yaml` mapping."""
    kind = cfg.get("policy")
    retry_after = float(cfg.get("retry_after_s", 1.0))
    if kind == "passthrough":
        return Passthrough()
    if kind == "hard_cap":
        return HardCap(int(cfg["capacity"]), retry_after_s=retry_after)
    if kind == "bounded_queue":
        capacity = int(cfg["capacity"])
        queue_limit = cfg.get("queue_limit")
        queue_limit = capacity if queue_limit in (None, "capacity") else int(queue_limit)
        return BoundedQueue(
            capacity, queue_limit, float(cfg.get("timeout_s", 1.0)), retry_after_s=retry_after
        )
    raise ValueError(f"unknown admission policy: {kind!r}")
