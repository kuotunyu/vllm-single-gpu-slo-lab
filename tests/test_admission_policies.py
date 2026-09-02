import asyncio

import pytest

from slo_lab.admission import (
    AdmissionPolicy,
    Admitted,
    BoundedQueue,
    HardCap,
    Passthrough,
    Rejected,
    policy_from_config,
)


async def test_passthrough_never_rejects():
    p = Passthrough()
    decisions = [await p.acquire() for _ in range(50)]
    assert all(isinstance(d, Admitted) for d in decisions)
    assert p.snapshot()["in_flight"] == 50
    for _ in decisions:
        await p.release()
    assert p.snapshot()["in_flight"] == 0


async def test_hard_cap_rejects_beyond_capacity_without_queueing():
    p = HardCap(2, retry_after_s=3.0)
    assert isinstance(await p.acquire(), Admitted)
    assert isinstance(await p.acquire(), Admitted)
    third = await p.acquire()
    assert third == Rejected("cap_full", 3.0)
    assert p.snapshot() == {"in_flight": 2, "waiting": 0, "capacity": 2}
    await p.release()
    assert isinstance(await p.acquire(), Admitted)


async def test_bounded_queue_waits_then_admits_and_rejects_when_full():
    p = BoundedQueue(capacity=1, queue_limit=1, timeout_s=5.0, retry_after_s=2.0)
    assert isinstance(await p.acquire(), Admitted)

    queued = asyncio.create_task(p.acquire())
    await asyncio.sleep(0)  # let it enter the queue
    assert p.snapshot()["waiting"] == 1

    overflow = await p.acquire()
    assert overflow == Rejected("queue_full", 2.0)

    await p.release()
    decision = await asyncio.wait_for(queued, 1.0)
    assert isinstance(decision, Admitted)
    assert decision.wait_s >= 0.0
    assert p.snapshot()["in_flight"] == 1
    assert p.snapshot()["waiting"] == 0


async def test_bounded_queue_times_out_and_leaves_state_consistent():
    p = BoundedQueue(capacity=1, queue_limit=4, timeout_s=0.05)
    assert isinstance(await p.acquire(), Admitted)
    decision = await p.acquire()
    assert decision == Rejected("queue_timeout", 1.0)
    assert p.snapshot()["waiting"] == 0
    assert p.snapshot()["in_flight"] == 1
    await p.release()
    assert isinstance(await p.acquire(), Admitted)


async def test_bounded_queue_is_fifo_and_newcomers_do_not_jump_the_queue():
    p = BoundedQueue(capacity=1, queue_limit=2, timeout_s=5.0)
    assert isinstance(await p.acquire(), Admitted)
    order: list[str] = []

    async def wait(name: str) -> None:
        d = await p.acquire()
        assert isinstance(d, Admitted)
        order.append(name)

    b = asyncio.create_task(wait("b"))
    await asyncio.sleep(0)
    c = asyncio.create_task(wait("c"))
    await asyncio.sleep(0)
    assert p.snapshot()["waiting"] == 2

    await p.release()  # frees one slot: b must get it, not a newcomer
    newcomer = await p.acquire()
    assert newcomer == Rejected("queue_full", 1.0)
    await asyncio.wait_for(b, 1.0)
    assert order == ["b"]
    await p.release()
    await asyncio.wait_for(c, 1.0)
    assert order == ["b", "c"]


async def test_bounded_queue_zero_queue_behaves_like_hard_cap():
    p = BoundedQueue(capacity=1, queue_limit=0, timeout_s=1.0)
    assert isinstance(await p.acquire(), Admitted)
    assert await p.acquire() == Rejected("queue_full", 1.0)


def test_policy_from_config_and_validation():
    assert isinstance(policy_from_config({"policy": "passthrough"}), Passthrough)
    cap = policy_from_config({"policy": "hard_cap", "capacity": 8, "retry_after_s": 2})
    assert isinstance(cap, HardCap) and cap.capacity == 8 and cap.retry_after_s == 2.0
    bq = policy_from_config({"policy": "bounded_queue", "capacity": 4, "queue_limit": "capacity"})
    assert isinstance(bq, BoundedQueue) and bq.queue_limit == 4 and bq.timeout_s == 1.0
    assert isinstance(bq, AdmissionPolicy)
    with pytest.raises(ValueError):
        policy_from_config({"policy": "magic"})
    with pytest.raises(ValueError):
        HardCap(0)
    with pytest.raises(ValueError):
        BoundedQueue(1, -1, 1.0)
