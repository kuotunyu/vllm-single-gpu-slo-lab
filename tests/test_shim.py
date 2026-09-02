"""HTTP-level tests of the admission shim against an in-process fake upstream (loopback only)."""

import asyncio
from collections.abc import AsyncIterator

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from slo_lab.admission import BoundedQueue, HardCap, Passthrough
from slo_lab.admission.shim import build_app


class FakeUpstream:
    """Echo server that blocks each request until `gate` is set; records arrivals."""

    def __init__(self) -> None:
        self.gate = asyncio.Event()
        self.arrived = asyncio.Event()
        self.seen: list[tuple[str, str, bytes]] = []

    async def handle(self, request: web.Request) -> web.StreamResponse:
        body = await request.read()
        self.seen.append((request.method, request.rel_url.path_qs, body))
        self.arrived.set()
        await self.gate.wait()
        if request.path == "/stream":
            resp = web.StreamResponse(headers={"X-Upstream": "yes"})
            resp.enable_chunked_encoding()
            await resp.prepare(request)
            for chunk in (b"data: one\n\n", b"data: two\n\n", b"data: [DONE]\n\n"):
                await resp.write(chunk)
                await asyncio.sleep(0.005)
            await resp.write_eof()
            return resp
        return web.json_response(
            {"echo": body.decode(), "path": request.rel_url.path_qs}, headers={"X-Upstream": "yes"}
        )

    def app(self) -> web.Application:
        app = web.Application()
        app.router.add_route("*", "/{tail:.*}", self.handle)
        return app


@pytest.fixture
async def upstream() -> AsyncIterator[tuple[FakeUpstream, str]]:
    fake = FakeUpstream()
    server = TestServer(fake.app())
    await server.start_server()
    yield fake, str(server.make_url("")).rstrip("/")
    await server.close()


async def test_passthrough_forwards_body_query_and_streams_chunks(upstream):
    fake, base = upstream
    fake.gate.set()
    async with TestClient(TestServer(build_app(base, Passthrough()))) as client:
        resp = await client.post("/v1/completions?x=1", data=b"hello")
        assert resp.status == 200
        assert resp.headers["X-Upstream"] == "yes"
        assert "X-Shim-Queue-Wait-Ms" in resp.headers
        assert await resp.json() == {"echo": "hello", "path": "/v1/completions?x=1"}
        assert fake.seen[-1] == ("POST", "/v1/completions?x=1", b"hello")

        streamed = await client.get("/stream")
        assert streamed.status == 200
        assert await streamed.read() == b"data: one\n\ndata: two\n\ndata: [DONE]\n\n"

        stats = await (await client.get("/_shim/stats")).json()
        assert stats["policy"] == "passthrough"
        assert stats["admitted"] == 2 and stats["completed"] == 2 and stats["rejected"] == {}


async def test_hard_cap_returns_429_with_retry_after_and_never_queues(upstream):
    fake, base = upstream
    async with TestClient(TestServer(build_app(base, HardCap(1, retry_after_s=3)))) as client:
        first = asyncio.create_task(client.post("/v1/chat/completions", data=b"a"))
        await asyncio.wait_for(fake.arrived.wait(), 2.0)

        second = await client.post("/v1/chat/completions", data=b"b")
        assert second.status == 429
        assert second.headers["Retry-After"] == "3"
        body = await second.json()
        assert body["error"] == {
            "type": "admission_rejected",
            "reason": "cap_full",
            "policy": "hard_cap",
        }
        assert len(fake.seen) == 1  # the rejected request never reached the upstream

        fake.gate.set()
        resp = await asyncio.wait_for(first, 2.0)
        assert resp.status == 200
        assert (await resp.json())["echo"] == "a"

        stats = await (await client.get("/_shim/stats")).json()
        assert stats["rejected"] == {"cap_full": 1}
        assert stats["admitted"] == 1 and stats["in_flight"] == 0


async def test_bounded_queue_admits_after_wait_and_times_out(upstream):
    fake, base = upstream
    policy = BoundedQueue(capacity=1, queue_limit=1, timeout_s=0.5, retry_after_s=1)
    async with TestClient(TestServer(build_app(base, policy))) as client:
        first = asyncio.create_task(client.post("/v1/completions", data=b"a"))
        await asyncio.wait_for(fake.arrived.wait(), 2.0)

        # second waits in the bounded queue; third finds the queue full immediately
        second = asyncio.create_task(client.post("/v1/completions", data=b"b"))
        await asyncio.sleep(0.05)
        third = await client.post("/v1/completions", data=b"c")
        assert third.status == 429
        assert (await third.json())["error"]["reason"] == "queue_full"

        fake.gate.set()  # first completes -> second is admitted within its 0.5 s budget
        assert (await asyncio.wait_for(first, 2.0)).status == 200
        resp_b = await asyncio.wait_for(second, 2.0)
        assert resp_b.status == 200
        assert float(resp_b.headers["X-Shim-Queue-Wait-Ms"]) >= 0.0

        # now hold the upstream again: a queued request must time out into 429
        fake.gate.clear()
        fake.arrived.clear()
        held = asyncio.create_task(client.post("/v1/completions", data=b"d"))
        await asyncio.wait_for(fake.arrived.wait(), 2.0)
        timed_out = await client.post("/v1/completions", data=b"e")
        assert timed_out.status == 429
        assert (await timed_out.json())["error"]["reason"] == "queue_timeout"
        fake.gate.set()
        assert (await asyncio.wait_for(held, 2.0)).status == 200

        stats = await (await client.get("/_shim/stats")).json()
        assert stats["rejected"] == {"queue_full": 1, "queue_timeout": 1}
        assert stats["completed"] == 3


async def test_unreachable_upstream_becomes_502_and_releases_slot():
    # nothing listens on this loopback port; the shim must answer 502 and free the slot
    policy = HardCap(1)
    async with TestClient(TestServer(build_app("http://127.0.0.1:9", policy))) as client:
        resp = await client.get("/v1/models")
        assert resp.status == 502
        assert (await resp.json())["error"]["type"] == "upstream_unreachable"
        assert policy.in_flight == 0
