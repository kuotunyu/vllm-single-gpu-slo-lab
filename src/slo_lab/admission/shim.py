"""aiohttp reverse-proxy shim that applies an admission policy in front of vLLM (spec §3.5).

The shim is intentionally tiny: read the request, ask the policy, either answer 429 with
Retry-After or forward the request byte-for-byte and stream the upstream response back
(SSE chunks included, no buffering, no decompression). Engine flags never change between
policies, so any difference in attainment/goodput/rejection is attributable to admission.

Every admitted response carries `X-Shim-Queue-Wait-Ms` so queue time can be separated from
engine time; `/_shim/stats` exposes counters for the harness to scrape.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field

import aiohttp
from aiohttp import web

from slo_lab.admission.policies import AdmissionPolicy, Rejected

HOP_BY_HOP = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)


@dataclass
class ShimStats:
    admitted: int = 0
    completed: int = 0
    upstream_errors: int = 0
    rejected: dict[str, int] = field(default_factory=dict)

    def as_dict(self, policy: AdmissionPolicy) -> dict[str, object]:
        return {
            "policy": policy.name,
            "admitted": self.admitted,
            "completed": self.completed,
            "upstream_errors": self.upstream_errors,
            "rejected": dict(self.rejected),
            **policy.snapshot(),
        }


UPSTREAM_KEY = web.AppKey("upstream", str)
POLICY_KEY = web.AppKey("policy", AdmissionPolicy)
SESSION_KEY = web.AppKey("session", aiohttp.ClientSession)
STATS_KEY = web.AppKey("stats", ShimStats)


def _forward_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP}


async def _stats_handler(request: web.Request) -> web.Response:
    return web.json_response(request.app[STATS_KEY].as_dict(request.app[POLICY_KEY]))


async def _proxy_handler(request: web.Request) -> web.StreamResponse:
    policy = request.app[POLICY_KEY]
    stats = request.app[STATS_KEY]
    decision = await policy.acquire()
    if isinstance(decision, Rejected):
        stats.rejected[decision.reason] = stats.rejected.get(decision.reason, 0) + 1
        return web.json_response(
            {
                "error": {
                    "type": "admission_rejected",
                    "reason": decision.reason,
                    "policy": policy.name,
                }
            },
            status=429,
            headers={"Retry-After": f"{decision.retry_after_s:g}"},
        )
    stats.admitted += 1
    try:
        body = await request.read()
        url = request.app[UPSTREAM_KEY] + request.rel_url.path_qs
        session = request.app[SESSION_KEY]
        async with session.request(
            request.method, url, headers=_forward_headers(request.headers), data=body
        ) as upstream:
            response = web.StreamResponse(status=upstream.status, reason=upstream.reason)
            for key, value in _forward_headers(upstream.headers).items():
                response.headers[key] = value
            response.headers["X-Shim-Queue-Wait-Ms"] = f"{decision.wait_s * 1000.0:.3f}"
            await response.prepare(request)
            async for chunk in upstream.content.iter_any():
                await response.write(chunk)
            await response.write_eof()
            stats.completed += 1
            return response
    except aiohttp.ClientError as exc:
        stats.upstream_errors += 1
        return web.json_response(
            {"error": {"type": "upstream_unreachable", "detail": str(exc)}}, status=502
        )
    finally:
        await policy.release()


def build_app(upstream_base_url: str, policy: AdmissionPolicy) -> web.Application:
    """Create the shim application. `upstream_base_url` e.g. http://localhost:8000 (no trailing /)."""
    app = web.Application()
    app[UPSTREAM_KEY] = upstream_base_url.rstrip("/")
    app[POLICY_KEY] = policy
    app[STATS_KEY] = ShimStats()

    async def _open_session(app: web.Application) -> None:
        app[SESSION_KEY] = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=None), auto_decompress=False
        )

    async def _close_session(app: web.Application) -> None:
        await app[SESSION_KEY].close()

    app.on_startup.append(_open_session)
    app.on_cleanup.append(_close_session)
    app.router.add_get("/_shim/stats", _stats_handler)
    app.router.add_route("*", "/{tail:.*}", _proxy_handler)
    return app


def run(upstream_base_url: str, policy: AdmissionPolicy, *, host: str, port: int) -> None:
    """Blocking entry point used by `slo-lab shim`."""
    web.run_app(build_app(upstream_base_url, policy), host=host, port=port, print=None)
