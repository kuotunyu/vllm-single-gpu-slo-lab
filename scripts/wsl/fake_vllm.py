"""A CPU-only stand-in for ``vllm serve`` used to dry-run the W3 driver end to end (plan Task A7).

It serves what the harness touches: ``GET /v1/models``, streaming ``POST /v1/completions`` (one SSE
chunk per output token, a usage chunk, ``[DONE]``), and ``GET /metrics`` with
``vllm:num_requests_running`` / ``vllm:num_requests_waiting`` and empty latency histograms. At most
``--slots`` requests decode at once; the rest wait FIFO, so a burst builds a real queue and the
admission policies behave as they would in front of the engine. It never touches the GPU.

    python fake_vllm.py --port 8013 --model Qwen/Qwen3-8B-FP8 --slots 8 --token-ms 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time

from aiohttp import web

HISTOGRAMS = (
    "vllm:time_to_first_token_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_time_per_output_token_seconds",
    "vllm:e2e_request_latency_seconds",
)


class Engine:
    def __init__(self, slots: int, token_s: float) -> None:
        self.slots = asyncio.Semaphore(slots)
        self.token_s = token_s
        self.running = 0
        self.waiting = 0
        self.done = 0


def build(model: str, slots: int, token_s: float) -> web.Application:
    engine = Engine(slots, token_s)

    async def models(_request: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [{"id": model, "object": "model"}]})

    async def metrics(_request: web.Request) -> web.Response:
        lines = [
            f'vllm:num_requests_running{{model_name="{model}"}} {engine.running}',
            f'vllm:num_requests_waiting{{model_name="{model}"}} {engine.waiting}',
            f'vllm:request_success_total{{model_name="{model}"}} {engine.done}',
        ]
        for name in HISTOGRAMS:
            lines += [f'{name}_bucket{{le="+Inf"}} 0', f"{name}_count 0", f"{name}_sum 0"]
        return web.Response(text="\n".join(lines) + "\n")

    async def completions(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        n_tokens = int(body.get("max_tokens") or 16)
        engine.waiting += 1
        await engine.slots.acquire()
        engine.waiting -= 1
        engine.running += 1
        try:
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            created = int(time.time())
            for _ in range(n_tokens):
                await asyncio.sleep(engine.token_s)
                chunk = {
                    "id": "cmpl-fake",
                    "object": "text_completion",
                    "created": created,
                    "model": model,
                    "choices": [{"index": 0, "text": " x", "finish_reason": None}],
                }
                await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
            usage = {
                "prompt_tokens": 108,
                "completion_tokens": n_tokens,
                "total_tokens": 108 + n_tokens,
            }
            final = {
                "id": "cmpl-fake",
                "object": "text_completion",
                "model": model,
                "choices": [],
                "usage": usage,
            }
            await resp.write(f"data: {json.dumps(final)}\n\n".encode())
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            engine.done += 1
            return resp
        finally:
            engine.running -= 1
            engine.slots.release()

    app = web.Application(client_max_size=1 << 22)
    app.router.add_get("/v1/models", models)
    app.router.add_get("/metrics", metrics)
    app.router.add_post("/v1/completions", completions)
    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8013)
    ap.add_argument("--model", default="Qwen/Qwen3-8B-FP8")
    ap.add_argument("--slots", type=int, default=8)
    ap.add_argument("--token-ms", type=float, default=2.0)
    a = ap.parse_args()
    web.run_app(build(a.model, a.slots, a.token_ms / 1000.0), host=a.host, port=a.port, print=None)


if __name__ == "__main__":
    main()
