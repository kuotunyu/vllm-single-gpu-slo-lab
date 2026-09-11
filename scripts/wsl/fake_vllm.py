"""A CPU-only stand-in for ``vllm serve`` used to dry-run the W3 and W4 drivers end to end.

It serves what the harness touches: ``GET /v1/models``, streaming ``POST /v1/completions`` (one SSE
chunk per output token, a usage chunk, ``[DONE]``), and ``GET /metrics`` with
``vllm:num_requests_running`` / ``vllm:num_requests_waiting`` / ``vllm:num_preemptions_total`` and
empty latency histograms. At most ``--slots`` requests decode at once; the rest wait FIFO, so a
burst builds a real queue and the admission policies behave as they would in front of the engine.

``--specdec`` (W4, ADR 0015) imitates speculative decoding: every engine step drafts three tokens
and accepts them position by position with fixed odds, emits one plus the accepted count in one
step time, and exposes the same counters vLLM 0.28 does (``vllm:spec_decode_num_drafts_total``,
``..._num_draft_tokens_total``, ``..._num_accepted_tokens_total``, ``..._per_pos_total{position}``),
so the harness sees a faster stream and a non-trivial acceptance rate. It never touches the GPU.

    python fake_vllm.py --port 8013 --model Qwen/Qwen3-8B-FP8 --slots 8 --token-ms 2 [--specdec]
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import time

from aiohttp import web

HISTOGRAMS = (
    "vllm:time_to_first_token_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_time_per_output_token_seconds",
    "vllm:e2e_request_latency_seconds",
)
DRAFT_K = 3
# odds that draft position i is accepted given every earlier position was (a plausible EAGLE-like
# profile: 0.7 / 0.5 / 0.3 -> acceptance rate about 0.42, mean acceptance length about 2.3)
ACCEPT_ODDS = (0.7, 0.5, 0.3)


class Engine:
    def __init__(self, slots: int, token_s: float, specdec: bool) -> None:
        self.slots = asyncio.Semaphore(slots)
        self.token_s = token_s
        self.specdec = specdec
        self.running = 0
        self.waiting = 0
        self.done = 0
        self.requests = 0
        self.drafts = 0
        self.draft_tokens = 0
        self.accepted_tokens = 0
        self.accepted_per_pos = [0] * DRAFT_K


def build(model: str, slots: int, token_s: float, specdec: bool) -> web.Application:
    engine = Engine(slots, token_s, specdec)

    async def models(_request: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [{"id": model, "object": "model"}]})

    async def metrics(_request: web.Request) -> web.Response:
        lines = [
            f'vllm:num_requests_running{{model_name="{model}"}} {engine.running}',
            f'vllm:num_requests_waiting{{model_name="{model}"}} {engine.waiting}',
            f'vllm:num_preemptions_total{{model_name="{model}"}} 0',
            f'vllm:request_success_total{{model_name="{model}"}} {engine.done}',
        ]
        if engine.specdec:
            lines += [
                f'vllm:spec_decode_num_drafts_total{{model_name="{model}"}} {engine.drafts}',
                f'vllm:spec_decode_num_draft_tokens_total{{model_name="{model}"}} {engine.draft_tokens}',
                f'vllm:spec_decode_num_accepted_tokens_total{{model_name="{model}"}} {engine.accepted_tokens}',
            ]
            lines += [
                f'vllm:spec_decode_num_accepted_tokens_per_pos_total{{model_name="{model}",position="{i}"}} {n}'
                for i, n in enumerate(engine.accepted_per_pos)
            ]
        for name in HISTOGRAMS:
            lines += [f'{name}_bucket{{le="+Inf"}} 0', f"{name}_count 0", f"{name}_sum 0"]
        return web.Response(text="\n".join(lines) + "\n")

    def step_tokens(rng: random.Random) -> int:
        """Tokens produced by one engine step: 1 without spec decode, 1 + accepted drafts with."""
        if not engine.specdec:
            return 1
        accepted = 0
        for odds in ACCEPT_ODDS:
            if rng.random() >= odds:
                break
            accepted += 1
        engine.drafts += 1
        engine.draft_tokens += DRAFT_K
        engine.accepted_tokens += accepted
        for i in range(accepted):
            engine.accepted_per_pos[i] += 1
        return 1 + accepted

    async def completions(request: web.Request) -> web.StreamResponse:
        body = await request.json()
        n_tokens = int(body.get("max_tokens") or 16)
        engine.requests += 1
        rng = random.Random(engine.requests)  # deterministic per request order
        engine.waiting += 1
        await engine.slots.acquire()
        engine.waiting -= 1
        engine.running += 1
        try:
            resp = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
            await resp.prepare(request)
            created = int(time.time())
            emitted = 0
            while emitted < n_tokens:
                await asyncio.sleep(engine.token_s)
                for _ in range(min(step_tokens(rng), n_tokens - emitted)):
                    chunk = {
                        "id": "cmpl-fake",
                        "object": "text_completion",
                        "created": created,
                        "model": model,
                        "choices": [{"index": 0, "text": " x", "finish_reason": None}],
                    }
                    await resp.write(f"data: {json.dumps(chunk)}\n\n".encode())
                    emitted += 1
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
    ap.add_argument("--specdec", action="store_true", help="imitate speculative decoding (W4)")
    a = ap.parse_args()
    web.run_app(
        build(a.model, a.slots, a.token_ms / 1000.0, a.specdec),
        host=a.host,
        port=a.port,
        print=None,
    )


if __name__ == "__main__":
    main()
