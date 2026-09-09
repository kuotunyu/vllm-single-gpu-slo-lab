"""Score a TMMLU+ slice against an OpenAI-compatible chat endpoint (quality track, spec §5).

usage: python scripts/tmmluplus_eval.py --slice eval/tmmluplus/slice-1.jsonl --base-url http://127.0.0.1:8013 \
           --model Qwen/Qwen3-8B-FP8 --out evidence/raw/w1/tmmluplus-dryrun.json [--limit 20] [--concurrency 8]

Greedy decoding (temperature 0), thinking disabled with the Qwen ``/no_think`` tag, ``max_tokens`` 8.
Writes per-item records and the accuracy with a Wilson 95% interval. Requires ``aiohttp`` (a
runtime dependency of this package); run it from the repo environment.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import time
from pathlib import Path

import aiohttp

from slo_lab.stats import wilson_ci
from slo_lab.tmmluplus import Item, parse_letter, read_slice, slice_digest


async def _ask(
    session: aiohttp.ClientSession, base_url: str, model: str, item: Item, sem: asyncio.Semaphore
) -> dict:
    """Score one item. Never raises: a failure becomes a record carrying the exception name.

    The full test set is 19,680 items per cell and ``asyncio.gather`` cancels every sibling when
    one coroutine raises, so a single transient connection error 25 minutes in would discard the
    whole cell's quality result. An errored item is recorded as unparsed (and therefore counted
    against accuracy); ``errors`` in the summary says how many there were, so a run with any
    error can be spotted and re-run rather than quietly published.
    """
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": item.prompt() + " /no_think"}],
        "max_tokens": 8,
        "temperature": 0,
    }
    base = {"subject": item.subject, "index": item.index, "answer": item.answer}
    try:
        async with sem:
            start = time.perf_counter()
            async with session.post(f"{base_url}/v1/chat/completions", json=payload) as resp:
                body = await resp.json()
                status = resp.status
            elapsed = time.perf_counter() - start
        text = body["choices"][0]["message"]["content"] if status == 200 else ""
        completion_tokens = (body.get("usage") or {}).get("completion_tokens")
    except Exception as exc:  # any failure must stay local to this item, never cancel the run
        return {
            **base,
            "predicted": None,
            "correct": False,
            "raw": "",
            "status": 0,
            "latency_s": None,
            "completion_tokens": None,
            "error": type(exc).__name__,
        }
    predicted = parse_letter(text or "")
    return {
        **base,
        "predicted": predicted,
        "correct": predicted == item.answer,
        "raw": (text or "")[:40],
        "status": status,
        "latency_s": round(elapsed, 4),
        "completion_tokens": completion_tokens,
        "error": None,
    }


async def main_async(args: argparse.Namespace) -> int:
    items = read_slice(Path(args.slice))
    if args.limit:
        items = items[: args.limit]
    sem = asyncio.Semaphore(args.concurrency)
    started = time.time()
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as session:
        records = await asyncio.gather(
            *(_ask(session, args.base_url, args.model, it, sem) for it in items)
        )
    wall = time.time() - started
    correct = sum(1 for r in records if r["correct"])
    ci = wilson_ci(correct, len(records))
    low, high = ci.lo, ci.hi
    result = {
        "slice": Path(args.slice).name,
        "slice_sha256": slice_digest(items) if not args.limit else None,
        "items_sha256_scored": hashlib.sha256(
            ("\n".join(f"{r['subject']}:{r['index']}" for r in records) + "\n").encode("utf-8")
        ).hexdigest(),
        "model": args.model,
        "base_url": args.base_url,
        "n": len(records),
        "correct": correct,
        "accuracy": correct / len(records),
        "wilson95": [low, high],
        "unparsed": sum(1 for r in records if r["predicted"] is None),
        "non_200": sum(1 for r in records if r["status"] != 200),
        "errors": sum(1 for r in records if r.get("error")),
        "wall_s": round(wall, 2),
        "items_per_s": round(len(records) / wall, 3),
        "decoding": {"temperature": 0, "max_tokens": 8, "thinking": False},
        "records": records,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "n",
                    "correct",
                    "accuracy",
                    "wilson95",
                    "unparsed",
                    "non_200",
                    "errors",
                    "wall_s",
                    "items_per_s",
                )
            }
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--slice", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=8)
    return asyncio.run(main_async(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
