"""Adapt inference-perf's per-request lifecycle output into canonical :class:`RequestRecord`s.

inference-perf 0.6.1 writes ``per_request_lifecycle_metrics.json``: a list of
``{start_time, end_time, request, response, info, error}`` where ``info.response_metrics``
carries ``output_tokens`` and ``chunk_times`` (monotonic seconds of every streamed chunk).
TTFT is therefore ``chunk_times[0] - start_time`` and e2e is ``end_time - start_time``.
``offered_at_s`` is made relative to the earliest ``start_time`` in the file (stage start).

Outcome mapping: ``error`` mentioning 429 -> rejected_429; timeout wording -> timeout; any other
error or a record without chunks -> error. The bulky raw file stays outside git; the adapted
JSONL is what the evidence contract commits.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from slo_lab.slo import Outcome, RequestRecord


def _outcome(error: Any, chunk_times: list[float]) -> Outcome:
    if error:
        text = str(error).lower()
        if "429" in text or "too many" in text:
            return Outcome.REJECTED_429
        if "timeout" in text or "timed out" in text:
            return Outcome.TIMEOUT
        return Outcome.ERROR
    if not chunk_times:
        return Outcome.ERROR
    return Outcome.OK


def adapt_record(raw: dict[str, Any], *, index: int, origin_s: float) -> RequestRecord:
    info = raw.get("info") or {}
    resp = info.get("response_metrics") or {}
    req = info.get("request_metrics") or {}
    chunk_times = [float(t) for t in (resp.get("chunk_times") or [])]
    start = float(raw["start_time"])
    end = float(raw["end_time"]) if raw.get("end_time") is not None else start
    outcome = _outcome(raw.get("error"), chunk_times)
    ttft = (chunk_times[0] - start) if chunk_times else None
    e2e = end - start
    if ttft is not None and e2e < ttft:
        e2e = ttft  # end_time can be recorded a hair before the last chunk timestamp settles
    output_tokens = resp.get("output_tokens")
    if output_tokens is None and resp.get("server_usage"):
        output_tokens = resp["server_usage"].get("completion_tokens")
    input_tokens = None
    text = req.get("text") if isinstance(req, dict) else None
    if isinstance(text, dict):
        input_tokens = text.get("input_tokens")
    if input_tokens is None:
        input_tokens = info.get("input_tokens")
    return RequestRecord(
        request_id=f"ipf-{index:06d}",
        offered_at_s=max(0.0, start - origin_s),
        outcome=outcome,
        ttft_s=max(0.0, ttft) if (ttft is not None and outcome is Outcome.OK) else None,
        e2e_s=max(0.0, e2e) if outcome is Outcome.OK else None,
        output_tokens=int(output_tokens)
        if (output_tokens is not None and outcome is Outcome.OK)
        else None,
        input_tokens=int(input_tokens) if input_tokens is not None else None,
    )


def adapt_file(path: Path) -> list[RequestRecord]:
    return adapt_file_with_origin(path)[0]


def adapt_file_with_origin(path: Path) -> tuple[list[RequestRecord], float]:
    """Records plus the raw origin (earliest ``start_time``, inference-perf's monotonic clock).

    The trace stage needs the origin to put scraped ``/metrics`` and shim samples (wall-clock)
    on the records' time axis: wall = origin + (time.time() - time.monotonic()) at the stage.
    """
    raw_list = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw_list, list) or not raw_list:
        raise ValueError(f"{path} holds no per-request records")
    origin = min(float(r["start_time"]) for r in raw_list)
    ordered = sorted(raw_list, key=lambda r: float(r["start_time"]))
    records = [adapt_record(r, index=i, origin_s=origin) for i, r in enumerate(ordered)]
    return records, origin


def write_records_jsonl(records: list[RequestRecord], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for rec in records:
            handle.write(rec.model_dump_json() + "\n")
