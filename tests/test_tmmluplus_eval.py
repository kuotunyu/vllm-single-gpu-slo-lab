"""The scorer must survive a single failed request: the full set is 19,680 items per cell.

`asyncio.gather` without `return_exceptions` cancels every sibling when one coroutine raises, so
one transient connection error 25 minutes into a full-set run would throw away the whole cell's
quality result. `_ask` therefore turns any failure into a scored record carrying the error name.
"""

from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

import pytest

from slo_lab.tmmluplus import Item

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "tmmluplus_eval.py"
_spec = importlib.util.spec_from_file_location("tmmluplus_eval", SCRIPT)
assert _spec and _spec.loader
tmmluplus_eval = importlib.util.module_from_spec(_spec)
sys.modules["tmmluplus_eval"] = tmmluplus_eval
_spec.loader.exec_module(tmmluplus_eval)

ITEM = Item(
    subject="accounting",
    index=7,
    question="X 公司的權益總額是多少？",
    A="100",
    B="200",
    C="300",
    D="400",
    answer="B",
)


class _Response:
    def __init__(self, status: int, body: dict) -> None:
        self.status = status
        self._body = body

    async def json(self) -> dict:
        return self._body

    async def __aenter__(self) -> _Response:
        return self

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _Session:
    """Minimal stand-in for aiohttp.ClientSession: answers, or raises what it was given."""

    def __init__(self, response: _Response | None = None, error: Exception | None = None) -> None:
        self._response = response
        self._error = error

    def post(self, url: str, json: dict) -> _Response:  # mirrors aiohttp signature
        if self._error is not None:
            raise self._error
        assert self._response is not None
        return self._response


def _run(session: _Session) -> dict:
    async def go() -> dict:
        sem = asyncio.Semaphore(1)
        return await tmmluplus_eval._ask(session, "http://x", "m", ITEM, sem)

    return asyncio.run(go())


def test_ask_scores_a_normal_answer() -> None:
    body = {
        "choices": [{"message": {"content": "B"}}],
        "usage": {"completion_tokens": 1},
    }
    record = _run(_Session(_Response(200, body)))
    assert record["predicted"] == "B" and record["correct"] is True
    assert record["status"] == 200 and record["error"] is None
    assert record["subject"] == "accounting" and record["index"] == 7
    assert record["completion_tokens"] == 1


def test_ask_marks_a_non_200_as_unparsed_rather_than_correct() -> None:
    record = _run(_Session(_Response(503, {"error": "overloaded"})))
    assert record["status"] == 503 and record["predicted"] is None
    assert record["correct"] is False and record["error"] is None


def test_ask_turns_a_raised_error_into_a_record_instead_of_cancelling_the_run() -> None:
    record = _run(_Session(error=ConnectionResetError("peer went away")))
    assert record["error"] == "ConnectionResetError"
    assert record["status"] == 0 and record["predicted"] is None and record["correct"] is False
    assert record["latency_s"] is None
    # the record still identifies the item, so the scored-item digest stays complete
    assert record["subject"] == "accounting" and record["index"] == 7 and record["answer"] == "B"


def test_ask_survives_a_malformed_body() -> None:
    record = _run(_Session(_Response(200, {"unexpected": True})))
    assert record["error"] in {"KeyError", "IndexError", "TypeError"}
    assert record["correct"] is False


@pytest.mark.parametrize("text", ["B", " b ", "答案是 B", "(B)"])
def test_letter_parsing_used_by_the_scorer(text: str) -> None:
    from slo_lab.tmmluplus import parse_letter

    assert parse_letter(text) == "B"
