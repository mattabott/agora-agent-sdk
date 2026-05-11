"""Test LogsPoller: format_log_line per i kind comuni, poll loop end-to-end
con httpx MockTransport."""
from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from agora_agent_sdk.logs_poller import LogsPoller, _format_log_line


def test_format_decision_line():
    line = _format_log_line({
        "tick": 17227, "kind": "decision",
        "content": {
            "action": "move", "via": "reflex",
            "direction": "north",
            "thought": "going to shelter for the night",
        },
    })
    assert "decision" in line
    assert "via=reflex" in line
    assert "direction=north" in line
    assert "going to shelter" in line


def test_format_action_result_ok_line():
    line = _format_log_line({
        "tick": 17227, "kind": "action_result",
        "content": {"action": "move", "ok": True, "to": [46, 54]},
    })
    assert "OK" in line and "move" in line


def test_format_action_result_fail_line():
    line = _format_log_line({
        "tick": 100, "kind": "action_result",
        "content": {"action": "propose", "ok": False, "reason": "close_kin"},
    })
    assert "FAIL" in line and "close_kin" in line


def test_format_dialogue_line():
    line = _format_log_line({
        "tick": 50, "kind": "dialogue_received",
        "content": {"from_name": "Roko", "content": "Are you well?"},
    })
    assert "dialogue from Roko" in line and "Are you well?" in line


def test_format_unknown_kind_falls_back_to_json():
    line = _format_log_line({
        "tick": 1, "kind": "exotic_kind",
        "content": {"foo": "bar"},
    })
    assert "exotic_kind" in line and "foo" in line


@pytest.mark.asyncio
async def test_poll_loop_advances_since_and_stops_on_event():
    calls: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        calls.append(params)
        n = len(calls)
        if n == 1:
            payload = {
                "agent_id": 7, "agent_name": "X", "current_tick": 100,
                "alive": True, "since": int(params["since"]),
                "limit": int(params["limit"]), "kind_filter": [],
                "count": 1,
                "logs": [{"tick": 50, "kind": "decision",
                          "content": {"action": "wait", "via": "reflex"}}],
                "next_since": 51,
            }
        else:
            payload = {
                "agent_id": 7, "agent_name": "X", "current_tick": 100,
                "alive": True, "since": int(params["since"]),
                "limit": int(params["limit"]), "kind_filter": [],
                "count": 0, "logs": [], "next_since": 51,
            }
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    # Patch LogsPoller.run con il transport: piu' semplice = sostituisco
    # AsyncClient via monkeypatch sul modulo.
    import agora_agent_sdk.logs_poller as mod
    orig_client = mod.httpx.AsyncClient

    class _FakeClient(httpx.AsyncClient):
        def __init__(self, *a, **kw):
            kw["transport"] = transport
            super().__init__(*a, **kw)

    mod.httpx.AsyncClient = _FakeClient
    try:
        stop = asyncio.Event()
        poller = LogsPoller(
            server="http://fake", agent_id=7, interval_s=0.05,
        )
        task = asyncio.create_task(poller.run(stop))
        await asyncio.sleep(0.18)  # 3-4 poll cicli
        stop.set()
        await asyncio.wait_for(task, timeout=1.0)
    finally:
        mod.httpx.AsyncClient = orig_client

    assert len(calls) >= 2
    # Il primo poll ha since=0 (default), il secondo since=51 (next_since)
    assert calls[0]["since"] == "0"
    assert calls[1]["since"] == "51"
