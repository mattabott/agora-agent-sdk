"""In-process FastAPI mock of the agora server.

Used only by E2E tests. It scripts a fixed sequence of perceptions and
records the actions received from the client.
"""
from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass, field

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from agora_core.protocol import ACTION_SCHEMA_VERSION
from agora_core.world_mirror import pack_walkable_mask


@dataclass
class MockState:
    next_id: int = 1
    tokens: dict[int, str] = field(default_factory=dict)
    name_to_id: dict[str, int] = field(default_factory=dict)
    actions_received: list[dict] = field(default_factory=list)
    ws_connected: asyncio.Event = field(default_factory=asyncio.Event)
    perception_script: list[dict] = field(default_factory=list)
    extra_messages: list[dict] = field(default_factory=list)


def _walkable_b64(w=8, h=8) -> str:
    return base64.b64encode(pack_walkable_mask([[True]*w for _ in range(h)])).decode()


def make_mock_app() -> tuple[FastAPI, MockState]:
    state = MockState()
    app = FastAPI()

    @app.post("/api/agents/join")
    async def join(req: dict) -> JSONResponse:
        if int(req.get("action_schema_version", 0)) != ACTION_SCHEMA_VERSION:
            return JSONResponse(status_code=426, content={
                "error": "schema_mismatch",
                "server_schema": ACTION_SCHEMA_VERSION,
                "client_schema": int(req.get("action_schema_version", 0)),
                "min_supported": ACTION_SCHEMA_VERSION,
            })
        name = req["name"]
        if name in state.name_to_id:
            return JSONResponse(status_code=409, content={
                "error": "name_taken",
                "suggestions": [f"{name}2", f"{name}A"],
            })
        agent_id = state.next_id
        state.next_id += 1
        state.name_to_id[name] = agent_id
        state.tokens[agent_id] = f"tk-{agent_id}"
        return JSONResponse(status_code=200, content={
            "agent_id": agent_id, "token": state.tokens[agent_id],
            "world_seed": 42, "tick_ms": 1000,
            "world_w": 8, "world_h": 8,
            "action_schema_version": ACTION_SCHEMA_VERSION,
        })

    @app.websocket("/ws/agents/{agent_id}")
    async def ws_endpoint(ws: WebSocket, agent_id: int, token: str = ""):
        if state.tokens.get(agent_id) != token:
            await ws.close(code=4401)
            return
        await ws.accept()
        state.ws_connected.set()
        await ws.send_text(json.dumps({
            "type": "snapshot", "tick": 0,
            "walkable_mask": _walkable_b64(),
            "agents": [{"id": agent_id, "name": "Self",
                         "x": 4, "y": 4, "color": "#fff", "sex": "F",
                         "alive": True, "born_tick": 0}],
            "structures": [], "resource_clusters": [],
            "storage_summary": {}, "world_events": [],
        }))
        def _snapshot_msg(tick: int = 0) -> str:
            return json.dumps({
                "type": "snapshot", "tick": tick,
                "walkable_mask": _walkable_b64(),
                "agents": [{"id": agent_id, "name": "Self",
                             "x": 4, "y": 4, "color": "#fff",
                             "sex": "F", "alive": True, "born_tick": 0}],
                "structures": [], "resource_clusters": [],
                "storage_summary": {}, "world_events": [],
            })

        try:
            # Drain any pre-loop messages (e.g. request_snapshot sent by client
            # immediately after connect, before the first perception is ready).
            import asyncio as _asyncio
            try:
                while True:
                    raw = await _asyncio.wait_for(ws.receive_text(), timeout=0.05)
                    msg = json.loads(raw)
                    if msg.get("type") == "request_snapshot":
                        await ws.send_text(_snapshot_msg(0))
                    # pong and other pre-loop messages are silently consumed
            except _asyncio.TimeoutError:
                pass

            for perc in state.perception_script:
                await ws.send_text(json.dumps(perc))
                while True:
                    raw = await ws.receive_text()
                    msg = json.loads(raw)
                    if msg.get("type") == "request_snapshot":
                        await ws.send_text(_snapshot_msg(int(perc.get("tick", 0))))
                        continue
                    if msg.get("type") == "pong":
                        continue
                    state.actions_received.append(msg)
                    await ws.send_text(json.dumps({
                        "type": "result", "tick_ack": msg.get("tick_ack", 0),
                        "action": msg.get("action", "wait"), "ok": True,
                    }))
                    break
            for extra in state.extra_messages:
                await ws.send_text(json.dumps(extra))
        except WebSocketDisconnect:
            return

    return app, state
