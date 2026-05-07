"""End-to-end: AgoraClient + Brain against the mock server.

Uses starlette TestClient for the WS leg (sync). The HTTP join uses httpx
with ASGITransport. We script perceptions and assert the actions returned.
"""
import asyncio
import json
import threading

import httpx
import pytest
from starlette.testclient import TestClient

from agora_agent_sdk.brain import Brain
from agora_agent_sdk.client import http_join
from agora_agent_sdk.llm import NoOpLLM
from agora_core.world_mirror import WorldMirror, pack_walkable_mask
from tests.mock_server import make_mock_app


def _hungry_perception(agent_id: int) -> dict:
    return {
        "type": "perception", "tick": 5,
        "agent_state": {"x": 4, "y": 4, "hp": 100, "energy": 80, "mood": 60,
                        "hunger": 75, "personality_current": "x",
                        "born_tick": 0, "wait_streak": 0, "sleep_streak": 0,
                        "inventory": {"berry": 1}},
        "terrain_here": "grass", "visible_around": "",
        "here_resource": None, "here_structure": None,
        "nearby_agents": [], "nearby_resources": [], "nearby_structures": [],
        "walkable_dirs": ["north", "south", "east", "west"],
        "relations": {}, "relations_inbound": {},
        "family": {"mother": None, "father": None, "children": []},
        "recent_dialogues": [], "world_events": [],
    }


@pytest.mark.asyncio
async def test_e2e_hungry_eats_berry():
    app, state = make_mock_app()
    transport = httpx.ASGITransport(app=app)
    # 1. Join via mock HTTP
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://mock") as http_client:
        r = await http_client.post("/api/agents/join", json={
            "name": "Maya", "personality_seed": "x", "sex": "F",
            "action_schema_version": 1, "client_version": "t/0",
        })
        assert r.status_code == 200
        data = r.json()
    state.perception_script = [_hungry_perception(data["agent_id"])]

    # 2. Open WS via starlette TestClient (sync)
    received_actions = []
    with TestClient(app) as tc:
        with tc.websocket_connect(
            f"/ws/agents/{data['agent_id']}?token={data['token']}"
        ) as ws:
            # snapshot
            snap = ws.receive_json()
            assert snap["type"] == "snapshot"
            # client would normally send request_snapshot; mock continues anyway
            ws.send_json({"type": "request_snapshot"})
            # mock responds with a snapshot too
            second_snap = ws.receive_json()
            assert second_snap["type"] == "snapshot"
            # Now perception
            perc = ws.receive_json()
            assert perc["type"] == "perception"
            # Build a brain locally and decide
            mirror = WorldMirror(
                world_w=8, world_h=8,
                walkable_mask=pack_walkable_mask([[True]*8 for _ in range(8)]),
            )
            mirror.apply_snapshot(snap)
            brain = Brain(
                mirror=mirror, llm=NoOpLLM(),
                agent_id=data["agent_id"], agent_name="Maya", sex="F",
                color="#fff", personality_seed="x",
            )
            decision = await brain.decide(perc)
            ws.send_json({
                "type": "action", "tick_ack": perc["tick"],
                **decision,
            })
            result = ws.receive_json()
            received_actions.append(decision)

    assert received_actions[0]["action"] == "eat"
    assert received_actions[0]["item"] == "berry"


@pytest.mark.asyncio
async def test_e2e_join_then_agent_died_exits():
    app, state = make_mock_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://mock") as http_client:
        r = await http_client.post("/api/agents/join", json={
            "name": "Doomed", "personality_seed": "x", "sex": "F",
            "action_schema_version": 1, "client_version": "t/0",
        })
        data = r.json()
    state.perception_script = []  # no perceptions
    state.extra_messages = [
        {"type": "event", "kind": "agent_died",
         "tick": 100, "agent_id": data["agent_id"]}
    ]
    with TestClient(app) as tc:
        with tc.websocket_connect(
            f"/ws/agents/{data['agent_id']}?token={data['token']}"
        ) as ws:
            snap = ws.receive_json()
            assert snap["type"] == "snapshot"
            died_event = ws.receive_json()
            assert died_event["kind"] == "agent_died"
            assert died_event["agent_id"] == data["agent_id"]
