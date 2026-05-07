"""WS loop tests using an in-memory WS double."""
import asyncio
import json

import pytest

from agora_agent_sdk.client import AgentDiedExit, AgoraClient
from agora_agent_sdk.brain import Brain
from agora_agent_sdk.llm import NoOpLLM
from agora_core.world_mirror import WorldMirror, pack_walkable_mask


class _FakeWS:
    """In-memory WS double: queues inbound and records outbound."""

    def __init__(self, inbound: list[dict]):
        self._inbound = list(inbound)
        self.outbound: list[dict] = []
        self._closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self._closed = True

    async def send(self, raw: str) -> None:
        self.outbound.append(json.loads(raw))

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._inbound:
            raise StopAsyncIteration
        return json.dumps(self._inbound.pop(0))


def _ws_factory(messages: list[dict]):
    fake = _FakeWS(messages)

    def factory(url: str):
        return fake

    factory.fake = fake
    return factory


def _b64_walkable(w=8, h=8) -> str:
    import base64
    return base64.b64encode(pack_walkable_mask([[True]*w for _ in range(h)])).decode()


def _make_client(messages: list[dict], agent_id=99):
    grid = [[True]*8 for _ in range(8)]
    mirror = WorldMirror(world_w=8, world_h=8,
                         walkable_mask=pack_walkable_mask(grid))
    brain = Brain(mirror=mirror, llm=NoOpLLM(), agent_id=agent_id,
                  agent_name="Self", sex="F", color="#fff",
                  personality_seed="x")
    factory = _ws_factory(messages)
    client = AgoraClient(
        "http://mock", agent_id=agent_id, token="tk", world_w=8, world_h=8,
        brain=brain, ws_factory=factory,
    )
    return client, factory.fake


@pytest.mark.asyncio
async def test_run_once_applies_snapshot_and_responds_to_perception():
    snap = {
        "type": "snapshot", "tick": 1, "walkable_mask": _b64_walkable(),
        "agents": [{"id": 99, "name": "Self", "x": 5, "y": 5, "color": "#fff",
                    "sex": "F", "alive": True, "born_tick": 0}],
        "structures": [], "resource_clusters": [],
        "storage_summary": {}, "world_events": [],
    }
    perc = {
        "type": "perception", "tick": 2,
        "agent_state": {"x": 5, "y": 5, "hp": 100, "energy": 80, "mood": 60,
                        "hunger": 30, "personality_current": "x", "born_tick": 0,
                        "wait_streak": 0, "sleep_streak": 0, "inventory": {}},
        "terrain_here": "grass", "visible_around": "",
        "here_resource": None, "here_structure": None,
        "nearby_agents": [], "nearby_resources": [], "nearby_structures": [],
        "walkable_dirs": ["north"],
        "relations": {}, "relations_inbound": {},
        "family": {"mother": None, "father": None, "children": []},
        "recent_dialogues": [], "world_events": [],
    }
    client, fake = _make_client([snap, perc])
    await client._run_once()
    types = [m["type"] for m in fake.outbound]
    assert types[0] == "request_snapshot"
    assert "action" in types


@pytest.mark.asyncio
async def test_run_once_pongs_to_ping():
    ping = {"type": "ping", "ts": 12345.0}
    client, fake = _make_client([ping])
    await client._run_once()
    pongs = [m for m in fake.outbound if m["type"] == "pong"]
    assert pongs and pongs[-1]["ts"] == 12345.0


@pytest.mark.asyncio
async def test_run_once_self_died_raises():
    died = {"type": "event", "kind": "agent_died", "tick": 100, "agent_id": 99}
    client, _ = _make_client([died])
    with pytest.raises(AgentDiedExit):
        await client._run_once()


@pytest.mark.asyncio
async def test_run_once_other_died_does_not_raise():
    died = {"type": "event", "kind": "agent_died", "tick": 100, "agent_id": 5}
    client, _ = _make_client([died])
    await client._run_once()


@pytest.mark.asyncio
async def test_run_once_requests_snapshot_on_tick_gap():
    snap = {
        "type": "snapshot", "tick": 1, "walkable_mask": _b64_walkable(),
        "agents": [], "structures": [], "resource_clusters": [],
        "storage_summary": {}, "world_events": [],
    }
    far_perc = {
        "type": "perception", "tick": 200,
        "agent_state": {"x": 0, "y": 0, "hp": 100, "energy": 0, "mood": 0,
                        "hunger": 0, "personality_current": "x", "born_tick": 0,
                        "wait_streak": 0, "sleep_streak": 0, "inventory": {}},
        "terrain_here": "grass", "visible_around": "",
        "here_resource": None, "here_structure": None,
        "nearby_agents": [], "nearby_resources": [], "nearby_structures": [],
        "walkable_dirs": [],
        "relations": {}, "relations_inbound": {},
        "family": {"mother": None, "father": None, "children": []},
        "recent_dialogues": [], "world_events": [],
    }
    client, fake = _make_client([snap, far_perc])
    await client._run_once()
    request_snaps = [m for m in fake.outbound if m["type"] == "request_snapshot"]
    assert len(request_snaps) >= 2
