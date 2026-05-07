import asyncio
import base64
import pytest

from agora_core.world_mirror import (
    WorldMirror, AgentSnap, pack_walkable_mask,
)
from agora_agent_sdk.brain import Brain
from agora_agent_sdk.llm import NoOpLLM


def _mirror() -> WorldMirror:
    grid = [[True] * 16 for _ in range(16)]
    return WorldMirror(world_w=16, world_h=16, walkable_mask=pack_walkable_mask(grid))


def _setup_brain(*, with_self=True) -> tuple[Brain, WorldMirror]:
    m = _mirror()
    if with_self:
        m.self_agent_id = 99
        m.agents[99] = AgentSnap(id=99, name="Self", x=5, y=5, color="#fff",
                                 sex="F", alive=True, born_tick=0)
    brain = Brain(
        mirror=m, llm=NoOpLLM(), agent_id=99, agent_name="Self",
        sex="F", color="#fff", personality_seed="test seed",
        llm_decide_interval=10, ring_buffer_size=30, policy=None,
    )
    return brain, m


def _basic_perception(**extra) -> dict:
    base = {
        "tick": 5,
        "agent_state": {
            "x": 5, "y": 5, "hp": 100, "energy": 80, "mood": 60, "hunger": 30,
            "personality_current": "x", "born_tick": 0,
            "wait_streak": 0, "sleep_streak": 0,
            "inventory": {},
        },
        "terrain_here": "grass",
        "visible_around": "(0,0)=grass",
        "here_resource": None,
        "here_structure": None,
        "nearby_agents": [],
        "nearby_resources": [],
        "nearby_structures": [],
        "walkable_dirs": ["north", "south", "east", "west"],
        "relations": {},
        "relations_inbound": {},
        "family": {"mother": None, "father": None, "children": []},
        "recent_dialogues": [],
        "world_events": [],
    }
    base.update(extra)
    return base


@pytest.mark.asyncio
async def test_decide_reflex_eats_when_hungry():
    brain, _ = _setup_brain()
    perc = _basic_perception(
        agent_state={"x": 5, "y": 5, "hp": 100, "energy": 80, "mood": 60,
                     "hunger": 70, "personality_current": "x", "born_tick": 0,
                     "wait_streak": 0, "sleep_streak": 0,
                     "inventory": {"berry": 1}},
    )
    out = await brain.decide(perc)
    assert out["action"] == "eat"
    assert out["item"] == "berry"
    assert out["decided_via"] == "reflex"


@pytest.mark.asyncio
async def test_decide_falls_back_to_wander_when_idle():
    brain, _ = _setup_brain()
    out = await brain.decide(_basic_perception())
    assert out["action"] in ("wander", "wait")
    assert out["decided_via"] in ("auto_cooldown", "social", "reflex")


@pytest.mark.asyncio
async def test_decide_logs_to_ring_buffer():
    brain, _ = _setup_brain()
    await brain.decide(_basic_perception())
    assert len(brain.episodic) == 1
    assert brain.episodic[-1]["kind"] == "decision"


@pytest.mark.asyncio
async def test_decide_dispatches_llm_after_interval():
    brain, m = _setup_brain()
    brain.last_llm_decide_tick = -100
    m.current_tick = 50
    perc = _basic_perception(tick=50)
    perc["agent_state"]["x"] = 5
    perc["agent_state"]["y"] = 5
    await brain.decide(perc)
    assert brain.pending_llm_task is not None
    await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_push_dialogue_received_appends_episodic_and_ring():
    brain, _ = _setup_brain()
    brain.push_dialogue_received(
        from_id=2, from_name="Niko", content="hello there friend",
        tick=10,
    )
    assert any(e["kind"] == "dialogue_received" for e in brain.episodic)
    assert "hello there friend" in brain.recent_lines_by_agent[2]


@pytest.mark.asyncio
async def test_validate_pre_send_short_talk_to_wait():
    brain, _ = _setup_brain()
    out = brain._validate_pre_send({"action": "talk", "target_id": 2,
                                    "content": "Hi", "thought": "x"})
    assert out["action"] == "wait"


@pytest.mark.asyncio
async def test_validate_pre_send_dedup_talk_to_wait():
    brain, _ = _setup_brain()
    brain.recent_lines_by_agent[brain.agent_id] = ["already said this once"]
    out = brain._validate_pre_send({"action": "talk", "target_id": 2,
                                    "content": "already said this once",
                                    "thought": "x"})
    assert out["action"] == "wait"


@pytest.mark.asyncio
async def test_validate_pre_send_blocked_move_to_wander():
    brain, _ = _setup_brain()
    brain.last_walkable_dirs = ["east"]
    out = brain._validate_pre_send({"action": "move", "direction": "north"})
    assert out["action"] == "wander"
