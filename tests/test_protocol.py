import pytest
from pydantic import ValidationError

from agora_core.protocol import (
    ACTION_SCHEMA_VERSION,
    ActionMsg,
    AgentSelfState,
    AgentSnapshot,
    EventMsg,
    JoinRequest,
    JoinResponse,
    PerceptionMsg,
    PingMsg,
    PongMsg,
    RequestSnapshotMsg,
    ResultMsg,
    SnapshotMsg,
    StructureSnapshot,
    WorldEventSnapshot,
    validate_action_dict,
)


def test_action_schema_version_is_1():
    assert ACTION_SCHEMA_VERSION == 1


def test_join_request_valid():
    req = JoinRequest(
        name="Maya", personality_seed="curious one", sex="F", color="#7fa9d4"
    )
    assert req.action_schema_version == 1
    assert req.client_version.startswith("agora-agent-sdk/")


def test_join_request_rejects_bad_name():
    with pytest.raises(ValidationError):
        JoinRequest(name="123Bad", personality_seed="x", sex="F")


def test_join_request_rejects_bad_color():
    with pytest.raises(ValidationError):
        JoinRequest(name="Maya", personality_seed="x", sex="F", color="bad")


def test_join_response_minimal():
    r = JoinResponse(agent_id=5, token="t", world_seed=42, tick_ms=1000,
                     world_w=64, world_h=64, action_schema_version=1)
    assert r.agent_id == 5


def test_snapshot_round_trip():
    snap = SnapshotMsg(
        tick=1,
        walkable_mask="AAAAAA==",
        agents=[AgentSnapshot(id=1, name="A", x=0, y=0, color="#fff", sex="F",
                              alive=True, born_tick=0)],
        structures=[StructureSnapshot(id=1, x=2, y=3, type="hut", owner_id=1, built_tick=0)],
        resource_clusters=[],
        storage_summary={},
        world_events=[],
    )
    raw = snap.model_dump()
    again = SnapshotMsg.model_validate(raw)
    assert again.tick == 1
    assert again.agents[0].name == "A"


def test_perception_round_trip():
    p = PerceptionMsg(
        tick=10,
        agent_state=AgentSelfState(
            x=5, y=5, hp=100, energy=80, mood=60, hunger=20,
            personality_current="...", born_tick=0,
        ),
        terrain_here="grass",
        visible_around="(0,0)=grass",
    )
    raw = p.model_dump()
    again = PerceptionMsg.model_validate(raw)
    assert again.agent_state.x == 5
    assert again.tick == 10


def test_event_msg_extra_payload_preserved():
    ev = EventMsg.model_validate(
        {"type": "event", "kind": "tile_update", "tick": 1,
         "x": 5, "y": 6, "resource_type": "wood", "resource_qty": 0}
    )
    assert ev.kind == "tile_update"
    # extra field accessible via dump
    assert ev.model_dump()["x"] == 5


def test_action_msg_minimal():
    a = ActionMsg(tick_ack=10, action="wait")
    assert a.action == "wait"
    assert a.thought == ""


def test_action_msg_talk_full():
    a = ActionMsg(tick_ack=10, action="talk", target_id=2, content="hey there",
                  thought="greeting")
    raw = a.model_dump(exclude_none=True)
    assert raw["target_id"] == 2
    assert raw["content"] == "hey there"


def test_ping_pong():
    p = PingMsg(ts=1.0)
    pong = PongMsg(ts=p.ts)
    assert pong.ts == 1.0


def test_request_snapshot():
    m = RequestSnapshotMsg()
    assert m.type == "request_snapshot"


def test_result_msg_failure():
    r = ResultMsg.model_validate(
        {"type": "result", "tick_ack": 10, "action": "build", "ok": False,
         "reason": "tile_occupied", "structure_type": "hut"}
    )
    assert not r.ok
    assert r.reason == "tile_occupied"


# === validate_action_dict ===

def test_validate_action_unknown():
    ok, reason = validate_action_dict({"action": "ascend"})
    assert not ok
    assert reason.startswith("unknown_action")


def test_validate_action_move_missing_direction():
    ok, reason = validate_action_dict({"action": "move"})
    assert not ok
    assert "missing_field:direction" in reason


def test_validate_action_move_bad_direction():
    ok, reason = validate_action_dict({"action": "move", "direction": "diagonal"})
    assert not ok
    assert reason == "invalid_direction"


def test_validate_action_move_ok():
    ok, _ = validate_action_dict({"action": "move", "direction": "north"})
    assert ok


def test_validate_action_talk_too_short():
    ok, reason = validate_action_dict({"action": "talk", "target_id": 2, "content": "Hi"})
    assert not ok
    assert reason == "talk_content_length"


def test_validate_action_talk_ok():
    ok, _ = validate_action_dict(
        {"action": "talk", "target_id": 2, "content": "Hey how are you"}
    )
    assert ok


def test_validate_action_wait_no_required():
    ok, _ = validate_action_dict({"action": "wait"})
    assert ok
