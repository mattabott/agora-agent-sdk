import base64

from agora_core.world_mirror import (
    WorldMirror, AgentSnap, pack_walkable_mask,
)


def _empty_mirror(w=8, h=8) -> WorldMirror:
    grid = [[True] * w for _ in range(h)]
    raw = pack_walkable_mask(grid)
    return WorldMirror(world_w=w, world_h=h, walkable_mask=raw)


def _b64_walkable(w=8, h=8) -> str:
    grid = [[True] * w for _ in range(h)]
    return base64.b64encode(pack_walkable_mask(grid)).decode("ascii")


def test_apply_snapshot_loads_agents_and_structures():
    m = _empty_mirror()
    snap = {
        "type": "snapshot",
        "tick": 100,
        "walkable_mask": _b64_walkable(),
        "agents": [
            {"id": 1, "name": "A", "x": 0, "y": 0, "color": "#fff",
             "sex": "F", "alive": True, "born_tick": 0,
             "sleep_streak": 0, "wait_streak": 0,
             "mother_id": None, "father_id": None},
        ],
        "structures": [
            {"id": 1, "x": 5, "y": 5, "type": "hut", "owner_id": 1,
             "built_tick": 10, "color": "#a06a3c", "label": "Hut"},
        ],
        "resource_clusters": [
            {"type": "wood", "cx": 2, "cy": 3, "total_qty": 4,
             "tiles": [[2, 3], [3, 3]]}
        ],
        "storage_summary": {"3": {"berry": 12, "wood": 5}},
        "world_events": [
            {"id": 99, "type": "rain", "x": 0, "y": 0, "radius": 4,
             "started_tick": 90, "ends_tick": 200}
        ],
    }
    m.apply_snapshot(snap)
    assert m.current_tick == 100
    assert m.agents[1].name == "A"
    assert m.structures[(5, 5)].type == "hut"
    assert (2, 3) in m.resources
    assert m.storage_summary[3]["berry"] == 12
    assert m.events[99].type == "rain"


def test_apply_event_tile_update_set_and_clear():
    m = _empty_mirror()
    m.apply_event({"type": "event", "kind": "tile_update", "tick": 1,
                   "x": 4, "y": 5, "resource_type": "wood", "resource_qty": 3})
    assert m.resources[(4, 5)] == ("wood", 3)
    m.apply_event({"type": "event", "kind": "tile_update", "tick": 2,
                   "x": 4, "y": 5, "resource_type": None, "resource_qty": 0})
    assert (4, 5) not in m.resources


def test_apply_event_structure_built_and_destroyed():
    m = _empty_mirror()
    m.apply_event({"type": "event", "kind": "structure_built", "tick": 5,
                   "structure_id": 7, "x": 1, "y": 2, "structure_type": "hut",
                   "owner_id": 3, "color": "#a06a3c", "label": "Hut"})
    assert m.structures[(1, 2)].id == 7
    m.apply_event({"type": "event", "kind": "structure_destroyed", "tick": 6,
                   "structure_id": 7, "x": 1, "y": 2})
    assert (1, 2) not in m.structures


def test_apply_event_agent_born_and_died():
    m = _empty_mirror()
    m.apply_event({"type": "event", "kind": "agent_born", "tick": 5,
                   "agent": {"id": 9, "name": "Born", "x": 3, "y": 3,
                             "color": "#fff", "sex": "M", "alive": True,
                             "born_tick": 5}})
    assert 9 in m.agents and m.agents[9].alive
    m.apply_event({"type": "event", "kind": "agent_died", "tick": 10,
                   "agent_id": 9})
    assert not m.agents[9].alive
    assert m.agents[9].died_tick == 10


def test_apply_event_agent_action_updates_position():
    m = _empty_mirror()
    m.agents[1] = AgentSnap(id=1, name="X", x=0, y=0, color="#fff",
                            sex="F", alive=True, born_tick=0)
    m.apply_event({"type": "event", "kind": "agent_action", "tick": 1,
                   "agent_id": 1, "x": 4, "y": 6})
    assert m.agents[1].x == 4 and m.agents[1].y == 6


def test_apply_event_storage_changed_set_and_zero():
    m = _empty_mirror()
    m.apply_event({"type": "event", "kind": "storage_changed", "tick": 1,
                   "structure_id": 5, "item": "wood", "qty": 8})
    assert m.storage_summary[5]["wood"] == 8
    m.apply_event({"type": "event", "kind": "storage_changed", "tick": 2,
                   "structure_id": 5, "item": "wood", "qty": 0})
    assert 5 not in m.storage_summary  # bucket empty -> removed


def test_apply_event_world_event_lifecycle():
    m = _empty_mirror()
    m.apply_event({"type": "event", "kind": "world_event_started", "tick": 1,
                   "event": {"id": 7, "type": "fire", "x": 5, "y": 5,
                             "radius": 3, "started_tick": 1, "ends_tick": 100}})
    assert m.events[7].type == "fire"
    m.apply_event({"type": "event", "kind": "world_event_ended", "tick": 2,
                   "event_id": 7, "reason": "rain"})
    assert 7 not in m.events


def test_apply_event_unknown_kind_noop():
    m = _empty_mirror()
    m.apply_event({"type": "event", "kind": "future_kind", "tick": 1})
    assert m.current_tick == 1
