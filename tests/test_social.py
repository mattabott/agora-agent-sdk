from agora_core.social import social_navigate
from agora_core.world_mirror import (
    WorldMirror, AgentSnap, pack_walkable_mask,
)


def _mirror() -> WorldMirror:
    grid = [[True] * 16 for _ in range(16)]
    return WorldMirror(world_w=16, world_h=16, walkable_mask=pack_walkable_mask(grid))


def _self(mirror: WorldMirror, x=5, y=5) -> AgentSnap:
    a = AgentSnap(id=99, name="Self", x=x, y=y, color="#fff", sex="F",
                  alive=True, born_tick=0)
    mirror.self_agent_id = a.id
    mirror.agents[a.id] = a
    return a


def _other(mirror: WorldMirror, oid=1, x=10, y=5, name="Other") -> AgentSnap:
    a = AgentSnap(id=oid, name=name, x=x, y=y, color="#fff", sex="M",
                  alive=True, born_tick=0)
    mirror.agents[a.id] = a
    return a


def test_social_returns_none_with_no_agents():
    m = _mirror()
    s = _self(m)
    out = social_navigate(m, s, {"walkable_dirs": ["north"], "nearby_agents": []},
                          current_tick=0, inventory={}, next_talk_line_ready=False,
                          wait_streak=0)
    assert out is None


def test_social_talks_when_partner_adjacent_and_line_ready():
    m = _mirror()
    s = _self(m, x=5, y=5)
    _other(m, oid=2, x=6, y=5, name="Niko")
    perc = {"walkable_dirs": ["north"],
            "nearby_agents": [{"id": 2, "name": "Niko", "x": 6, "y": 5, "sex": "M"}]}
    out = social_navigate(m, s, perc, current_tick=0,
                          inventory={"wood": 10, "stone": 10},
                          next_talk_line_ready=True, wait_streak=0)
    assert out["action"] == "talk"
    assert out["target_id"] == 2
    assert out["content"] == "<<USE_NEXT_TALK_LINE>>"


def test_social_waits_when_partner_adjacent_and_no_line():
    m = _mirror()
    s = _self(m, x=5, y=5)
    _other(m, oid=2, x=6, y=5)
    perc = {"walkable_dirs": ["north"],
            "nearby_agents": [{"id": 2, "name": "Niko", "x": 6, "y": 5, "sex": "M"}]}
    out = social_navigate(m, s, perc, current_tick=0,
                          inventory={"wood": 10, "stone": 10},
                          next_talk_line_ready=False, wait_streak=0)
    assert out["action"] == "wait"


def test_social_breaks_sticky_talk_after_4_waits():
    m = _mirror()
    s = _self(m, x=5, y=5)
    _other(m, oid=2, x=6, y=5)
    perc = {"walkable_dirs": ["north"],
            "nearby_agents": [{"id": 2, "name": "Niko", "x": 6, "y": 5, "sex": "M"}]}
    out = social_navigate(m, s, perc, current_tick=0,
                          inventory={"wood": 10, "stone": 10},
                          next_talk_line_ready=False, wait_streak=4)
    assert out is None


def test_social_pathfinds_to_visible_partner():
    m = _mirror()
    s = _self(m, x=0, y=0)
    perc = {"walkable_dirs": ["north", "south", "east", "west"],
            "nearby_agents": [{"id": 2, "name": "Far", "x": 5, "y": 0, "sex": "M"}]}
    _other(m, oid=2, x=5, y=0, name="Far")
    out = social_navigate(m, s, perc, current_tick=0,
                          inventory={"wood": 10, "stone": 10},
                          next_talk_line_ready=False, wait_streak=0)
    assert out["action"] == "move"
    assert out["direction"] == "east"


def test_social_pursues_resource_when_alone_and_low():
    m = _mirror()
    s = _self(m, x=0, y=0)
    m.resources[(3, 0)] = ("wood", 1)
    out = social_navigate(m, s, {"walkable_dirs": ["east"], "nearby_agents": []},
                          current_tick=0, inventory={"wood": 0},
                          next_talk_line_ready=False, wait_streak=0)
    assert out["action"] == "move"
    assert out["direction"] == "east"


def test_social_long_range_gravitate_when_no_resources():
    m = _mirror()
    s = _self(m, x=0, y=0)
    _other(m, oid=2, x=8, y=0)
    out = social_navigate(m, s, {"walkable_dirs": ["east"], "nearby_agents": []},
                          current_tick=0, inventory={"wood": 10, "stone": 10},
                          next_talk_line_ready=False, wait_streak=0)
    assert out["action"] == "move"
    assert out["direction"] == "east"
    assert "seeking company" in out["thought"]
