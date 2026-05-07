"""Reflex parity tests: known scenarios → expected decision."""
from agora_core.reflex import try_reflex, INV_TARGET
from agora_core.world_mirror import (
    WorldMirror, AgentSnap, StructureInfo, pack_walkable_mask,
)


def _basic_mirror(w=16, h=16) -> WorldMirror:
    grid = [[True] * w for _ in range(h)]
    return WorldMirror(world_w=w, world_h=h, walkable_mask=pack_walkable_mask(grid))


def _self_agent(mirror: WorldMirror, x=5, y=5) -> AgentSnap:
    a = AgentSnap(id=99, name="Self", x=x, y=y, color="#fff", sex="F",
                  alive=True, born_tick=0)
    mirror.self_agent_id = a.id
    mirror.agents[a.id] = a
    return a


def _basic_perception(walkable=("north", "south", "east", "west"), tick=300, **extra):
    base = {
        "tick": tick,
        "here_resource": None,
        "here_structure": None,
        "nearby_agents": [],
        "nearby_resources": [],
        "walkable_dirs": list(walkable),
    }
    base.update(extra)
    return base


def test_reflex_eats_when_hungry_with_berry():
    m = _basic_mirror()
    agent = _self_agent(m)
    out = try_reflex(m, agent, _basic_perception(), {"berry": 1},
                     hunger=70, energy=80)
    assert out["action"] == "eat" and out["item"] == "berry"


def test_reflex_no_eat_if_not_hungry():
    m = _basic_mirror()
    agent = _self_agent(m)
    inv = {"berry": 5, "wood": 6, "stone": 4, "iron_ore": 2,
           "axe": 1, "pickaxe": 1}
    out = try_reflex(m, agent, _basic_perception(), inv,
                     hunger=30, energy=80)
    assert out is None or out["action"] != "eat"


def test_reflex_gathers_berry_on_tile_when_low():
    m = _basic_mirror()
    agent = _self_agent(m)
    perc = _basic_perception(here_resource={"type": "berry", "qty": 1})
    out = try_reflex(m, agent, perc, {"berry": 0}, hunger=10, energy=80)
    assert out["action"] == "gather"


def test_reflex_moves_to_nearest_berry():
    m = _basic_mirror()
    agent = _self_agent(m, x=0, y=0)
    m.resources[(5, 0)] = ("berry", 1)
    perc = _basic_perception()
    out = try_reflex(m, agent, perc, {}, hunger=80, energy=80)
    assert out["action"] == "move"
    assert out["direction"] == "east"


def test_reflex_crafts_axe_when_no_axe_yet():
    m = _basic_mirror()
    agent = _self_agent(m)
    inv = {"wood": 2, "stone": 1}
    out = try_reflex(m, agent, _basic_perception(), inv, hunger=10, energy=80)
    assert out == {"action": "craft", "recipe": "axe", "thought": "making an axe"}


def test_reflex_skips_axe_when_already_owned():
    m = _basic_mirror()
    agent = _self_agent(m)
    inv = {"wood": 2, "stone": 1, "axe": 1}
    out = try_reflex(m, agent, _basic_perception(), inv, hunger=10, energy=80)
    assert out is None or out["recipe"] != "axe"


def test_reflex_builds_hut_when_5_wood_and_no_hut_nearby():
    m = _basic_mirror()
    agent = _self_agent(m, x=8, y=8)
    inv = {"wood": 6}
    perc = _basic_perception(tick=60)  # day
    out = try_reflex(m, agent, perc, inv, hunger=10, energy=80)
    assert out is not None
    assert out["action"] == "build"
    assert out["structure"] == "hut"


def test_reflex_moves_toward_hut_at_night():
    m = _basic_mirror()
    agent = _self_agent(m, x=0, y=0)
    m.structures[(5, 0)] = StructureInfo(
        id=1, x=5, y=0, type="hut", owner_id=1, built_tick=0,
    )
    perc = _basic_perception(tick=int(0.8 * 600))  # night
    inv = {}
    out = try_reflex(m, agent, perc, inv, hunger=10, energy=80)
    assert out["action"] == "move"
    assert out["direction"] == "east"


def test_reflex_rests_when_already_in_hut_at_night():
    m = _basic_mirror()
    agent = _self_agent(m, x=5, y=5)
    m.structures[(5, 5)] = StructureInfo(
        id=1, x=5, y=5, type="hut", owner_id=1, built_tick=0,
    )
    perc = _basic_perception(tick=int(0.8 * 600), here_structure={"type": "hut"})
    out = try_reflex(m, agent, perc, {}, hunger=10, energy=80)
    assert out["action"] == "wait"


def test_reflex_rests_when_low_energy_at_night_no_hut():
    m = _basic_mirror()
    agent = _self_agent(m)
    perc = _basic_perception(tick=int(0.8 * 600))
    out = try_reflex(m, agent, perc, {}, hunger=10, energy=10)
    assert out["action"] == "wait"


def test_reflex_returns_none_when_idle():
    m = _basic_mirror()
    agent = _self_agent(m)
    # Inventory tuned to NOT trigger any reflex branch:
    # - berry/wood/stone at INV_MIN exactly → no withdraw
    # - wood<3 → no build storage; wood<5 → no build hut
    # - axe + pickaxe owned → no craft
    # - no excess to deposit, no nearby agents, daytime → no propose/shelter
    inv = {"berry": 5, "wood": 2, "stone": 1, "axe": 1, "pickaxe": 1}
    perc = _basic_perception(tick=60)
    out = try_reflex(m, agent, perc, inv, hunger=10, energy=80)
    assert out is None
