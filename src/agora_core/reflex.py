"""Reflex layer: deterministic emergency rules. Ported from agora.agents.reflex.

Operates on a `WorldMirror` instead of `WorldRuntime`, but the decision logic
is identical. Returns a decision dict or None if no reflex applies.

Priority order (matches private repo):
  1. eat berry if hunger >= 60 and has berry
  2. gather berry on tile if hungry or low stock
  3. move toward nearest berry if hunger >= 65
  4. opportunistic gather on tile
  4b. craft tools if mats available and tool not owned
  4b2. step away from hut if surplus wood + already on/adjacent to hut
  4c. build hut if 5+ wood and no hut nearby (cap = max(4, 2*alive))
  4c1. withdraw need: low inventory + nearby storage with item
  4c2. build storage if 3w+2s and no storage within 8 tiles
  4d. deposit excess at storage
  4e. propose if M-F adjacent + mutual affinity >= 20 (low prob/tick)
  5. shelter pull at night (move toward nearest hut)
  6. rest if night + low energy + no hut
"""
from __future__ import annotations

import random as _random
from typing import TYPE_CHECKING

from agora_core.age import age_in_days, age_stage
from agora_core.daynight import is_night, time_of_day
from agora_core.grid import DIRECTIONS

if TYPE_CHECKING:
    from agora_core.world_mirror import WorldMirror, AgentSnap


EAT_HUNGER_TH = 60
SEEK_FOOD_HUNGER_TH = 65
NIGHT_REST_ENERGY_TH = 20

INV_TARGET = {
    "berry": 2,
    "wood": 6,
    "stone": 4,
    "iron_ore": 2,
}


def _cardinal_fallback(
    mirror: "WorldMirror",
    ax: int, ay: int, tx: int, ty: int,
    walkable_dirs: list[str],
) -> str | None:
    """When BFS fails, try a single cardinal step toward target if walkable."""
    dx = tx - ax
    dy = ty - ay
    walkable = set(walkable_dirs)
    if abs(dx) >= abs(dy):
        cands = [
            ("east" if dx > 0 else "west"),
            ("south" if dy > 0 else "north"),
        ]
    else:
        cands = [
            ("south" if dy > 0 else "north"),
            ("east" if dx > 0 else "west"),
        ]
    for c in cands:
        if c in walkable:
            return c
    return None


def try_reflex(
    mirror: "WorldMirror",
    agent: "AgentSnap",
    perception: dict,
    inventory: dict[str, int],
    aff_out: dict[int, int] | None = None,
    aff_in: dict[int, int] | None = None,
    *,
    sex: str = "F",
    born_tick: int = 0,
    hunger: int = 0,
    energy: int = 100,
) -> dict | None:
    """Emergency reflex. Returns a decision dict or None.

    `agent` is the AgentSnap of self in mirror. `sex/born_tick/hunger/energy`
    come from perception.agent_state (not present on AgentSnap).
    """
    walkable_dirs = perception.get("walkable_dirs") or []

    # 1. Eat if hungry and has berry
    if hunger >= EAT_HUNGER_TH and inventory.get("berry", 0) > 0:
        return {"action": "eat", "item": "berry",
                "thought": "I'm hungry, eating a berry"}

    here = perception.get("here_resource")

    # 2. Berry on tile + hungry or low stock
    if here is not None:
        h_type = here.get("type")
        if h_type == "berry" and (
            hunger >= SEEK_FOOD_HUNGER_TH
            or inventory.get("berry", 0) < INV_TARGET["berry"]
        ):
            return {"action": "gather", "thought": "picking a berry"}

    # 3. Hunger high → move to nearest berry
    if hunger >= SEEK_FOOD_HUNGER_TH:
        target = mirror.nearest_resource(agent.x, agent.y, "berry")
        if target is not None:
            d = mirror.find_path_step(agent.x, agent.y, target[0], target[1])
            if d is not None:
                return {"action": "move", "direction": d,
                        "thought": "hungry, heading to the berries"}

    # 4. Opportunistic gather: useful resource on this tile
    if here is not None:
        h_type = here.get("type")
        target_qty = INV_TARGET.get(h_type, 0)
        if h_type and inventory.get(h_type, 0) < target_qty:
            return {"action": "gather",
                    "thought": f"picking up {h_type} while passing"}

    # 4b. Craft tools
    have_axe = inventory.get("axe", 0) > 0
    have_pick = inventory.get("pickaxe", 0) > 0
    if (not have_axe and inventory.get("wood", 0) >= 2
            and inventory.get("stone", 0) >= 1):
        return {"action": "craft", "recipe": "axe", "thought": "making an axe"}
    if (not have_pick and inventory.get("wood", 0) >= 1
            and inventory.get("stone", 0) >= 2):
        return {"action": "craft", "recipe": "pickaxe", "thought": "making a pickaxe"}

    phase = time_of_day(perception.get("tick", mirror.current_tick))["phase"]

    # 4b2. Step away from hut if surplus wood + adjacent/on hut
    wood_qty = inventory.get("wood", 0)
    stone_target = INV_TARGET.get("stone", 0)
    stone_done = inventory.get("stone", 0) >= stone_target
    n_huts = sum(1 for s in mirror.structures.values() if s.type == "hut")
    n_alive = sum(1 for a in mirror.agents.values() if a.alive)
    enough_huts = n_huts >= n_alive
    if wood_qty >= 10 and stone_done and not is_night(phase) and not enough_huts:
        nearest_h, nearest_d = None, 10**9
        for (sx, sy), info in mirror.structures.items():
            if info.type != "hut":
                continue
            d = max(abs(sx - agent.x), abs(sy - agent.y))
            if d < nearest_d:
                nearest_d = d
                nearest_h = (sx, sy)
        if nearest_h is not None and nearest_d <= 1:
            dx = agent.x - nearest_h[0]
            dy = agent.y - nearest_h[1]
            walkable = set(walkable_dirs)
            cands = (
                [("east" if dx >= 0 else "west"),
                 ("south" if dy >= 0 else "north")]
                if abs(dx) >= abs(dy) else
                [("south" if dy >= 0 else "north"),
                 ("east" if dx >= 0 else "west")]
            )
            for c in cands:
                if c in walkable:
                    return {"action": "move", "direction": c,
                            "thought": "stepping away to make room for a new hut"}

    # 4c. Build hut
    n_alive_agents = sum(1 for a in mirror.agents.values() if a.alive)
    n_huts = sum(1 for info in mirror.structures.values() if info.type == "hut")
    hut_cap = max(4, n_alive_agents * 2)
    if inventory.get("wood", 0) >= 5 and n_huts < hut_cap:
        on_resource = perception.get("here_resource") is not None
        on_struct = perception.get("here_structure") is not None
        if not on_resource and not on_struct:
            min_d = 10**9
            for (sx, sy), info in mirror.structures.items():
                if info.type != "hut":
                    continue
                d = max(abs(sx - agent.x), abs(sy - agent.y))
                if d < min_d:
                    min_d = d
            should_build = (
                (wood_qty >= 10 and min_d >= 5)
                or (not is_night(phase) and min_d >= 4)
            )
            if should_build:
                return {"action": "build", "structure": "hut",
                        "thought": "building a hut here"}

    # 4c1. Withdraw need
    INV_MIN = {"berry": 5, "wood": 2, "stone": 1}
    needed: list[tuple[str, int]] = []
    for it, low in INV_MIN.items():
        have = inventory.get(it, 0)
        if have < low:
            needed.append((it, low - have + 2))
    if needed:
        candidates = []
        for (sx, sy), info in mirror.structures.items():
            if info.type != "storage":
                continue
            cache = mirror.storage_summary.get(info.id, {})
            for it, qty_need in needed:
                if cache.get(it, 0) >= qty_need:
                    d = max(abs(sx - agent.x), abs(sy - agent.y))
                    candidates.append((d, sx, sy, it, qty_need))
                    break
        if candidates:
            candidates.sort()
            d_st, bx, by, item_w, qty_w = candidates[0]
            here_struct = perception.get("here_structure")
            if (here_struct and here_struct.get("type") == "storage"
                    and (agent.x, agent.y) == (bx, by)):
                return {"action": "withdraw", "item": item_w, "qty": qty_w,
                        "thought": f"taking {qty_w} {item_w} from storage"}
            if d_st <= 12:
                d = mirror.find_path_step(agent.x, agent.y, bx, by)
                if d is None:
                    d = _cardinal_fallback(mirror, agent.x, agent.y, bx, by, walkable_dirs)
                if d is not None:
                    return {"action": "move", "direction": d,
                            "thought": f"going to storage to fetch {item_w}"}

    # 4c2. Build storage
    storages = [(sx, sy) for (sx, sy), info in mirror.structures.items()
                if info.type == "storage"]
    if (inventory.get("wood", 0) >= 3 and inventory.get("stone", 0) >= 2
            and perception.get("here_resource") is None
            and perception.get("here_structure") is None):
        nearest_st_d = min(
            (max(abs(sx - agent.x), abs(sy - agent.y)) for sx, sy in storages),
            default=10**9,
        )
        if nearest_st_d > 8:
            return {"action": "build", "structure": "storage",
                    "thought": "building a storage for shared items"}

    # 4d. Deposit excess
    INV_KEEP = {"wood": 8, "stone": 4, "berry": 25, "iron_ore": 2}
    excess: list[tuple[str, int]] = []
    for it, keep in INV_KEEP.items():
        have = inventory.get(it, 0)
        if have > keep:
            excess.append((it, have - keep))
    if excess and storages:
        here_struct = perception.get("here_structure")
        if here_struct and here_struct.get("type") == "storage":
            it_dep, qty_dep = max(excess, key=lambda x: x[1])
            return {"action": "deposit", "item": it_dep, "qty": qty_dep,
                    "thought": f"depositing {qty_dep} {it_dep} to storage"}
        best_st, best_d = None, 10**9
        for sx, sy in storages:
            d = max(abs(sx - agent.x), abs(sy - agent.y))
            if d < best_d:
                best_d = d
                best_st = (sx, sy)
        if best_st is not None and best_d <= 10:
            d = mirror.find_path_step(agent.x, agent.y, best_st[0], best_st[1])
            if d is None:
                d = _cardinal_fallback(mirror, agent.x, agent.y,
                                       best_st[0], best_st[1], walkable_dirs)
            if d is not None:
                return {"action": "move", "direction": d,
                        "thought": "going to storage to drop surplus"}

    # 4e. Propose
    if aff_out is not None and aff_in is not None:
        nearby_a = perception.get("nearby_agents") or []
        days = age_in_days(born_tick, mirror.current_tick)
        my_stage = age_stage(days)
        if my_stage in ("young", "adult"):
            for other in nearby_a:
                ox, oy = other.get("x", 0), other.get("y", 0)
                d = max(abs(ox - agent.x), abs(oy - agent.y))
                if d > 3 or other.get("sex") == sex:
                    continue
                tid = int(other["id"])
                a_to_b = aff_out.get(tid, 0)
                b_to_a = aff_in.get(tid, 0)
                if a_to_b < 20 or b_to_a < 20:
                    continue
                partner = mirror.agents.get(tid)
                if partner is None:
                    continue
                p_days = age_in_days(partner.born_tick, mirror.current_tick)
                if age_stage(p_days) not in ("young", "adult"):
                    continue
                rng = _random.Random(agent.id * 31 + mirror.current_tick + tid)
                if rng.random() < (1 / 120):
                    return {"action": "propose", "target_id": tid,
                            "thought": f"I want a child with {other.get('name', '')}".strip()}

    # 5. Shelter pull at night
    if is_night(phase):
        best_hut, best_d = None, 10**9
        for (sx, sy), info in mirror.structures.items():
            if info.type != "hut":
                continue
            d = max(abs(sx - agent.x), abs(sy - agent.y))
            if d < best_d:
                best_d = d
                best_hut = (sx, sy)
        if best_hut is not None:
            if best_d == 0:
                return {"action": "wait", "thought": "resting inside the hut"}
            d = mirror.find_path_step(agent.x, agent.y, best_hut[0], best_hut[1])
            if d is None:
                d = _cardinal_fallback(mirror, agent.x, agent.y,
                                       best_hut[0], best_hut[1], walkable_dirs)
            if d is not None:
                return {"action": "move", "direction": d,
                        "thought": "going to shelter for the night"}

    # 6. Night + low energy + no hut: rest anyway
    if is_night(phase) and energy < NIGHT_REST_ENERGY_TH:
        return {"action": "wait", "thought": "resting, it's night and I'm tired"}

    return None
