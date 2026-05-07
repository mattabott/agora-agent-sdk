"""Social navigation: gravitate toward nearby agents and talk when adjacent.

Ported from agora.agents.brain._social_navigate. Operates on WorldMirror.
"""
from __future__ import annotations

import random as _r
from typing import TYPE_CHECKING

from agora_core.reflex import INV_TARGET, _cardinal_fallback

if TYPE_CHECKING:
    from agora_core.world_mirror import WorldMirror, AgentSnap


def social_navigate(
    mirror: "WorldMirror",
    agent: "AgentSnap",
    perception: dict,
    current_tick: int,
    inventory: dict[str, int],
    next_talk_line_ready: bool,
    wait_streak: int,
) -> dict | None:
    """Society building. Returns decision dict or None.

    `next_talk_line_ready`: whether the brain has an LLM-generated line in
    `next_talk_line` ready to be spoken. If False and partner is adjacent, we
    return wait("(waiting for words)") unless wait_streak >= 4 (then give up
    sticky talk).
    """
    walkable_dirs = perception.get("walkable_dirs") or []
    nearby = perception.get("nearby_agents") or []
    inv = inventory or {}

    # Detect critical material need (wood / stone below INV_TARGET)
    needed_resource = None
    for r in ("wood", "stone"):
        if inv.get(r, 0) < INV_TARGET.get(r, 0):
            needed_resource = r
            break

    # 1. Closest agent
    if nearby:
        closest = min(
            nearby,
            key=lambda o: max(abs(o.get("x", 0) - agent.x), abs(o.get("y", 0) - agent.y)),
        )
        target_id = int(closest["id"])
        name = closest.get("name", "")
        cx, cy = closest.get("x", 0), closest.get("y", 0)
        d = max(abs(cx - agent.x), abs(cy - agent.y))
        rng = _r.Random(agent.id * 31 + current_tick * 7 + target_id)

        # Errand override: 70% gathering when near and material missing
        if d <= 2 and needed_resource is not None and rng.random() < 0.7:
            tgt = mirror.nearest_resource(agent.x, agent.y, needed_resource)
            if tgt is not None:
                if max(abs(tgt[0] - agent.x), abs(tgt[1] - agent.y)) <= 1:
                    return {"action": "gather",
                            "thought": f"gathering {needed_resource}"}
                pf_target = (tgt[0], tgt[1])
                if not mirror.is_walkable_terrain(tgt[0], tgt[1]):
                    best, bd = None, 10**9
                    for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                        nx, ny = tgt[0] + dx, tgt[1] + dy
                        if (mirror.is_walkable_terrain(nx, ny)
                                and not mirror.is_occupied(nx, ny)):
                            dd = max(abs(nx - agent.x), abs(ny - agent.y))
                            if dd < bd:
                                bd = dd
                                best = (nx, ny)
                    if best is not None:
                        pf_target = best
                step = mirror.find_path_step(agent.x, agent.y,
                                             pf_target[0], pf_target[1])
                if step is None:
                    step = _cardinal_fallback(mirror, agent.x, agent.y,
                                              pf_target[0], pf_target[1],
                                              walkable_dirs)
                if step is not None:
                    return {"action": "move", "direction": step,
                            "thought": f"going to get {needed_resource}"}

        if d <= 2:
            if next_talk_line_ready:
                return {"action": "talk", "target_id": target_id,
                        "content": "<<USE_NEXT_TALK_LINE>>",
                        "thought": f"talking with {name}".strip()}
            if wait_streak >= 4:
                return None
            return {"action": "wait", "thought": "(waiting for words)"}

        # In view but not adjacent → pathfind
        direction = mirror.find_path_step(agent.x, agent.y, cx, cy)
        if direction is None:
            direction = _cardinal_fallback(mirror, agent.x, agent.y, cx, cy,
                                           walkable_dirs)
        if direction is not None:
            return {"action": "move", "direction": direction,
                    "thought": f"heading to {name}".strip()}

    # 2. No one nearby: pursue missing resource (wood/stone)
    for resource in ("wood", "stone"):
        if inventory.get(resource, 0) < INV_TARGET.get(resource, 0):
            tgt = mirror.nearest_resource(agent.x, agent.y, resource)
            if tgt is not None:
                d_now = max(abs(tgt[0] - agent.x), abs(tgt[1] - agent.y))
                if d_now <= 1:
                    return {"action": "gather",
                            "thought": f"gathering {resource}"}
                pf_target = (tgt[0], tgt[1])
                if not mirror.is_walkable_terrain(tgt[0], tgt[1]):
                    best, best_d = None, 10**9
                    for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0)):
                        nx, ny = tgt[0] + dx, tgt[1] + dy
                        if (mirror.is_walkable_terrain(nx, ny)
                                and not mirror.is_occupied(nx, ny)):
                            dd = max(abs(nx - agent.x), abs(ny - agent.y))
                            if dd < best_d:
                                best_d = dd
                                best = (nx, ny)
                    if best is not None:
                        pf_target = best
                d = mirror.find_path_step(agent.x, agent.y,
                                          pf_target[0], pf_target[1])
                if d is None:
                    d = _cardinal_fallback(mirror, agent.x, agent.y,
                                           pf_target[0], pf_target[1],
                                           walkable_dirs)
                if d is not None:
                    return {"action": "move", "direction": d,
                            "thought": f"looking for {resource}"}

    # 3. Long-range gravitate to closest live agent
    others = [a for a in mirror.agents.values()
              if a.id != agent.id and a.alive]
    if not others:
        return None
    closest_global = min(
        others, key=lambda a: max(abs(a.x - agent.x), abs(a.y - agent.y)),
    )
    direction = mirror.find_path_step(agent.x, agent.y,
                                      closest_global.x, closest_global.y)
    if direction is None:
        direction = _cardinal_fallback(mirror, agent.x, agent.y,
                                       closest_global.x, closest_global.y,
                                       walkable_dirs)
    if direction is not None:
        return {"action": "move", "direction": direction,
                "thought": "seeking company"}
    return None
