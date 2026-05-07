"""WorldMirror: client-side mirror of the agora world state.

Updated by `apply_snapshot` (full override) and `apply_event` (delta).
Provides BFS pathfinding + walkability checks used by reflex/social.
"""
from __future__ import annotations

import base64
from collections import deque
from dataclasses import dataclass, field

from agora_core.grid import DIRECTIONS


# ============ walkable mask packing ============

def pack_walkable_mask(grid: list[list[bool]]) -> bytes:
    """Pack a 2D bool grid into a row-major LSB-first bitmap."""
    if not grid:
        return b""
    h = len(grid)
    w = len(grid[0])
    n_bits = w * h
    n_bytes = (n_bits + 7) // 8
    out = bytearray(n_bytes)
    for y in range(h):
        for x in range(w):
            if grid[y][x]:
                idx = y * w + x
                out[idx // 8] |= 1 << (idx % 8)
    return bytes(out)


def unpack_walkable_mask(mask_b64: str, w: int, h: int) -> bytes:
    """Decode a base64 raw walkable mask. Returns the raw bytes."""
    raw = base64.b64decode(mask_b64)
    expected = (w * h + 7) // 8
    if len(raw) != expected:
        raise ValueError(f"walkable_mask: expected {expected} bytes, got {len(raw)}")
    return raw


def mask_bit(raw: bytes, w: int, x: int, y: int) -> bool:
    """Read bit (x,y) from a packed mask. No bounds-check (caller's responsibility)."""
    idx = y * w + x
    return bool(raw[idx // 8] & (1 << (idx % 8)))


# ============ dataclasses ============

@dataclass
class AgentSnap:
    id: int
    name: str
    x: int
    y: int
    color: str
    sex: str
    alive: bool
    born_tick: int
    died_tick: int = 0
    sleep_streak: int = 0
    wait_streak: int = 0
    mother_id: int | None = None
    father_id: int | None = None


@dataclass
class StructureInfo:
    id: int
    x: int
    y: int
    type: str
    owner_id: int
    built_tick: int
    color: str = "#888"
    label: str = ""


@dataclass
class WorldEvent:
    id: int
    type: str
    x: int
    y: int
    radius: int
    started_tick: int
    ends_tick: int


@dataclass
class WorldMirror:
    world_w: int
    world_h: int
    walkable_mask: bytes  # raw packed bitmap
    self_agent_id: int = 0
    current_tick: int = 0
    agents: dict[int, AgentSnap] = field(default_factory=dict)
    structures: dict[tuple[int, int], StructureInfo] = field(default_factory=dict)
    resources: dict[tuple[int, int], tuple[str, int]] = field(default_factory=dict)
    resource_clusters: list[dict] = field(default_factory=list)
    storage_summary: dict[int, dict[str, int]] = field(default_factory=dict)
    events: dict[int, WorldEvent] = field(default_factory=dict)

    # ============ apply_snapshot ============

    def apply_snapshot(self, snap: dict) -> None:
        """Override the mirror with a fresh snapshot dict.

        snap is the dict-form of SnapshotMsg (after .model_dump() or raw JSON).
        """
        self.current_tick = int(snap["tick"])
        self.walkable_mask = unpack_walkable_mask(
            snap["walkable_mask"], self.world_w, self.world_h
        )
        self.agents = {}
        for a in snap.get("agents", []):
            self.agents[int(a["id"])] = AgentSnap(
                id=int(a["id"]),
                name=a["name"],
                x=int(a["x"]),
                y=int(a["y"]),
                color=a.get("color", "#888"),
                sex=a.get("sex", "F"),
                alive=bool(a.get("alive", True)),
                born_tick=int(a.get("born_tick", 0)),
                sleep_streak=int(a.get("sleep_streak", 0)),
                wait_streak=int(a.get("wait_streak", 0)),
                mother_id=a.get("mother_id"),
                father_id=a.get("father_id"),
            )
        self.structures = {}
        for s in snap.get("structures", []):
            info = StructureInfo(
                id=int(s["id"]),
                x=int(s["x"]),
                y=int(s["y"]),
                type=s["type"],
                owner_id=int(s["owner_id"]),
                built_tick=int(s.get("built_tick", 0)),
                color=s.get("color", "#888"),
                label=s.get("label", ""),
            )
            self.structures[(info.x, info.y)] = info
        self.resources = {}
        self.resource_clusters = list(snap.get("resource_clusters", []))
        for cluster in self.resource_clusters:
            rtype = cluster["type"]
            for tx, ty in cluster.get("tiles", []):
                self.resources[(int(tx), int(ty))] = (
                    rtype,
                    1,
                )
        self.storage_summary = {
            int(sid): {it: int(q) for it, q in items.items()}
            for sid, items in (snap.get("storage_summary") or {}).items()
        }
        self.events = {}
        for ev in snap.get("world_events", []):
            self.events[int(ev["id"])] = WorldEvent(
                id=int(ev["id"]),
                type=ev["type"],
                x=int(ev.get("x", 0)),
                y=int(ev.get("y", 0)),
                radius=int(ev.get("radius", 0)),
                started_tick=int(ev.get("started_tick", 0)),
                ends_tick=int(ev.get("ends_tick", 0)),
            )

    # ============ apply_event ============

    def apply_event(self, ev: dict) -> None:
        """Apply a delta event dict (EventMsg.model_dump()).

        Unknown kinds are ignored (forward-compat).
        """
        kind = ev.get("kind")
        tick = int(ev.get("tick", self.current_tick))
        if tick > self.current_tick:
            self.current_tick = tick
        handler = _EVENT_HANDLERS.get(kind)
        if handler is None:
            return
        handler(self, ev)

    # ============ walkability + pathfinding ============

    def is_walkable_terrain(self, x: int, y: int) -> bool:
        """Tile in-bounds and walkable per the static mask. Does NOT account for
        agents or structures occupying the tile."""
        if not (0 <= x < self.world_w and 0 <= y < self.world_h):
            return False
        return mask_bit(self.walkable_mask, self.world_w, x, y)

    def is_walkable(self, x: int, y: int) -> bool:
        """Walkable terrain AND not occupied by a live agent."""
        if not self.is_walkable_terrain(x, y):
            return False
        if self.is_occupied(x, y):
            return False
        return True

    def is_occupied(self, x: int, y: int) -> bool:
        return any(a.x == x and a.y == y and a.alive for a in self.agents.values())

    def find_path_step(self, sx: int, sy: int, tx: int, ty: int,
                       max_nodes: int = 256) -> str | None:
        """BFS from (sx,sy) to (tx,ty); return the direction name of the FIRST step.

        Ported from agora.agents.reflex.find_path_step (private repo).
        """
        if (sx, sy) == (tx, ty):
            return None
        queue: deque[tuple[int, int]] = deque([(sx, sy)])
        parent: dict[tuple[int, int], tuple[int, int] | None] = {(sx, sy): None}
        found = False
        while queue and len(parent) < max_nodes:
            x, y = queue.popleft()
            if (x, y) == (tx, ty):
                found = True
                break
            for _, (dx, dy) in DIRECTIONS.items():
                nx, ny = x + dx, y + dy
                if (nx, ny) in parent:
                    continue
                if not self.is_walkable_terrain(nx, ny):
                    continue
                if (nx, ny) != (tx, ty) and self.is_occupied(nx, ny):
                    continue
                parent[(nx, ny)] = (x, y)
                queue.append((nx, ny))
                if (nx, ny) == (tx, ty):
                    found = True
                    break
            if found:
                break
        if not found:
            return None
        cur = (tx, ty)
        while parent[cur] != (sx, sy):
            prev = parent[cur]
            if prev is None:
                return None
            cur = prev
        step_dx = cur[0] - sx
        step_dy = cur[1] - sy
        for name, (dx, dy) in DIRECTIONS.items():
            if (dx, dy) == (step_dx, step_dy):
                return name
        return None

    def nearest_resource(self, ax: int, ay: int, item_type: str) -> tuple[int, int] | None:
        """Find the closest known tile of `item_type` reachable from (ax,ay).
        Falls back to cluster centroids when no exact tile is known."""
        best: tuple[int, int] | None = None
        best_d = 10**9
        for (rx, ry), (rtype, rqty) in self.resources.items():
            if rtype != item_type or rqty <= 0:
                continue
            if not self.is_walkable_terrain(rx, ry):
                has_access = any(
                    self.is_walkable_terrain(rx + dx, ry + dy)
                    for dx, dy in ((0, 1), (0, -1), (1, 0), (-1, 0))
                )
                if not has_access:
                    continue
            d = max(abs(rx - ax), abs(ry - ay))
            if d < best_d:
                best_d = d
                best = (rx, ry)
        if best is not None:
            return best
        for cluster in self.resource_clusters:
            if cluster.get("type") != item_type:
                continue
            cx, cy = int(cluster["cx"]), int(cluster["cy"])
            d = max(abs(cx - ax), abs(cy - ay))
            if d < best_d:
                best_d = d
                best = (cx, cy)
        return best

    # ============ apply_perception (refresh nearby tiles) ============

    def apply_perception(self, perc: dict) -> None:
        """Apply incremental updates from a perception dict.

        - Updates `current_tick` and self-agent stats.
        - Refreshes `resources` for the perception's nearby_resources (raggio 3).
        - Refreshes `structures` for nearby_structures (in case server pushed
          them via perception but missed a delta).
        """
        self.current_tick = max(self.current_tick, int(perc.get("tick", 0)))
        sa = perc.get("agent_state") or {}
        if self.self_agent_id and self.self_agent_id in self.agents:
            agent = self.agents[self.self_agent_id]
            if "x" in sa:
                agent.x = int(sa["x"])
            if "y" in sa:
                agent.y = int(sa["y"])
            if "wait_streak" in sa:
                agent.wait_streak = int(sa["wait_streak"])
            if "sleep_streak" in sa:
                agent.sleep_streak = int(sa["sleep_streak"])
        ax = int(sa.get("x", 0))
        ay = int(sa.get("y", 0))
        nearby_set = {(int(r["x"]), int(r["y"])): (r["type"], int(r["qty"]))
                      for r in perc.get("nearby_resources", [])}
        for (rx, ry) in list(self.resources.keys()):
            if max(abs(rx - ax), abs(ry - ay)) <= 3 and (rx, ry) not in nearby_set:
                del self.resources[(rx, ry)]
        for (rx, ry), v in nearby_set.items():
            self.resources[(rx, ry)] = v


# ============ event handlers ============

def _on_tile_update(m: WorldMirror, ev: dict) -> None:
    x, y = int(ev["x"]), int(ev["y"])
    rtype = ev.get("resource_type")
    rqty = int(ev.get("resource_qty", 0))
    if rtype is None or rqty <= 0:
        m.resources.pop((x, y), None)
    else:
        m.resources[(x, y)] = (rtype, rqty)


def _on_structure_built(m: WorldMirror, ev: dict) -> None:
    info = StructureInfo(
        id=int(ev["structure_id"]),
        x=int(ev["x"]),
        y=int(ev["y"]),
        type=ev["structure_type"],
        owner_id=int(ev["owner_id"]),
        built_tick=int(ev.get("tick", m.current_tick)),
        color=ev.get("color", "#888"),
        label=ev.get("label", ""),
    )
    m.structures[(info.x, info.y)] = info


def _on_structure_destroyed(m: WorldMirror, ev: dict) -> None:
    sid = int(ev.get("structure_id", 0))
    pos = next((p for p, s in m.structures.items() if s.id == sid), None)
    if pos is not None:
        m.structures.pop(pos, None)


def _on_agent_born(m: WorldMirror, ev: dict) -> None:
    a = ev["agent"]
    m.agents[int(a["id"])] = AgentSnap(
        id=int(a["id"]),
        name=a["name"],
        x=int(a["x"]),
        y=int(a["y"]),
        color=a.get("color", "#888"),
        sex=a.get("sex", "F"),
        alive=bool(a.get("alive", True)),
        born_tick=int(a.get("born_tick", 0)),
        mother_id=a.get("mother_id"),
        father_id=a.get("father_id"),
    )


def _on_agent_died(m: WorldMirror, ev: dict) -> None:
    aid = int(ev["agent_id"])
    a = m.agents.get(aid)
    if a is not None:
        a.alive = False
        a.died_tick = int(ev.get("tick", m.current_tick))


def _on_agent_stats(m: WorldMirror, ev: dict) -> None:
    pass


def _on_agent_moved(m: WorldMirror, ev: dict) -> None:
    aid = int(ev["agent_id"])
    a = m.agents.get(aid)
    if a is not None and a.alive:
        a.x = int(ev["x"])
        a.y = int(ev["y"])


def _on_agent_action(m: WorldMirror, ev: dict) -> None:
    """Reuse path: server may broadcast `agent_action` (not `agent_moved`).
    Treat the embedded x,y as a position update."""
    aid = ev.get("agent_id")
    if aid is None:
        return
    a = m.agents.get(int(aid))
    if a is not None and a.alive and "x" in ev and "y" in ev:
        a.x = int(ev["x"])
        a.y = int(ev["y"])


def _on_storage_changed(m: WorldMirror, ev: dict) -> None:
    sid = int(ev["structure_id"])
    item = ev["item"]
    qty = int(ev["qty"])
    bucket = m.storage_summary.setdefault(sid, {})
    if qty <= 0:
        bucket.pop(item, None)
        if not bucket:
            m.storage_summary.pop(sid, None)
    else:
        bucket[item] = qty


def _on_world_event_started(m: WorldMirror, ev: dict) -> None:
    e = ev["event"]
    m.events[int(e["id"])] = WorldEvent(
        id=int(e["id"]),
        type=e["type"],
        x=int(e.get("x", 0)),
        y=int(e.get("y", 0)),
        radius=int(e.get("radius", 0)),
        started_tick=int(e.get("started_tick", m.current_tick)),
        ends_tick=int(e.get("ends_tick", 0)),
    )


def _on_world_event_ended(m: WorldMirror, ev: dict) -> None:
    eid = int(ev.get("event_id", 0))
    m.events.pop(eid, None)


def _on_relation_update(m: WorldMirror, ev: dict) -> None:
    pass


def _on_episodic_buffer_event(m: WorldMirror, ev: dict) -> None:
    pass


_EVENT_HANDLERS = {
    "tile_update": _on_tile_update,
    "structure_built": _on_structure_built,
    "structure_destroyed": _on_structure_destroyed,
    "agent_born": _on_agent_born,
    "agent_died": _on_agent_died,
    "agent_stats": _on_agent_stats,
    "agent_moved": _on_agent_moved,
    "agent_action": _on_agent_action,
    "storage_changed": _on_storage_changed,
    "world_event_started": _on_world_event_started,
    "world_event_ended": _on_world_event_ended,
    "relation_update": _on_relation_update,
    "dialogue_received": _on_episodic_buffer_event,
    "gift_received": _on_episodic_buffer_event,
    "loss": _on_episodic_buffer_event,
    "user_message": _on_episodic_buffer_event,
}
