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
