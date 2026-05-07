"""Pydantic schemas for the agora-agent-sdk WS protocol.

Reference: docs/specs/2026-05-07-agora-agent-sdk-design.md §5
"""
from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


ACTION_SCHEMA_VERSION = 1


# ============ HTTP join ============

class JoinRequest(BaseModel):
    name: str = Field(min_length=1, max_length=32, pattern=r"^[A-Za-z][A-Za-z0-9_-]*$")
    personality_seed: str = Field(min_length=1, max_length=500)
    sex: Literal["F", "M"]
    color: str | None = Field(default=None, pattern=r"^#[0-9a-fA-F]{6}$")
    action_schema_version: int = ACTION_SCHEMA_VERSION
    client_version: str = "agora-agent-sdk/0.1.0"


class JoinResponse(BaseModel):
    agent_id: int
    token: str
    world_seed: int
    tick_ms: int
    world_w: int
    world_h: int
    action_schema_version: int


class JoinError(BaseModel):
    error: str
    suggestions: list[str] | None = None
    server_schema: int | None = None
    client_schema: int | None = None
    min_supported: int | None = None
    field: str | None = None
    reason: str | None = None


# ============ WS structures ============

class AgentSnapshot(BaseModel):
    id: int
    name: str
    x: int
    y: int
    color: str
    sex: str
    alive: bool
    born_tick: int
    sleep_streak: int = 0
    wait_streak: int = 0
    mother_id: int | None = None
    father_id: int | None = None


class StructureSnapshot(BaseModel):
    id: int
    x: int
    y: int
    type: str
    owner_id: int
    built_tick: int
    color: str = "#888"
    label: str = ""


class ResourceClusterSnapshot(BaseModel):
    type: str
    cx: int
    cy: int
    total_qty: int
    tiles: list[tuple[int, int]]


class WorldEventSnapshot(BaseModel):
    id: int
    type: str
    x: int
    y: int
    radius: int = 0
    started_tick: int
    ends_tick: int


# ============ Server → Client ============

class SnapshotMsg(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["snapshot"] = "snapshot"
    tick: int
    walkable_mask: str  # base64 raw
    agents: list[AgentSnapshot]
    structures: list[StructureSnapshot]
    resource_clusters: list[ResourceClusterSnapshot]
    storage_summary: dict[str, dict[str, int]]
    world_events: list[WorldEventSnapshot]


class AgentSelfState(BaseModel):
    x: int
    y: int
    hp: int
    energy: int
    mood: int
    hunger: int
    personality_current: str
    current_goal: str = ""
    sleep_streak: int = 0
    wait_streak: int = 0
    born_tick: int
    mother_id: int | None = None
    father_id: int | None = None
    last_thought: str = ""
    last_action: str = ""
    inventory: dict[str, int] = Field(default_factory=dict)


class NearbyAgent(BaseModel):
    id: int
    name: str
    x: int
    y: int
    sex: str


class NearbyResource(BaseModel):
    x: int
    y: int
    type: str
    qty: int


class NearbyStructure(BaseModel):
    x: int
    y: int
    type: str
    owner_id: int | None = None
    label: str = ""


class HereResource(BaseModel):
    type: str
    qty: int


class HereStructure(BaseModel):
    type: str
    label: str = ""


class FamilyEntry(BaseModel):
    id: int
    name: str
    alive: bool


class Family(BaseModel):
    mother: FamilyEntry | None = None
    father: FamilyEntry | None = None
    children: list[FamilyEntry] = Field(default_factory=list)


class RecentDialogue(BaseModel):
    tick: int
    from_id: int
    from_name: str
    content: str


class PerceptionMsg(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["perception"] = "perception"
    tick: int
    agent_state: AgentSelfState
    terrain_here: str
    visible_around: str
    here_resource: HereResource | None = None
    here_structure: HereStructure | None = None
    nearby_agents: list[NearbyAgent] = Field(default_factory=list)
    nearby_resources: list[NearbyResource] = Field(default_factory=list)
    nearby_structures: list[NearbyStructure] = Field(default_factory=list)
    walkable_dirs: list[str] = Field(default_factory=list)
    relations: dict[str, int] = Field(default_factory=dict)
    relations_inbound: dict[str, int] = Field(default_factory=dict)
    family: Family = Field(default_factory=Family)
    recent_dialogues: list[RecentDialogue] = Field(default_factory=list)
    world_events: list[WorldEventSnapshot] = Field(default_factory=list)


class EventMsg(BaseModel):
    """A delta event. The `kind` field selects the payload schema (see design §5.2.3)."""
    model_config = ConfigDict(extra="allow")
    type: Literal["event"] = "event"
    kind: str
    tick: int


class ResultMsg(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["result"] = "result"
    tick_ack: int
    action: str
    ok: bool
    reason: str | None = None


class PingMsg(BaseModel):
    type: Literal["ping"] = "ping"
    ts: float


# ============ Client → Server ============

class PongMsg(BaseModel):
    type: Literal["pong"] = "pong"
    ts: float


class ActionMsg(BaseModel):
    model_config = ConfigDict(extra="allow")
    type: Literal["action"] = "action"
    tick_ack: int
    action: str
    direction: str | None = None
    target_id: int | None = None
    content: str | None = None
    item: str | None = None
    qty: int | None = None
    recipe: str | None = None
    structure: str | None = None
    thought: str = ""
    decided_via: str = ""


class RequestSnapshotMsg(BaseModel):
    type: Literal["request_snapshot"] = "request_snapshot"


# ============ Action schema validation ============

VALID_ACTIONS: set[str] = {
    "move", "wait", "wander", "note", "talk", "gather",
    "eat", "craft", "build", "give", "deposit", "withdraw", "propose",
}

REQUIRED_FIELDS_BY_ACTION: dict[str, set[str]] = {
    "move": {"direction"},
    "wait": set(),
    "wander": set(),
    "note": {"content"},
    "talk": {"target_id", "content"},
    "gather": set(),
    "eat": {"item"},
    "craft": {"recipe"},
    "build": {"structure"},
    "give": {"target_id", "item", "qty"},
    "deposit": {"item", "qty"},
    "withdraw": {"item", "qty"},
    "propose": {"target_id"},
}


def validate_action_dict(d: dict) -> tuple[bool, str]:
    """Return (ok, reason). reason is empty when ok."""
    a = d.get("action")
    if a not in VALID_ACTIONS:
        return False, f"unknown_action:{a}"
    required = REQUIRED_FIELDS_BY_ACTION[a]
    for f in required:
        v = d.get(f)
        if v is None or (isinstance(v, str) and not v.strip()):
            return False, f"missing_field:{f}"
    if a == "move" and d.get("direction") not in ("north", "south", "east", "west"):
        return False, "invalid_direction"
    if a == "talk":
        content = (d.get("content") or "").strip()
        if not (3 <= len(content) <= 280):
            return False, "talk_content_length"
    return True, ""
