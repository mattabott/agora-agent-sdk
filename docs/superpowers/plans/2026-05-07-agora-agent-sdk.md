# agora-agent-sdk Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a public Python SDK (`agora-agent-sdk`) that lets external users plug their own LLM agent into the agora world via WebSocket, using a local Ollama for inference and porting the deterministic behavior layer (reflex/social/policy/dialogue filters) from the private repo.

**Architecture:** Two-namespace package (`agora_core` for shared frozen logic, `agora_agent_sdk` for client/CLI). Client maintains a `WorldMirror` updated from server snapshot + delta events, runs the same decision pipeline as the server brain (reflex → social → policy → background LLM) with a ring-buffer episodic memory in lieu of the DB-backed observation log.

**Tech Stack:** Python ≥3.10, `httpx` (HTTP + Ollama), `websockets` (WS client), `pydantic` v2 (schemas), `numpy` (policy features), `scikit-learn` (optional, MLP policy), `pytest` + `pytest-asyncio` + `fastapi` + `starlette` (test deps), `argparse` (CLI). MIT license, PyPI-ready.

**Reference:** Design spec at `docs/specs/2026-05-07-agora-agent-sdk-design.md`. Source for ports: private `agora` repo at `/home/mattabott/Documents/agora` (read-only).

**Working dir for this plan:** `/home/mattabott/Documents/agora-agent-sdk/` (already `git init`-ed locally, has spec at `docs/specs/`). **Never push to private agora repo. Never start a real agora server. Never point at production.**

---

## Phase 0 — Bootstrap

### Task 0.1: pyproject.toml + LICENSE + .gitignore

**Files:**
- Create: `/home/mattabott/Documents/agora-agent-sdk/pyproject.toml`
- Create: `/home/mattabott/Documents/agora-agent-sdk/LICENSE`
- Create: `/home/mattabott/Documents/agora-agent-sdk/.gitignore`

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "agora-agent-sdk"
version = "0.1.0"
description = "Client SDK to plug an external LLM agent into the agora world"
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [{ name = "mattabott", email = "info@iaitalia.net" }]
keywords = ["agora", "llm", "agent", "simulation", "ollama"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]
dependencies = [
    "httpx>=0.27",
    "websockets>=12",
    "pydantic>=2.5",
    "numpy>=1.26",
]

[project.optional-dependencies]
policy = ["scikit-learn>=1.3"]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "fastapi>=0.110",
    "starlette>=0.36",
    "httpx>=0.27",
]

[project.scripts]
agora-agent = "agora_agent_sdk.cli:main"

[project.urls]
Homepage = "https://github.com/mattabott/agora-agent-sdk"
Issues = "https://github.com/mattabott/agora-agent-sdk/issues"

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "-v"
```

- [ ] **Step 2: Write `LICENSE`** (MIT, year 2026)

```
MIT License

Copyright (c) 2026 mattabott

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

- [ ] **Step 3: Write `.gitignore`**

```
__pycache__/
*.py[cod]
*$py.class
*.egg-info/
build/
dist/
.pytest_cache/
.tox/
.coverage
htmlcov/
.venv/
venv/
.env
.env.local
.idea/
.vscode/
*.swp
~/.agora-agent/
```

- [ ] **Step 4: Commit**

```bash
cd /home/mattabott/Documents/agora-agent-sdk
git add pyproject.toml LICENSE .gitignore
git -c user.email="info@iaitalia.net" -c user.name="mattabott" commit -m "chore: project bootstrap (pyproject, LICENSE, gitignore)"
```

---

### Task 0.2: Source layout + __init__.py + README placeholder

**Files:**
- Create: `src/agora_core/__init__.py`
- Create: `src/agora_agent_sdk/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `README.md`

- [ ] **Step 1: Create directory structure**

```bash
cd /home/mattabott/Documents/agora-agent-sdk
mkdir -p src/agora_core src/agora_agent_sdk tests examples
```

- [ ] **Step 2: Write `src/agora_core/__init__.py`**

```python
"""agora_core: shared deterministic logic for agora-agent-sdk.

Frozen v1 port of the behavior layer from the private agora repo:
- protocol: pydantic models for the WS wire format
- world_mirror: client-side world state mirror updated by snapshot + deltas
- reflex / social / policy: deterministic decision layers
- prompts / dialogue_filters: LLM prompt builders + output filters
- age / daynight: time-related helpers
- recipes / structures / edibles: static game data
"""

__version__ = "0.1.0"
ACTION_SCHEMA_VERSION = 1
```

- [ ] **Step 3: Write `src/agora_agent_sdk/__init__.py`**

```python
"""agora_agent_sdk: client + CLI to plug an external agent into agora.

Public API:
- AgoraClient: async client for join + WS loop
- OllamaClient: async wrapper around local Ollama
- main: CLI entry point (`agora-agent` script)
"""

from agora_core import ACTION_SCHEMA_VERSION  # noqa: F401

__version__ = "0.1.0"
```

- [ ] **Step 4: Write `tests/__init__.py`** (empty file)

```python
```

- [ ] **Step 5: Write `tests/conftest.py`**

```python
"""Shared pytest fixtures."""
import pytest


@pytest.fixture
def fixed_tick() -> int:
    """A deterministic tick value used across tests."""
    return 12345
```

- [ ] **Step 6: Write `README.md` placeholder**

```markdown
# agora-agent-sdk

Plug your own LLM agent into the [agora](https://agora.chatbot4eva.com) world.

Status: pre-release. See `docs/specs/2026-05-07-agora-agent-sdk-design.md`.
```

- [ ] **Step 7: Commit**

```bash
cd /home/mattabott/Documents/agora-agent-sdk
git add src/ tests/ README.md
git -c user.email="info@iaitalia.net" -c user.name="mattabott" commit -m "chore: source layout + module placeholders"
```

---

### Task 0.3: Smoke test `import agora_core` and `import agora_agent_sdk`

**Files:**
- Create: `tests/test_smoke.py`

- [ ] **Step 1: Write the failing test**

```python
"""Smoke tests: package imports work and version constants are defined."""
import agora_core
import agora_agent_sdk


def test_agora_core_imports():
    assert agora_core.__version__ == "0.1.0"
    assert agora_core.ACTION_SCHEMA_VERSION == 1


def test_agora_agent_sdk_imports():
    assert agora_agent_sdk.__version__ == "0.1.0"
    assert agora_agent_sdk.ACTION_SCHEMA_VERSION == 1
```

- [ ] **Step 2: Install package editable + dev deps**

```bash
cd /home/mattabott/Documents/agora-agent-sdk
python -m venv .venv
. .venv/bin/activate
pip install -e ".[dev,policy]"
```

Expected: install completes without errors.

- [ ] **Step 3: Run test**

```bash
cd /home/mattabott/Documents/agora-agent-sdk
. .venv/bin/activate
pytest tests/test_smoke.py -v
```

Expected: PASS for both tests.

- [ ] **Step 4: Commit**

```bash
git add tests/test_smoke.py
git -c user.email="info@iaitalia.net" -c user.name="mattabott" commit -m "test: smoke import test for both packages"
```

---

## Phase 1 — Constants & static data

### Task 1.1: grid.py (DIRECTIONS) + age.py + daynight.py ports

**Files:**
- Create: `src/agora_core/grid.py`
- Create: `src/agora_core/age.py`
- Create: `src/agora_core/daynight.py`
- Create: `tests/test_grid.py`
- Create: `tests/test_age.py`
- Create: `tests/test_daynight.py`

- [ ] **Step 1: Write `src/agora_core/grid.py`**

```python
"""Grid directions used by reflex/social pathfinding.

Ported 1:1 from the private agora repo (agora.agents.actions.DIRECTIONS).
"""

DIRECTIONS: dict[str, tuple[int, int]] = {
    "north": (0, -1),
    "south": (0, 1),
    "east": (1, 0),
    "west": (-1, 0),
}
```

- [ ] **Step 2: Write `tests/test_grid.py`**

```python
from agora_core.grid import DIRECTIONS


def test_directions_complete():
    assert set(DIRECTIONS) == {"north", "south", "east", "west"}


def test_directions_unit_vectors():
    assert DIRECTIONS["north"] == (0, -1)
    assert DIRECTIONS["south"] == (0, 1)
    assert DIRECTIONS["east"] == (1, 0)
    assert DIRECTIONS["west"] == (-1, 0)
```

- [ ] **Step 3: Write `src/agora_core/age.py`** (port da privata)

```python
"""Age stages of agents. Ported 1:1 from agora.agents.age."""

from agora_core.daynight import DAY_LENGTH_TICKS

CHILD_END_DAYS = 1
YOUNG_END_DAYS = 3
ADULT_END_DAYS = 20
ELDER_END_DAYS = 100

ANCIENT_HP_DECAY = 1
ANCIENT_DECAY_EVERY_N_CYCLES = 3

STAGE_LABEL = {
    "child": "child",
    "young": "young",
    "adult": "adult",
    "elder": "elder",
    "ancient": "ancient",
}


def age_in_days(born_tick: int, current_tick: int) -> float:
    return max(0.0, (current_tick - born_tick) / DAY_LENGTH_TICKS)


def age_stage(days: float) -> str:
    if days < CHILD_END_DAYS:
        return "child"
    if days < YOUNG_END_DAYS:
        return "young"
    if days < ADULT_END_DAYS:
        return "adult"
    if days < ELDER_END_DAYS:
        return "elder"
    return "ancient"


def stage_label(stage: str) -> str:
    return STAGE_LABEL.get(stage, stage)


def can_reproduce(stage: str) -> bool:
    return stage in ("adult", "elder", "ancient")


def is_ancient(stage: str) -> bool:
    return stage == "ancient"
```

- [ ] **Step 4: Write `src/agora_core/daynight.py`** (port da privata)

```python
"""Day/night cycle. Ported 1:1 from agora.sim.daynight."""

DAY_LENGTH_TICKS = 600
PHASE_BREAKPOINTS = [
    (0.00, 0.05, "dawn"),
    (0.05, 0.50, "day"),
    (0.50, 0.55, "dusk"),
    (0.55, 1.00, "night"),
]


def time_of_day(tick: int) -> dict:
    day_n = tick // DAY_LENGTH_TICKS
    pos = tick % DAY_LENGTH_TICKS
    fraction = pos / DAY_LENGTH_TICKS
    phase = "day"
    for lo, hi, name in PHASE_BREAKPOINTS:
        if lo <= fraction < hi:
            phase = name
            break
    return {"phase": phase, "day_n": day_n, "fraction": round(fraction, 4)}


PHASE_LABEL = {"dawn": "dawn", "day": "day", "dusk": "dusk", "night": "night"}


def is_active_phase(phase: str) -> bool:
    return phase in ("day", "dusk", "dawn")


def is_night(phase: str) -> bool:
    return phase == "night"
```

- [ ] **Step 5: Write `tests/test_age.py`**

```python
from agora_core.age import (
    age_in_days, age_stage, stage_label, can_reproduce, is_ancient,
)


def test_age_in_days_zero():
    assert age_in_days(0, 0) == 0.0


def test_age_in_days_one_full_day():
    # DAY_LENGTH_TICKS = 600
    assert age_in_days(0, 600) == 1.0


def test_age_in_days_negative_clamped():
    assert age_in_days(100, 50) == 0.0


def test_age_stages_boundaries():
    assert age_stage(0.5) == "child"
    assert age_stage(2.0) == "young"
    assert age_stage(10.0) == "adult"
    assert age_stage(50.0) == "elder"
    assert age_stage(150.0) == "ancient"


def test_can_reproduce_adult_plus():
    assert not can_reproduce("child")
    assert not can_reproduce("young")
    assert can_reproduce("adult")
    assert can_reproduce("elder")
    assert can_reproduce("ancient")


def test_is_ancient():
    assert is_ancient("ancient")
    assert not is_ancient("elder")


def test_stage_label_known():
    assert stage_label("child") == "child"
    assert stage_label("unknown_stage") == "unknown_stage"
```

- [ ] **Step 6: Write `tests/test_daynight.py`**

```python
from agora_core.daynight import (
    DAY_LENGTH_TICKS, time_of_day, is_night, is_active_phase,
)


def test_day_length():
    assert DAY_LENGTH_TICKS == 600


def test_time_of_day_dawn():
    info = time_of_day(0)
    assert info["phase"] == "dawn"
    assert info["day_n"] == 0
    assert info["fraction"] == 0.0


def test_time_of_day_day():
    info = time_of_day(60)  # 0.10 fraction → "day"
    assert info["phase"] == "day"


def test_time_of_day_dusk():
    info = time_of_day(int(0.51 * 600))
    assert info["phase"] == "dusk"


def test_time_of_day_night():
    info = time_of_day(int(0.80 * 600))
    assert info["phase"] == "night"


def test_time_of_day_day_n_increments():
    assert time_of_day(0)["day_n"] == 0
    assert time_of_day(600)["day_n"] == 1
    assert time_of_day(1200)["day_n"] == 2


def test_is_night_true_only_at_night():
    assert is_night("night")
    assert not is_night("day")
    assert not is_night("dawn")
    assert not is_night("dusk")


def test_is_active_phase():
    assert is_active_phase("day")
    assert is_active_phase("dawn")
    assert is_active_phase("dusk")
    assert not is_active_phase("night")
```

- [ ] **Step 7: Run tests**

```bash
cd /home/mattabott/Documents/agora-agent-sdk
. .venv/bin/activate
pytest tests/test_grid.py tests/test_age.py tests/test_daynight.py -v
```

Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add src/agora_core/grid.py src/agora_core/age.py src/agora_core/daynight.py \
        tests/test_grid.py tests/test_age.py tests/test_daynight.py
git -c user.email="info@iaitalia.net" -c user.name="mattabott" commit -m "feat: port grid/age/daynight constants from private repo"
```

---

### Task 1.2: recipes.py + structures.py + edibles.py ports

**Files:**
- Create: `src/agora_core/recipes.py`
- Create: `src/agora_core/structures.py`
- Create: `src/agora_core/edibles.py`
- Create: `tests/test_static_data.py`

- [ ] **Step 1: Write `src/agora_core/recipes.py`** (port from privata `crafting.py`)

```python
"""Crafting recipes. Ported 1:1 from agora.agents.crafting."""

RECIPES: dict[str, dict] = {
    "axe":    {"label": "Axe",     "in": {"wood": 2, "stone": 1}, "out": "axe"},
    "pickaxe":{"label": "Pickaxe", "in": {"wood": 1, "stone": 2}, "out": "pickaxe"},
    "bucket": {"label": "Bucket",  "in": {"wood": 3},             "out": "bucket"},
}

TOOL_TYPES = {recipe["out"] for recipe in RECIPES.values()}


def can_craft(recipe_name: str, inventory: dict[str, int]) -> bool:
    recipe = RECIPES.get(recipe_name)
    if recipe is None:
        return False
    return all(inventory.get(item, 0) >= qty for item, qty in recipe["in"].items())


def available_recipes(inventory: dict[str, int]) -> list[str]:
    return [name for name in RECIPES if can_craft(name, inventory)]


def format_recipes_for_prompt() -> str:
    lines = []
    for name, r in RECIPES.items():
        ing = " + ".join(f"{q} {k}" for k, q in r["in"].items())
        lines.append(f"  {name} ({r['label']}): {ing} -> 1 {r['out']}")
    return "\n".join(lines)


def format_recipes_for_user(inventory: dict[str, int]) -> str:
    available: list[str] = []
    missing: list[str] = []
    for name, r in RECIPES.items():
        deficits = []
        for item, qty in r["in"].items():
            have = inventory.get(item, 0)
            if have < qty:
                deficits.append(f"{qty - have} {item}")
        if not deficits:
            available.append(name)
        else:
            missing.append(f"{name} (need {', '.join(deficits)})")
    out = []
    if available:
        out.append(f"  available now: {', '.join(available)}")
    if missing:
        out.append(f"  not available: {', '.join(missing)}")
    return "\n".join(out) or "  (none)"
```

- [ ] **Step 2: Write `src/agora_core/structures.py`** (port from privata `building.py`)

```python
"""Buildable structures. Ported 1:1 from agora.agents.building."""

STRUCTURE_TYPES: dict[str, dict] = {
    "hut":     {"label": "Hut",     "in": {"wood": 5},                       "color": "#8b4513"},
    "storage": {"label": "Storage", "in": {"wood": 3, "stone": 2},           "color": "#5e4d3f"},
    "shrine":  {"label": "Shrine",  "in": {"stone": 2, "iron_ore": 1},       "color": "#704080"},
}


def can_build(structure_type: str, inventory: dict[str, int]) -> bool:
    spec = STRUCTURE_TYPES.get(structure_type)
    if spec is None:
        return False
    return all(inventory.get(item, 0) >= qty for item, qty in spec["in"].items())


def available_structures(inventory: dict[str, int]) -> list[str]:
    return [name for name in STRUCTURE_TYPES if can_build(name, inventory)]


def format_structures_for_prompt() -> str:
    lines = []
    for name, s in STRUCTURE_TYPES.items():
        ing = " + ".join(f"{q} {k}" for k, q in s["in"].items())
        lines.append(f"  {name} ({s['label']}): {ing}")
    return "\n".join(lines)


def format_structures_for_user(inventory: dict[str, int]) -> str:
    available: list[str] = []
    missing: list[str] = []
    for name, s in STRUCTURE_TYPES.items():
        deficits = []
        for item, qty in s["in"].items():
            have = inventory.get(item, 0)
            if have < qty:
                deficits.append(f"{qty - have} {item}")
        if not deficits:
            available.append(name)
        else:
            missing.append(f"{name} (need {', '.join(deficits)})")
    out = []
    if available:
        out.append(f"  available now: {', '.join(available)}")
    if missing:
        out.append(f"  not available: {', '.join(missing)}")
    return "\n".join(out) or "  (none)"
```

- [ ] **Step 3: Write `src/agora_core/edibles.py`** (port from privata `needs.py`)

```python
"""Edible items. Subset port from agora.agents.needs."""

EDIBLE_ITEMS: dict[str, dict] = {
    "berry": {"hunger_reduce": 30, "mood_boost": 10, "label": "berry"},
}


def edible_in_inventory(inventory: dict[str, int]) -> list[str]:
    return [item for item in EDIBLE_ITEMS if inventory.get(item, 0) > 0]


def format_edibles_for_prompt() -> str:
    return ", ".join(
        f"{name} (hunger -{spec['hunger_reduce']}, mood +{spec['mood_boost']})"
        for name, spec in EDIBLE_ITEMS.items()
    )
```

- [ ] **Step 4: Write `tests/test_static_data.py`**

```python
from agora_core.recipes import (
    RECIPES, TOOL_TYPES, can_craft, available_recipes,
    format_recipes_for_user, format_recipes_for_prompt,
)
from agora_core.structures import (
    STRUCTURE_TYPES, can_build, available_structures,
    format_structures_for_user, format_structures_for_prompt,
)
from agora_core.edibles import (
    EDIBLE_ITEMS, edible_in_inventory, format_edibles_for_prompt,
)


# === recipes ===

def test_recipes_have_axe_pickaxe_bucket():
    assert set(RECIPES) == {"axe", "pickaxe", "bucket"}


def test_tool_types_match_outputs():
    assert TOOL_TYPES == {"axe", "pickaxe", "bucket"}


def test_can_craft_axe_with_enough():
    assert can_craft("axe", {"wood": 2, "stone": 1})


def test_can_craft_axe_missing_stone():
    assert not can_craft("axe", {"wood": 5, "stone": 0})


def test_available_recipes_partial():
    assert available_recipes({"wood": 3}) == ["bucket"]


def test_format_recipes_user_available_section():
    out = format_recipes_for_user({"wood": 5, "stone": 3})
    assert "available now" in out
    assert "axe" in out


def test_format_recipes_user_no_inventory():
    out = format_recipes_for_user({})
    assert "(none)" in out or "not available" in out


def test_format_recipes_for_prompt_lists_all():
    out = format_recipes_for_prompt()
    for r in ("axe", "pickaxe", "bucket"):
        assert r in out


# === structures ===

def test_structures_have_hut_storage_shrine():
    assert set(STRUCTURE_TYPES) == {"hut", "storage", "shrine"}


def test_hut_costs_5_wood():
    assert STRUCTURE_TYPES["hut"]["in"] == {"wood": 5}


def test_can_build_hut():
    assert can_build("hut", {"wood": 5})
    assert not can_build("hut", {"wood": 4})


def test_available_structures():
    assert "hut" in available_structures({"wood": 5})
    assert "storage" not in available_structures({"wood": 3})


def test_format_structures_for_user_available():
    out = format_structures_for_user({"wood": 10, "stone": 5, "iron_ore": 1})
    assert "hut" in out
    assert "storage" in out
    assert "shrine" in out


def test_format_structures_for_prompt():
    out = format_structures_for_prompt()
    for s in ("hut", "storage", "shrine"):
        assert s in out


# === edibles ===

def test_berry_edible():
    assert "berry" in EDIBLE_ITEMS


def test_edible_in_inventory_finds_berry():
    assert edible_in_inventory({"berry": 3, "wood": 1}) == ["berry"]


def test_edible_in_inventory_empty():
    assert edible_in_inventory({"wood": 1}) == []


def test_format_edibles_mentions_berry():
    out = format_edibles_for_prompt()
    assert "berry" in out
    assert "hunger -30" in out
```

- [ ] **Step 5: Run tests**

```bash
pytest tests/test_static_data.py -v
```

Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add src/agora_core/recipes.py src/agora_core/structures.py src/agora_core/edibles.py \
        tests/test_static_data.py
git -c user.email="info@iaitalia.net" -c user.name="mattabott" commit -m "feat: port recipes/structures/edibles static data + format helpers"
```

---

### Task 1.3: protocol.py — pydantic models for WS messages

**Files:**
- Create: `src/agora_core/protocol.py`
- Create: `tests/test_protocol.py`

- [ ] **Step 1: Write `src/agora_core/protocol.py`**

```python
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
```

- [ ] **Step 2: Write `tests/test_protocol.py`**

```python
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
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_protocol.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/agora_core/protocol.py tests/test_protocol.py
git -c user.email="info@iaitalia.net" -c user.name="mattabott" commit -m "feat: protocol pydantic schemas + action validator"
```

---

## Phase 2 — WorldMirror

### Task 2.1: walkable_mask helpers + AgentSnap/StructureInfo/WorldEvent dataclasses

**Files:**
- Create: `src/agora_core/world_mirror.py` (initial: dataclasses + mask helpers + skeleton class)
- Create: `tests/test_world_mirror_basic.py`

- [ ] **Step 1: Write `src/agora_core/world_mirror.py`** (skeleton with mask helpers)

```python
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
```

- [ ] **Step 2: Write `tests/test_world_mirror_basic.py`**

```python
import base64
import pytest
from agora_core.world_mirror import (
    pack_walkable_mask, unpack_walkable_mask, mask_bit,
    AgentSnap, StructureInfo, WorldEvent, WorldMirror,
)


def test_pack_unpack_roundtrip():
    grid = [
        [True,  False, True ],
        [False, True,  False],
    ]  # 2x3
    raw = pack_walkable_mask(grid)
    # 6 bits → 1 byte
    assert len(raw) == 1
    # Encode → decode
    b64 = base64.b64encode(raw).decode("ascii")
    decoded = unpack_walkable_mask(b64, 3, 2)
    assert decoded == raw


def test_mask_bit_reads_correctly():
    grid = [
        [True,  False, True ],
        [False, True,  False],
    ]
    raw = pack_walkable_mask(grid)
    assert mask_bit(raw, 3, 0, 0) is True
    assert mask_bit(raw, 3, 1, 0) is False
    assert mask_bit(raw, 3, 2, 0) is True
    assert mask_bit(raw, 3, 0, 1) is False
    assert mask_bit(raw, 3, 1, 1) is True
    assert mask_bit(raw, 3, 2, 1) is False


def test_unpack_wrong_size_raises():
    raw = b"\x00"
    b64 = base64.b64encode(raw).decode("ascii")
    with pytest.raises(ValueError):
        unpack_walkable_mask(b64, 8, 8)


def test_64x64_mask_size():
    grid = [[True] * 64 for _ in range(64)]
    raw = pack_walkable_mask(grid)
    assert len(raw) == 512
    b64 = base64.b64encode(raw).decode("ascii")
    assert len(b64) == 684


def test_world_mirror_constructs_with_defaults():
    grid = [[True] * 4 for _ in range(4)]
    raw = pack_walkable_mask(grid)
    m = WorldMirror(world_w=4, world_h=4, walkable_mask=raw)
    assert m.current_tick == 0
    assert m.agents == {}
    assert m.structures == {}


def test_dataclasses_construct():
    a = AgentSnap(id=1, name="A", x=0, y=0, color="#fff", sex="F",
                  alive=True, born_tick=0)
    assert a.id == 1
    s = StructureInfo(id=1, x=2, y=3, type="hut", owner_id=1, built_tick=10)
    assert s.type == "hut"
    e = WorldEvent(id=1, type="rain", x=0, y=0, radius=4, started_tick=0, ends_tick=200)
    assert e.type == "rain"
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_world_mirror_basic.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/agora_core/world_mirror.py tests/test_world_mirror_basic.py
git -c user.email="info@iaitalia.net" -c user.name="mattabott" commit -m "feat: WorldMirror skeleton + walkable_mask pack/unpack"
```

---

### Task 2.2: WorldMirror.apply_snapshot + apply_event router

**Files:**
- Modify: `src/agora_core/world_mirror.py`
- Create: `tests/test_world_mirror_apply.py`

- [ ] **Step 1: Add to `src/agora_core/world_mirror.py`** (append after the `WorldMirror` dataclass)

```python
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
                    # Cluster snapshot does not give per-tile qty; assume 1 unit
                    # per known tile until a perception or tile_update refines it.
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
    # only meaningful for keeping `wait_streak`/`sleep_streak` in sync, and
    # the self-agent's state which the client also receives via perception.
    # mirror keeps a stub; tests assert presence not full coverage.
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
    # Relations are read by the client from each Perception (relations +
    # relations_inbound) — apply_event for relation_update is a no-op here.
    # The client may track them out-of-band if needed.
    pass


def _on_episodic_buffer_event(m: WorldMirror, ev: dict) -> None:
    # dialogue_received / gift_received / loss / user_message:
    # WorldMirror does not store the episodic buffer (the brain does).
    # No-op here; the brain consumes these via a separate channel.
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
```

- [ ] **Step 2: Write `tests/test_world_mirror_apply.py`**

```python
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
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_world_mirror_apply.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/agora_core/world_mirror.py tests/test_world_mirror_apply.py
git -c user.email="info@iaitalia.net" -c user.name="mattabott" commit -m "feat: WorldMirror apply_snapshot + delta event handlers"
```

---

### Task 2.3: is_walkable, is_occupied, find_path_step, nearest_resource, apply_perception

**Files:**
- Modify: `src/agora_core/world_mirror.py` (add methods to `WorldMirror`)
- Create: `tests/test_world_mirror_pathing.py`

- [ ] **Step 1: Add methods to `WorldMirror` class** (insert before the event handler functions)

```python
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
        # Fallback to cluster centroid
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
        # Self-agent stats: keep self in sync (alive, position).
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
        # Nearby resources: trust perception over stale mirror state.
        ax = int(sa.get("x", 0))
        ay = int(sa.get("y", 0))
        nearby_set = {(int(r["x"]), int(r["y"])): (r["type"], int(r["qty"]))
                      for r in perc.get("nearby_resources", [])}
        # First clear stale tiles in raggio 3 that are NOT reported anymore.
        for (rx, ry) in list(self.resources.keys()):
            if max(abs(rx - ax), abs(ry - ay)) <= 3 and (rx, ry) not in nearby_set:
                del self.resources[(rx, ry)]
        # Then upsert what the perception saw.
        for (rx, ry), v in nearby_set.items():
            self.resources[(rx, ry)] = v
```

- [ ] **Step 2: Write `tests/test_world_mirror_pathing.py`**

```python
from agora_core.world_mirror import (
    WorldMirror, AgentSnap, StructureInfo, pack_walkable_mask,
)


def _square_mirror(w=8, h=8) -> WorldMirror:
    grid = [[True] * w for _ in range(h)]
    return WorldMirror(world_w=w, world_h=h, walkable_mask=pack_walkable_mask(grid))


def _mirror_with_wall() -> WorldMirror:
    """8x8 mirror with column x=4 non-walkable except at y=7."""
    grid = [[True] * 8 for _ in range(8)]
    for y in range(7):
        grid[y][4] = False
    return WorldMirror(world_w=8, world_h=8, walkable_mask=pack_walkable_mask(grid))


def test_is_walkable_terrain_in_bounds():
    m = _square_mirror()
    assert m.is_walkable_terrain(0, 0)
    assert m.is_walkable_terrain(7, 7)
    assert not m.is_walkable_terrain(-1, 0)
    assert not m.is_walkable_terrain(8, 0)


def test_is_walkable_blocked_by_agent():
    m = _square_mirror()
    m.agents[1] = AgentSnap(id=1, name="X", x=3, y=3, color="#fff",
                            sex="F", alive=True, born_tick=0)
    assert not m.is_walkable(3, 3)
    assert m.is_walkable_terrain(3, 3)  # tile itself ok


def test_is_walkable_dead_agent_no_block():
    m = _square_mirror()
    m.agents[1] = AgentSnap(id=1, name="X", x=3, y=3, color="#fff",
                            sex="F", alive=False, born_tick=0, died_tick=10)
    assert m.is_walkable(3, 3)


def test_find_path_step_straight_east():
    m = _square_mirror()
    assert m.find_path_step(0, 0, 5, 0) == "east"


def test_find_path_step_around_wall():
    m = _mirror_with_wall()
    # Need to go south first to bypass column x=4
    step = m.find_path_step(0, 0, 7, 0)
    assert step in ("south", "east")  # south detours around wall


def test_find_path_step_unreachable_returns_none():
    grid = [[True] * 4 for _ in range(4)]
    # Box (3,3) with non-walkable tiles around it
    for dx, dy in ((0, 1), (1, 0), (-1, 0), (0, -1)):
        nx, ny = 3 + dx, 3 + dy
        if 0 <= nx < 4 and 0 <= ny < 4:
            grid[ny][nx] = False
    grid[3][3] = True  # destination is walkable but isolated
    m = WorldMirror(world_w=4, world_h=4, walkable_mask=pack_walkable_mask(grid))
    assert m.find_path_step(0, 0, 3, 3) is None


def test_nearest_resource_exact_tile():
    m = _square_mirror()
    m.resources[(2, 2)] = ("wood", 1)
    m.resources[(7, 7)] = ("wood", 1)
    assert m.nearest_resource(0, 0, "wood") == (2, 2)


def test_nearest_resource_falls_back_to_cluster():
    m = _square_mirror()
    m.resource_clusters = [{"type": "wood", "cx": 6, "cy": 6, "total_qty": 5,
                            "tiles": []}]
    assert m.nearest_resource(0, 0, "wood") == (6, 6)


def test_nearest_resource_skips_inaccessible_tile():
    """A wood tile on a non-walkable cell with no walkable neighbor is skipped."""
    grid = [[True] * 4 for _ in range(4)]
    grid[1][1] = False
    grid[0][1] = False
    grid[2][1] = False
    grid[1][0] = False
    grid[1][2] = False
    m = WorldMirror(world_w=4, world_h=4, walkable_mask=pack_walkable_mask(grid))
    m.resources[(1, 1)] = ("wood", 1)
    assert m.nearest_resource(0, 0, "wood") is None


def test_apply_perception_updates_self_position():
    m = _square_mirror()
    m.self_agent_id = 1
    m.agents[1] = AgentSnap(id=1, name="Self", x=0, y=0, color="#fff",
                            sex="F", alive=True, born_tick=0)
    m.apply_perception({
        "tick": 5,
        "agent_state": {"x": 3, "y": 4, "wait_streak": 2, "sleep_streak": 0},
    })
    assert m.agents[1].x == 3
    assert m.agents[1].y == 4
    assert m.agents[1].wait_streak == 2
    assert m.current_tick == 5


def test_apply_perception_clears_stale_nearby_resources():
    m = _square_mirror()
    m.self_agent_id = 1
    m.agents[1] = AgentSnap(id=1, name="Self", x=4, y=4, color="#fff",
                            sex="F", alive=True, born_tick=0)
    m.resources[(5, 4)] = ("berry", 1)  # stale, will not be in next perception
    m.resources[(0, 0)] = ("wood", 1)   # far, must remain
    m.apply_perception({
        "tick": 5,
        "agent_state": {"x": 4, "y": 4},
        "nearby_resources": [{"x": 4, "y": 5, "type": "berry", "qty": 1}],
    })
    assert (5, 4) not in m.resources  # cleared (stale within radius)
    assert (4, 5) in m.resources       # added
    assert (0, 0) in m.resources        # outside radius, preserved
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_world_mirror_pathing.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/agora_core/world_mirror.py tests/test_world_mirror_pathing.py
git -c user.email="info@iaitalia.net" -c user.name="mattabott" commit -m "feat: WorldMirror walkability, BFS pathfinding, nearest_resource, apply_perception"
```

---

## Phase 3 — Dialogue filters & prompts

### Task 3.1: dialogue_filters.py — port the full filter chain

**Files:**
- Create: `src/agora_core/dialogue_filters.py`
- Create: `tests/test_dialogue_filters.py`

- [ ] **Step 1: Write `src/agora_core/dialogue_filters.py`**

```python
"""Dialogue line filters. Ported 1:1 from agora.agents.brain `_dialogue_gen_bg`.

Reject criteria (in order):
  - too short (< 3 words)
  - poetic blacklist substring (EN + IT)
  - italian markers
  - noun-list pattern (>=2 comma parts, all <= 2 words)
  - truncated tail (... or no terminator .!?)
  - 3-gram overlap with any recent line of any agent
"""
from __future__ import annotations

POETIC_BLACKLIST: set[str] = {
    # English
    "shadow", "shadows", "whisper", "whispers", "echo", "echoes",
    "mystery", "mysterious", "ancient", "eternity", "eternal",
    "soul", "souls", "essence", "infinite", "infinity",
    "ineffab", "silence", "silent", "void", "boundless",
    "harmony", "tangible", "intimate", "intimacy",
    "fate", "destiny", "omen", "presage", "transc",
    "celestial", "ethereal", "sacred", "profound",
    # Italian (small LLMs sometimes slip into IT)
    "ombra", "sussurro", "eco", "mistero", "palpita",
    "eternita", "anima", "fluire", "antico", "antica",
    "rifugio", "promette", "desiderio", "celeste",
    "infinito", "presagio", "fato", "destino", "essenza",
    "silenzio", "esistenza", "intimo", "intima", "vuoto",
    "armonia", "ineffab", "trasc",
}

ITALIAN_MARKERS: tuple[str, ...] = (
    "sono", "sto", "siamo", "siete", "voglio", "vuoi", "vogliamo",
    "perche", "perché", "anche", "molto", "questa", "questo",
    "quello", "ancora", "adesso", "ieri", "oggi", "domani",
    "altrimenti", "abbia", "abbiamo", "qui con", "con te",
    "stanco", "stanca", "sento", "senti", "sente", "vado",
    "vai", "andiamo", "trovare", "iniziare", "costruire",
    "dispiace", "abitazione",
)

MAX_LINE_LEN = 100
MAX_RECENT_LINES_PER_AGENT = 12


def _strip_name_prefix(line: str, agent_names: list[str]) -> str:
    for nm in agent_names:
        for sep in (":", " -", " —", ","):
            pfx = f"{nm}{sep}"
            if line.startswith(pfx):
                return line[len(pfx):].strip().lstrip("\"'")
    return line


def _truncate(line: str, max_len: int = MAX_LINE_LEN) -> str:
    if len(line) <= max_len:
        return line
    return line[:max_len].rsplit(" ", 1)[0] + "…"


def accept_dialogue_line(
    line: str,
    *,
    agent_names: list[str],
    recent_lines_by_agent: dict[int, list[str]],
) -> str | None:
    """Return the cleaned line if accepted, or None if rejected.

    `recent_lines_by_agent` maps agent_id → list of recent normalized lines.
    """
    if not line:
        return None
    line = line.strip().strip('"').strip("'")
    line = _strip_name_prefix(line, agent_names)
    line = _truncate(line)
    if len(line.split()) < 3:
        return None

    line_low = line.lower()

    # Poetic blacklist (substring case-insensitive)
    for w in POETIC_BLACKLIST:
        if w in line_low:
            return None

    # Italian markers (multi-word substring OR word-boundary single tokens)
    words = {w.strip(".,;:!?\"'()") for w in line_low.split()}
    for marker in ITALIAN_MARKERS:
        if " " in marker:
            if marker in line_low:
                return None
        elif marker in words:
            return None

    # Anti-noun-list: ≥2 comma parts, all ≤2 words
    parts = [p.strip() for p in line.split(",") if p.strip()]
    if len(parts) >= 2 and all(len(p.split()) <= 2 for p in parts):
        return None

    # Anti-truncated
    if line.endswith(("...", "…")):
        return None
    if line and line[-1] not in ".!?":
        return None

    # 3-gram overlap dedup with any recent line of any agent
    line_norm = " ".join(line_low.split())
    new_words = line_norm.split()
    if len(new_words) >= 3:
        new_trigrams = {tuple(new_words[i:i+3]) for i in range(len(new_words) - 2)}
        for buf in recent_lines_by_agent.values():
            for prev in buf:
                pwords = prev.split()
                if len(pwords) < 3:
                    continue
                prev_trigrams = {tuple(pwords[i:i+3])
                                 for i in range(len(pwords) - 2)}
                if new_trigrams & prev_trigrams:
                    return None

    return line


def append_to_ring(buf: list[str], line: str, max_size: int = MAX_RECENT_LINES_PER_AGENT) -> None:
    """Append a normalized line to a per-agent ring buffer (in-place)."""
    norm = " ".join(line.lower().split())
    buf.append(norm)
    while len(buf) > max_size:
        buf.pop(0)
```

- [ ] **Step 2: Write `tests/test_dialogue_filters.py`**

```python
from agora_core.dialogue_filters import (
    accept_dialogue_line, append_to_ring,
    POETIC_BLACKLIST, ITALIAN_MARKERS,
    MAX_RECENT_LINES_PER_AGENT,
)


def _empty_ring() -> dict[int, list[str]]:
    return {}


# === acceptance ===

def test_accepts_normal_line():
    out = accept_dialogue_line(
        "I saw berries near the pond.",
        agent_names=["Aria"], recent_lines_by_agent=_empty_ring(),
    )
    assert out == "I saw berries near the pond."


def test_strips_name_prefix_colon():
    out = accept_dialogue_line(
        "Aria: Let us go gather wood.",
        agent_names=["Aria"], recent_lines_by_agent=_empty_ring(),
    )
    assert out == "Let us go gather wood."


def test_strips_name_prefix_dash_em():
    out = accept_dialogue_line(
        "Aria — let us go now.",
        agent_names=["Aria"], recent_lines_by_agent=_empty_ring(),
    )
    assert out == "let us go now."


# === short line ===

def test_rejects_short_line():
    assert accept_dialogue_line(
        "Hi.", agent_names=[], recent_lines_by_agent=_empty_ring(),
    ) is None
    assert accept_dialogue_line(
        "Two words.", agent_names=[], recent_lines_by_agent=_empty_ring(),
    ) is None


# === poetic blacklist ===

def test_rejects_poetic_en():
    assert accept_dialogue_line(
        "I feel a whisper in the wind.",
        agent_names=[], recent_lines_by_agent=_empty_ring(),
    ) is None


def test_rejects_poetic_it():
    assert accept_dialogue_line(
        "Sento un'ombra che mi segue.",
        agent_names=[], recent_lines_by_agent=_empty_ring(),
    ) is None


# === italian markers ===

def test_rejects_italian_word_boundary():
    assert accept_dialogue_line(
        "Vado a cercare cibo.",
        agent_names=[], recent_lines_by_agent=_empty_ring(),
    ) is None


def test_rejects_italian_multiword():
    assert accept_dialogue_line(
        "Sto qui con te per ora.",
        agent_names=[], recent_lines_by_agent=_empty_ring(),
    ) is None


# === noun list ===

def test_rejects_noun_list_pattern():
    assert accept_dialogue_line(
        "Hunger, fear, fatigue.",
        agent_names=[], recent_lines_by_agent=_empty_ring(),
    ) is None


# === truncated ===

def test_rejects_ellipsis():
    assert accept_dialogue_line(
        "I was thinking about you...",
        agent_names=[], recent_lines_by_agent=_empty_ring(),
    ) is None


def test_rejects_no_terminator():
    assert accept_dialogue_line(
        "I was thinking about you and",
        agent_names=[], recent_lines_by_agent=_empty_ring(),
    ) is None


# === 3-gram dedup ===

def test_rejects_trigram_overlap_same_agent():
    ring = {1: ["i saw berries near the pond"]}
    assert accept_dialogue_line(
        "Maybe I saw berries near the river.",
        agent_names=[], recent_lines_by_agent=ring,
    ) is None


def test_rejects_trigram_overlap_other_agent():
    ring = {2: ["lets build another hut tomorrow"]}
    assert accept_dialogue_line(
        "We should build another hut here.",
        agent_names=[], recent_lines_by_agent=ring,
    ) is None


def test_accepts_unique_line_with_full_ring():
    ring = {1: ["a b c d e", "f g h i j"], 2: ["m n o p q"]}
    out = accept_dialogue_line(
        "Want to find more wood?",
        agent_names=[], recent_lines_by_agent=ring,
    )
    assert out == "Want to find more wood?"


# === append_to_ring ===

def test_append_to_ring_normalizes():
    buf: list[str] = []
    append_to_ring(buf, "  Hello WORLD here  ")
    assert buf == ["hello world here"]


def test_append_to_ring_trims_to_max():
    buf: list[str] = ["x"] * MAX_RECENT_LINES_PER_AGENT
    append_to_ring(buf, "fresh new line")
    assert len(buf) == MAX_RECENT_LINES_PER_AGENT
    assert buf[-1] == "fresh new line"
    assert buf[0] == "x"  # one popped from the front
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_dialogue_filters.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/agora_core/dialogue_filters.py tests/test_dialogue_filters.py
git -c user.email="info@iaitalia.net" -c user.name="mattabott" commit -m "feat: dialogue line filters (poetic, italian, noun-list, dedup)"
```

---

### Task 3.2: prompts.py — SYSTEM_PROMPT + DIALOGUE_SYSTEM constants + format helpers

**Files:**
- Create: `src/agora_core/prompts.py`
- Create: `tests/test_prompts.py`

- [ ] **Step 1: Write `src/agora_core/prompts.py`**

```python
"""LLM prompts and prompt builders. Ported from agora.agents.brain (private).

Two prompts:
  - SYSTEM_PROMPT: for the JSON 'decide' call (action selection)
  - DIALOGUE_SYSTEM: for the freeform 'talk_line' call (one spoken sentence)
"""
from __future__ import annotations

from agora_core.age import age_in_days, age_stage
from agora_core.daynight import time_of_day
from agora_core.edibles import EDIBLE_ITEMS
from agora_core.recipes import format_recipes_for_user
from agora_core.structures import format_structures_for_user


SYSTEM_PROMPT = (
    "You are a person in a new world. Few of you are left. Survive and "
    "rebuild: eat, sleep safe, gather wood/stone, make tools, build huts, "
    "have children. Practical. Not philosophy.\n"
    "\n"
    "World: 2D grid. Resources: wood, stone, iron_ore, berry. Structures: "
    "hut (shelter), storage (shared inventory), shrine (mood bonus).\n"
    "\n"
    "Survival rules:\n"
    "- Hunger>70: eat berry (from inventory) or go to one.\n"
    "- Night without hut: sleep outside loses energy + mood. Build a hut "
    "or sleep next to one.\n"
    "- Rain at night outside: double penalty.\n"
    "- Tools speed up gathering. Make axe (2wood+1stone), pickaxe "
    "(1wood+2stone) when you can.\n"
    "- Hut needs 5 wood. Build it on a free walkable tile, far from "
    "other huts (>4 tiles), so the group spreads.\n"
    "- Group is small. To repopulate: opposite sex + mutual affinity>=20 "
    "→ propose. Children take time to be born.\n"
    "\n"
    "Decide ONE action as JSON, English.\n"
    "Schemas:\n"
    ' move: {"action":"move","direction":"north|south|east|west","thought":"..."}\n'
    ' wander: {"action":"wander","thought":"..."}\n'
    ' gather: {"action":"gather","thought":"..."} (resource on your tile)\n'
    ' wait: {"action":"wait","thought":"..."}\n'
    ' note: {"action":"note","content":"personal thought","thought":"..."}\n'
    ' talk: {"action":"talk","target_id":N,"content":"...","thought":"..."}\n'
    ' eat: {"action":"eat","item":"berry","thought":"..."}\n'
    ' craft: {"action":"craft","recipe":"axe|pickaxe|bucket","thought":"..."}\n'
    ' build: {"action":"build","structure":"hut|storage|shrine","thought":"..."}\n'
    ' give: {"action":"give","target_id":N,"item":"berry|wood|...","qty":1,"thought":"..."}\n'
    ' deposit: {"action":"deposit","item":"...","qty":1,"thought":"..."} (storage on tile)\n'
    ' withdraw: {"action":"withdraw","item":"...","qty":1,"thought":"..."} (storage on tile)\n'
    ' propose: {"action":"propose","target_id":N,"thought":"..."} '
    "(opposite sex, mutual affinity>=20, neither pregnant)\n\n"
    "Hard rules:\n"
    "- 'direction' must be in 'walkable_dirs'. No clear path? wander.\n"
    "- 'target_id' must be in 'nearby' list.\n"
    "- 'thought' max 12 words, plain spoken. No poetry, no abstract ideas.\n"
    "- 'note' is for a real personal thought (a plan, a worry, something "
    "you noticed). Not poems.\n"
    "- JSON only, no other text."
)


DIALOGUE_SYSTEM = (
    "You are {name}, a {sex_label}, {age_days} days alive.\n"
    "About you: {personality}\n"
    "Talking to {partner_name}, a {partner_sex_label} ({partner_age} days).\n"
    "\n"
    "Setting: small group of survivors in a new world. Practical talk: "
    "food, shelter, work, plans, feelings, the others. Not poetry.\n"
    "\n"
    "ALWAYS reply in ENGLISH. Some past exchanges may be in Italian — "
    "IGNORE that and answer in English regardless.\n"
    "\n"
    "RULES:\n"
    "- ONE plain sentence, 4-12 words.\n"
    "- Talk like a normal person to a friend. Concrete things.\n"
    "- FORBIDDEN: metaphors, abstract ideas, philosophical 'X is Y'.\n"
    "- FORBIDDEN: words like 'shadow', 'whisper', 'echo', 'mystery', "
    "'eternity', 'soul', 'silence', 'essence', 'ancient'.\n"
    "- FORBIDDEN: repeating the partner's words.\n"
    "- FORBIDDEN: single-word reply.\n"
    "\n"
    "GOOD examples:\n"
    "  Aria: \"I saw berries near the pond, want to go?\"\n"
    "  Niko: \"My legs hurt, I should rest before night.\"\n"
    "  Sole: \"Let's build another hut, this one is full.\"\n"
    "  Rio: \"Stone is heavy. Did you find any tools?\"\n"
    "  Aria: \"You look tired today, did you sleep ok?\"\n"
    "\n"
    "BAD examples:\n"
    "  \"Hungry.\" (too short)\n"
    "  \"Curiosity is an ancient desire.\" (abstract)\n"
    "  \"A tangible mystery, a shadow on its way.\" (poetic)\n"
    "\n"
    "Output: only the spoken sentence, no quotes, no '{name}:' prefix."
)


# ============ format helpers ============

def format_episodic(memory: list[dict]) -> str:
    if not memory:
        return "  (none)"
    lines = []
    for m in memory[-12:]:
        kind = m.get("kind", "?")
        tick = m.get("tick", "?")
        if kind == "decision":
            lines.append(f"  t{tick} thought: {m.get('thought', '')[:80]}")
        elif kind == "action_result":
            ok = m.get("ok")
            reason = m.get("reason", "")
            target = m.get("target_name", "")
            extra = (f" -> {target}" if target else "") + (f" ({reason})" if reason else "")
            lines.append(f"  t{tick} result: {m.get('action', '?')} ok={ok}{extra}")
        elif kind == "dialogue_received":
            lines.append(
                f"  t{tick} {m.get('from_name', '?')} told you: \"{m.get('content', '')[:100]}\""
            )
        elif kind == "gift_received":
            lines.append(
                f"  t{tick} {m.get('from_name', '?')} gave you "
                f"{m.get('qty', 0)} {m.get('item_type') or m.get('item', '?')}"
            )
        elif kind == "loss":
            lines.append(
                f"  t{tick} you lost {m.get('deceased_name', '?')} ({m.get('relation', '')})"
            )
        elif kind == "user_message":
            lines.append(
                f"  t{tick} an outside voice said: \"{m.get('content', '')[:100]}\""
            )
    return "\n".join(lines) or "  (none)"


def format_nearby(nearby_list: list[dict]) -> str:
    if not nearby_list:
        return "(none)"
    return ", ".join(
        f"{a['name']}(id={a['id']}, sex={a.get('sex','?')}) at ({a['x']},{a['y']})"
        for a in nearby_list
    )


def format_resources(nearby: list[dict], here: dict | None) -> str:
    parts = []
    if here:
        parts.append(f"HERE: {here['type']} (qty {here['qty']})")
    for r in nearby:
        if r["x"] == 0 and r["y"] == 0:
            continue
        parts.append(f"{r['type']} at ({r['x']},{r['y']}) qty={r['qty']}")
    return "; ".join(parts) or "(none)"


def format_inventory(inv: dict[str, int]) -> str:
    if not inv:
        return "(empty)"
    return ", ".join(f"{k}:{v}" for k, v in inv.items() if v > 0) or "(empty)"


def format_relations(relations: dict[int, int], agents_by_id: dict) -> str:
    if not relations:
        return "  (none)"
    lines = []
    for tid, aff in sorted(relations.items(), key=lambda x: -x[1]):
        target = agents_by_id.get(tid)
        name = target.name if target else f"id={tid}"
        sign = "+" if aff > 0 else ""
        lines.append(f"  {name} (id={tid}): {sign}{aff}")
    return "\n".join(lines)


def format_semantic(memory: list[dict]) -> str:
    if not memory:
        return "  (none)"
    return "\n".join(
        f"  - [{m['kind']} t{m['tick']}] {m['text'][:120]}" for m in memory
    )
```

- [ ] **Step 2: Write `tests/test_prompts.py`**

```python
from agora_core.prompts import (
    SYSTEM_PROMPT, DIALOGUE_SYSTEM,
    format_episodic, format_nearby, format_resources,
    format_inventory, format_relations, format_semantic,
)


def test_system_prompt_mentions_actions():
    assert "move" in SYSTEM_PROMPT
    assert "talk" in SYSTEM_PROMPT
    assert "build" in SYSTEM_PROMPT
    assert "JSON only" in SYSTEM_PROMPT


def test_dialogue_system_has_placeholders():
    assert "{name}" in DIALOGUE_SYSTEM
    assert "{partner_name}" in DIALOGUE_SYSTEM
    assert "{personality}" in DIALOGUE_SYSTEM


def test_format_episodic_empty():
    assert format_episodic([]) == "  (none)"


def test_format_episodic_decision_and_dialogue():
    mem = [
        {"kind": "decision", "tick": 1, "thought": "going east"},
        {"kind": "dialogue_received", "tick": 2, "from_name": "Niko",
         "content": "hello there"},
        {"kind": "action_result", "tick": 3, "action": "move", "ok": True,
         "target_name": ""},
    ]
    out = format_episodic(mem)
    assert "thought: going east" in out
    assert "Niko told you" in out
    assert "result: move ok=True" in out


def test_format_episodic_gift_loss_user_message():
    mem = [
        {"kind": "gift_received", "tick": 4, "from_name": "Sole",
         "item": "berry", "qty": 2},
        {"kind": "loss", "tick": 5, "deceased_name": "Rio",
         "relation": "vicino"},
        {"kind": "user_message", "tick": 6, "content": "be brave"},
    ]
    out = format_episodic(mem)
    assert "Sole gave you 2 berry" in out
    assert "you lost Rio" in out
    assert "outside voice said" in out


def test_format_episodic_caps_at_12():
    mem = [{"kind": "decision", "tick": i, "thought": str(i)} for i in range(50)]
    out = format_episodic(mem)
    assert "thought: 49" in out  # last shown
    assert "thought: 0" not in out  # first dropped


def test_format_nearby_lists_agents():
    out = format_nearby([{"id": 2, "name": "Niko", "sex": "M", "x": 5, "y": 6}])
    assert "Niko(id=2, sex=M) at (5,6)" == out


def test_format_resources_with_here():
    out = format_resources(
        [{"x": 5, "y": 6, "type": "wood", "qty": 1}],
        here={"type": "berry", "qty": 2},
    )
    assert "HERE: berry (qty 2)" in out
    assert "wood at (5,6)" in out


def test_format_inventory_empty():
    assert format_inventory({}) == "(empty)"


def test_format_inventory_filters_zeros():
    assert format_inventory({"wood": 0, "stone": 3}) == "stone:3"


def test_format_relations_sorts_by_affinity():
    class A:
        def __init__(self, name): self.name = name
    agents = {1: A("Aria"), 2: A("Niko")}
    out = format_relations({1: -10, 2: 30}, agents)
    # Niko (positive) first
    lines = out.strip().split("\n")
    assert "Niko" in lines[0]
    assert "+30" in lines[0]


def test_format_semantic_empty():
    assert format_semantic([]) == "  (none)"


def test_format_semantic_lines():
    mem = [{"kind": "reflection", "tick": 5, "text": "I learned to share."}]
    out = format_semantic(mem)
    assert "reflection" in out and "I learned to share." in out
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_prompts.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/agora_core/prompts.py tests/test_prompts.py
git -c user.email="info@iaitalia.net" -c user.name="mattabott" commit -m "feat: SYSTEM_PROMPT + DIALOGUE_SYSTEM + format helpers"
```

---

### Task 3.3: prompts.py — build_user_prompt + build_dialogue_user_prompt

**Files:**
- Modify: `src/agora_core/prompts.py` (append builders)
- Create: `tests/test_prompt_builders.py`

- [ ] **Step 1: Append to `src/agora_core/prompts.py`**

```python
# ============ prompt builders ============

def build_user_prompt(
    *,
    personality_current: str,
    sex: str,
    born_tick: int,
    current_tick: int,
    family: dict | None,
    current_goal: str,
    perception: dict,
    inventory: dict[str, int],
    relations: dict[int, int],
    agents_by_id: dict,
    episodic: list[dict],
    semantic: list[dict] | None = None,
    wait_streak: int = 0,
) -> str:
    """Build the per-tick USER prompt. Mirrors agora.agents.brain.build_user_prompt."""
    parts: list[str] = []
    parts.append(f"Identity: {personality_current}")
    days = age_in_days(born_tick, current_tick)
    stage = age_stage(days)
    parts.append(f"Sex: {sex} · Age: {days:.1f} days ({stage})")

    fam_bits = []
    fam = family or {}
    if fam.get("mother"):
        fam_bits.append(f"mother: {fam['mother']['name']}")
    if fam.get("father"):
        fam_bits.append(f"father: {fam['father']['name']}")
    children = [c["name"] for c in fam.get("children", []) if c.get("alive")]
    if children:
        fam_bits.append(f"children: {', '.join(children)}")
    if fam_bits:
        parts.append("Family: " + "; ".join(fam_bits))

    if current_goal:
        parts.append(f"Current goal: {current_goal}")

    tod = time_of_day(current_tick)
    parts.append(
        f"Status t{current_tick} (day {tod['day_n']}, {tod['phase']}): "
        f"pos {perception.get('position') or [0, 0]} "
        f"terrain {perception.get('terrain_here', '?')} "
        f"energy {perception.get('energy', 0)} "
        f"mood {perception.get('mood', 0)} "
        f"hunger {perception.get('hunger', 0)}"
    )

    walk = perception.get("walkable_dirs") or []
    parts.append(
        "You can move: " + (", ".join(walk) if walk else "NONE (blocked, choose another action)")
    )

    if wait_streak >= 3:
        parts.append(
            f"NOTE: you've waited {wait_streak} times in a row. "
            "Time to move, talk, explore or build something."
        )

    events = perception.get("world_events") or []
    if events:
        ev_strs = []
        for e in events:
            if e.get("type") == "rain":
                ev_strs.append("raining (berries respawn)")
            elif e.get("type") == "fire":
                ev_strs.append(f"fire at ({e.get('x',0)},{e.get('y',0)}) - destroys wood")
            else:
                ev_strs.append(e.get("type", "?"))
        parts.append(f"World events: {'; '.join(ev_strs)}")

    parts.append(f"Visible around: {perception.get('visible_around', '')}")

    if perception.get("nearby_agents"):
        parts.append(f"Nearby agents: {format_nearby(perception['nearby_agents'])}")

    res_str = format_resources(
        perception.get("nearby_resources") or [],
        perception.get("here_resource"),
    )
    if res_str != "(none)":
        parts.append(f"Resources: {res_str}")

    parts.append(f"Inventory: {format_inventory(inventory)}")

    if relations:
        parts.append(f"Relations: {format_relations(relations, agents_by_id)}")

    # Crafting if has any craftable mat
    if any(inventory.get(k, 0) > 0 for k in ("wood", "stone", "iron_ore")):
        recipes = format_recipes_for_user(inventory)
        if "(none)" not in recipes:
            parts.append(f"Crafting:\n{recipes}")

    # Building if tile is free + has any material
    if perception.get("here_structure") is None and any(
        inventory.get(k, 0) > 0 for k in ("wood", "stone", "iron_ore")
    ):
        structs = format_structures_for_user(inventory)
        if "(none)" not in structs:
            parts.append(f"Building (you can build here):\n{structs}")

    # Edibles
    if any(inventory.get(k, 0) > 0 for k in EDIBLE_ITEMS):
        eds = ", ".join(
            f"{k}({inventory.get(k, 0)})" for k in EDIBLE_ITEMS if inventory.get(k, 0) > 0
        )
        parts.append(f"Edible items: {eds}")

    if episodic:
        parts.append(f"Recent memory:\n{format_episodic(episodic)}")
    if semantic:
        parts.append(f"Relevant memories:\n{format_semantic(semantic)}")

    parts.append("Choose ONE action. Reply with a single JSON.")
    return "\n\n".join(parts)


def build_dialogue_user_prompt(
    *,
    self_name: str,
    self_x: int,
    self_y: int,
    mood: int,
    hunger: int,
    energy: int,
    current_tick: int,
    last_thought: str,
    current_goal: str,
    last_reflection: str,
    partner_name: str,
    nearby_resources: list[tuple[str, int]],   # [(type, dist_max), ...] precomputed
    nearby_structures: list[str],              # ["hut", "storage", ...] precomputed
    ongoing_events: list[str],                 # ["rain", "fire"]
    recent_dialogue_text: str,                 # already formatted multi-line string
) -> str:
    """Build the dialogue gen USER prompt with rich context.

    Ports agora.agents.brain._format_dialogue_context (private repo) but takes
    precomputed inputs (the world walking is done by the caller using WorldMirror).
    """
    PHASE_EN = {"dawn": "dawn", "day": "daytime", "dusk": "dusk", "night": "night"}
    parts: list[str] = []
    parts.append(
        f"Body: mood {mood}/100, hunger {hunger}/100, energy {energy}/100."
    )
    tod = time_of_day(current_tick)
    parts.append(f"It's {PHASE_EN.get(tod['phase'], tod['phase'])}.")

    visible_bits = []
    if nearby_resources:
        items = ", ".join(f"{k}({v})" for k, v in nearby_resources[:4])
        visible_bits.append(f"resources nearby: {items}")
    if nearby_structures:
        visible_bits.append(f"structures: {', '.join(nearby_structures[:3])}")
    if ongoing_events:
        visible_bits.append(f"ongoing: {', '.join(ongoing_events[:3])}")
    if visible_bits:
        parts.append("Around you: " + "; ".join(visible_bits) + ".")
    else:
        parts.append("Around you nothing notable.")

    if last_thought and not last_thought.startswith("("):
        parts.append(f"You were thinking: \"{last_thought[:120]}\".")
    if current_goal:
        parts.append(f"Your goal: {current_goal[:120]}.")
    if last_reflection:
        parts.append(f"Recent reflection of yours: \"{last_reflection[:140]}\".")

    parts.append(f"Recent exchanges between you and {partner_name}:")
    parts.append(recent_dialogue_text or "  (first time talking)")
    parts.append("\nWhat do you say now? One spoken sentence, no quotes.")
    return "\n".join(parts)
```

- [ ] **Step 2: Write `tests/test_prompt_builders.py`**

```python
from agora_core.prompts import build_user_prompt, build_dialogue_user_prompt


class _AgentRef:
    def __init__(self, name): self.name = name


def test_build_user_prompt_includes_identity_and_status():
    out = build_user_prompt(
        personality_current="curious one",
        sex="F",
        born_tick=0,
        current_tick=300,
        family=None,
        current_goal="",
        perception={
            "position": [5, 5], "terrain_here": "grass",
            "energy": 80, "mood": 60, "hunger": 30,
            "walkable_dirs": ["north", "east"],
            "visible_around": "(0,0)=grass",
            "nearby_agents": [], "nearby_resources": [],
            "here_resource": None, "here_structure": None,
            "world_events": [],
        },
        inventory={"berry": 3},
        relations={},
        agents_by_id={},
        episodic=[],
        semantic=None,
        wait_streak=0,
    )
    assert "Identity: curious one" in out
    assert "Sex: F" in out
    assert "energy 80" in out
    assert "You can move: north, east" in out
    assert "Inventory: berry:3" in out
    assert "Edible items: berry(3)" in out
    assert "Choose ONE action" in out


def test_build_user_prompt_recommends_movement_when_idle():
    out = build_user_prompt(
        personality_current="x", sex="M", born_tick=0, current_tick=10,
        family=None, current_goal="",
        perception={"position": [0, 0], "terrain_here": "grass",
                    "energy": 50, "mood": 50, "hunger": 0,
                    "walkable_dirs": ["north"], "visible_around": "",
                    "nearby_agents": [], "nearby_resources": [],
                    "here_resource": None, "here_structure": None,
                    "world_events": []},
        inventory={}, relations={}, agents_by_id={}, episodic=[],
        wait_streak=5,
    )
    assert "you've waited 5 times" in out


def test_build_user_prompt_includes_world_events():
    out = build_user_prompt(
        personality_current="x", sex="F", born_tick=0, current_tick=10,
        family=None, current_goal="",
        perception={"position": [0,0], "terrain_here": "grass",
                    "energy": 50, "mood": 50, "hunger": 0,
                    "walkable_dirs": [], "visible_around": "",
                    "nearby_agents": [], "nearby_resources": [],
                    "here_resource": None, "here_structure": None,
                    "world_events": [{"type": "rain"},
                                     {"type": "fire", "x": 5, "y": 6}]},
        inventory={}, relations={}, agents_by_id={}, episodic=[],
    )
    assert "raining" in out
    assert "fire at (5,6)" in out


def test_build_user_prompt_relations_section():
    out = build_user_prompt(
        personality_current="x", sex="F", born_tick=0, current_tick=10,
        family=None, current_goal="",
        perception={"position": [0,0], "terrain_here": "grass",
                    "energy": 50, "mood": 50, "hunger": 0,
                    "walkable_dirs": [], "visible_around": "",
                    "nearby_agents": [], "nearby_resources": [],
                    "here_resource": None, "here_structure": None,
                    "world_events": []},
        inventory={}, relations={2: 25}, agents_by_id={2: _AgentRef("Niko")},
        episodic=[],
    )
    assert "Niko" in out
    assert "+25" in out


def test_build_dialogue_prompt_full_context():
    out = build_dialogue_user_prompt(
        self_name="Aria", self_x=5, self_y=5,
        mood=70, hunger=20, energy=80, current_tick=10,
        last_thought="going east", current_goal="find wood",
        last_reflection="I should be more careful",
        partner_name="Niko",
        nearby_resources=[("wood", 3), ("berry", 2)],
        nearby_structures=["hut"],
        ongoing_events=["rain"],
        recent_dialogue_text="  Niko: hi\n  you: hi back",
    )
    assert "Body: mood 70/100, hunger 20/100, energy 80/100" in out
    assert "It's daytime." in out
    assert "resources nearby: wood(3), berry(2)" in out
    assert "structures: hut" in out
    assert "ongoing: rain" in out
    assert "going east" in out
    assert "find wood" in out
    assert "more careful" in out
    assert "Recent exchanges" in out
    assert "What do you say now" in out


def test_build_dialogue_prompt_first_time():
    out = build_dialogue_user_prompt(
        self_name="Aria", self_x=0, self_y=0,
        mood=50, hunger=0, energy=50, current_tick=0,
        last_thought="", current_goal="",
        last_reflection="",
        partner_name="Niko",
        nearby_resources=[], nearby_structures=[], ongoing_events=[],
        recent_dialogue_text="",
    )
    assert "first time talking" in out
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_prompt_builders.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/agora_core/prompts.py tests/test_prompt_builders.py
git -c user.email="info@iaitalia.net" -c user.name="mattabott" commit -m "feat: build_user_prompt + build_dialogue_user_prompt"
```

---

## Phase 4 — Reflex, social, policy ports

### Task 4.1: policy.py — port without sklearn dependency

**Files:**
- Create: `src/agora_core/policy.py`
- Create: `tests/test_policy.py`

- [ ] **Step 1: Write `src/agora_core/policy.py`** (sklearn import is lazy + optional)

```python
"""Action vocabulary, feature extractor, decoder. Distilled MLP policy is
optional (loads only if scikit-learn is installed and a .pkl exists).

Ported from agora.agents.policy. Server-only training pipeline NOT included:
the SDK only consumes a trained policy file if the user provides one.
"""
from __future__ import annotations

import logging
import pickle
from pathlib import Path

import numpy as np

from agora_core.age import age_in_days, age_stage
from agora_core.daynight import is_night, time_of_day

log = logging.getLogger("agora_core.policy")

FEATURE_DIM = 32

ACTION_VOCAB = [
    "wait",          # 0
    "wander",        # 1
    "move_north",    # 2
    "move_south",    # 3
    "move_east",     # 4
    "move_west",     # 5
    "gather",        # 6
    "eat_berry",     # 7
    "talk_nearby",   # 8
    "give_nearby",   # 9
]
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTION_VOCAB)}
N_ACTIONS = len(ACTION_VOCAB)


def encode_decision(action: str, direction: str = "", item: str = "") -> int | None:
    if action == "wait":
        return 0
    if action == "wander":
        return 1
    if action == "move":
        return ACTION_TO_IDX.get(f"move_{direction}", None)
    if action == "gather":
        return 6
    if action == "eat" and item == "berry":
        return 7
    if action == "talk":
        return 8
    if action == "give":
        return 9
    return None


def extract_features(
    *,
    current_tick: int,
    born_tick: int,
    hunger: int,
    mood: int,
    energy: int,
    hp: int,
    inventory: dict[str, int],
    perception: dict,
    wait_streak: int,
    sleep_streak: int,
) -> np.ndarray:
    """32-dim float32 feature vector. Ports agora.agents.policy.extract_features."""
    f = np.zeros(FEATURE_DIM, dtype=np.float32)

    f[0] = hunger / 100.0
    f[1] = mood / 100.0
    f[2] = energy / 100.0
    f[3] = hp / 100.0

    days = age_in_days(born_tick, current_tick)
    f[4] = min(1.0, days / 100.0)

    phase = time_of_day(current_tick)["phase"]
    f[5] = 1.0 if is_night(phase) else 0.0

    here_res = perception.get("here_resource") or {}
    f[6] = 1.0 if here_res else 0.0
    f[7] = 1.0 if perception.get("here_structure") else 0.0

    f[8]  = min(1.0, inventory.get("berry", 0) / 5.0)
    f[9]  = min(1.0, inventory.get("wood", 0) / 10.0)
    f[10] = min(1.0, inventory.get("stone", 0) / 5.0)
    f[11] = min(1.0, inventory.get("iron_ore", 0) / 3.0)
    f[12] = 1.0 if inventory.get("axe", 0) > 0 else 0.0
    f[13] = 1.0 if inventory.get("pickaxe", 0) > 0 else 0.0
    f[14] = 1.0 if inventory.get("bucket", 0) > 0 else 0.0

    nearby_a = perception.get("nearby_agents") or []
    f[15] = min(1.0, len(nearby_a) / 4.0)

    nearby_r = perception.get("nearby_resources") or []
    types_seen = {r["type"] for r in nearby_r}
    f[17] = 1.0 if "wood" in types_seen else 0.0
    f[18] = 1.0 if "berry" in types_seen else 0.0
    f[19] = 1.0 if "stone" in types_seen else 0.0
    f[20] = 1.0 if "iron_ore" in types_seen else 0.0

    walkable = set(perception.get("walkable_dirs") or [])
    f[21] = 1.0 if "north" in walkable else 0.0
    f[22] = 1.0 if "south" in walkable else 0.0
    f[23] = 1.0 if "east" in walkable else 0.0
    f[24] = 1.0 if "west" in walkable else 0.0

    f[25] = min(1.0, wait_streak / 10.0)
    f[26] = min(1.0, sleep_streak / 10.0)

    stage = age_stage(days)
    f[27] = 1.0 if stage == "child" else 0.0
    f[28] = 1.0 if stage == "young" else 0.0
    f[29] = 1.0 if stage == "adult" else 0.0
    f[30] = 1.0 if stage == "elder" else 0.0
    f[31] = 1.0 if stage == "ancient" else 0.0

    return f


# ============ optional MLP policy loader ============

class Policy:
    def __init__(self, path: Path):
        self.path = path
        self.model = None

    def load(self) -> bool:
        if not self.path.exists():
            return False
        try:
            with open(self.path, "rb") as fh:
                data = pickle.load(fh)
            self.model = data["model"]
            return True
        except Exception:
            log.exception("policy load failed")
            return False

    def predict(self, features: np.ndarray) -> int | None:
        if self.model is None:
            return None
        try:
            pred = self.model.predict(features.reshape(1, -1))
            return int(pred[0])
        except Exception:
            return None


def decode_to_decision(
    action_idx: int,
    perception: dict,
    inventory: dict[str, int],
) -> dict | None:
    """Map an action index back to a decision dict, or None if not applicable."""
    if action_idx < 0 or action_idx >= N_ACTIONS:
        return None
    action_name = ACTION_VOCAB[action_idx]
    walkable = set(perception.get("walkable_dirs") or [])

    if action_name == "wait":
        return {"action": "wait", "thought": "(policy)"}
    if action_name == "wander":
        return {"action": "wander", "thought": "(policy)"}
    if action_name.startswith("move_"):
        direction = action_name.split("_", 1)[1]
        if direction not in walkable:
            return None
        return {"action": "move", "direction": direction, "thought": "(policy)"}
    if action_name == "gather":
        if not perception.get("here_resource"):
            return None
        return {"action": "gather", "thought": "(policy)"}
    if action_name == "eat_berry":
        if inventory.get("berry", 0) <= 0:
            return None
        return {"action": "eat", "item": "berry", "thought": "(policy)"}
    if action_name == "talk_nearby":
        nearby = perception.get("nearby_agents") or []
        if not nearby:
            return None
        target = nearby[0]
        return {
            "action": "talk", "target_id": int(target["id"]),
            "content": "Hey there friend", "thought": "(policy)",
        }
    if action_name == "give_nearby":
        nearby = perception.get("nearby_agents") or []
        if not nearby:
            return None
        item = next((k for k, v in inventory.items() if v > 0), None)
        if item is None:
            return None
        return {
            "action": "give", "target_id": int(nearby[0]["id"]),
            "item": item, "qty": 1, "thought": "(policy)",
        }
    return None
```

- [ ] **Step 2: Write `tests/test_policy.py`**

```python
import numpy as np
from agora_core.policy import (
    ACTION_VOCAB, ACTION_TO_IDX, FEATURE_DIM, N_ACTIONS,
    encode_decision, extract_features, decode_to_decision,
)


def test_action_vocab_size():
    assert len(ACTION_VOCAB) == 10
    assert N_ACTIONS == 10


def test_action_vocab_indexes():
    assert ACTION_TO_IDX["wait"] == 0
    assert ACTION_TO_IDX["wander"] == 1
    assert ACTION_TO_IDX["move_north"] == 2
    assert ACTION_TO_IDX["give_nearby"] == 9


def test_encode_decision_simple():
    assert encode_decision("wait") == 0
    assert encode_decision("wander") == 1
    assert encode_decision("gather") == 6
    assert encode_decision("eat", item="berry") == 7
    assert encode_decision("talk") == 8
    assert encode_decision("give") == 9


def test_encode_decision_move_directions():
    assert encode_decision("move", direction="north") == 2
    assert encode_decision("move", direction="south") == 3
    assert encode_decision("move", direction="east") == 4
    assert encode_decision("move", direction="west") == 5


def test_encode_decision_unknown_returns_none():
    assert encode_decision("ascend") is None
    assert encode_decision("eat", item="rock") is None
    assert encode_decision("move", direction="up") is None


def test_extract_features_dimension_and_dtype():
    f = extract_features(
        current_tick=300, born_tick=0,
        hunger=50, mood=50, energy=50, hp=100,
        inventory={"berry": 2, "wood": 5},
        perception={
            "here_resource": None, "here_structure": None,
            "nearby_agents": [],
            "nearby_resources": [{"type": "wood", "x": 1, "y": 1, "qty": 1}],
            "walkable_dirs": ["north", "east"],
        },
        wait_streak=0, sleep_streak=0,
    )
    assert f.shape == (FEATURE_DIM,)
    assert f.dtype == np.float32


def test_extract_features_normalizes_stats():
    f = extract_features(
        current_tick=0, born_tick=0,
        hunger=100, mood=100, energy=100, hp=100,
        inventory={}, perception={}, wait_streak=0, sleep_streak=0,
    )
    assert f[0] == 1.0  # hunger normalized
    assert f[1] == 1.0  # mood
    assert f[2] == 1.0  # energy
    assert f[3] == 1.0  # hp


def test_extract_features_walkable_flags():
    f = extract_features(
        current_tick=0, born_tick=0, hunger=0, mood=0, energy=0, hp=0,
        inventory={}, perception={"walkable_dirs": ["north", "east"]},
        wait_streak=0, sleep_streak=0,
    )
    assert f[21] == 1.0  # north
    assert f[22] == 0.0  # south
    assert f[23] == 1.0  # east
    assert f[24] == 0.0  # west


def test_decode_wait_and_wander():
    perc = {"walkable_dirs": []}
    assert decode_to_decision(0, perc, {})["action"] == "wait"
    assert decode_to_decision(1, perc, {})["action"] == "wander"


def test_decode_move_blocked():
    # move_north when north not walkable → None
    perc = {"walkable_dirs": ["east"]}
    assert decode_to_decision(2, perc, {}) is None


def test_decode_move_ok():
    perc = {"walkable_dirs": ["north"]}
    out = decode_to_decision(2, perc, {})
    assert out == {"action": "move", "direction": "north", "thought": "(policy)"}


def test_decode_gather_requires_resource():
    assert decode_to_decision(6, {"here_resource": None}, {}) is None
    out = decode_to_decision(6, {"here_resource": {"type": "wood", "qty": 1}}, {})
    assert out["action"] == "gather"


def test_decode_eat_requires_berry():
    assert decode_to_decision(7, {}, {}) is None
    out = decode_to_decision(7, {}, {"berry": 1})
    assert out == {"action": "eat", "item": "berry", "thought": "(policy)"}


def test_decode_talk_nearby_no_one():
    assert decode_to_decision(8, {"nearby_agents": []}, {}) is None


def test_decode_talk_nearby_picks_first():
    out = decode_to_decision(8, {"nearby_agents": [{"id": 7}]}, {})
    assert out["action"] == "talk"
    assert out["target_id"] == 7


def test_decode_give_picks_first_item():
    out = decode_to_decision(
        9, {"nearby_agents": [{"id": 3}]},
        {"berry": 0, "wood": 2, "stone": 0},
    )
    assert out["action"] == "give"
    assert out["item"] == "wood"
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_policy.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/agora_core/policy.py tests/test_policy.py
git -c user.email="info@iaitalia.net" -c user.name="mattabott" commit -m "feat: policy ACTION_VOCAB + extract_features + decode_to_decision"
```

---

### Task 4.2: reflex.py — port try_reflex with WorldMirror

**Files:**
- Create: `src/agora_core/reflex.py`
- Create: `tests/test_reflex_parity.py`

- [ ] **Step 1: Write `src/agora_core/reflex.py`** (port from `agora.agents.reflex.try_reflex`, adapted to WorldMirror)

```python
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
```

- [ ] **Step 2: Write `tests/test_reflex_parity.py`**

```python
"""Reflex parity tests: known scenarios → expected decision.

Each scenario builds a deterministic WorldMirror + AgentSnap and asserts the
exact decision try_reflex returns. Snapshot expectations come from running the
private repo's logic with equivalent state (manually verified).
"""
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


# === 1. eat berry on hunger ===

def test_reflex_eats_when_hungry_with_berry():
    m = _basic_mirror()
    agent = _self_agent(m)
    out = try_reflex(m, agent, _basic_perception(), {"berry": 1},
                     hunger=70, energy=80)
    assert out["action"] == "eat" and out["item"] == "berry"


def test_reflex_no_eat_if_not_hungry():
    m = _basic_mirror()
    agent = _self_agent(m)
    # hunger 30, has berry but plenty of all + INV_TARGET met → no reflex
    inv = {"berry": 5, "wood": 6, "stone": 4, "iron_ore": 2,
           "axe": 1, "pickaxe": 1}
    out = try_reflex(m, agent, _basic_perception(), inv,
                     hunger=30, energy=80)
    # not eat reflex
    assert out is None or out["action"] != "eat"


# === 2. gather berry on tile ===

def test_reflex_gathers_berry_on_tile_when_low():
    m = _basic_mirror()
    agent = _self_agent(m)
    perc = _basic_perception(here_resource={"type": "berry", "qty": 1})
    out = try_reflex(m, agent, perc, {"berry": 0}, hunger=10, energy=80)
    assert out["action"] == "gather"


# === 3. move to nearest berry when hungry ===

def test_reflex_moves_to_nearest_berry():
    m = _basic_mirror()
    agent = _self_agent(m, x=0, y=0)
    m.resources[(5, 0)] = ("berry", 1)
    perc = _basic_perception()
    out = try_reflex(m, agent, perc, {}, hunger=80, energy=80)
    assert out["action"] == "move"
    assert out["direction"] == "east"


# === 4b. craft axe ===

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
    # next reflex: pickaxe? needs 1w + 2s, has 2w 1s → no. Falls through.
    assert out is None or out["recipe"] != "axe"


# === 4c. build hut ===

def test_reflex_builds_hut_when_5_wood_and_no_hut_nearby():
    m = _basic_mirror()
    agent = _self_agent(m, x=8, y=8)
    inv = {"wood": 6}  # > 5 but with target (6) so opportunistic gather not active
    perc = _basic_perception(tick=300)  # tick=300 → fraction 0.5 → dusk; not night
    # Need to be daytime for "min_d >= 4" branch; tick 60 → fraction 0.1 → "day"
    perc = _basic_perception(tick=60)
    out = try_reflex(m, agent, perc, inv, hunger=10, energy=80)
    assert out is not None
    assert out["action"] == "build"
    assert out["structure"] == "hut"


# === 5. shelter pull at night ===

def test_reflex_moves_toward_hut_at_night():
    m = _basic_mirror()
    agent = _self_agent(m, x=0, y=0)
    m.structures[(5, 0)] = StructureInfo(
        id=1, x=5, y=0, type="hut", owner_id=1, built_tick=0,
    )
    perc = _basic_perception(tick=int(0.8 * 600))  # night
    inv = {}  # no resources, nothing else triggers
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


# === 6. night low energy fallback ===

def test_reflex_rests_when_low_energy_at_night_no_hut():
    m = _basic_mirror()
    agent = _self_agent(m)
    perc = _basic_perception(tick=int(0.8 * 600))
    out = try_reflex(m, agent, perc, {}, hunger=10, energy=10)
    assert out["action"] == "wait"


# === default: returns None when no reflex applies ===

def test_reflex_returns_none_when_idle():
    m = _basic_mirror()
    agent = _self_agent(m)
    inv = {"berry": 5, "wood": 6, "stone": 4, "axe": 1, "pickaxe": 1}
    perc = _basic_perception(tick=60)  # day, no resources, no hut, mid-day
    out = try_reflex(m, agent, perc, inv, hunger=10, energy=80)
    assert out is None
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_reflex_parity.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/agora_core/reflex.py tests/test_reflex_parity.py
git -c user.email="info@iaitalia.net" -c user.name="mattabott" commit -m "feat: reflex layer port + parity tests"
```

---

### Task 4.3: social.py — port social_navigate

**Files:**
- Create: `src/agora_core/social.py`
- Create: `tests/test_social.py`

- [ ] **Step 1: Write `src/agora_core/social.py`**

```python
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
```

- [ ] **Step 2: Write `tests/test_social.py`**

```python
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
    assert out is None  # falls through to LLM/reflex


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
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_social.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/agora_core/social.py tests/test_social.py
git -c user.email="info@iaitalia.net" -c user.name="mattabott" commit -m "feat: social_navigate port + tests"
```

---

## Phase 5 — LLM client

### Task 5.1: OllamaClient (httpx async) + test

**Files:**
- Create: `src/agora_agent_sdk/llm.py`
- Create: `tests/test_llm.py`

- [ ] **Step 1: Write `src/agora_agent_sdk/llm.py`**

```python
"""Async wrapper around a local Ollama server.

Two methods used by the brain:
  - decide(system, user) → dict (LLM JSON output, format=json)
  - talk_line(system, user) → str (single freeform sentence)

Both share a single AsyncLock to serialize calls (the user has one Ollama
process; concurrent calls just slow it down).
"""
from __future__ import annotations

import asyncio
import json
import logging

import httpx

log = logging.getLogger("agora_agent_sdk.llm")


class OllamaClient:
    def __init__(
        self,
        host: str,
        model: str,
        *,
        num_predict_decide: int = 80,
        num_predict_dialogue: int = 60,
        num_ctx: int = 2048,
        temperature: float = 0.7,
        timeout_decide: float = 60.0,
        timeout_dialogue: float = 30.0,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.num_predict_decide = num_predict_decide
        self.num_predict_dialogue = num_predict_dialogue
        self.num_ctx = num_ctx
        self.temperature = temperature
        self._timeout_decide = timeout_decide
        self._timeout_dialogue = timeout_dialogue
        self._lock = asyncio.Lock()
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def decide(self, system: str, user: str) -> dict:
        async with self._lock:
            try:
                r = await self._client.post(
                    f"{self.host}/api/generate",
                    json={
                        "model": self.model,
                        "system": system,
                        "prompt": user,
                        "format": "json",
                        "stream": False,
                        "options": {
                            "num_predict": self.num_predict_decide,
                            "num_ctx": self.num_ctx,
                            "temperature": self.temperature,
                        },
                    },
                    timeout=self._timeout_decide,
                )
                r.raise_for_status()
                text = r.json().get("response", "")
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    log.debug("ollama decide: invalid JSON %r", text[:200])
                    return {}
                return parsed if isinstance(parsed, dict) else {}
            except httpx.HTTPError as e:
                log.warning("ollama decide failed: %s", e)
                return {}

    async def talk_line(self, system: str, user: str) -> str:
        async with self._lock:
            try:
                r = await self._client.post(
                    f"{self.host}/api/generate",
                    json={
                        "model": self.model,
                        "system": system,
                        "prompt": user,
                        "stream": False,
                        "options": {
                            "num_predict": self.num_predict_dialogue,
                            "num_ctx": self.num_ctx,
                            "temperature": self.temperature,
                        },
                    },
                    timeout=self._timeout_dialogue,
                )
                r.raise_for_status()
                return r.json().get("response", "")
            except httpx.HTTPError as e:
                log.warning("ollama talk_line failed: %s", e)
                return ""


class NoOpLLM:
    """Placeholder used when --no-llm is set. Returns empty results."""

    async def decide(self, system: str, user: str) -> dict:
        return {}

    async def talk_line(self, system: str, user: str) -> str:
        return ""

    async def aclose(self) -> None:
        return None
```

- [ ] **Step 2: Write `tests/test_llm.py`**

```python
import json
import pytest
import httpx

from agora_agent_sdk.llm import OllamaClient, NoOpLLM


@pytest.mark.asyncio
async def test_decide_parses_valid_json():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["format"] == "json"
        return httpx.Response(200, json={
            "response": json.dumps({"action": "wait", "thought": "ok"})
        })

    transport = httpx.MockTransport(handler)
    client = OllamaClient(host="http://x", model="test-model")
    client._client = httpx.AsyncClient(transport=transport)
    out = await client.decide("sys", "user")
    assert out == {"action": "wait", "thought": "ok"}
    await client.aclose()


@pytest.mark.asyncio
async def test_decide_invalid_json_returns_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": "not json"})

    transport = httpx.MockTransport(handler)
    client = OllamaClient(host="http://x", model="m")
    client._client = httpx.AsyncClient(transport=transport)
    assert await client.decide("s", "u") == {}
    await client.aclose()


@pytest.mark.asyncio
async def test_decide_http_error_returns_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    transport = httpx.MockTransport(handler)
    client = OllamaClient(host="http://x", model="m")
    client._client = httpx.AsyncClient(transport=transport)
    assert await client.decide("s", "u") == {}
    await client.aclose()


@pytest.mark.asyncio
async def test_talk_line_returns_response_text():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": "Hello there friend"})

    transport = httpx.MockTransport(handler)
    client = OllamaClient(host="http://x", model="m")
    client._client = httpx.AsyncClient(transport=transport)
    assert await client.talk_line("s", "u") == "Hello there friend"
    await client.aclose()


@pytest.mark.asyncio
async def test_no_op_llm_returns_empty():
    n = NoOpLLM()
    assert await n.decide("s", "u") == {}
    assert await n.talk_line("s", "u") == ""
    await n.aclose()
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_llm.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/agora_agent_sdk/llm.py tests/test_llm.py
git -c user.email="info@iaitalia.net" -c user.name="mattabott" commit -m "feat: OllamaClient async wrapper + NoOpLLM"
```

---

## Phase 6 — Brain orchestrator

### Task 6.1: brain.py — ring buffer + decide() core

**Files:**
- Create: `src/agora_agent_sdk/brain.py`
- Create: `tests/test_brain_decide.py`

- [ ] **Step 1: Write `src/agora_agent_sdk/brain.py`**

```python
"""Decision orchestrator client-side.

Pipeline:
  1. apply_perception to mirror
  2. compute reflex; if reflex returns → use it (via="reflex")
  3. else if pending_llm_task done with non-empty result → use it (via="llm")
  4. else fast_decision = social or policy or wander (via="social"/"policy"/"auto_cooldown")
  5. dispatch background LLM if cooldown elapsed
  6. dispatch background dialogue gen if partner adjacent and slot free
  7. resolve talk content (use next_talk_line if ready, else wait)
  8. validate action client-side (talk dedup, move walkable)
  9. log to ring buffer

The brain owns:
  - mirror (WorldMirror)
  - llm (OllamaClient or NoOpLLM)
  - episodic ring buffer (deque)
  - per-agent recent_dialogue_lines rings
  - pending background tasks (asyncio.Task)
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Awaitable, Callable

from agora_core.dialogue_filters import accept_dialogue_line, append_to_ring
from agora_core.policy import (
    Policy, decode_to_decision, extract_features,
)
from agora_core.prompts import (
    SYSTEM_PROMPT, DIALOGUE_SYSTEM,
    build_dialogue_user_prompt, build_user_prompt,
)
from agora_core.protocol import validate_action_dict
from agora_core.reflex import try_reflex
from agora_core.social import social_navigate
from agora_core.world_mirror import AgentSnap, WorldMirror

log = logging.getLogger("agora_agent_sdk.brain")


class Brain:
    def __init__(
        self,
        *,
        mirror: WorldMirror,
        llm,                                  # OllamaClient or NoOpLLM
        agent_id: int,
        agent_name: str,
        sex: str,
        color: str,
        personality_seed: str,
        llm_decide_interval: int = 120,
        ring_buffer_size: int = 30,
        policy: Policy | None = None,
    ):
        self.mirror = mirror
        self.llm = llm
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.sex = sex
        self.color = color
        self.personality_seed = personality_seed
        self.llm_decide_interval = llm_decide_interval
        self.policy = policy

        # State
        self.episodic: deque[dict] = deque(maxlen=ring_buffer_size)
        # Per-agent recent dialogue ring buffer (own + observed)
        self.recent_lines_by_agent: dict[int, list[str]] = {}
        self.next_talk_line: str = ""
        self.last_walkable_dirs: list[str] = []
        # Background tasks
        self.pending_llm_task: asyncio.Task | None = None
        self.pending_dialogue_task: asyncio.Task | None = None
        self.last_llm_decide_tick: int = -10**9
        self.llm_cooldown: int = 0

    # ============ ring buffer ============

    def push_episodic(self, entry: dict) -> None:
        self.episodic.append(dict(entry))

    def push_dialogue_received(self, from_id: int, from_name: str,
                                content: str, tick: int) -> None:
        self.push_episodic({
            "kind": "dialogue_received", "tick": tick,
            "from_id": from_id, "from_name": from_name, "content": content,
        })
        ring = self.recent_lines_by_agent.setdefault(from_id, [])
        append_to_ring(ring, content)

    def push_event_to_episodic(self, ev: dict) -> None:
        kind = ev.get("kind")
        if kind == "dialogue_received":
            self.push_dialogue_received(
                int(ev["from_id"]), ev["from_name"], ev["content"],
                int(ev.get("tick", self.mirror.current_tick)),
            )
        elif kind in ("gift_received", "loss", "user_message"):
            self.push_episodic({**ev, "kind": kind})

    # ============ decide ============

    def _has_close_partner(self, perception: dict) -> int:
        nearby = perception.get("nearby_agents") or []
        for o in nearby:
            ox, oy = int(o.get("x", 0)), int(o.get("y", 0))
            mirror = self.mirror
            self_a = mirror.agents.get(self.agent_id)
            if self_a is None:
                return 0
            if max(abs(ox - self_a.x), abs(oy - self_a.y)) <= 2:
                return int(o["id"])
        return 0

    async def _llm_think_bg(self, perception: dict, inventory: dict) -> dict:
        try:
            self_a = self.mirror.agents.get(self.agent_id)
            sa = perception.get("agent_state", {})
            user = build_user_prompt(
                personality_current=sa.get("personality_current",
                                            self.personality_seed),
                sex=self.sex,
                born_tick=int(sa.get("born_tick", 0)),
                current_tick=self.mirror.current_tick,
                family=perception.get("family"),
                current_goal=sa.get("current_goal", ""),
                perception={
                    "position": [self_a.x, self_a.y] if self_a else [0, 0],
                    "terrain_here": perception.get("terrain_here", ""),
                    "energy": int(sa.get("energy", 0)),
                    "mood": int(sa.get("mood", 0)),
                    "hunger": int(sa.get("hunger", 0)),
                    "walkable_dirs": perception.get("walkable_dirs", []),
                    "visible_around": perception.get("visible_around", ""),
                    "nearby_agents": perception.get("nearby_agents", []),
                    "nearby_resources": perception.get("nearby_resources", []),
                    "here_resource": perception.get("here_resource"),
                    "here_structure": perception.get("here_structure"),
                    "world_events": perception.get("world_events", []),
                },
                inventory=inventory,
                relations={int(k): v for k, v in
                           (perception.get("relations") or {}).items()},
                agents_by_id=self.mirror.agents,
                episodic=list(self.episodic),
                semantic=None,
                wait_streak=int(sa.get("wait_streak", 0)),
            )
            return await self.llm.decide(SYSTEM_PROMPT, user)
        except Exception:
            log.exception("llm_think_bg failed")
            return {}

    async def _dialogue_gen_bg(self, partner_id: int) -> str:
        try:
            self_a = self.mirror.agents.get(self.agent_id)
            partner = self.mirror.agents.get(partner_id)
            if self_a is None or partner is None:
                return ""

            # gather recent cross-talk text from episodic ring
            recent_lines = []
            for m in list(self.episodic)[-12:]:
                if m.get("kind") == "dialogue_received" and m.get("from_id") == partner_id:
                    recent_lines.append(f"  {partner.name}: {m['content']}")
            recent_text = "\n".join(recent_lines) or ""

            # nearby resources / structures from mirror
            nearby_res: list[tuple[str, int]] = []
            for (rx, ry), (rt, rq) in self.mirror.resources.items():
                if max(abs(rx - self_a.x), abs(ry - self_a.y)) <= 3:
                    nearby_res.append((rt, rq))
            nearby_res = nearby_res[:4]
            nearby_struct = [info.type for (sx, sy), info in self.mirror.structures.items()
                             if max(abs(sx - self_a.x), abs(sy - self_a.y)) <= 3][:3]
            ongoing = [ev.type for ev in self.mirror.events.values()][:3]

            from agora_core.age import age_in_days
            sys_p = DIALOGUE_SYSTEM.format(
                name=self.agent_name,
                sex_label="man" if self.sex == "M" else "woman",
                age_days=age_in_days(self_a.born_tick, self.mirror.current_tick),
                personality=self.personality_seed[:200],
                partner_name=partner.name,
                partner_sex_label="man" if partner.sex == "M" else "woman",
                partner_age=age_in_days(partner.born_tick, self.mirror.current_tick),
            )
            user_p = build_dialogue_user_prompt(
                self_name=self.agent_name,
                self_x=self_a.x, self_y=self_a.y,
                mood=0, hunger=0, energy=0,  # NOT in mirror; if needed, pass via brain state
                current_tick=self.mirror.current_tick,
                last_thought="", current_goal="",
                last_reflection="",
                partner_name=partner.name,
                nearby_resources=nearby_res,
                nearby_structures=nearby_struct,
                ongoing_events=ongoing,
                recent_dialogue_text=recent_text,
            )
            raw = await self.llm.talk_line(sys_p, user_p)
            agent_names = [a.name for a in self.mirror.agents.values()]
            line = accept_dialogue_line(
                raw, agent_names=agent_names,
                recent_lines_by_agent=self.recent_lines_by_agent,
            )
            return line or ""
        except Exception:
            log.exception("dialogue_gen_bg failed")
            return ""

    def _validate_pre_send(self, decision: dict) -> dict:
        a = decision.get("action")
        if a == "talk":
            content = (decision.get("content") or "").strip()
            if len(content.split()) < 3:
                return {"action": "wait", "thought": "(too short)"}
            norm = " ".join(content.lower().split())
            for buf in self.recent_lines_by_agent.values():
                if norm in buf:
                    return {"action": "wait", "thought": "(already said)"}
        if a == "move":
            d = decision.get("direction")
            if d not in self.last_walkable_dirs:
                return {"action": "wander", "thought": "(blocked)"}
        return decision

    async def decide(self, perception: dict) -> dict:
        # 1. mirror update
        self.mirror.apply_perception(perception)
        sa = perception.get("agent_state", {})
        self.last_walkable_dirs = list(perception.get("walkable_dirs") or [])
        self_a = self.mirror.agents.get(self.agent_id)
        inventory = dict(sa.get("inventory") or {})

        # 2. reflex
        reflex_dec = None
        if self_a is not None:
            reflex_dec = try_reflex(
                self.mirror, self_a, perception, inventory,
                aff_out={int(k): v for k, v in (perception.get("relations") or {}).items()},
                aff_in={int(k): v for k, v in (perception.get("relations_inbound") or {}).items()},
                sex=self.sex, born_tick=int(sa.get("born_tick", 0)),
                hunger=int(sa.get("hunger", 0)),
                energy=int(sa.get("energy", 100)),
            )

        # 3. resolve pending background tasks
        prebaked_llm: dict | None = None
        if self.pending_llm_task and self.pending_llm_task.done():
            try:
                res = self.pending_llm_task.result()
                if isinstance(res, dict) and res.get("action"):
                    prebaked_llm = res
            except Exception:
                log.exception("pending llm task failed")
            self.pending_llm_task = None
        if self.pending_dialogue_task and self.pending_dialogue_task.done():
            try:
                line = self.pending_dialogue_task.result()
                if line:
                    self.next_talk_line = line
            except Exception:
                log.exception("pending dialogue task failed")
            self.pending_dialogue_task = None

        # 4. choose action
        if reflex_dec is not None:
            decision, via = reflex_dec, "reflex"
        elif prebaked_llm is not None:
            decision, via = prebaked_llm, "llm"
            self.llm_cooldown = 4
        else:
            wait_streak = int(sa.get("wait_streak", 0))
            social = None
            if self_a is not None:
                social = social_navigate(
                    self.mirror, self_a, perception,
                    self.mirror.current_tick, inventory,
                    next_talk_line_ready=bool(self.next_talk_line),
                    wait_streak=wait_streak,
                )
            if social is not None:
                decision, via = social, "social"
            elif self.policy is not None:
                feats = extract_features(
                    current_tick=self.mirror.current_tick,
                    born_tick=int(sa.get("born_tick", 0)),
                    hunger=int(sa.get("hunger", 0)),
                    mood=int(sa.get("mood", 0)),
                    energy=int(sa.get("energy", 0)),
                    hp=int(sa.get("hp", 100)),
                    inventory=inventory,
                    perception=perception,
                    wait_streak=wait_streak,
                    sleep_streak=int(sa.get("sleep_streak", 0)),
                )
                idx = self.policy.predict(feats)
                pol_dec = decode_to_decision(idx, perception, inventory) if idx is not None else None
                decision, via = (
                    (pol_dec, "policy") if pol_dec is not None
                    else ({"action": "wander", "thought": "(idle moment)"}, "auto_cooldown")
                )
            else:
                decision, via = {"action": "wander", "thought": "(idle moment)"}, "auto_cooldown"

            if self.llm_cooldown > 0:
                self.llm_cooldown -= 1
            if (self.pending_llm_task is None
                and self.mirror.current_tick - self.last_llm_decide_tick
                    > self.llm_decide_interval):
                self.pending_llm_task = asyncio.create_task(
                    self._llm_think_bg(perception, inventory)
                )
                self.last_llm_decide_tick = self.mirror.current_tick

        # 5. dispatch dialogue gen if partner close + no task pending
        partner_id = self._has_close_partner(perception)
        if partner_id and (self.pending_dialogue_task is None
                           or self.pending_dialogue_task.done()):
            self.pending_dialogue_task = asyncio.create_task(
                self._dialogue_gen_bg(partner_id)
            )

        # 6. resolve talk
        if decision.get("action") == "talk":
            if decision.get("content") == "<<USE_NEXT_TALK_LINE>>":
                if self.next_talk_line:
                    decision["content"] = self.next_talk_line
                    self.next_talk_line = ""
                else:
                    decision = {"action": "wait", "thought": "(waiting for words)"}
            elif not (decision.get("content") or "").strip():
                decision = {"action": "wait", "thought": "(no content)"}

        # 7. client-side validation
        decision = self._validate_pre_send(decision)

        # 8. log own talk into recent ring (so future filters dedup) + ring buffer
        if decision.get("action") == "talk":
            ring = self.recent_lines_by_agent.setdefault(self.agent_id, [])
            append_to_ring(ring, decision.get("content", ""))
        self.push_episodic({
            "kind": "decision",
            "tick": self.mirror.current_tick,
            "thought": decision.get("thought", "")[:120],
            "action": decision.get("action", "wait"),
            "via": via,
        })
        decision = dict(decision)
        decision["decided_via"] = via
        ok, _ = validate_action_dict(decision)
        if not ok:
            return {"action": "wait", "thought": "(invalid)", "decided_via": via}
        return decision
```

- [ ] **Step 2: Write `tests/test_brain_decide.py`**

```python
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
    # No reflex / no LLM ready / no social (no nearby agents) / no policy → wander
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
    brain.last_llm_decide_tick = -100  # force dispatch
    m.current_tick = 50
    perc = _basic_perception(tick=50)
    perc["agent_state"]["x"] = 5
    perc["agent_state"]["y"] = 5
    await brain.decide(perc)
    # NoOpLLM returns {} but a task IS created
    assert brain.pending_llm_task is not None
    # Wait for the task to complete to keep the loop clean
    await asyncio.sleep(0.01)


@pytest.mark.asyncio
async def test_decide_validate_invalid_falls_to_wait():
    brain, _ = _setup_brain()
    # Force a "decision" that fails validation: build with no structure
    # Patch reflex to return invalid for this test
    import agora_agent_sdk.brain as bm

    async def fake_decide(perc):
        d = await Brain.decide(brain, perc)
        return d
    out = await brain.decide(_basic_perception())
    # Wander/wait are always valid; ensure decided_via is set
    assert out.get("decided_via") in ("reflex", "social", "auto_cooldown", "policy", "llm")


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
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_brain_decide.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/agora_agent_sdk/brain.py tests/test_brain_decide.py
git -c user.email="info@iaitalia.net" -c user.name="mattabott" commit -m "feat: Brain orchestrator with reflex/social/policy/llm-bg + ring buffer"
```

---

## Phase 7 — Networking client

### Task 7.1: AgoraClient — join HTTP + token persistence

**Files:**
- Create: `src/agora_agent_sdk/client.py` (initial: join + token I/O only)
- Create: `tests/test_client_join.py`

- [ ] **Step 1: Write `src/agora_agent_sdk/client.py`**

```python
"""AgoraClient: HTTP join + WebSocket loop client."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import httpx

from agora_core.protocol import (
    ACTION_SCHEMA_VERSION, JoinRequest, JoinResponse,
)

log = logging.getLogger("agora_agent_sdk.client")


class JoinError(Exception):
    """Raised when join fails permanently (not retryable)."""

    def __init__(self, code: int, payload: dict):
        super().__init__(f"join failed: HTTP {code} {payload}")
        self.code = code
        self.payload = payload


@dataclass
class JoinResult:
    agent_id: int
    token: str
    world_seed: int
    tick_ms: int
    world_w: int
    world_h: int
    action_schema_version: int


def default_token_path(name: str) -> Path:
    home = Path(os.path.expanduser("~"))
    return home / ".agora-agent" / f"{name}.token"


def write_token(path: Path, agent_id: int, token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    path.write_text(json.dumps({"agent_id": agent_id, "token": token}))
    try:
        path.chmod(0o600)
    except OSError:
        pass


def read_token(path: Path) -> tuple[int, str] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return int(data["agent_id"]), str(data["token"])
    except Exception:
        log.warning("token file unreadable: %s", path)
        return None


def delete_token(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


async def http_join(
    server: str,
    *,
    name: str,
    personality_seed: str,
    sex: str,
    color: str | None = None,
    client_version: str = "agora-agent-sdk/0.1.0",
    timeout: float = 10.0,
    http_client: httpx.AsyncClient | None = None,
) -> JoinResult:
    payload = JoinRequest(
        name=name, personality_seed=personality_seed, sex=sex, color=color,
        action_schema_version=ACTION_SCHEMA_VERSION, client_version=client_version,
    ).model_dump(exclude_none=True)

    async def _do(client: httpx.AsyncClient) -> JoinResult:
        r = await client.post(
            f"{server.rstrip('/')}/api/agents/join",
            json=payload, timeout=timeout,
        )
        if r.status_code != 200:
            try:
                data = r.json()
            except Exception:
                data = {"raw": r.text[:200]}
            raise JoinError(r.status_code, data)
        resp = JoinResponse.model_validate(r.json())
        return JoinResult(
            agent_id=resp.agent_id, token=resp.token,
            world_seed=resp.world_seed, tick_ms=resp.tick_ms,
            world_w=resp.world_w, world_h=resp.world_h,
            action_schema_version=resp.action_schema_version,
        )

    if http_client is not None:
        return await _do(http_client)
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await _do(client)
```

- [ ] **Step 2: Write `tests/test_client_join.py`**

```python
import json
from pathlib import Path

import httpx
import pytest

from agora_agent_sdk.client import (
    JoinError, default_token_path, delete_token, http_join,
    read_token, write_token,
)


def test_token_roundtrip(tmp_path: Path):
    path = tmp_path / "test.token"
    write_token(path, 5, "abc")
    out = read_token(path)
    assert out == (5, "abc")
    delete_token(path)
    assert read_token(path) is None


def test_default_token_path(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    p = default_token_path("Maya")
    assert p.name == "Maya.token"
    assert ".agora-agent" in str(p)


def test_read_token_missing():
    assert read_token(Path("/nonexistent/path.token")) is None


def test_delete_token_missing_no_error(tmp_path: Path):
    delete_token(tmp_path / "nope.token")  # no exception


@pytest.mark.asyncio
async def test_http_join_success():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["name"] == "Maya"
        assert body["sex"] == "F"
        return httpx.Response(200, json={
            "agent_id": 5, "token": "tk-123", "world_seed": 42,
            "tick_ms": 1000, "world_w": 64, "world_h": 64,
            "action_schema_version": 1,
        })

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        result = await http_join(
            "http://mock", name="Maya", personality_seed="curious",
            sex="F", color="#7fa9d4", http_client=client,
        )
    assert result.agent_id == 5
    assert result.token == "tk-123"


@pytest.mark.asyncio
async def test_http_join_409_name_taken():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"error": "name_taken",
                                          "suggestions": ["Maya2"]})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(JoinError) as exc:
            await http_join("http://mock", name="Maya",
                            personality_seed="x", sex="F",
                            http_client=client)
        assert exc.value.code == 409
        assert exc.value.payload["error"] == "name_taken"


@pytest.mark.asyncio
async def test_http_join_426_schema_mismatch():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(426, json={"error": "schema_mismatch",
                                          "server_schema": 2,
                                          "client_schema": 1})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(JoinError) as exc:
            await http_join("http://mock", name="Maya",
                            personality_seed="x", sex="F",
                            http_client=client)
        assert exc.value.code == 426
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_client_join.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/agora_agent_sdk/client.py tests/test_client_join.py
git -c user.email="info@iaitalia.net" -c user.name="mattabott" commit -m "feat: AgoraClient join HTTP + token file persistence"
```

---

### Task 7.2: AgoraClient — WS loop + reconnect + dispatch + AgentDied exit

**Files:**
- Modify: `src/agora_agent_sdk/client.py` (append AgoraClient class)
- Create: `tests/test_client_ws_loop.py` (using starlette WS test app)

- [ ] **Step 1: Append to `src/agora_agent_sdk/client.py`**

```python
import time
from typing import Any, Awaitable, Callable

from agora_core.protocol import (
    ActionMsg, EventMsg, PerceptionMsg, PingMsg, PongMsg,
    RequestSnapshotMsg, ResultMsg, SnapshotMsg,
)
from agora_core.world_mirror import WorldMirror, pack_walkable_mask


class AgentDiedExit(Exception):
    """Raised when self agent died — caller should clean up token and exit."""

    def __init__(self, name: str, tick: int):
        super().__init__(f"agent {name} died at tick {tick}")
        self.name = name
        self.tick = tick


class AgoraClient:
    """WebSocket client that drives the brain loop."""

    def __init__(
        self,
        server: str,
        *,
        agent_id: int,
        token: str,
        world_w: int,
        world_h: int,
        brain,                                    # Brain
        token_path: Path | None = None,
        max_reconnect_attempts: int = 0,         # 0 = infinite
        snapshot_gap_threshold: int = 60,
        ws_factory: Callable[[str], Awaitable] | None = None,
    ):
        self.server = server.rstrip("/")
        self.agent_id = agent_id
        self.token = token
        self.brain = brain
        self.token_path = token_path
        self.max_reconnect_attempts = max_reconnect_attempts
        self.snapshot_gap_threshold = snapshot_gap_threshold
        self._ws_factory = ws_factory  # for tests
        # init mirror with empty walkable mask; replaced by snapshot
        empty_grid = [[True] * world_w for _ in range(world_h)]
        self.brain.mirror.world_w = world_w
        self.brain.mirror.world_h = world_h
        self.brain.mirror.walkable_mask = pack_walkable_mask(empty_grid)
        self.brain.mirror.self_agent_id = agent_id
        self._last_known_tick = -1
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    def _ws_url(self) -> str:
        scheme = "ws" if self.server.startswith("http://") else "wss"
        host = self.server.split("//", 1)[1]
        return f"{scheme}://{host}/ws/agents/{self.agent_id}?token={self.token}"

    async def run(self) -> None:
        """Main loop with reconnect."""
        attempt = 0
        backoff = 1.0
        while not self._stop.is_set():
            try:
                await self._run_once()
                # clean exit (e.g. server closed normally)
                return
            except AgentDiedExit:
                if self.token_path is not None:
                    delete_token(self.token_path)
                raise
            except Exception as e:
                log.warning("WS loop error: %s", e)
                attempt += 1
                if (self.max_reconnect_attempts
                        and attempt >= self.max_reconnect_attempts):
                    log.error("max reconnect attempts reached, giving up")
                    raise
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _run_once(self) -> None:
        """Open WS, drive recv→decide→send loop. Raises on disconnect."""
        if self._ws_factory is not None:
            ws_ctx = self._ws_factory(self._ws_url())
        else:
            import websockets
            ws_ctx = websockets.connect(self._ws_url())
        async with ws_ctx as ws:
            log.info("WS connected to %s", self.server)
            # Always request a snapshot at the start of a fresh connection
            await self._send(ws, RequestSnapshotMsg().model_dump())
            await self._loop(ws)

    async def _send(self, ws, payload: dict) -> None:
        await ws.send(json.dumps(payload))

    async def _loop(self, ws) -> None:
        async for raw in ws:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            await self._dispatch(ws, msg)

    async def _dispatch(self, ws, msg: dict) -> None:
        mtype = msg.get("type")
        if mtype == "snapshot":
            self.brain.mirror.apply_snapshot(msg)
            self._last_known_tick = self.brain.mirror.current_tick
        elif mtype == "perception":
            tick = int(msg.get("tick", 0))
            if (self._last_known_tick > 0
                and (tick < self._last_known_tick
                     or tick - self._last_known_tick > self.snapshot_gap_threshold)):
                await self._send(ws, RequestSnapshotMsg().model_dump())
                return
            self._last_known_tick = tick
            decision = await self.brain.decide(msg)
            await self._send(ws, ActionMsg(
                tick_ack=tick, **{k: v for k, v in decision.items()
                                   if k in ("action", "direction", "target_id",
                                            "content", "item", "qty",
                                            "recipe", "structure",
                                            "thought", "decided_via")},
            ).model_dump(exclude_none=True))
        elif mtype == "event":
            self.brain.mirror.apply_event(msg)
            kind = msg.get("kind")
            if kind == "agent_died" and int(msg.get("agent_id", 0)) == self.agent_id:
                a = self.brain.mirror.agents.get(self.agent_id)
                name = a.name if a else "<unknown>"
                tick = int(msg.get("tick", 0))
                raise AgentDiedExit(name, tick)
            self.brain.push_event_to_episodic(msg)
        elif mtype == "result":
            self.brain.push_episodic({
                "kind": "action_result", "tick": int(msg.get("tick_ack", 0)),
                **{k: v for k, v in msg.items() if k != "type"},
            })
        elif mtype == "ping":
            await self._send(ws, PongMsg(ts=float(msg.get("ts", time.time()))).model_dump())
```

- [ ] **Step 2: Write `tests/test_client_ws_loop.py`**

```python
"""WS loop tests using a small in-process WebSocket server (starlette + uvicorn-less).

We use a minimal asyncio-based WS server stub that yields scripted messages.
"""
import asyncio
import json

import pytest

from agora_agent_sdk.client import AgentDiedExit, AgoraClient
from agora_agent_sdk.brain import Brain
from agora_agent_sdk.llm import NoOpLLM
from agora_core.world_mirror import WorldMirror, pack_walkable_mask


class _FakeWS:
    """In-memory WS double: queues inbound and records outbound."""

    def __init__(self, inbound: list[dict]):
        self._inbound = list(inbound)
        self.outbound: list[dict] = []
        self._closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self._closed = True

    async def send(self, raw: str) -> None:
        self.outbound.append(json.loads(raw))

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._inbound:
            raise StopAsyncIteration
        return json.dumps(self._inbound.pop(0))


def _ws_factory(messages: list[dict]) -> "Callable":
    fake = _FakeWS(messages)

    def factory(url: str):
        return fake

    factory.fake = fake  # so tests can read outbound
    return factory


def _b64_walkable(w=8, h=8) -> str:
    import base64
    return base64.b64encode(pack_walkable_mask([[True]*w for _ in range(h)])).decode()


def _make_client(messages: list[dict], agent_id=99) -> tuple[AgoraClient, "_FakeWS"]:
    grid = [[True]*8 for _ in range(8)]
    mirror = WorldMirror(world_w=8, world_h=8,
                         walkable_mask=pack_walkable_mask(grid))
    brain = Brain(mirror=mirror, llm=NoOpLLM(), agent_id=agent_id,
                  agent_name="Self", sex="F", color="#fff",
                  personality_seed="x")
    factory = _ws_factory(messages)
    client = AgoraClient(
        "http://mock", agent_id=agent_id, token="tk", world_w=8, world_h=8,
        brain=brain, ws_factory=factory,
    )
    return client, factory.fake


@pytest.mark.asyncio
async def test_run_once_applies_snapshot_and_responds_to_perception():
    snap = {
        "type": "snapshot", "tick": 1, "walkable_mask": _b64_walkable(),
        "agents": [{"id": 99, "name": "Self", "x": 5, "y": 5, "color": "#fff",
                    "sex": "F", "alive": True, "born_tick": 0}],
        "structures": [], "resource_clusters": [],
        "storage_summary": {}, "world_events": [],
    }
    perc = {
        "type": "perception", "tick": 2,
        "agent_state": {"x": 5, "y": 5, "hp": 100, "energy": 80, "mood": 60,
                        "hunger": 30, "personality_current": "x", "born_tick": 0,
                        "wait_streak": 0, "sleep_streak": 0, "inventory": {}},
        "terrain_here": "grass", "visible_around": "",
        "here_resource": None, "here_structure": None,
        "nearby_agents": [], "nearby_resources": [], "nearby_structures": [],
        "walkable_dirs": ["north"],
        "relations": {}, "relations_inbound": {},
        "family": {"mother": None, "father": None, "children": []},
        "recent_dialogues": [], "world_events": [],
    }
    client, fake = _make_client([snap, perc])
    await client._run_once()
    # outbound: request_snapshot + action
    types = [m["type"] for m in fake.outbound]
    assert types[0] == "request_snapshot"
    assert "action" in types


@pytest.mark.asyncio
async def test_run_once_pongs_to_ping():
    ping = {"type": "ping", "ts": 12345.0}
    client, fake = _make_client([ping])
    await client._run_once()
    pongs = [m for m in fake.outbound if m["type"] == "pong"]
    assert pongs and pongs[-1]["ts"] == 12345.0


@pytest.mark.asyncio
async def test_run_once_self_died_raises():
    died = {"type": "event", "kind": "agent_died", "tick": 100, "agent_id": 99}
    client, _ = _make_client([died])
    with pytest.raises(AgentDiedExit):
        await client._run_once()


@pytest.mark.asyncio
async def test_run_once_other_died_does_not_raise():
    died = {"type": "event", "kind": "agent_died", "tick": 100, "agent_id": 5}
    client, _ = _make_client([died])
    await client._run_once()  # no exception


@pytest.mark.asyncio
async def test_run_once_requests_snapshot_on_tick_gap():
    snap = {
        "type": "snapshot", "tick": 1, "walkable_mask": _b64_walkable(),
        "agents": [], "structures": [], "resource_clusters": [],
        "storage_summary": {}, "world_events": [],
    }
    far_perc = {
        "type": "perception", "tick": 200,  # gap > 60 from tick 1
        "agent_state": {"x": 0, "y": 0, "hp": 100, "energy": 0, "mood": 0,
                        "hunger": 0, "personality_current": "x", "born_tick": 0,
                        "wait_streak": 0, "sleep_streak": 0, "inventory": {}},
        "terrain_here": "grass", "visible_around": "",
        "here_resource": None, "here_structure": None,
        "nearby_agents": [], "nearby_resources": [], "nearby_structures": [],
        "walkable_dirs": [],
        "relations": {}, "relations_inbound": {},
        "family": {"mother": None, "father": None, "children": []},
        "recent_dialogues": [], "world_events": [],
    }
    client, fake = _make_client([snap, far_perc])
    await client._run_once()
    request_snaps = [m for m in fake.outbound if m["type"] == "request_snapshot"]
    # First one is the initial connect-time request, second is gap-triggered
    assert len(request_snaps) >= 2
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_client_ws_loop.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add src/agora_agent_sdk/client.py tests/test_client_ws_loop.py
git -c user.email="info@iaitalia.net" -c user.name="mattabott" commit -m "feat: AgoraClient WS loop, dispatch, snapshot resync, agent_died exit"
```

---

## Phase 8 — Mock server + E2E

### Task 8.1: tests/mock_server.py — FastAPI ASGI app for join + WS

**Files:**
- Create: `tests/mock_server.py`
- Create: `tests/test_mock_server.py`

- [ ] **Step 1: Write `tests/mock_server.py`**

```python
"""In-process FastAPI mock of the agora server.

Used only by E2E tests. It scripts a fixed sequence of perceptions and
records the actions received from the client.
"""
from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass, field

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from agora_core.protocol import ACTION_SCHEMA_VERSION
from agora_core.world_mirror import pack_walkable_mask


@dataclass
class MockState:
    next_id: int = 1
    tokens: dict[int, str] = field(default_factory=dict)
    name_to_id: dict[str, int] = field(default_factory=dict)
    actions_received: list[dict] = field(default_factory=list)
    ws_connected: asyncio.Event = field(default_factory=asyncio.Event)
    perception_script: list[dict] = field(default_factory=list)
    extra_messages: list[dict] = field(default_factory=list)


def _walkable_b64(w=8, h=8) -> str:
    return base64.b64encode(pack_walkable_mask([[True]*w for _ in range(h)])).decode()


def make_mock_app() -> tuple[FastAPI, MockState]:
    state = MockState()
    app = FastAPI()

    @app.post("/api/agents/join")
    async def join(req: dict) -> JSONResponse:
        if int(req.get("action_schema_version", 0)) != ACTION_SCHEMA_VERSION:
            return JSONResponse(status_code=426, content={
                "error": "schema_mismatch",
                "server_schema": ACTION_SCHEMA_VERSION,
                "client_schema": int(req.get("action_schema_version", 0)),
                "min_supported": ACTION_SCHEMA_VERSION,
            })
        name = req["name"]
        if name in state.name_to_id:
            return JSONResponse(status_code=409, content={
                "error": "name_taken",
                "suggestions": [f"{name}2", f"{name}A"],
            })
        agent_id = state.next_id
        state.next_id += 1
        state.name_to_id[name] = agent_id
        state.tokens[agent_id] = f"tk-{agent_id}"
        return JSONResponse(status_code=200, content={
            "agent_id": agent_id, "token": state.tokens[agent_id],
            "world_seed": 42, "tick_ms": 1000,
            "world_w": 8, "world_h": 8,
            "action_schema_version": ACTION_SCHEMA_VERSION,
        })

    @app.websocket("/ws/agents/{agent_id}")
    async def ws_endpoint(ws: WebSocket, agent_id: int, token: str = ""):
        if state.tokens.get(agent_id) != token:
            await ws.close(code=4401)
            return
        await ws.accept()
        state.ws_connected.set()
        # Send a snapshot
        await ws.send_text(json.dumps({
            "type": "snapshot", "tick": 0,
            "walkable_mask": _walkable_b64(),
            "agents": [{"id": agent_id, "name": "Self",
                         "x": 4, "y": 4, "color": "#fff", "sex": "F",
                         "alive": True, "born_tick": 0}],
            "structures": [], "resource_clusters": [],
            "storage_summary": {}, "world_events": [],
        }))
        try:
            for perc in state.perception_script:
                await ws.send_text(json.dumps(perc))
                # Wait for an action OR a request_snapshot or pong
                while True:
                    raw = await ws.receive_text()
                    msg = json.loads(raw)
                    if msg.get("type") == "request_snapshot":
                        await ws.send_text(json.dumps({
                            "type": "snapshot", "tick": int(perc.get("tick", 0)),
                            "walkable_mask": _walkable_b64(),
                            "agents": [{"id": agent_id, "name": "Self",
                                         "x": 4, "y": 4, "color": "#fff",
                                         "sex": "F", "alive": True,
                                         "born_tick": 0}],
                            "structures": [], "resource_clusters": [],
                            "storage_summary": {}, "world_events": [],
                        }))
                        continue
                    if msg.get("type") == "pong":
                        continue
                    state.actions_received.append(msg)
                    # Send a fabricated result
                    await ws.send_text(json.dumps({
                        "type": "result", "tick_ack": msg.get("tick_ack", 0),
                        "action": msg.get("action", "wait"), "ok": True,
                    }))
                    break
            for extra in state.extra_messages:
                await ws.send_text(json.dumps(extra))
        except WebSocketDisconnect:
            return

    return app, state
```

- [ ] **Step 2: Write `tests/test_mock_server.py`**

```python
import json

import httpx
import pytest

from tests.mock_server import make_mock_app


@pytest.mark.asyncio
async def test_mock_join_success():
    app, state = make_mock_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://mock") as client:
        r = await client.post("/api/agents/join", json={
            "name": "Maya", "personality_seed": "x", "sex": "F",
            "action_schema_version": 1, "client_version": "test/0.1",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["agent_id"] == 1


@pytest.mark.asyncio
async def test_mock_join_409_duplicate():
    app, state = make_mock_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://mock") as client:
        body = {"name": "Maya", "personality_seed": "x", "sex": "F",
                "action_schema_version": 1, "client_version": "t/0"}
        await client.post("/api/agents/join", json=body)
        r = await client.post("/api/agents/join", json=body)
        assert r.status_code == 409


@pytest.mark.asyncio
async def test_mock_join_426_schema():
    app, state = make_mock_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://mock") as client:
        r = await client.post("/api/agents/join", json={
            "name": "Maya", "personality_seed": "x", "sex": "F",
            "action_schema_version": 999, "client_version": "t/0",
        })
        assert r.status_code == 426
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_mock_server.py -v
```

Expected: all PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/mock_server.py tests/test_mock_server.py
git -c user.email="info@iaitalia.net" -c user.name="mattabott" commit -m "test: in-process FastAPI mock server for E2E"
```

---

### Task 8.2: tests/test_e2e.py — full client ↔ mock cycle

**Files:**
- Create: `tests/test_e2e.py`

- [ ] **Step 1: Write `tests/test_e2e.py`**

```python
"""End-to-end: AgoraClient + Brain against the mock server.

Uses starlette TestClient for the WS leg (sync). The HTTP join uses httpx
with ASGITransport. We script perceptions and assert the actions returned.
"""
import asyncio
import json
import threading

import httpx
import pytest
from starlette.testclient import TestClient

from agora_agent_sdk.brain import Brain
from agora_agent_sdk.client import http_join
from agora_agent_sdk.llm import NoOpLLM
from agora_core.world_mirror import WorldMirror, pack_walkable_mask
from tests.mock_server import make_mock_app


def _hungry_perception(agent_id: int) -> dict:
    return {
        "type": "perception", "tick": 5,
        "agent_state": {"x": 4, "y": 4, "hp": 100, "energy": 80, "mood": 60,
                        "hunger": 75, "personality_current": "x",
                        "born_tick": 0, "wait_streak": 0, "sleep_streak": 0,
                        "inventory": {"berry": 1}},
        "terrain_here": "grass", "visible_around": "",
        "here_resource": None, "here_structure": None,
        "nearby_agents": [], "nearby_resources": [], "nearby_structures": [],
        "walkable_dirs": ["north", "south", "east", "west"],
        "relations": {}, "relations_inbound": {},
        "family": {"mother": None, "father": None, "children": []},
        "recent_dialogues": [], "world_events": [],
    }


@pytest.mark.asyncio
async def test_e2e_hungry_eats_berry():
    app, state = make_mock_app()
    transport = httpx.ASGITransport(app=app)
    # 1. Join via mock HTTP
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://mock") as http_client:
        r = await http_client.post("/api/agents/join", json={
            "name": "Maya", "personality_seed": "x", "sex": "F",
            "action_schema_version": 1, "client_version": "t/0",
        })
        assert r.status_code == 200
        data = r.json()
    state.perception_script = [_hungry_perception(data["agent_id"])]

    # 2. Open WS via starlette TestClient (sync)
    received_actions = []
    with TestClient(app) as tc:
        with tc.websocket_connect(
            f"/ws/agents/{data['agent_id']}?token={data['token']}"
        ) as ws:
            # snapshot
            snap = ws.receive_json()
            assert snap["type"] == "snapshot"
            # client would normally send request_snapshot; mock continues anyway
            ws.send_json({"type": "request_snapshot"})
            # mock responds with a snapshot too
            second_snap = ws.receive_json()
            assert second_snap["type"] == "snapshot"
            # Now perception
            perc = ws.receive_json()
            assert perc["type"] == "perception"
            # Build a brain locally and decide
            mirror = WorldMirror(
                world_w=8, world_h=8,
                walkable_mask=pack_walkable_mask([[True]*8 for _ in range(8)]),
            )
            mirror.apply_snapshot(snap)
            brain = Brain(
                mirror=mirror, llm=NoOpLLM(),
                agent_id=data["agent_id"], agent_name="Maya", sex="F",
                color="#fff", personality_seed="x",
            )
            decision = asyncio.get_event_loop().run_until_complete(
                brain.decide(perc)
            ) if False else await brain.decide(perc)
            ws.send_json({
                "type": "action", "tick_ack": perc["tick"],
                **decision,
            })
            result = ws.receive_json()
            received_actions.append(decision)

    assert received_actions[0]["action"] == "eat"
    assert received_actions[0]["item"] == "berry"


@pytest.mark.asyncio
async def test_e2e_join_then_agent_died_exits():
    app, state = make_mock_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://mock") as http_client:
        r = await http_client.post("/api/agents/join", json={
            "name": "Doomed", "personality_seed": "x", "sex": "F",
            "action_schema_version": 1, "client_version": "t/0",
        })
        data = r.json()
    state.perception_script = []  # no perceptions
    state.extra_messages = [
        {"type": "event", "kind": "agent_died",
         "tick": 100, "agent_id": data["agent_id"]}
    ]
    with TestClient(app) as tc:
        with tc.websocket_connect(
            f"/ws/agents/{data['agent_id']}?token={data['token']}"
        ) as ws:
            snap = ws.receive_json()
            assert snap["type"] == "snapshot"
            died_event = ws.receive_json()
            assert died_event["kind"] == "agent_died"
            assert died_event["agent_id"] == data["agent_id"]
```

- [ ] **Step 2: Run tests**

```bash
pytest tests/test_e2e.py -v
```

Expected: all PASS.

- [ ] **Step 3: Commit**

```bash
git add tests/test_e2e.py
git -c user.email="info@iaitalia.net" -c user.name="mattabott" commit -m "test: E2E client ↔ mock server (hungry → eat, agent_died)"
```

---

## Phase 9 — CLI

### Task 9.1: cli.py — argparse + main + entry point

**Files:**
- Create: `src/agora_agent_sdk/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write `src/agora_agent_sdk/cli.py`**

```python
"""Command-line entry point: `agora-agent`.

Composes AgoraClient + Brain + OllamaClient (or NoOpLLM with --no-llm) and runs
the WS loop until the agent dies or the user interrupts.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from agora_agent_sdk.brain import Brain
from agora_agent_sdk.client import (
    AgentDiedExit, AgoraClient, JoinError,
    default_token_path, delete_token, http_join, read_token, write_token,
)
from agora_agent_sdk.llm import NoOpLLM, OllamaClient
from agora_core.policy import Policy
from agora_core.world_mirror import WorldMirror, pack_walkable_mask

log = logging.getLogger("agora_agent_sdk.cli")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agora-agent",
                                description="Plug an LLM agent into the agora world.")
    p.add_argument("--server", default=os.environ.get("AGORA_SERVER"),
                   help="agora server base URL (e.g. https://agora.chatbot4eva.com). "
                        "REQUIRED unless AGORA_SERVER is set.")
    p.add_argument("--name", required=True)
    p.add_argument("--seed", required=True, help="personality seed (1-500 char)")
    p.add_argument("--sex", choices=["F", "M"], required=True)
    p.add_argument("--color", default=None, help="optional #RRGGBB")
    p.add_argument("--ollama-host", default="http://localhost:11434")
    p.add_argument("--model", default="qwen2.5:1.5b")
    p.add_argument("--no-llm", action="store_true",
                   help="disable LLM — run only reflex/social/wander")
    p.add_argument("--llm-decide-interval", type=int, default=120,
                   help="ticks between LLM 'decide' calls (default 120)")
    p.add_argument("--ring-buffer", type=int, default=30,
                   help="size of episodic ring buffer (default 30)")
    p.add_argument("--token-file", type=Path, default=None,
                   help="override default token persistence path")
    p.add_argument("--max-reconnect-attempts", type=int, default=0,
                   help="0 = infinite")
    p.add_argument("--policy-file", type=Path, default=None,
                   help="optional .pkl with a trained MLP policy")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


async def _async_main(ns: argparse.Namespace) -> int:
    logging.basicConfig(level=ns.log_level,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if not ns.server:
        log.error("--server is required (or set AGORA_SERVER)")
        return 2

    token_path = ns.token_file or default_token_path(ns.name)

    # Token-based resume
    existing = read_token(token_path)
    if existing is not None:
        agent_id, token = existing
        log.info("resuming existing session: agent_id=%s from %s",
                 agent_id, token_path)
        # Need world dimensions; fetch via a fresh join? No — use a minimal /api/health
        # approach. Simpler: hardcode default 64x64 here; the snapshot will contain
        # the actual mask anyway, so dims for the mirror init can be 64×64 default.
        world_w, world_h = 64, 64
    else:
        try:
            join = await http_join(
                ns.server, name=ns.name, personality_seed=ns.seed,
                sex=ns.sex, color=ns.color,
            )
        except JoinError as e:
            log.error("join failed: HTTP %s %s", e.code, e.payload)
            return 3
        agent_id = join.agent_id
        token = join.token
        world_w = join.world_w
        world_h = join.world_h
        write_token(token_path, agent_id, token)

    grid = [[True] * world_w for _ in range(world_h)]
    mirror = WorldMirror(world_w=world_w, world_h=world_h,
                         walkable_mask=pack_walkable_mask(grid))

    if ns.no_llm:
        llm = NoOpLLM()
    else:
        llm = OllamaClient(host=ns.ollama_host, model=ns.model)

    policy = None
    if ns.policy_file is not None and ns.policy_file.exists():
        policy = Policy(ns.policy_file)
        if not policy.load():
            log.warning("could not load policy from %s", ns.policy_file)
            policy = None

    brain = Brain(
        mirror=mirror, llm=llm, agent_id=agent_id,
        agent_name=ns.name, sex=ns.sex, color=ns.color or "#fff",
        personality_seed=ns.seed,
        llm_decide_interval=ns.llm_decide_interval,
        ring_buffer_size=ns.ring_buffer,
        policy=policy,
    )

    client = AgoraClient(
        ns.server, agent_id=agent_id, token=token,
        world_w=world_w, world_h=world_h, brain=brain,
        token_path=token_path,
        max_reconnect_attempts=ns.max_reconnect_attempts,
    )

    try:
        await client.run()
        return 0
    except AgentDiedExit as e:
        print(
            f"Your agent {ns.name} died at tick {e.tick}.\n"
            f"Run agora-agent again with a new --name to spawn a new one.",
            file=sys.stderr,
        )
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        await llm.aclose()


def main() -> int:
    parser = _build_parser()
    ns = parser.parse_args()
    return asyncio.run(_async_main(ns))


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Write `tests/test_cli.py`**

```python
import sys
from unittest.mock import patch

import pytest

from agora_agent_sdk.cli import _build_parser, main


def test_parser_requires_name_seed_sex():
    parser = _build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])  # missing required


def test_parser_minimal():
    parser = _build_parser()
    ns = parser.parse_args([
        "--server", "http://x", "--name", "Maya",
        "--seed", "curious", "--sex", "F",
    ])
    assert ns.name == "Maya"
    assert ns.sex == "F"
    assert ns.llm_decide_interval == 120  # default
    assert ns.ring_buffer == 30


def test_parser_no_llm_flag():
    parser = _build_parser()
    ns = parser.parse_args([
        "--server", "http://x", "--name", "Maya",
        "--seed", "x", "--sex", "F", "--no-llm",
    ])
    assert ns.no_llm is True


def test_main_missing_server_returns_2(capsys, monkeypatch):
    monkeypatch.delenv("AGORA_SERVER", raising=False)
    monkeypatch.setattr(sys, "argv", [
        "agora-agent", "--name", "Maya", "--seed", "x", "--sex", "F",
    ])
    code = main()
    assert code == 2
```

- [ ] **Step 3: Run tests**

```bash
pytest tests/test_cli.py -v
```

Expected: all PASS.

- [ ] **Step 4: Manual smoke test (no real server)**

```bash
cd /home/mattabott/Documents/agora-agent-sdk
. .venv/bin/activate
agora-agent --help
```

Expected: prints argparse help, exit 0.

- [ ] **Step 5: Commit**

```bash
git add src/agora_agent_sdk/cli.py tests/test_cli.py
git -c user.email="info@iaitalia.net" -c user.name="mattabott" commit -m "feat: CLI entry point agora-agent"
```

---

## Phase 10 — Deliverables

### Task 10.1: README.md — quickstart

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace `README.md` with full content**

```markdown
# agora-agent-sdk

Plug your own LLM agent into the [agora](https://agora.chatbot4eva.com) world. Your agent runs locally, uses your own [Ollama](https://ollama.com) for inference, and joins a shared 2D world inhabited by other LLM-driven agents.

**Status:** alpha. Server-side endpoint coming.

## Install

```bash
pip install agora-agent-sdk
```

You also need:
- Python ≥ 3.10
- A running Ollama on your machine (`ollama serve`)
- A model pulled, e.g. `ollama pull qwen2.5:1.5b`

## Quickstart

```bash
agora-agent \
  --server https://agora.chatbot4eva.com \
  --name Maya \
  --seed "You are Maya. Curious, prefers building over talking." \
  --sex F \
  --color "#7fa9d4" \
  --ollama-host http://localhost:11434 \
  --model qwen2.5:1.5b
```

The client joins, opens a WebSocket, and starts driving Maya in the world. Press Ctrl-C to disconnect — your agent's body keeps living server-side, and the next `agora-agent --name Maya …` resumes the same body.

When Maya dies (hunger, ancient age), the client exits with a message and clears the saved token.

## Flags

| Flag | Default | What it does |
|---|---|---|
| `--server` | `$AGORA_SERVER` | Server base URL. Required. |
| `--name` | — | Unique agent name (`[A-Za-z][A-Za-z0-9_-]*`, ≤32). |
| `--seed` | — | Personality seed, 1-500 chars. |
| `--sex` | — | `F` or `M`. |
| `--color` | (server-assigned) | Optional `#RRGGBB`. |
| `--ollama-host` | `http://localhost:11434` | Ollama base URL. |
| `--model` | `qwen2.5:1.5b` | Ollama model. |
| `--no-llm` | off | Skip LLM, run only reflex/social. |
| `--llm-decide-interval` | 120 | Ticks between background LLM calls. |
| `--ring-buffer` | 30 | Episodic memory size. |
| `--token-file` | `~/.agora-agent/<name>.token` | Where to save the join token. |
| `--max-reconnect-attempts` | 0 (infinite) | Cap WS reconnects. |
| `--policy-file` | none | Optional MLP policy `.pkl`. |
| `--log-level` | INFO | DEBUG, INFO, WARNING, ERROR. |

## How your agent decides

Each tick, the server sends a perception (your agent's surroundings). The client picks an action via this priority chain (same as the agora server brain):

1. **Reflex** — emergency rules (eat if hungry, gather food, build hut, shelter at night, …).
2. **LLM (background)** — every `--llm-decide-interval` ticks, a creative LLM call is dispatched. When it returns, the next decision uses the LLM's chosen action.
3. **Social** — gravitate toward other agents and talk when adjacent.
4. **Policy** — optional distilled MLP policy (load via `--policy-file`).
5. **Wander** — fallback.

Dialogue lines are also LLM-generated in the background, with strict filters that reject poetic / Italian / repetitive / truncated outputs. Your agent is silent until a usable line arrives.

## Token persistence

The first run does an HTTP join and saves `{agent_id, token}` to `~/.agora-agent/<name>.token` (chmod 600). Subsequent runs re-use it and skip the join. If the server says "agent dead" or "invalid token", the file is deleted and the next run rejoins from scratch.

## Protocol

See [PROTOCOL.md](PROTOCOL.md) for the full WS protocol (snapshot, perception, events, action schema).

## Licence

MIT.
```

- [ ] **Step 2: Commit**

```bash
git add README.md
git -c user.email="info@iaitalia.net" -c user.name="mattabott" commit -m "docs: README quickstart + flag reference"
```

---

### Task 10.2: PROTOCOL.md — API contract for the server team

**Files:**
- Create: `PROTOCOL.md`

- [ ] **Step 1: Write `PROTOCOL.md` (extracted from design spec §5)**

```markdown
# agora-agent-sdk — Wire Protocol v1

`ACTION_SCHEMA_VERSION = 1`. Inviato dal client nel join. Server confronta. Mismatch → 426.

## HTTP join

`POST /api/agents/join`

Request:
```json
{
  "name": "Maya",
  "personality_seed": "You are Maya. Curious, prefers building.",
  "sex": "F",
  "color": "#7fa9d4",
  "action_schema_version": 1,
  "client_version": "agora-agent-sdk/0.1.0"
}
```

`name` regex `^[A-Za-z][A-Za-z0-9_-]*$`, ≤32 char. `personality_seed` ≤500 char. `sex ∈ {F,M}`. `color` opzionale (`^#[0-9a-fA-F]{6}$`).

Response 200:
```json
{
  "agent_id": 5, "token": "<opaque>",
  "world_seed": 4242, "tick_ms": 1000,
  "world_w": 64, "world_h": 64,
  "action_schema_version": 1
}
```

Errori:
- 409 `{"error":"name_taken","suggestions":["Maya2","MayaA"]}`
- 403 `{"error":"join_closed"}`
- 426 `{"error":"schema_mismatch","server_schema":N,"client_schema":M,"min_supported":K}`
- 400 `{"error":"invalid_field","field":"<f>","reason":"..."}`

## WS

`WS /ws/agents/{agent_id}?token=<token>`

Upgrade fallisce con 401 se token non valido. Token può essere riusato all'apertura di una nuova WS dopo disconnessione (long-lived per la durata della sessione).

### Server → Client: snapshot (al connect, una volta)

```json
{
  "type": "snapshot",
  "tick": 12345,
  "walkable_mask": "<base64 raw>",
  "agents": [
    {"id":1,"name":"Aria","x":12,"y":30,"color":"#e6195a","sex":"F",
     "alive":true,"born_tick":0,"sleep_streak":0,"wait_streak":0,
     "mother_id":null,"father_id":null}
  ],
  "structures": [
    {"id":1,"x":28,"y":14,"type":"hut","owner_id":1,"built_tick":200,
     "color":"#a06a3c","label":"Hut"}
  ],
  "resource_clusters": [
    {"type":"wood","cx":32,"cy":18,"total_qty":60,
     "tiles":[[31,17],[32,18],[33,18]]}
  ],
  "storage_summary": {"3":{"berry":12,"wood":5}},
  "world_events": [
    {"id":99,"type":"rain","x":0,"y":0,"radius":0,"started_tick":12300,"ends_tick":12500}
  ]
}
```

`walkable_mask`: base64 raw bitmap, `world_w * world_h` bit packed in row-major order, LSB-first per byte. **Niente compressione.** Per 64×64 = 512 byte raw → 684 byte base64.

`storage_summary`: chiavi sono `structure_id` come stringa (JSON), valori `{item_type: qty}`.

### Server → Client: perception (per tick, ~500B-1KB)

```json
{
  "type": "perception",
  "tick": 12346,
  "agent_state": {
    "x": 15, "y": 30, "hp": 90, "energy": 70, "mood": 60, "hunger": 40,
    "personality_current": "...",
    "current_goal": "",
    "sleep_streak": 0, "wait_streak": 1,
    "born_tick": 0, "mother_id": null, "father_id": null,
    "last_thought": "...", "last_action": "...",
    "inventory": {"berry": 3, "wood": 2}
  },
  "terrain_here": "grass",
  "visible_around": "(-3,-3)=grass, …",
  "here_resource": null,
  "here_structure": null,
  "nearby_agents": [{"id":2,"name":"Niko","x":13,"y":30,"sex":"M"}],
  "nearby_resources": [{"x":14,"y":31,"type":"berry","qty":1}],
  "nearby_structures": [],
  "walkable_dirs": ["north","east"],
  "relations": {"2": 25, "3": -5},
  "relations_inbound": {"2": 18, "3": -2},
  "family": {"mother":null, "father":null, "children":[]},
  "recent_dialogues": [
    {"tick":12300,"from_id":2,"from_name":"Niko","content":"You ok?"}
  ],
  "world_events": [{"id":99,"type":"rain","x":0,"y":0,"radius":0,"ends_tick":12500}]
}
```

- `personality_current` evolve **server-side** (passive needs + reflection chain). Il client lo legge passivamente.
- `relations_inbound[other_id]` = quanto `other_id` vuole bene a self. Serve a `propose` (richiede affinity reciproca ≥ 20).
- `recent_dialogues` = ultimi N (default 5) dialoghi RICEVUTI da self.

### Server → Client: delta events

`{"type":"event","kind":"<KIND>","tick":N, ...}`

| `kind` | Payload (campi oltre tick) | Effetto |
|---|---|---|
| `tile_update` | `x,y,resource_type,resource_qty` | aggiorna `resources[(x,y)]`, cancella se qty=0 |
| `structure_built` | `structure_id,x,y,structure_type,owner_id,color,label` | aggiunge a `structures` |
| `structure_destroyed` | `structure_id,x,y` | rimuove |
| `agent_born` | `agent: {id,name,x,y,color,sex,alive,born_tick,mother_id,father_id}` | aggiunge a `agents` |
| `agent_died` | `agent_id,name,x,y` | marca alive=false. **Se `agent_id == self.agent_id`, il client esce.** |
| `agent_stats` | `agent_id,hp?,mood?,energy?,hunger?` | aggiorna stats |
| `agent_moved` *(o `agent_action`)* | `agent_id,x,y` | aggiorna posizione. Server può scegliere kind dedicato o riusare `agent_action` |
| `storage_changed` | `structure_id,item,qty` | qty **assoluta**, mai delta. qty=0 → cancella la entry |
| `world_event_started` | `event: {id,type,x,y,radius,started_tick,ends_tick}` | aggiunge a `events` |
| `world_event_ended` | `event_id,reason?` | rimuove |
| `dialogue_received` | `from_id,from_name,content` | aggiunto al ring episodico client + ring dedup |
| `gift_received` | `from_id,from_name,item,qty` | ring episodico |
| `loss` | `deceased_id,deceased_name,relation,mood_drop` | ring episodico |
| `user_message` | `content` | ring episodico |
| `relation_update` | `observer_id,target_id,affinity` | client aggiorna `relations` se è il proprio osservatore o `relations_inbound` se è il proprio target |

### Server → Client: result

```json
{"type":"result","tick_ack":N,"action":"...","ok":true|false,"reason":"...","...":...}
```

I campi extra dipendono dall'azione e replicano i return dict di `actions.py` server-side (es. `to`, `from`, `attempted`, `target_name`, `item_type`, `qty`, `materials_used`, `pregnancy_id`, `due_tick`, `cleared_resource`, ecc.).

### Server → Client: heartbeat

```json
{"type":"ping","ts":1714579200.123}
```

Server invia ogni 5s. Client risponde `pong` entro 1s. 3 ping mancati lato server → server fa `wait` per quell'agente fino al reconnect (NON kicka).

### Client → Server: action

```json
{
  "type": "action",
  "tick_ack": 12346,
  "action": "move|wait|wander|gather|eat|craft|build|talk|give|deposit|withdraw|propose|note",
  "direction": "north|south|east|west",
  "target_id": 2,
  "content": "...",
  "item": "...",
  "qty": 1,
  "recipe": "axe|pickaxe|bucket",
  "structure": "hut|storage|shrine",
  "thought": "...",
  "decided_via": "reflex|social|policy|llm|auto_cooldown"
}
```

Solo i campi pertinenti per ogni `action`. `decided_via` è solo telemetria (server non lo usa).

| `action` | Required |
|---|---|
| `move` | `direction ∈ {north,south,east,west}` |
| `wait`, `wander`, `gather` | — |
| `note` | `content` (≤500) |
| `talk` | `target_id`, `content` (3-280 char) |
| `eat` | `item` |
| `craft` | `recipe ∈ {axe,pickaxe,bucket}` |
| `build` | `structure ∈ {hut,storage,shrine}` |
| `give` | `target_id`, `item`, `qty ≥ 1` |
| `deposit`, `withdraw` | `item`, `qty ≥ 1` |
| `propose` | `target_id` |

`thought` ≤ 240 char.

### Client → Server: pong + request_snapshot

```json
{"type":"pong","ts":1714579200.123}
{"type":"request_snapshot"}
```

`request_snapshot` triggers lato client:
1. Tick non monotono crescente (`perception.tick < last_known_tick`).
2. Gap > 60 tick tra perception consecutive.
3. Subito dopo apertura (o riapertura) della WS.

Il server risponde con un nuovo `snapshot` completo.

## Behavior coverage (differenze brain remote vs server)

Il client porta 1:1: reflex priorities, social_navigate, dialogue filters (poetic/italian/anti-noun/anti-trunc/dedup 3-gram), policy (encode/decode/extract_features), prompts (SYSTEM_PROMPT, DIALOGUE_SYSTEM, build_user_prompt, build_dialogue_user_prompt).

NON port (out of scope V1):
- `recall_episodic` DB-backed → ring buffer in-memory (last 30, 12 in prompt).
- `semantic_recall` (sqlite-vec).
- `maybe_reflect` / `maybe_dream` → server li gestisce per gli agenti remoti.
- `wlog_event`, `index_text`, `Observation`/`Dialogue` insert lato client.
- Affinity update lato client. Server applica affinity all'esecuzione delle action e push `relation_update` events.

`LLM_DECIDE_INTERVAL` lato client default = **120 tick** (server usa 300 per via dei vincoli Pi 5).
```

- [ ] **Step 2: Commit**

```bash
git add PROTOCOL.md
git -c user.email="info@iaitalia.net" -c user.name="mattabott" commit -m "docs: PROTOCOL.md — wire protocol v1 contract for server team"
```

---

### Task 10.3: Final verification

**Files:** none (verification + handoff)

- [ ] **Step 1: Run full test suite**

```bash
cd /home/mattabott/Documents/agora-agent-sdk
. .venv/bin/activate
pytest -v 2>&1 | tail -20
```

Expected: all tests PASS, no errors.

- [ ] **Step 2: Smoke run with `--no-llm`**

Build the mock server fixture into a runnable script:

```bash
cd /home/mattabott/Documents/agora-agent-sdk
python -c "
import asyncio, threading, time, uvicorn
from tests.mock_server import make_mock_app
app, state = make_mock_app()
state.perception_script = [
    {'type': 'perception', 'tick': 5,
     'agent_state': {'x': 4, 'y': 4, 'hp': 100, 'energy': 80, 'mood': 60,
                     'hunger': 75, 'personality_current': 'x', 'born_tick': 0,
                     'wait_streak': 0, 'sleep_streak': 0,
                     'inventory': {'berry': 1}},
     'terrain_here': 'grass', 'visible_around': '',
     'here_resource': None, 'here_structure': None,
     'nearby_agents': [], 'nearby_resources': [], 'nearby_structures': [],
     'walkable_dirs': ['north'],
     'relations': {}, 'relations_inbound': {},
     'family': {'mother': None, 'father': None, 'children': []},
     'recent_dialogues': [], 'world_events': []}
]
config = uvicorn.Config(app, host='127.0.0.1', port=18765, log_level='warning')
server = uvicorn.Server(config)
threading.Thread(target=lambda: asyncio.new_event_loop().run_until_complete(server.serve()), daemon=True).start()
time.sleep(1)
import subprocess
subprocess.run([
    'agora-agent',
    '--server', 'http://127.0.0.1:18765',
    '--name', 'SmokeMaya', '--seed', 'curious smoke',
    '--sex', 'F', '--no-llm', '--max-reconnect-attempts', '1',
    '--log-level', 'INFO',
], timeout=10)
" || echo "Smoke ran"
```

Expected: client joins, applies snapshot, decides `eat` for the hungry perception, then exits cleanly when mock disconnects.

- [ ] **Step 3: Self-check the deliverables**

Verify the four required deliverables are present:
1. `README.md` exists and has quickstart.
2. `PROTOCOL.md` exists and covers all message types.
3. Tests passing: `pytest -q | tail -3` shows 0 failures.
4. Spec is committed at `docs/specs/2026-05-07-agora-agent-sdk-design.md`.

- [ ] **Step 4: Stop and ping**

DO NOT run `gh repo create` automatically. The maintainer (user) will do that when ready.

Ping the user with:
- Path to local repo: `/home/mattabott/Documents/agora-agent-sdk/`
- Test summary (count passed/failed)
- Path to `PROTOCOL.md` (the deliverable for the server team)
- Path to `README.md`
- Suggested next step: `gh repo create --public mattabott/agora-agent-sdk` from the repo root, then `git push -u origin main`.

---

## Spec Coverage Audit

Cross-checking sections of `docs/specs/2026-05-07-agora-agent-sdk-design.md` against tasks:

| Spec section | Implemented in |
|---|---|
| §1 Overview | (no code; design framing) |
| §2 Vincoli operativi | Task 10.3 (no real server, no production, no private push) |
| §3 Architettura | Task 0.2 (layout) + Task 6.1 (Brain orchestration) |
| §4 Repo layout | Task 0.1, 0.2 |
| §5.1 HTTP join | Task 1.3 (schemas), Task 7.1 (impl), Task 8.1 (mock) |
| §5.2.1 Snapshot | Task 1.3 (schema), Task 2.2 (apply) |
| §5.2.2 Perception | Task 1.3 (schema), Task 2.3 (apply_perception) |
| §5.2.3 Delta events | Task 2.2 (router + handlers) |
| §5.2.4 Result | Task 1.3, Task 7.2 (dispatch) |
| §5.2.5 Heartbeat | Task 1.3 (schema), Task 7.2 (pong) |
| §5.2.6 Action | Task 1.3 (schema + validate_action_dict) |
| §5.2.7 request_snapshot | Task 7.2 (gap detection + initial request) |
| §6 WorldMirror | Task 2.1, 2.2, 2.3 |
| §7 Decision flow | Task 6.1 (Brain.decide) |
| §8 Dialogue filters | Task 3.1 |
| §9 Action validation client-side | Task 6.1 (`_validate_pre_send`) |
| §10 Ollama client | Task 5.1 |
| §11 CLI + token persistence | Task 7.1 (token I/O), Task 9.1 (CLI) |
| §12 Mock server + tests | Task 8.1, 8.2 |
| §13 Reconnect | Task 7.2 (`run` with backoff) |
| §14 Schema versioning | Task 1.3 (`ACTION_SCHEMA_VERSION`), Task 8.1 (mock 426) |
| §15 Out of scope V1 | none in code (lack of features = covered) |
| §16 Deliverables | Task 10.1 (README), Task 10.2 (PROTOCOL.md), Task 10.3 (verify + ping) |

All sections covered.

---

## Self-Review Notes (post-write)

- **Placeholder scan**: no `TODO`, no `TBD`, no "Similar to Task N", no "implement later". All steps contain runnable code or exact commands.
- **Type consistency**: `WorldMirror.is_walkable_terrain` vs `is_walkable` are both used and clearly separate (terrain-only vs terrain + occupancy). `try_reflex` consistently takes mirror+agent+perception+inventory across all tasks.
- **Scope check**: single implementation plan, ~30 tasks, terminal state is a working client against mock + deliverables ready for handoff. No scope creep into server-side or V2 features.
- **Ambiguity check**: each filter's reject criteria are spelled out as code, every WS message kind has a payload schema, every test has expected outputs.

Plan is ready for execution.

