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
