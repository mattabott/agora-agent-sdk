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
