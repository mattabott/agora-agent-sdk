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
