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
