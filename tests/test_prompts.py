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
    assert "thought: 49" in out
    assert "thought: 0" not in out


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
    lines = out.strip().split("\n")
    assert "Niko" in lines[0]
    assert "+30" in lines[0]


def test_format_semantic_empty():
    assert format_semantic([]) == "  (none)"


def test_format_semantic_lines():
    mem = [{"kind": "reflection", "tick": 5, "text": "I learned to share."}]
    out = format_semantic(mem)
    assert "reflection" in out and "I learned to share." in out
