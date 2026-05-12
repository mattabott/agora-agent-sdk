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
    # Look-ahead (server v0.2+): se sei ristretto, suggerisci verso quale
    # quadrante c'e' open space entro 5 step. Il primo move puo' essere in
    # altra direzione (vai in diagonale aggirando ostacoli). Solo se
    # walkable_dirs e' ristretto (≤2), altrimenti e' rumore.
    if len(walk) <= 2:
        escape = perception.get("escape_dirs") or []
        useful = [d for d in escape if d not in walk]
        if useful:
            parts.append(
                f"Open space within 5 steps toward: {', '.join(useful)} "
                "(go around obstacles)"
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

    if any(inventory.get(k, 0) > 0 for k in ("wood", "stone", "iron_ore")):
        recipes = format_recipes_for_user(inventory)
        if "(none)" not in recipes:
            parts.append(f"Crafting:\n{recipes}")

    if perception.get("here_structure") is None and any(
        inventory.get(k, 0) > 0 for k in ("wood", "stone", "iron_ore")
    ):
        structs = format_structures_for_user(inventory)
        if "(none)" not in structs:
            parts.append(f"Building (you can build here):\n{structs}")

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
    nearby_resources: list[tuple[str, int]],
    nearby_structures: list[str],
    ongoing_events: list[str],
    recent_dialogue_text: str,
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
