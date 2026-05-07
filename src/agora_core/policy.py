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
            "content": "<<USE_NEXT_TALK_LINE>>", "thought": "(policy)",
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
