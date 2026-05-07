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
    assert f[0] == 1.0
    assert f[1] == 1.0
    assert f[2] == 1.0
    assert f[3] == 1.0


def test_extract_features_walkable_flags():
    f = extract_features(
        current_tick=0, born_tick=0, hunger=0, mood=0, energy=0, hp=0,
        inventory={}, perception={"walkable_dirs": ["north", "east"]},
        wait_streak=0, sleep_streak=0,
    )
    assert f[21] == 1.0
    assert f[22] == 0.0
    assert f[23] == 1.0
    assert f[24] == 0.0


def test_decode_wait_and_wander():
    perc = {"walkable_dirs": []}
    assert decode_to_decision(0, perc, {})["action"] == "wait"
    assert decode_to_decision(1, perc, {})["action"] == "wander"


def test_decode_move_blocked():
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
