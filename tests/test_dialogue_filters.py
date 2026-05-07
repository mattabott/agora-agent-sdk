from agora_core.dialogue_filters import (
    accept_dialogue_line, append_to_ring,
    POETIC_BLACKLIST, ITALIAN_MARKERS,
    MAX_RECENT_LINES_PER_AGENT,
)


def _empty_ring() -> dict[int, list[str]]:
    return {}


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


def test_rejects_short_line():
    assert accept_dialogue_line(
        "Hi.", agent_names=[], recent_lines_by_agent=_empty_ring(),
    ) is None
    assert accept_dialogue_line(
        "Two words.", agent_names=[], recent_lines_by_agent=_empty_ring(),
    ) is None


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


def test_rejects_noun_list_pattern():
    assert accept_dialogue_line(
        "Hunger, fear, fatigue.",
        agent_names=[], recent_lines_by_agent=_empty_ring(),
    ) is None


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


def test_append_to_ring_normalizes():
    buf: list[str] = []
    append_to_ring(buf, "  Hello WORLD here  ")
    assert buf == ["hello world here"]


def test_append_to_ring_trims_to_max():
    buf: list[str] = ["x"] * MAX_RECENT_LINES_PER_AGENT
    append_to_ring(buf, "fresh new line")
    assert len(buf) == MAX_RECENT_LINES_PER_AGENT
    assert buf[-1] == "fresh new line"
    assert buf[0] == "x"
