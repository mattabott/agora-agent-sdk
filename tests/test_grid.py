from agora_core.grid import DIRECTIONS


def test_directions_complete():
    assert set(DIRECTIONS) == {"north", "south", "east", "west"}


def test_directions_unit_vectors():
    assert DIRECTIONS["north"] == (0, -1)
    assert DIRECTIONS["south"] == (0, 1)
    assert DIRECTIONS["east"] == (1, 0)
    assert DIRECTIONS["west"] == (-1, 0)
