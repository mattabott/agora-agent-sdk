"""Grid directions used by reflex/social pathfinding.

Ported 1:1 from the private agora repo (agora.agents.actions.DIRECTIONS).
"""

DIRECTIONS: dict[str, tuple[int, int]] = {
    "north": (0, -1),
    "south": (0, 1),
    "east": (1, 0),
    "west": (-1, 0),
}
