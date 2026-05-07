from agora_core.world_mirror import (
    WorldMirror, AgentSnap, StructureInfo, pack_walkable_mask,
)


def _square_mirror(w=8, h=8) -> WorldMirror:
    grid = [[True] * w for _ in range(h)]
    return WorldMirror(world_w=w, world_h=h, walkable_mask=pack_walkable_mask(grid))


def _mirror_with_wall() -> WorldMirror:
    """8x8 mirror with column x=4 non-walkable except at y=7."""
    grid = [[True] * 8 for _ in range(8)]
    for y in range(7):
        grid[y][4] = False
    return WorldMirror(world_w=8, world_h=8, walkable_mask=pack_walkable_mask(grid))


def test_is_walkable_terrain_in_bounds():
    m = _square_mirror()
    assert m.is_walkable_terrain(0, 0)
    assert m.is_walkable_terrain(7, 7)
    assert not m.is_walkable_terrain(-1, 0)
    assert not m.is_walkable_terrain(8, 0)


def test_is_walkable_blocked_by_agent():
    m = _square_mirror()
    m.agents[1] = AgentSnap(id=1, name="X", x=3, y=3, color="#fff",
                            sex="F", alive=True, born_tick=0)
    assert not m.is_walkable(3, 3)
    assert m.is_walkable_terrain(3, 3)


def test_is_walkable_dead_agent_no_block():
    m = _square_mirror()
    m.agents[1] = AgentSnap(id=1, name="X", x=3, y=3, color="#fff",
                            sex="F", alive=False, born_tick=0, died_tick=10)
    assert m.is_walkable(3, 3)


def test_find_path_step_straight_east():
    m = _square_mirror()
    assert m.find_path_step(0, 0, 5, 0) == "east"


def test_find_path_step_around_wall():
    m = _mirror_with_wall()
    step = m.find_path_step(0, 0, 7, 0)
    assert step in ("south", "east")


def test_find_path_step_unreachable_returns_none():
    grid = [[True] * 4 for _ in range(4)]
    for dx, dy in ((0, 1), (1, 0), (-1, 0), (0, -1)):
        nx, ny = 3 + dx, 3 + dy
        if 0 <= nx < 4 and 0 <= ny < 4:
            grid[ny][nx] = False
    grid[3][3] = True
    m = WorldMirror(world_w=4, world_h=4, walkable_mask=pack_walkable_mask(grid))
    assert m.find_path_step(0, 0, 3, 3) is None


def test_nearest_resource_exact_tile():
    m = _square_mirror()
    m.resources[(2, 2)] = ("wood", 1)
    m.resources[(7, 7)] = ("wood", 1)
    assert m.nearest_resource(0, 0, "wood") == (2, 2)


def test_nearest_resource_falls_back_to_cluster():
    m = _square_mirror()
    m.resource_clusters = [{"type": "wood", "cx": 6, "cy": 6, "total_qty": 5,
                            "tiles": []}]
    assert m.nearest_resource(0, 0, "wood") == (6, 6)


def test_nearest_resource_skips_inaccessible_tile():
    grid = [[True] * 4 for _ in range(4)]
    grid[1][1] = False
    grid[0][1] = False
    grid[2][1] = False
    grid[1][0] = False
    grid[1][2] = False
    m = WorldMirror(world_w=4, world_h=4, walkable_mask=pack_walkable_mask(grid))
    m.resources[(1, 1)] = ("wood", 1)
    assert m.nearest_resource(0, 0, "wood") is None


def test_apply_perception_updates_self_position():
    m = _square_mirror()
    m.self_agent_id = 1
    m.agents[1] = AgentSnap(id=1, name="Self", x=0, y=0, color="#fff",
                            sex="F", alive=True, born_tick=0)
    m.apply_perception({
        "tick": 5,
        "agent_state": {"x": 3, "y": 4, "wait_streak": 2, "sleep_streak": 0},
    })
    assert m.agents[1].x == 3
    assert m.agents[1].y == 4
    assert m.agents[1].wait_streak == 2
    assert m.current_tick == 5


def test_apply_perception_clears_stale_nearby_resources():
    m = _square_mirror()
    m.self_agent_id = 1
    m.agents[1] = AgentSnap(id=1, name="Self", x=4, y=4, color="#fff",
                            sex="F", alive=True, born_tick=0)
    m.resources[(5, 4)] = ("berry", 1)
    m.resources[(0, 0)] = ("wood", 1)
    m.apply_perception({
        "tick": 5,
        "agent_state": {"x": 4, "y": 4},
        "nearby_resources": [{"x": 4, "y": 5, "type": "berry", "qty": 1}],
    })
    assert (5, 4) not in m.resources
    assert (4, 5) in m.resources
    assert (0, 0) in m.resources
