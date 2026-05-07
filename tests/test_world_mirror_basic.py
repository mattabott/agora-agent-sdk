import base64
import pytest
from agora_core.world_mirror import (
    pack_walkable_mask, unpack_walkable_mask, mask_bit,
    AgentSnap, StructureInfo, WorldEvent, WorldMirror,
)


def test_pack_unpack_roundtrip():
    grid = [
        [True,  False, True ],
        [False, True,  False],
    ]  # 2x3
    raw = pack_walkable_mask(grid)
    # 6 bits → 1 byte
    assert len(raw) == 1
    # Encode → decode
    b64 = base64.b64encode(raw).decode("ascii")
    decoded = unpack_walkable_mask(b64, 3, 2)
    assert decoded == raw


def test_mask_bit_reads_correctly():
    grid = [
        [True,  False, True ],
        [False, True,  False],
    ]
    raw = pack_walkable_mask(grid)
    assert mask_bit(raw, 3, 0, 0) is True
    assert mask_bit(raw, 3, 1, 0) is False
    assert mask_bit(raw, 3, 2, 0) is True
    assert mask_bit(raw, 3, 0, 1) is False
    assert mask_bit(raw, 3, 1, 1) is True
    assert mask_bit(raw, 3, 2, 1) is False


def test_unpack_wrong_size_raises():
    raw = b"\x00"
    b64 = base64.b64encode(raw).decode("ascii")
    with pytest.raises(ValueError):
        unpack_walkable_mask(b64, 8, 8)


def test_64x64_mask_size():
    grid = [[True] * 64 for _ in range(64)]
    raw = pack_walkable_mask(grid)
    assert len(raw) == 512
    b64 = base64.b64encode(raw).decode("ascii")
    assert len(b64) == 684


def test_world_mirror_constructs_with_defaults():
    grid = [[True] * 4 for _ in range(4)]
    raw = pack_walkable_mask(grid)
    m = WorldMirror(world_w=4, world_h=4, walkable_mask=raw)
    assert m.current_tick == 0
    assert m.agents == {}
    assert m.structures == {}


def test_dataclasses_construct():
    a = AgentSnap(id=1, name="A", x=0, y=0, color="#fff", sex="F",
                  alive=True, born_tick=0)
    assert a.id == 1
    s = StructureInfo(id=1, x=2, y=3, type="hut", owner_id=1, built_tick=10)
    assert s.type == "hut"
    e = WorldEvent(id=1, type="rain", x=0, y=0, radius=4, started_tick=0, ends_tick=200)
    assert e.type == "rain"
