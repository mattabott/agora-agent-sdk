from agora_core.daynight import (
    DAY_LENGTH_TICKS, time_of_day, is_night, is_active_phase,
)


def test_day_length():
    assert DAY_LENGTH_TICKS == 600


def test_time_of_day_dawn():
    info = time_of_day(0)
    assert info["phase"] == "dawn"
    assert info["day_n"] == 0
    assert info["fraction"] == 0.0


def test_time_of_day_day():
    info = time_of_day(60)  # 0.10 fraction → "day"
    assert info["phase"] == "day"


def test_time_of_day_dusk():
    info = time_of_day(int(0.51 * 600))
    assert info["phase"] == "dusk"


def test_time_of_day_night():
    info = time_of_day(int(0.80 * 600))
    assert info["phase"] == "night"


def test_time_of_day_day_n_increments():
    assert time_of_day(0)["day_n"] == 0
    assert time_of_day(600)["day_n"] == 1
    assert time_of_day(1200)["day_n"] == 2


def test_is_night_true_only_at_night():
    assert is_night("night")
    assert not is_night("day")
    assert not is_night("dawn")
    assert not is_night("dusk")


def test_is_active_phase():
    assert is_active_phase("day")
    assert is_active_phase("dawn")
    assert is_active_phase("dusk")
    assert not is_active_phase("night")
