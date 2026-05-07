from agora_core.age import (
    age_in_days, age_stage, stage_label, can_reproduce, is_ancient,
)


def test_age_in_days_zero():
    assert age_in_days(0, 0) == 0.0


def test_age_in_days_one_full_day():
    # DAY_LENGTH_TICKS = 600
    assert age_in_days(0, 600) == 1.0


def test_age_in_days_negative_clamped():
    assert age_in_days(100, 50) == 0.0


def test_age_stages_boundaries():
    assert age_stage(0.5) == "child"
    assert age_stage(2.0) == "young"
    assert age_stage(10.0) == "adult"
    assert age_stage(50.0) == "elder"
    assert age_stage(150.0) == "ancient"


def test_can_reproduce_adult_plus():
    assert not can_reproduce("child")
    assert not can_reproduce("young")
    assert can_reproduce("adult")
    assert can_reproduce("elder")
    assert can_reproduce("ancient")


def test_is_ancient():
    assert is_ancient("ancient")
    assert not is_ancient("elder")


def test_stage_label_known():
    assert stage_label("child") == "child"
    assert stage_label("unknown_stage") == "unknown_stage"
