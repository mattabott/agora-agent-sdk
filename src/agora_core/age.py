"""Age stages of agents. Ported 1:1 from agora.agents.age."""

from agora_core.daynight import DAY_LENGTH_TICKS

CHILD_END_DAYS = 1
YOUNG_END_DAYS = 3
ADULT_END_DAYS = 20
ELDER_END_DAYS = 100

ANCIENT_HP_DECAY = 1
ANCIENT_DECAY_EVERY_N_CYCLES = 3

STAGE_LABEL = {
    "child": "child",
    "young": "young",
    "adult": "adult",
    "elder": "elder",
    "ancient": "ancient",
}


def age_in_days(born_tick: int, current_tick: int) -> float:
    return max(0.0, (current_tick - born_tick) / DAY_LENGTH_TICKS)


def age_stage(days: float) -> str:
    if days < CHILD_END_DAYS:
        return "child"
    if days < YOUNG_END_DAYS:
        return "young"
    if days < ADULT_END_DAYS:
        return "adult"
    if days < ELDER_END_DAYS:
        return "elder"
    return "ancient"


def stage_label(stage: str) -> str:
    return STAGE_LABEL.get(stage, stage)


def can_reproduce(stage: str) -> bool:
    return stage in ("adult", "elder", "ancient")


def is_ancient(stage: str) -> bool:
    return stage == "ancient"
