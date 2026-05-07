"""Day/night cycle. Ported 1:1 from agora.sim.daynight."""

DAY_LENGTH_TICKS = 600
PHASE_BREAKPOINTS = [
    (0.00, 0.05, "dawn"),
    (0.05, 0.50, "day"),
    (0.50, 0.55, "dusk"),
    (0.55, 1.00, "night"),
]


def time_of_day(tick: int) -> dict:
    day_n = tick // DAY_LENGTH_TICKS
    pos = tick % DAY_LENGTH_TICKS
    fraction = pos / DAY_LENGTH_TICKS
    phase = "day"
    for lo, hi, name in PHASE_BREAKPOINTS:
        if lo <= fraction < hi:
            phase = name
            break
    return {"phase": phase, "day_n": day_n, "fraction": round(fraction, 4)}


PHASE_LABEL = {"dawn": "dawn", "day": "day", "dusk": "dusk", "night": "night"}


def is_active_phase(phase: str) -> bool:
    return phase in ("day", "dusk", "dawn")


def is_night(phase: str) -> bool:
    return phase == "night"
