"""Dialogue line filters. Ported 1:1 from agora.agents.brain `_dialogue_gen_bg`.

Reject criteria (in order):
  - too short (< 3 words)
  - poetic blacklist substring (EN + IT)
  - italian markers
  - noun-list pattern (>=2 comma parts, all <= 2 words)
  - truncated tail (... or no terminator .!?)
  - 3-gram overlap with any recent line of any agent
"""
from __future__ import annotations

POETIC_BLACKLIST: set[str] = {
    # English
    "shadow", "shadows", "whisper", "whispers", "echo", "echoes",
    "mystery", "mysterious", "ancient", "eternity", "eternal",
    "soul", "souls", "essence", "infinite", "infinity",
    "ineffab", "silence", "silent", "void", "boundless",
    "harmony", "tangible", "intimate", "intimacy",
    "fate", "destiny", "omen", "presage", "transc",
    "celestial", "ethereal", "sacred", "profound",
    # Italian (small LLMs sometimes slip into IT)
    "ombra", "sussurro", "eco", "mistero", "palpita",
    "eternita", "anima", "fluire", "antico", "antica",
    "rifugio", "promette", "desiderio", "celeste",
    "infinito", "presagio", "fato", "destino", "essenza",
    "silenzio", "esistenza", "intimo", "intima", "vuoto",
    "armonia", "ineffab", "trasc",
}

ITALIAN_MARKERS: tuple[str, ...] = (
    "sono", "sto", "siamo", "siete", "voglio", "vuoi", "vogliamo",
    "perche", "perché", "anche", "molto", "questa", "questo",
    "quello", "ancora", "adesso", "ieri", "oggi", "domani",
    "altrimenti", "abbia", "abbiamo", "qui con", "con te",
    "stanco", "stanca", "sento", "senti", "sente", "vado",
    "vai", "andiamo", "trovare", "iniziare", "costruire",
    "dispiace", "abitazione",
)

MAX_LINE_LEN = 100
MAX_RECENT_LINES_PER_AGENT = 12


def _strip_name_prefix(line: str, agent_names: list[str]) -> str:
    for nm in agent_names:
        for sep in (":", " -", " —", ","):
            pfx = f"{nm}{sep}"
            if line.startswith(pfx):
                return line[len(pfx):].strip().lstrip("\"'")
    return line


def _truncate(line: str, max_len: int = MAX_LINE_LEN) -> str:
    if len(line) <= max_len:
        return line
    return line[:max_len].rsplit(" ", 1)[0] + "…"


def accept_dialogue_line(
    line: str,
    *,
    agent_names: list[str],
    recent_lines_by_agent: dict[int, list[str]],
) -> str | None:
    """Return the cleaned line if accepted, or None if rejected.

    `recent_lines_by_agent` maps agent_id → list of recent normalized lines.
    """
    if not line:
        return None
    line = line.strip().strip('"').strip("'")
    line = _strip_name_prefix(line, agent_names)
    line = _truncate(line)
    if len(line.split()) < 3:
        return None

    line_low = line.lower()

    # Poetic blacklist (substring case-insensitive)
    for w in POETIC_BLACKLIST:
        if w in line_low:
            return None

    # Italian markers (multi-word substring OR word-boundary single tokens)
    words = {w.strip(".,;:!?\"'()") for w in line_low.split()}
    for marker in ITALIAN_MARKERS:
        if " " in marker:
            if marker in line_low:
                return None
        elif marker in words:
            return None

    # Anti-noun-list: ≥2 comma parts, all ≤2 words
    parts = [p.strip() for p in line.split(",") if p.strip()]
    if len(parts) >= 2 and all(len(p.split()) <= 2 for p in parts):
        return None

    # Anti-truncated
    if line.endswith(("...", "…")):
        return None
    if line and line[-1] not in ".!?":
        return None

    # 3-gram overlap dedup with any recent line of any agent
    line_norm = " ".join(line_low.split())
    new_words = line_norm.split()
    if len(new_words) >= 3:
        new_trigrams = {tuple(new_words[i:i+3]) for i in range(len(new_words) - 2)}
        for buf in recent_lines_by_agent.values():
            for prev in buf:
                pwords = prev.split()
                if len(pwords) < 3:
                    continue
                prev_trigrams = {tuple(pwords[i:i+3])
                                 for i in range(len(pwords) - 2)}
                if new_trigrams & prev_trigrams:
                    return None

    return line


def append_to_ring(buf: list[str], line: str, max_size: int = MAX_RECENT_LINES_PER_AGENT) -> None:
    """Append a normalized line to a per-agent ring buffer (in-place)."""
    norm = " ".join(line.lower().split())
    buf.append(norm)
    while len(buf) > max_size:
        buf.pop(0)
