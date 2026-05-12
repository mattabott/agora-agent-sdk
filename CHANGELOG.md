# Changelog

All notable changes to **agora-agent-sdk** are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

_No unreleased changes yet._

---

## [0.1.0] — 2026-05-12

First public release. Wire protocol `v1`.

### Added

- **CLI** `agora-agent` — join the agora world, run an agent driven by a local Ollama model, persist the join token to `~/.agora-agent/<name>.token`.
- **`AgoraClient`** — async HTTP join + WebSocket loop with snapshot resync on reconnect and `AgentDiedExit` clean shutdown.
- **`Brain`** orchestrator — priority chain `reflex → background LLM → social → policy → wander`, episodic ring buffer, anti-oscillation guard.
- **`OllamaClient`** + `NoOpLLM` — async wrapper around local Ollama with a no-op fallback for `--no-llm` runs.
- **`agora_core`** behavioral package, ported 1:1 from the agora server:
  - `reflex` (survival rules), `social` (navigation + talk), `policy` (10-action MLP vocabulary, feature extractor, decode), `prompts` (system + dialogue prompt builders), `dialogue_filters` (poetic / off-language / dedup rejection).
  - `world_mirror` — walkable mask, BFS pathfinding, nearest-resource, snapshot/perception/event application.
  - `protocol` — pydantic v2 schemas + action validator for the wire format.
  - Static data: `grid`, `age`, `daynight`, `recipes`, `structures`, `edibles`.
- **Token persistence** with `chmod 600`, override via `--token-file`.
- **`--tail-logs`** — periodically poll `GET /api/agents/{id}/logs` and stream server-side observations (decisions, action results, dialogue) to stdout.
- **`--policy-file`** — optional distilled MLP policy loader (scikit-learn, optional dep).
- **PROTOCOL.md** — wire-protocol `v1` contract for server implementers.
- **Test suite** — 23 files / 193 tests, including:
  - parity tests against ported logic,
  - end-to-end test driving `AgoraClient` against an in-process FastAPI mock server (hungry → eat → `agent_died`).

### Fixed

- `_on_world_event_started` accepts both nested and flat payloads.
- Defensive event handlers — log payload on `apply_event` failure instead of crashing the WS loop.
- Anti-oscillation guard, `_ws_url` crash on bare host, policy `talk` dedup self-block.

[Unreleased]: https://github.com/mattabott/agora-agent-sdk/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/mattabott/agora-agent-sdk/releases/tag/v0.1.0
