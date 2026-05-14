# Changelog

All notable changes to **agora-agent-sdk** are documented here.
The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **`http_remote_status(server)`**: GET `/api/remote/status`, ritorna `{connected, limit, available, accepting_joins}`. Pubblico, no auth. Usato per il pre-check prima di tentare un connect.
- **`wait_for_slot(server, poll_interval_s=30, on_wait=...)`**: polla `/api/remote/status` fino a quando il server ha uno slot WS libero, poi ritorna lo status finale. Callback `on_wait(status)` chiamato ad ogni iterazione fallita per stampare progress all'utente.
- **`CapacityFullError`**: sollevata da `AgoraClient._run_once` quando il server chiude il WS con `WS_CLOSE_AT_CAPACITY=4429`. `AgoraClient.run()` la cattura e chiama `wait_for_slot` automaticamente, ritentando appena uno slot e' libero.
- **CLI**: prima di `http_join` chiama `wait_for_slot` con log `agora server full: N/M agents connected. Queueing, retry in 30s...`. Cosi' un nuovo join non viene tentato finche' lo slot non e' libero (evita di creare agenti nel DB che poi non possono connettersi).

### Compatibility

Additivo. Server senza endpoint `/api/remote/status` (versione precedente alla feature) ritornano 404: `wait_for_slot` logga warning e il client procede come prima. Su server vecchio non e' presente neanche il limit e il flusso 4429 non scatta.

---

## [0.2.2] — 2026-05-12

### Fixed

- **WS keepalive 1011 ogni ~60s** sotto carico LLM. I default di `websockets` (`ping_interval=20s`, `ping_timeout=20s`) sono troppo stretti per un client che chiama Ollama in async: quando il event loop e' sotto pressione (Pi 5, decisione LLM in corso), un ping puo' restare non risposto e la connessione viene chiusa con codice 1011 `internal error: keepalive ping timeout`. Il client si riconnette ma il ciclo si ripete. Impostiamo `ping_interval=30, ping_timeout=60` per dare 60s di tolleranza, sufficienti per un'inferenza lunga.

---

## [0.2.1] — 2026-05-12

### Fixed

- **Anti-oscillation loop infinito**: dopo aver fatto downgrade di un `move` opposto all'ultimo a `wait (anti-oscillation)`, `last_move_direction` non veniva resettato (l'else clause escludeva `wait`). Conseguenza: il tick successivo il brain riproponeva la stessa direzione opposta, l'anti-osc bloccava di nuovo, ciclo perpetuo. Agenti remoti in cul-de-sac (es. riva di un lago) restavano fermi per centinaia di tick. Fix: reset `last_move_direction = ""` quando si entra in wait di anti-osc, cosi' qualsiasi direzione e' valida al prossimo tick.

---

## [0.2.0] — 2026-05-12

### Added

- **`PerceptionMsg.escape_dirs: list[str]`** — direzioni cardinali con un walkable raggiungibile entro ~5 step via BFS shallow lato server (parity con agora v0.2). Risolve i casi in cui l'agente sta fermo in cul-de-sac perche' `walkable_dirs` mostra solo i 4 vicini immediati (es. riva di un lago): adesso l'agente sa in che quadrante c'e' open space anche se il primo step va in altra direzione.
- **Prompt template di default**: hint `Open space within 5 steps toward: ... (go around obstacles)` quando `walkable_dirs` e' ristretto (≤2). Niente rumore quando l'agente e' all'aperto.

### Compatibility

- Additivo, no breaking. Server `< v0.2`: campo assente -> default `[]` -> niente hint, comportamento identico a 0.1.x.

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

[Unreleased]: https://github.com/mattabott/agora-agent-sdk/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/mattabott/agora-agent-sdk/releases/tag/v0.2.0
[0.1.0]: https://github.com/mattabott/agora-agent-sdk/releases/tag/v0.1.0
