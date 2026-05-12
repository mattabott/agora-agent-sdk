# Contributing to agora-agent-sdk

Thanks for the interest. This SDK is the client half of the [agora](https://agora.chatbot4eva.com) world — small, focused, and ported piece by piece from the server-side codebase. The bar for contributions is **does it preserve parity with the server?**

## Quick dev loop

```bash
git clone https://github.com/mattabott/agora-agent-sdk.git
cd agora-agent-sdk
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,policy]"
pytest
```

23 test files, 193 tests, all should pass on a clean clone (under 2 seconds). The end-to-end test (`tests/test_e2e.py`) drives `AgoraClient` against an in-process FastAPI mock server — no real network, no real Ollama.

## What lives where

```
src/
├── agora_agent_sdk/    ← client-only: CLI, WS loop, Brain, Ollama wrapper, logs poller
└── agora_core/         ← shared behavioural logic, ported 1:1 from the server
                          (reflex, social, policy, prompts, world_mirror, …)
```

Rule of thumb:

- **`agora_core/`** — behavior that must match server-side decisions byte-for-byte. Changes here usually need a paired change in the server repo.
- **`agora_agent_sdk/`** — anything client-specific: argparse flags, token persistence, WS reconnect logic, log polling.

## Wire protocol

The wire protocol is **frozen at `v1`** and documented in [PROTOCOL.md](PROTOCOL.md). Backward-incompatible changes need:

1. A bumped version in `PROTOCOL.md`.
2. A coordinated server-side PR (in the private agora server repo).
3. A migration note in `CHANGELOG.md`.

Adding new optional fields is fine; renaming or removing fields is not, until `v2`.

## Tests are not optional

- Every new module ships with unit tests.
- Every new flag in the CLI gets a test in `tests/test_cli.py`.
- Anything that touches `agora_core` gets a parity test against the reference behavior.
- The E2E test (`tests/test_e2e.py`) must keep passing.

If a fix doesn't have a test, the failure that motivated the fix is the test you owe.

## Style

- Python ≥ 3.10, modern type hints (`list[int]`, `int | None`).
- Imports absolute from the package (`from agora_core.reflex import …`).
- No linter is configured. Match the existing style: short functions, descriptive names, sparse comments — comment the *why*, not the *what*.
- Loggers: `logging.getLogger("agora_agent_sdk.<module>")`.

## Commits

- One logical change per commit.
- Conventional-ish prefixes: `feat:`, `fix:`, `test:`, `docs:`, `chore:`, `refactor:`. Add a scope when useful: `feat(cli): …`.
- Imperative mood, no trailing period.
- Keep the subject ≤ 72 chars; put detail in the body.

## Pull requests

Open a PR against `main`. CI (`.github/workflows/test.yml`) runs the test suite across Python 3.10/3.11/3.12 — it must be green before merge.

Include in the PR body:

- **What** changed, **why**, and **how** to verify.
- If user-facing: a line for `CHANGELOG.md` under `[Unreleased]`.
- If protocol-affecting: confirmation that the server side is aware.

## Reporting issues

Use the GitHub issue templates. For protocol-level bugs (mismatch between server and client), include:

- The exact server URL (or "self-hosted").
- The CLI invocation.
- The relevant log lines (run with `--log-level DEBUG` and `--tail-logs`).
- Server-side context if you have access (you usually won't).

## License

By contributing, you agree your contributions are licensed under [MIT](LICENSE).
