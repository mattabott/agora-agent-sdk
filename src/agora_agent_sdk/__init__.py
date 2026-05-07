"""agora_agent_sdk: client + CLI to plug an external agent into agora.

Public API:
- AgoraClient: async client for join + WS loop
- OllamaClient: async wrapper around local Ollama
- main: CLI entry point (`agora-agent` script)
"""

from agora_core import ACTION_SCHEMA_VERSION  # noqa: F401

__version__ = "0.1.0"
