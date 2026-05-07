"""agora_core: shared deterministic logic for agora-agent-sdk.

Frozen v1 port of the behavior layer from the private agora repo:
- protocol: pydantic models for the WS wire format
- world_mirror: client-side world state mirror updated by snapshot + deltas
- reflex / social / policy: deterministic decision layers
- prompts / dialogue_filters: LLM prompt builders + output filters
- age / daynight: time-related helpers
- recipes / structures / edibles: static game data
"""

__version__ = "0.1.0"
ACTION_SCHEMA_VERSION = 1
