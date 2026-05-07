"""Async wrapper around a local Ollama server.

Two methods used by the brain:
  - decide(system, user) → dict (LLM JSON output, format=json)
  - talk_line(system, user) → str (single freeform sentence)

Both share a single AsyncLock to serialize calls (the user has one Ollama
process; concurrent calls just slow it down).
"""
from __future__ import annotations

import asyncio
import json
import logging

import httpx

log = logging.getLogger("agora_agent_sdk.llm")


class OllamaClient:
    def __init__(
        self,
        host: str,
        model: str,
        *,
        num_predict_decide: int = 80,
        num_predict_dialogue: int = 60,
        num_ctx: int = 2048,
        temperature: float = 0.7,
        timeout_decide: float = 60.0,
        timeout_dialogue: float = 30.0,
    ):
        self.host = host.rstrip("/")
        self.model = model
        self.num_predict_decide = num_predict_decide
        self.num_predict_dialogue = num_predict_dialogue
        self.num_ctx = num_ctx
        self.temperature = temperature
        self._timeout_decide = timeout_decide
        self._timeout_dialogue = timeout_dialogue
        self._lock = asyncio.Lock()
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(60.0))

    async def aclose(self) -> None:
        await self._client.aclose()

    async def decide(self, system: str, user: str) -> dict:
        async with self._lock:
            try:
                r = await self._client.post(
                    f"{self.host}/api/generate",
                    json={
                        "model": self.model,
                        "system": system,
                        "prompt": user,
                        "format": "json",
                        "stream": False,
                        "options": {
                            "num_predict": self.num_predict_decide,
                            "num_ctx": self.num_ctx,
                            "temperature": self.temperature,
                        },
                    },
                    timeout=self._timeout_decide,
                )
                r.raise_for_status()
                text = r.json().get("response", "")
                try:
                    parsed = json.loads(text)
                except json.JSONDecodeError:
                    log.debug("ollama decide: invalid JSON %r", text[:200])
                    return {}
                return parsed if isinstance(parsed, dict) else {}
            except httpx.HTTPError as e:
                log.warning("ollama decide failed: %s", e)
                return {}

    async def talk_line(self, system: str, user: str) -> str:
        async with self._lock:
            try:
                r = await self._client.post(
                    f"{self.host}/api/generate",
                    json={
                        "model": self.model,
                        "system": system,
                        "prompt": user,
                        "stream": False,
                        "options": {
                            "num_predict": self.num_predict_dialogue,
                            "num_ctx": self.num_ctx,
                            "temperature": self.temperature,
                        },
                    },
                    timeout=self._timeout_dialogue,
                )
                r.raise_for_status()
                return r.json().get("response", "")
            except httpx.HTTPError as e:
                log.warning("ollama talk_line failed: %s", e)
                return ""


class NoOpLLM:
    """Placeholder used when --no-llm is set. Returns empty results."""

    async def decide(self, system: str, user: str) -> dict:
        return {}

    async def talk_line(self, system: str, user: str) -> str:
        return ""

    async def aclose(self) -> None:
        return None
