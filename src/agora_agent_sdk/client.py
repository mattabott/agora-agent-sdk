"""AgoraClient: HTTP join + WebSocket loop client."""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import httpx

from agora_core.protocol import (
    ACTION_SCHEMA_VERSION, JoinRequest, JoinResponse,
)

log = logging.getLogger("agora_agent_sdk.client")


class JoinError(Exception):
    """Raised when join fails permanently (not retryable)."""

    def __init__(self, code: int, payload: dict):
        super().__init__(f"join failed: HTTP {code} {payload}")
        self.code = code
        self.payload = payload


@dataclass
class JoinResult:
    agent_id: int
    token: str
    world_seed: int
    tick_ms: int
    world_w: int
    world_h: int
    action_schema_version: int


def default_token_path(name: str) -> Path:
    home = Path(os.path.expanduser("~"))
    return home / ".agora-agent" / f"{name}.token"


def write_token(path: Path, agent_id: int, token: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
    path.write_text(json.dumps({"agent_id": agent_id, "token": token}))
    try:
        path.chmod(0o600)
    except OSError:
        pass


def read_token(path: Path) -> tuple[int, str] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
        return int(data["agent_id"]), str(data["token"])
    except Exception:
        log.warning("token file unreadable: %s", path)
        return None


def delete_token(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


async def http_join(
    server: str,
    *,
    name: str,
    personality_seed: str,
    sex: str,
    color: str | None = None,
    client_version: str = "agora-agent-sdk/0.1.0",
    timeout: float = 10.0,
    http_client: httpx.AsyncClient | None = None,
) -> JoinResult:
    payload = JoinRequest(
        name=name, personality_seed=personality_seed, sex=sex, color=color,
        action_schema_version=ACTION_SCHEMA_VERSION, client_version=client_version,
    ).model_dump(exclude_none=True)

    async def _do(client: httpx.AsyncClient) -> JoinResult:
        r = await client.post(
            f"{server.rstrip('/')}/api/agents/join",
            json=payload, timeout=timeout,
        )
        if r.status_code != 200:
            try:
                data = r.json()
            except Exception:
                data = {"raw": r.text[:200]}
            raise JoinError(r.status_code, data)
        resp = JoinResponse.model_validate(r.json())
        return JoinResult(
            agent_id=resp.agent_id, token=resp.token,
            world_seed=resp.world_seed, tick_ms=resp.tick_ms,
            world_w=resp.world_w, world_h=resp.world_h,
            action_schema_version=resp.action_schema_version,
        )

    if http_client is not None:
        return await _do(http_client)
    async with httpx.AsyncClient(timeout=timeout) as client:
        return await _do(client)
