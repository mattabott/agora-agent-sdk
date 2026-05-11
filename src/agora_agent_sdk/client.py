"""AgoraClient: HTTP join + WebSocket loop client."""
from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Awaitable, Callable

import httpx

from agora_core.protocol import (
    ACTION_SCHEMA_VERSION, JoinRequest, JoinResponse,
    ActionMsg, EventMsg, PerceptionMsg, PingMsg, PongMsg,
    RequestSnapshotMsg, ResultMsg, SnapshotMsg,
)
from agora_core.world_mirror import WorldMirror, pack_walkable_mask

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


class AgentDiedExit(Exception):
    """Raised when self agent died — caller should clean up token and exit."""

    def __init__(self, name: str, tick: int):
        super().__init__(f"agent {name} died at tick {tick}")
        self.name = name
        self.tick = tick


class AgoraClient:
    """WebSocket client that drives the brain loop."""

    def __init__(
        self,
        server: str,
        *,
        agent_id: int,
        token: str,
        world_w: int,
        world_h: int,
        brain,
        token_path: Path | None = None,
        max_reconnect_attempts: int = 0,
        snapshot_gap_threshold: int = 60,
        ws_factory: Callable[[str], Awaitable] | None = None,
    ):
        self.server = server.rstrip("/")
        self.agent_id = agent_id
        self.token = token
        self.brain = brain
        self.token_path = token_path
        self.max_reconnect_attempts = max_reconnect_attempts
        self.snapshot_gap_threshold = snapshot_gap_threshold
        self._ws_factory = ws_factory
        empty_grid = [[True] * world_w for _ in range(world_h)]
        self.brain.mirror.world_w = world_w
        self.brain.mirror.world_h = world_h
        self.brain.mirror.walkable_mask = pack_walkable_mask(empty_grid)
        self.brain.mirror.self_agent_id = agent_id
        self._last_known_tick = -1
        self._stop = asyncio.Event()

    def stop(self) -> None:
        self._stop.set()

    def _ws_url(self) -> str:
        scheme = "ws" if self.server.startswith("http://") else "wss"
        host = self.server.split("//", 1)[-1]
        return f"{scheme}://{host}/ws/agents/{self.agent_id}?token={self.token}"

    async def run(self) -> None:
        """Main loop with reconnect."""
        attempt = 0
        backoff = 1.0
        while not self._stop.is_set():
            try:
                await self._run_once()
                return
            except AgentDiedExit:
                if self.token_path is not None:
                    delete_token(self.token_path)
                raise
            except Exception as e:
                log.warning("WS loop error: %s", e)
                attempt += 1
                if (self.max_reconnect_attempts
                        and attempt >= self.max_reconnect_attempts):
                    log.error("max reconnect attempts reached, giving up")
                    raise
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 30.0)

    async def _run_once(self) -> None:
        """Open WS, drive recv→decide→send loop. Raises on disconnect."""
        if self._ws_factory is not None:
            ws_ctx = self._ws_factory(self._ws_url())
        else:
            import websockets
            ws_ctx = websockets.connect(self._ws_url())
        async with ws_ctx as ws:
            log.info("WS connected to %s", self.server)
            await self._send(ws, RequestSnapshotMsg().model_dump())
            await self._loop(ws)

    async def _send(self, ws, payload: dict) -> None:
        await ws.send(json.dumps(payload))

    async def _loop(self, ws) -> None:
        async for raw in ws:
            if isinstance(raw, bytes):
                raw = raw.decode("utf-8")
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue
            await self._dispatch(ws, msg)

    async def _dispatch(self, ws, msg: dict) -> None:
        mtype = msg.get("type")
        if mtype == "snapshot":
            self.brain.mirror.apply_snapshot(msg)
            self._last_known_tick = self.brain.mirror.current_tick
        elif mtype == "perception":
            tick = int(msg.get("tick", 0))
            if (self._last_known_tick > 0
                and (tick < self._last_known_tick
                     or tick - self._last_known_tick > self.snapshot_gap_threshold)):
                await self._send(ws, RequestSnapshotMsg().model_dump())
                return
            self._last_known_tick = tick
            decision = await self.brain.decide(msg)
            await self._send(ws, ActionMsg(
                tick_ack=tick, **{k: v for k, v in decision.items()
                                   if k in ("action", "direction", "target_id",
                                            "content", "item", "qty",
                                            "recipe", "structure",
                                            "thought", "decided_via")},
            ).model_dump(exclude_none=True))
        elif mtype == "event":
            try:
                self.brain.mirror.apply_event(msg)
            except Exception:
                log.exception("apply_event failed for event=%r", msg)
            kind = msg.get("kind")
            if kind == "agent_died" and int(msg.get("agent_id", 0)) == self.agent_id:
                a = self.brain.mirror.agents.get(self.agent_id)
                name = a.name if a else "<unknown>"
                tick = int(msg.get("tick", 0))
                raise AgentDiedExit(name, tick)
            try:
                self.brain.push_event_to_episodic(msg)
            except Exception:
                log.exception("push_event_to_episodic failed for event=%r", msg)
        elif mtype == "result":
            self.brain.push_episodic({
                "kind": "action_result", "tick": int(msg.get("tick_ack", 0)),
                **{k: v for k, v in msg.items() if k != "type"},
            })
        elif mtype == "ping":
            await self._send(ws, PongMsg(ts=float(msg.get("ts", time.time()))).model_dump())
