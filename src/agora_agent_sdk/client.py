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


class CapacityFullError(Exception):
    """Server ha rifiutato la connessione WS con WS_CLOSE_AT_CAPACITY (4429):
    le sessioni remote attive sono al limite. Il caller dovrebbe pollare
    /api/remote/status finche' available > 0, poi ritentare."""

    def __init__(self, status: dict | None = None, reason: str = ""):
        super().__init__(f"server at capacity: {status or reason}")
        self.status = status or {}
        self.reason = reason


# Close code numero (matched con server: ws_agents.WS_CLOSE_AT_CAPACITY)
WS_CLOSE_AT_CAPACITY = 4429


async def http_remote_status(
    server: str, *, timeout: float = 5.0,
) -> dict:
    """GET /api/remote/status. Ritorna dict con connected, limit, available,
    accepting_joins. Usato prima del connect per sapere se ci sono slot."""
    async with httpx.AsyncClient(timeout=timeout) as client:
        r = await client.get(f"{server.rstrip('/')}/api/remote/status")
        r.raise_for_status()
        return r.json()


async def wait_for_slot(
    server: str,
    *,
    poll_interval_s: float = 30.0,
    on_wait: Callable[[dict], None] | None = None,
    timeout_s: float | None = None,
) -> dict:
    """Polla /api/remote/status finche' available > 0 (almeno uno slot WS
    libero) e ritorna lo status finale. `on_wait(status)` chiamato ad ogni
    iterazione in cui lo slot non e' libero — utile per stampare un messaggio
    all'utente esterno. `timeout_s` opzionale: se superato raise TimeoutError.
    """
    loop = asyncio.get_event_loop()
    start = loop.time()
    while True:
        try:
            status = await http_remote_status(server)
        except Exception as e:
            log.warning("remote status check failed: %s", e)
            status = {"connected": -1, "limit": -1, "available": 0,
                      "accepting_joins": False, "error": str(e)}
        if status.get("available", 0) > 0:
            return status
        if on_wait is not None:
            try:
                on_wait(status)
            except Exception:
                log.exception("on_wait callback failed")
        if timeout_s is not None and (loop.time() - start) >= timeout_s:
            raise asyncio.TimeoutError(
                f"wait_for_slot timed out after {timeout_s}s, last status={status}"
            )
        await asyncio.sleep(poll_interval_s)


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
        """Main loop with reconnect.

        Su WS_CLOSE_AT_CAPACITY (4429) il server e' pieno: aspettiamo che
        si liberi uno slot pollando /api/remote/status, poi ritentiamo.
        Lo user-facing log dice 'queueing for free slot...'.
        """
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
            except CapacityFullError as e:
                log.info(
                    "server at capacity (%s), waiting for a free slot...",
                    e.status,
                )
                def _on_wait(st: dict) -> None:
                    log.info(
                        "still waiting: %d/%d connected, retry in 30s...",
                        st.get("connected", 0), st.get("limit", 0),
                    )
                await wait_for_slot(self.server, on_wait=_on_wait)
                log.info("slot available, reconnecting...")
                backoff = 1.0  # reset backoff post-queue
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
            # ping_interval=30, ping_timeout=60: default websockets (20/20) e'
            # troppo stretto su CPU sotto carico (Pi 5 con LLM call attive
            # via Ollama). Il event loop puo' restare bloccato 30-40s sotto
            # peak -> ping non risponde in tempo -> close 1011 -> reconnect
            # loop ogni ~60s. A 60s di tolleranza il client si mantiene
            # connesso anche durante un'inferenza lunga.
            ws_ctx = websockets.connect(
                self._ws_url(),
                ping_interval=30,
                ping_timeout=60,
            )
        try:
            async with ws_ctx as ws:
                log.info("WS connected to %s", self.server)
                await self._send(ws, RequestSnapshotMsg().model_dump())
                await self._loop(ws)
        except Exception as e:
            # Captura close-code 4429 (server pieno) e lo rilancia come
            # CapacityFullError, cosi' run() lo distingue dal retry normale.
            code = None
            rcvd = getattr(e, "rcvd", None)
            if rcvd is not None:
                code = getattr(rcvd, "code", None)
            if code is None:
                code = getattr(e, "code", None)
            if code == WS_CLOSE_AT_CAPACITY:
                raise CapacityFullError(
                    reason=getattr(e, "reason", "") or str(e),
                )
            raise

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
