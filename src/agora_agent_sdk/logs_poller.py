"""Background log poller: pulls the agent's persisted observation log from
the server periodically and prints each new entry to stdout.

Useful when you want to see what the server says happened to your agent
(decisions, action results, dialogues, gifts received, etc.) without
opening the web UI inspector. Polls GET /api/agents/{id}/logs?since=N
and uses the `next_since` in the response to advance the cursor.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass

import httpx

log = logging.getLogger("agora_agent_sdk.logs")


# Compact pretty-printer for the most common kinds. Falls back to a json dump.
def _format_log_line(entry: dict) -> str:
    tick = entry.get("tick", 0)
    kind = entry.get("kind", "?")
    c = entry.get("content", {}) or {}
    if kind == "decision":
        action = c.get("action", "?")
        via = c.get("via", "?")
        extras = " ".join(
            f"{k}={c[k]}" for k in ("direction", "target_id", "item",
                                    "qty", "recipe", "structure", "content")
            if k in c and c[k] is not None
        )
        thought = c.get("thought") or ""
        return (f"t{tick:>6}  decision      via={via:<8} {action} {extras}"
                + (f"  | {thought[:60]}" if thought else ""))
    if kind == "action_result":
        action = c.get("action", "?")
        ok = c.get("ok")
        marker = "OK" if ok else "FAIL"
        reason = f" reason={c.get('reason')}" if not ok else ""
        return f"t{tick:>6}  result        {marker:<4} {action}{reason}"
    if kind == "dialogue_received":
        who = c.get("from_name", "?")
        what = (c.get("content") or "")[:80]
        return f't{tick:>6}  dialogue from {who:<10}: "{what}"'
    if kind == "perception":
        return f"t{tick:>6}  perception    (visible_around / nearby snapshot)"
    return f"t{tick:>6}  {kind:<14} {json.dumps(c)[:120]}"


@dataclass
class LogsPoller:
    server: str
    agent_id: int
    interval_s: float = 30.0
    start_since: int = 0
    # Filtro server-side: comma-separated kind. "" = tutti.
    kind: str = ""

    async def run(self, stop: asyncio.Event) -> None:
        """Poll until `stop` is set. Each new entry is logged via the
        'agora_agent_sdk.logs' logger (INFO level by default)."""
        since = self.start_since
        url = f"{self.server.rstrip('/')}/api/agents/{self.agent_id}/logs"
        async with httpx.AsyncClient(timeout=10.0) as http:
            log.info("logs poller started: every %.0fs, since=%d", self.interval_s, since)
            while not stop.is_set():
                try:
                    params = {"since": since, "limit": 200}
                    if self.kind:
                        params["kind"] = self.kind
                    r = await http.get(url, params=params)
                    if r.status_code != 200:
                        log.warning("logs poll HTTP %d: %s", r.status_code, r.text[:200])
                    else:
                        data = r.json()
                        for entry in data.get("logs", []):
                            log.info("%s", _format_log_line(entry))
                        since = data.get("next_since", since)
                except Exception as e:
                    log.warning("logs poll error: %s", e)
                try:
                    await asyncio.wait_for(stop.wait(), timeout=self.interval_s)
                except asyncio.TimeoutError:
                    pass
            log.info("logs poller stopped")
