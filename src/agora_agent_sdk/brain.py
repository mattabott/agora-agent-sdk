"""Decision orchestrator client-side.

Pipeline:
  1. apply_perception to mirror
  2. compute reflex; if reflex returns → use it (via="reflex")
  3. else if pending_llm_task done with non-empty result → use it (via="llm")
  4. else fast_decision = social or policy or wander (via="social"/"policy"/"auto_cooldown")
  5. dispatch background LLM if cooldown elapsed
  6. dispatch background dialogue gen if partner adjacent and slot free
  7. resolve talk content (use next_talk_line if ready, else wait)
  8. validate action client-side (talk dedup, move walkable)
  9. log to ring buffer
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from typing import Awaitable, Callable

from agora_core.dialogue_filters import accept_dialogue_line, append_to_ring
from agora_core.policy import (
    Policy, decode_to_decision, extract_features,
)
from agora_core.prompts import (
    SYSTEM_PROMPT, DIALOGUE_SYSTEM,
    build_dialogue_user_prompt, build_user_prompt,
)
from agora_core.protocol import validate_action_dict
from agora_core.reflex import try_reflex
from agora_core.social import social_navigate
from agora_core.world_mirror import AgentSnap, WorldMirror

log = logging.getLogger("agora_agent_sdk.brain")


class Brain:
    def __init__(
        self,
        *,
        mirror: WorldMirror,
        llm,
        agent_id: int,
        agent_name: str,
        sex: str,
        color: str,
        personality_seed: str,
        llm_decide_interval: int = 120,
        ring_buffer_size: int = 30,
        policy: Policy | None = None,
    ):
        self.mirror = mirror
        self.llm = llm
        self.agent_id = agent_id
        self.agent_name = agent_name
        self.sex = sex
        self.color = color
        self.personality_seed = personality_seed
        self.llm_decide_interval = llm_decide_interval
        self.policy = policy

        self.episodic: deque[dict] = deque(maxlen=ring_buffer_size)
        self.recent_lines_by_agent: dict[int, list[str]] = {}
        self.next_talk_line: str = ""
        self.last_walkable_dirs: list[str] = []
        self.last_move_direction: str = ""
        self.pending_llm_task: asyncio.Task | None = None
        self.pending_dialogue_task: asyncio.Task | None = None
        self.last_llm_decide_tick: int = -10**9
        self.llm_cooldown: int = 0

    def push_episodic(self, entry: dict) -> None:
        self.episodic.append(dict(entry))

    def push_dialogue_received(self, from_id: int, from_name: str,
                                content: str, tick: int) -> None:
        self.push_episodic({
            "kind": "dialogue_received", "tick": tick,
            "from_id": from_id, "from_name": from_name, "content": content,
        })
        ring = self.recent_lines_by_agent.setdefault(from_id, [])
        append_to_ring(ring, content)

    def push_event_to_episodic(self, ev: dict) -> None:
        kind = ev.get("kind")
        if kind == "dialogue_received":
            self.push_dialogue_received(
                int(ev["from_id"]), ev["from_name"], ev["content"],
                int(ev.get("tick", self.mirror.current_tick)),
            )
        elif kind in ("gift_received", "loss", "user_message"):
            self.push_episodic({**ev, "kind": kind})

    def _has_close_partner(self, perception: dict) -> int:
        nearby = perception.get("nearby_agents") or []
        for o in nearby:
            ox, oy = int(o.get("x", 0)), int(o.get("y", 0))
            mirror = self.mirror
            self_a = mirror.agents.get(self.agent_id)
            if self_a is None:
                return 0
            if max(abs(ox - self_a.x), abs(oy - self_a.y)) <= 2:
                return int(o["id"])
        return 0

    async def _llm_think_bg(self, perception: dict, inventory: dict) -> dict:
        try:
            self_a = self.mirror.agents.get(self.agent_id)
            sa = perception.get("agent_state", {})
            user = build_user_prompt(
                personality_current=sa.get("personality_current",
                                            self.personality_seed),
                sex=self.sex,
                born_tick=int(sa.get("born_tick", 0)),
                current_tick=self.mirror.current_tick,
                family=perception.get("family"),
                current_goal=sa.get("current_goal", ""),
                perception={
                    "position": [self_a.x, self_a.y] if self_a else [0, 0],
                    "terrain_here": perception.get("terrain_here", ""),
                    "energy": int(sa.get("energy", 0)),
                    "mood": int(sa.get("mood", 0)),
                    "hunger": int(sa.get("hunger", 0)),
                    "walkable_dirs": perception.get("walkable_dirs", []),
                    "escape_dirs": perception.get("escape_dirs", []),
                    "visible_around": perception.get("visible_around", ""),
                    "nearby_agents": perception.get("nearby_agents", []),
                    "nearby_resources": perception.get("nearby_resources", []),
                    "here_resource": perception.get("here_resource"),
                    "here_structure": perception.get("here_structure"),
                    "world_events": perception.get("world_events", []),
                },
                inventory=inventory,
                relations={int(k): v for k, v in
                           (perception.get("relations") or {}).items()},
                agents_by_id=self.mirror.agents,
                episodic=list(self.episodic),
                semantic=None,
                wait_streak=int(sa.get("wait_streak", 0)),
            )
            return await self.llm.decide(SYSTEM_PROMPT, user)
        except Exception:
            log.exception("llm_think_bg failed")
            return {}

    async def _dialogue_gen_bg(self, partner_id: int) -> str:
        try:
            self_a = self.mirror.agents.get(self.agent_id)
            partner = self.mirror.agents.get(partner_id)
            if self_a is None or partner is None:
                return ""

            recent_lines = []
            for m in list(self.episodic)[-12:]:
                if m.get("kind") == "dialogue_received" and m.get("from_id") == partner_id:
                    recent_lines.append(f"  {partner.name}: {m['content']}")
            recent_text = "\n".join(recent_lines) or ""

            nearby_res: list[tuple[str, int]] = []
            for (rx, ry), (rt, rq) in self.mirror.resources.items():
                if max(abs(rx - self_a.x), abs(ry - self_a.y)) <= 3:
                    nearby_res.append((rt, rq))
            nearby_res = nearby_res[:4]
            nearby_struct = [info.type for (sx, sy), info in self.mirror.structures.items()
                             if max(abs(sx - self_a.x), abs(sy - self_a.y)) <= 3][:3]
            ongoing = [ev.type for ev in self.mirror.events.values()][:3]

            from agora_core.age import age_in_days
            sys_p = DIALOGUE_SYSTEM.format(
                name=self.agent_name,
                sex_label="man" if self.sex == "M" else "woman",
                age_days=age_in_days(self_a.born_tick, self.mirror.current_tick),
                personality=self.personality_seed[:200],
                partner_name=partner.name,
                partner_sex_label="man" if partner.sex == "M" else "woman",
                partner_age=age_in_days(partner.born_tick, self.mirror.current_tick),
            )
            user_p = build_dialogue_user_prompt(
                self_name=self.agent_name,
                self_x=self_a.x, self_y=self_a.y,
                mood=0, hunger=0, energy=0,
                current_tick=self.mirror.current_tick,
                last_thought="", current_goal="",
                last_reflection="",
                partner_name=partner.name,
                nearby_resources=nearby_res,
                nearby_structures=nearby_struct,
                ongoing_events=ongoing,
                recent_dialogue_text=recent_text,
            )
            raw = await self.llm.talk_line(sys_p, user_p)
            agent_names = [a.name for a in self.mirror.agents.values()]
            line = accept_dialogue_line(
                raw, agent_names=agent_names,
                recent_lines_by_agent=self.recent_lines_by_agent,
            )
            return line or ""
        except Exception:
            log.exception("dialogue_gen_bg failed")
            return ""

    def _validate_pre_send(self, decision: dict) -> dict:
        a = decision.get("action")
        if a == "talk":
            content = (decision.get("content") or "").strip()
            if len(content.split()) < 3:
                return {"action": "wait", "thought": "(too short)"}
            norm = " ".join(content.lower().split())
            for buf in self.recent_lines_by_agent.values():
                if norm in buf:
                    return {"action": "wait", "thought": "(already said)"}
        if a == "move":
            d = decision.get("direction")
            # Anti-oscillation: if proposed direction is opposite of last move,
            # downgrade to wait to break a potential ping-pong.
            OPPOSITES = {
                "north": "south", "south": "north",
                "east": "west", "west": "east",
            }
            if d and OPPOSITES.get(self.last_move_direction) == d:
                return {"action": "wait", "thought": "(anti-oscillation)"}
            if d not in self.last_walkable_dirs:
                return {"action": "wander", "thought": "(blocked)"}
        return decision

    async def decide(self, perception: dict) -> dict:
        self.mirror.apply_perception(perception)
        sa = perception.get("agent_state", {})
        self.last_walkable_dirs = list(perception.get("walkable_dirs") or [])
        self_a = self.mirror.agents.get(self.agent_id)
        inventory = dict(sa.get("inventory") or {})

        reflex_dec = None
        if self_a is not None:
            reflex_dec = try_reflex(
                self.mirror, self_a, perception, inventory,
                aff_out={int(k): v for k, v in (perception.get("relations") or {}).items()},
                aff_in={int(k): v for k, v in (perception.get("relations_inbound") or {}).items()},
                sex=self.sex, born_tick=int(sa.get("born_tick", 0)),
                hunger=int(sa.get("hunger", 0)),
                energy=int(sa.get("energy", 100)),
            )

        prebaked_llm: dict | None = None
        if self.pending_llm_task and self.pending_llm_task.done():
            try:
                res = self.pending_llm_task.result()
                if isinstance(res, dict) and res.get("action"):
                    prebaked_llm = res
            except Exception:
                log.exception("pending llm task failed")
            self.pending_llm_task = None
        if self.pending_dialogue_task and self.pending_dialogue_task.done():
            try:
                line = self.pending_dialogue_task.result()
                if line:
                    self.next_talk_line = line
            except Exception:
                log.exception("pending dialogue task failed")
            self.pending_dialogue_task = None

        if reflex_dec is not None:
            decision, via = reflex_dec, "reflex"
        elif prebaked_llm is not None:
            decision, via = prebaked_llm, "llm"
            self.llm_cooldown = 4
        else:
            wait_streak = int(sa.get("wait_streak", 0))
            social = None
            if self_a is not None:
                social = social_navigate(
                    self.mirror, self_a, perception,
                    self.mirror.current_tick, inventory,
                    next_talk_line_ready=bool(self.next_talk_line),
                    wait_streak=wait_streak,
                )
            if social is not None:
                decision, via = social, "social"
            elif self.policy is not None:
                feats = extract_features(
                    current_tick=self.mirror.current_tick,
                    born_tick=int(sa.get("born_tick", 0)),
                    hunger=int(sa.get("hunger", 0)),
                    mood=int(sa.get("mood", 0)),
                    energy=int(sa.get("energy", 0)),
                    hp=int(sa.get("hp", 100)),
                    inventory=inventory,
                    perception=perception,
                    wait_streak=wait_streak,
                    sleep_streak=int(sa.get("sleep_streak", 0)),
                )
                idx = self.policy.predict(feats)
                pol_dec = decode_to_decision(idx, perception, inventory) if idx is not None else None
                decision, via = (
                    (pol_dec, "policy") if pol_dec is not None
                    else ({"action": "wander", "thought": "(idle moment)"}, "auto_cooldown")
                )
            else:
                decision, via = {"action": "wander", "thought": "(idle moment)"}, "auto_cooldown"

            if self.llm_cooldown > 0:
                self.llm_cooldown -= 1
            if (self.pending_llm_task is None
                and self.mirror.current_tick - self.last_llm_decide_tick
                    > self.llm_decide_interval):
                self.pending_llm_task = asyncio.create_task(
                    self._llm_think_bg(perception, inventory)
                )
                self.last_llm_decide_tick = self.mirror.current_tick

        partner_id = self._has_close_partner(perception)
        if partner_id and (self.pending_dialogue_task is None
                           or self.pending_dialogue_task.done()):
            self.pending_dialogue_task = asyncio.create_task(
                self._dialogue_gen_bg(partner_id)
            )

        if decision.get("action") == "talk":
            if decision.get("content") == "<<USE_NEXT_TALK_LINE>>":
                if self.next_talk_line:
                    decision["content"] = self.next_talk_line
                    self.next_talk_line = ""
                else:
                    decision = {"action": "wait", "thought": "(waiting for words)"}
            elif not (decision.get("content") or "").strip():
                decision = {"action": "wait", "thought": "(no content)"}

        decision = self._validate_pre_send(decision)

        if decision.get("action") == "talk":
            ring = self.recent_lines_by_agent.setdefault(self.agent_id, [])
            append_to_ring(ring, decision.get("content", ""))
        self.push_episodic({
            "kind": "decision",
            "tick": self.mirror.current_tick,
            "thought": decision.get("thought", "")[:120],
            "action": decision.get("action", "wait"),
            "via": via,
        })
        # Track last move direction for anti-oscillation in next decide()
        if decision.get("action") == "move":
            self.last_move_direction = decision.get("direction", "")
        elif decision.get("action") not in ("wait",):
            # Reset when not moving and not waiting (waits don't change position)
            self.last_move_direction = ""

        decision = dict(decision)
        decision["decided_via"] = via
        ok, _ = validate_action_dict(decision)
        if not ok:
            return {"action": "wait", "thought": "(invalid)", "decided_via": via}
        return decision
