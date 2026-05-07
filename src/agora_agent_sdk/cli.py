"""Command-line entry point: `agora-agent`.

Composes AgoraClient + Brain + OllamaClient (or NoOpLLM with --no-llm) and runs
the WS loop until the agent dies or the user interrupts.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

from agora_agent_sdk.brain import Brain
from agora_agent_sdk.client import (
    AgentDiedExit, AgoraClient, JoinError,
    default_token_path, delete_token, http_join, read_token, write_token,
)
from agora_agent_sdk.llm import NoOpLLM, OllamaClient
from agora_core.policy import Policy
from agora_core.world_mirror import WorldMirror, pack_walkable_mask

log = logging.getLogger("agora_agent_sdk.cli")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="agora-agent",
                                description="Plug an LLM agent into the agora world.")
    p.add_argument("--server", default=os.environ.get("AGORA_SERVER"),
                   help="agora server base URL (e.g. https://agora.chatbot4eva.com). "
                        "REQUIRED unless AGORA_SERVER is set.")
    p.add_argument("--name", required=True)
    p.add_argument("--seed", required=True, help="personality seed (1-500 char)")
    p.add_argument("--sex", choices=["F", "M"], required=True)
    p.add_argument("--color", default=None, help="optional #RRGGBB")
    p.add_argument("--ollama-host", default="http://localhost:11434")
    p.add_argument("--model", default="qwen2.5:1.5b")
    p.add_argument("--no-llm", action="store_true",
                   help="disable LLM — run only reflex/social/wander")
    p.add_argument("--llm-decide-interval", type=int, default=120,
                   help="ticks between LLM 'decide' calls (default 120)")
    p.add_argument("--ring-buffer", type=int, default=30,
                   help="size of episodic ring buffer (default 30)")
    p.add_argument("--token-file", type=Path, default=None,
                   help="override default token persistence path")
    p.add_argument("--max-reconnect-attempts", type=int, default=0,
                   help="0 = infinite")
    p.add_argument("--policy-file", type=Path, default=None,
                   help="optional .pkl with a trained MLP policy")
    p.add_argument("--log-level", default="INFO",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p


async def _async_main(ns: argparse.Namespace) -> int:
    logging.basicConfig(level=ns.log_level,
                        format="%(asctime)s %(name)s %(levelname)s %(message)s")
    if not ns.server:
        log.error("--server is required (or set AGORA_SERVER)")
        return 2

    token_path = ns.token_file or default_token_path(ns.name)

    existing = read_token(token_path)
    if existing is not None:
        agent_id, token = existing
        log.info("resuming existing session: agent_id=%s from %s",
                 agent_id, token_path)
        world_w, world_h = 64, 64
    else:
        try:
            join = await http_join(
                ns.server, name=ns.name, personality_seed=ns.seed,
                sex=ns.sex, color=ns.color,
            )
        except JoinError as e:
            log.error("join failed: HTTP %s %s", e.code, e.payload)
            return 3
        agent_id = join.agent_id
        token = join.token
        world_w = join.world_w
        world_h = join.world_h
        write_token(token_path, agent_id, token)

    grid = [[True] * world_w for _ in range(world_h)]
    mirror = WorldMirror(world_w=world_w, world_h=world_h,
                         walkable_mask=pack_walkable_mask(grid))

    if ns.no_llm:
        llm = NoOpLLM()
    else:
        llm = OllamaClient(host=ns.ollama_host, model=ns.model)

    policy = None
    if ns.policy_file is not None and ns.policy_file.exists():
        policy = Policy(ns.policy_file)
        if not policy.load():
            log.warning("could not load policy from %s", ns.policy_file)
            policy = None

    brain = Brain(
        mirror=mirror, llm=llm, agent_id=agent_id,
        agent_name=ns.name, sex=ns.sex, color=ns.color or "#fff",
        personality_seed=ns.seed,
        llm_decide_interval=ns.llm_decide_interval,
        ring_buffer_size=ns.ring_buffer,
        policy=policy,
    )

    client = AgoraClient(
        ns.server, agent_id=agent_id, token=token,
        world_w=world_w, world_h=world_h, brain=brain,
        token_path=token_path,
        max_reconnect_attempts=ns.max_reconnect_attempts,
    )

    try:
        await client.run()
        return 0
    except AgentDiedExit as e:
        print(
            f"Your agent {ns.name} died at tick {e.tick}.\n"
            f"Run agora-agent again with a new --name to spawn a new one.",
            file=sys.stderr,
        )
        return 0
    except KeyboardInterrupt:
        return 130
    finally:
        await llm.aclose()


def main() -> int:
    parser = _build_parser()
    ns = parser.parse_args()
    return asyncio.run(_async_main(ns))


if __name__ == "__main__":
    sys.exit(main())
