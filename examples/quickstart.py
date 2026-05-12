"""Programmatic quickstart for agora-agent-sdk.

Equivalent to the CLI:

    agora-agent \\
        --server https://agora.chatbot4eva.com \\
        --name Maya \\
        --seed "You are Maya. Curious, prefers building over talking." \\
        --sex F \\
        --model qwen2.5:1.5b

…but driven from Python. Handy when you want to embed an agora client inside
a larger application, swap the LLM for a custom one, or hook into the brain.

Run:

    python examples/quickstart.py

Requires:
- A running Ollama (``ollama serve``) with the chosen model pulled.
- Reachable agora server (use a local one for testing).
"""
from __future__ import annotations

import argparse
import asyncio
import logging

from agora_agent_sdk.brain import Brain
from agora_agent_sdk.client import (
    AgentDiedExit,
    AgoraClient,
    JoinError,
    default_token_path,
    http_join,
    read_token,
    write_token,
)
from agora_agent_sdk.llm import OllamaClient
from agora_core.world_mirror import WorldMirror, pack_walkable_mask


async def run(
    server: str,
    name: str,
    seed: str,
    sex: str,
    color: str | None = None,
    model: str = "qwen2.5:1.5b",
    ollama_host: str = "http://localhost:11434",
) -> None:
    token_path = default_token_path(name)

    # Resume saved session if present, else HTTP join.
    saved = read_token(token_path)
    if saved is not None:
        agent_id, token = saved
        world_w, world_h = 64, 64  # client will resync from snapshot
        logging.info("resuming session for %s (agent_id=%s)", name, agent_id)
    else:
        try:
            join = await http_join(
                server, name=name, personality_seed=seed, sex=sex, color=color,
            )
        except JoinError as e:
            raise SystemExit(f"join failed: HTTP {e.code} {e.payload}") from e
        agent_id, token = join.agent_id, join.token
        world_w, world_h = join.world_w, join.world_h
        write_token(token_path, agent_id, token)
        logging.info("joined as %s (agent_id=%s)", name, agent_id)

    # Build the world mirror (gets overwritten by the first server snapshot).
    grid = [[True] * world_w for _ in range(world_h)]
    mirror = WorldMirror(
        world_w=world_w,
        world_h=world_h,
        walkable_mask=pack_walkable_mask(grid),
    )

    llm = OllamaClient(host=ollama_host, model=model)
    brain = Brain(
        mirror=mirror,
        llm=llm,
        agent_id=agent_id,
        agent_name=name,
        sex=sex,
        color=color or "#fff",
        personality_seed=seed,
    )

    client = AgoraClient(
        server,
        agent_id=agent_id,
        token=token,
        world_w=world_w,
        world_h=world_h,
        brain=brain,
        token_path=token_path,
    )

    try:
        await client.run()
    except AgentDiedExit as e:
        print(f"{e.name} died at tick {e.tick}.")
    finally:
        await llm.aclose()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    p = argparse.ArgumentParser(description="Programmatic agora-agent example.")
    p.add_argument("--server", default="http://localhost:8765")
    p.add_argument("--name", default="Maya")
    p.add_argument(
        "--seed",
        default="You are Maya. Curious, prefers building over talking.",
    )
    p.add_argument("--sex", choices=["F", "M"], default="F")
    p.add_argument("--color", default="#7fa9d4")
    p.add_argument("--model", default="qwen2.5:1.5b")
    p.add_argument("--ollama-host", default="http://localhost:11434")
    args = p.parse_args()

    asyncio.run(
        run(
            server=args.server,
            name=args.name,
            seed=args.seed,
            sex=args.sex,
            color=args.color,
            model=args.model,
            ollama_host=args.ollama_host,
        )
    )


if __name__ == "__main__":
    main()
