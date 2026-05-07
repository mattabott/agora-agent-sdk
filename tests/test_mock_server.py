import json

import httpx
import pytest

from tests.mock_server import make_mock_app


@pytest.mark.asyncio
async def test_mock_join_success():
    app, state = make_mock_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://mock") as client:
        r = await client.post("/api/agents/join", json={
            "name": "Maya", "personality_seed": "x", "sex": "F",
            "action_schema_version": 1, "client_version": "test/0.1",
        })
        assert r.status_code == 200
        data = r.json()
        assert data["agent_id"] == 1


@pytest.mark.asyncio
async def test_mock_join_409_duplicate():
    app, state = make_mock_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://mock") as client:
        body = {"name": "Maya", "personality_seed": "x", "sex": "F",
                "action_schema_version": 1, "client_version": "t/0"}
        await client.post("/api/agents/join", json=body)
        r = await client.post("/api/agents/join", json=body)
        assert r.status_code == 409


@pytest.mark.asyncio
async def test_mock_join_426_schema():
    app, state = make_mock_app()
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport,
                                 base_url="http://mock") as client:
        r = await client.post("/api/agents/join", json={
            "name": "Maya", "personality_seed": "x", "sex": "F",
            "action_schema_version": 999, "client_version": "t/0",
        })
        assert r.status_code == 426
