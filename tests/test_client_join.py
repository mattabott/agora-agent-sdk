import json
from pathlib import Path

import httpx
import pytest

from agora_agent_sdk.client import (
    JoinError, default_token_path, delete_token, http_join,
    read_token, write_token,
)


def test_token_roundtrip(tmp_path: Path):
    path = tmp_path / "test.token"
    write_token(path, 5, "abc")
    out = read_token(path)
    assert out == (5, "abc")
    delete_token(path)
    assert read_token(path) is None


def test_default_token_path(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    p = default_token_path("Maya")
    assert p.name == "Maya.token"
    assert ".agora-agent" in str(p)


def test_read_token_missing():
    assert read_token(Path("/nonexistent/path.token")) is None


def test_delete_token_missing_no_error(tmp_path: Path):
    delete_token(tmp_path / "nope.token")  # no exception


@pytest.mark.asyncio
async def test_http_join_success():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["name"] == "Maya"
        assert body["sex"] == "F"
        return httpx.Response(200, json={
            "agent_id": 5, "token": "tk-123", "world_seed": 42,
            "tick_ms": 1000, "world_w": 64, "world_h": 64,
            "action_schema_version": 1,
        })

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        result = await http_join(
            "http://mock", name="Maya", personality_seed="curious",
            sex="F", color="#7fa9d4", http_client=client,
        )
    assert result.agent_id == 5
    assert result.token == "tk-123"


@pytest.mark.asyncio
async def test_http_join_409_name_taken():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(409, json={"error": "name_taken",
                                          "suggestions": ["Maya2"]})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(JoinError) as exc:
            await http_join("http://mock", name="Maya",
                            personality_seed="x", sex="F",
                            http_client=client)
        assert exc.value.code == 409
        assert exc.value.payload["error"] == "name_taken"


@pytest.mark.asyncio
async def test_http_join_426_schema_mismatch():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(426, json={"error": "schema_mismatch",
                                          "server_schema": 2,
                                          "client_schema": 1})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        with pytest.raises(JoinError) as exc:
            await http_join("http://mock", name="Maya",
                            personality_seed="x", sex="F",
                            http_client=client)
        assert exc.value.code == 426
