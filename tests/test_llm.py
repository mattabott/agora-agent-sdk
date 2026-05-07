import json
import pytest
import httpx

from agora_agent_sdk.llm import OllamaClient, NoOpLLM


@pytest.mark.asyncio
async def test_decide_parses_valid_json():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert body["model"] == "test-model"
        assert body["format"] == "json"
        return httpx.Response(200, json={
            "response": json.dumps({"action": "wait", "thought": "ok"})
        })

    transport = httpx.MockTransport(handler)
    client = OllamaClient(host="http://x", model="test-model")
    client._client = httpx.AsyncClient(transport=transport)
    out = await client.decide("sys", "user")
    assert out == {"action": "wait", "thought": "ok"}
    await client.aclose()


@pytest.mark.asyncio
async def test_decide_invalid_json_returns_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": "not json"})

    transport = httpx.MockTransport(handler)
    client = OllamaClient(host="http://x", model="m")
    client._client = httpx.AsyncClient(transport=transport)
    assert await client.decide("s", "u") == {}
    await client.aclose()


@pytest.mark.asyncio
async def test_decide_http_error_returns_empty():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500)

    transport = httpx.MockTransport(handler)
    client = OllamaClient(host="http://x", model="m")
    client._client = httpx.AsyncClient(transport=transport)
    assert await client.decide("s", "u") == {}
    await client.aclose()


@pytest.mark.asyncio
async def test_talk_line_returns_response_text():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"response": "Hello there friend"})

    transport = httpx.MockTransport(handler)
    client = OllamaClient(host="http://x", model="m")
    client._client = httpx.AsyncClient(transport=transport)
    assert await client.talk_line("s", "u") == "Hello there friend"
    await client.aclose()


@pytest.mark.asyncio
async def test_no_op_llm_returns_empty():
    n = NoOpLLM()
    assert await n.decide("s", "u") == {}
    assert await n.talk_line("s", "u") == ""
    await n.aclose()
