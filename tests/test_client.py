import asyncio
import json

import httpx
import pytest

from agent.client import DEMO_MESSAGES, DEMO_TOOLS, LLMClient, LLMError
from agent.config import load_config

CFG = load_config().llm


def sse(*chunks) -> bytes:
    return b"".join(f"data: {c if isinstance(c, str) else json.dumps(c)}\n\n".encode() for c in chunks)


def delta(d, finish=None):
    return {"choices": [{"index": 0, "delta": d, "finish_reason": finish}]}


def tc(args, **first):
    return delta({"tool_calls": [{"index": 0, "function": {"arguments": args, **first.pop("fn", {})}, **first}]})


CANNED_TOOL_STREAM = sse(
    delta({"role": "assistant", "content": None}),
    tc("{", id="abc", type="function", fn={"name": "python"}),
    tc('"code": "print(1'),
    tc(')"}'),
    delta({}, finish="tool_calls"),
    {"choices": [], "timings": {"cache_n": 5, "prompt_n": 3, "predicted_n": 9}, "usage": {"prompt_tokens": 8}},
    "[DONE]",
)


def run_canned(body: bytes, status=200):
    client = LLMClient(CFG)
    client.http = httpx.AsyncClient(
        base_url="http://test", transport=httpx.MockTransport(lambda req: httpx.Response(status, content=body))
    )
    return asyncio.run(client.chat(DEMO_MESSAGES, DEMO_TOOLS))


def test_reassembles_tool_call_fragments():
    r = run_canned(CANNED_TOOL_STREAM)
    assert r.message == {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"id": "abc", "type": "function", "function": {"name": "python", "arguments": '{"code": "print(1)"}'}}],
    }
    assert r.finish_reason == "tool_calls"
    assert r.timings == {"cache_n": 5, "prompt_n": 3, "predicted_n": 9}
    assert r.usage == {"prompt_tokens": 8}
    assert r.t_send <= r.t_first_chunk <= r.t_first_delta <= r.t_end


def test_http_error_raises():
    with pytest.raises(LLMError, match="400"):
        run_canned(b'{"error": "exceeds context"}', status=400)


# ---- live: needs the llama.cpp server at config base_url ----

def server_up() -> bool:
    try:
        return httpx.get(f"{CFG.base_url}/health", timeout=2).status_code == 200
    except httpx.HTTPError:
        return False


live = pytest.mark.skipif(not server_up(), reason="llama.cpp server not reachable")


@live
@pytest.mark.parametrize(
    "messages,tools",
    [(DEMO_MESSAGES, DEMO_TOOLS), ([{"role": "user", "content": "Name three primary colors, briefly."}], None)],
    ids=["tool_call", "text"],
)
def test_stream_matches_non_stream(messages, tools):
    """The message we reassemble from SSE deltas == what the server returns without streaming."""

    async def both():
        client = LLMClient(CFG)
        try:
            streamed = await client.chat(messages, tools)
            body = client.build_body(messages, tools)
            body["stream"] = False
            del body["stream_options"]
            plain = (await client.http.post("/v1/chat/completions", json=body)).json()
        finally:
            await client.aclose()
        return streamed, plain

    streamed, plain = asyncio.run(both())
    expected = plain["choices"][0]["message"]
    got = streamed.message

    assert streamed.finish_reason == plain["choices"][0]["finish_reason"]
    assert got["content"] == expected["content"]
    strip_id = lambda calls: [{k: v for k, v in c.items() if k != "id"} for c in calls or []]
    assert strip_id(got.get("tool_calls")) == strip_id(expected.get("tool_calls"))
    t = streamed.timings
    assert streamed.usage["prompt_tokens"] == t["cache_n"] + t["prompt_n"]
    assert streamed.usage["completion_tokens"] == t["predicted_n"]
