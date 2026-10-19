"""Offline tests for the agent loop: a scripted fake client replaces the server."""

import asyncio
import copy
import json

from agent.client import ChatResult, LLMError
from agent.loop import run_agent
from agent.tools import Tool


def reply(content="", calls=(), finish=None):
    """A ChatResult as the client would return it. calls: [(name, arguments_json_string), ...]"""
    msg = {"role": "assistant", "content": content}
    if calls:
        msg["tool_calls"] = [
            {"id": f"id{i}", "type": "function", "function": {"name": n, "arguments": a}} for i, (n, a) in enumerate(calls)
        ]
    finish = finish or ("tool_calls" if calls else "stop")
    return ChatResult(msg, finish, {"cache_n": 0, "prompt_n": 1, "predicted_n": 1}, {}, 0.0, 0.0, 0.0, 0.0)


class FakeClient:
    """Returns scripted replies in order; records a snapshot of the messages it was sent."""

    def __init__(self, *script):
        self.script = list(script)
        self.sent = []
        self.headers = []

    async def chat(self, messages, tools, headers=None):
        self.sent.append(copy.deepcopy(messages))
        self.headers.append(headers)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


tool_runs = []


async def add(a: int, b: int) -> str:
    tool_runs.append((a, b))
    return str(a + b)


async def boom() -> str:
    raise ValueError("kaboom")


ADD = Tool("add", "add two ints", {"type": "object"}, add)
BOOM = Tool("boom", "always fails", {"type": "object"}, boom)


def run(client, tools=(ADD, BOOM), max_steps=10):
    tool_runs.clear()
    return asyncio.run(run_agent("q?", "sys", list(tools), client, max_steps))


def test_direct_answer():
    res = run(FakeClient(reply("42")))
    assert (res.status, res.final_answer, res.steps) == ("answered", "42", 1)


def test_tool_call_then_answer_message_layout():
    client = FakeClient(reply(calls=[("add", '{"a": 2, "b": 3}')]), reply("5"))
    res = run(client)
    assert (res.status, res.final_answer, res.steps) == ("answered", "5", 2)
    assert [m["role"] for m in res.messages] == ["system", "user", "assistant", "tool", "assistant"]
    # assistant message appended exactly as returned; tool message points back at its id
    assert res.messages[2] is res.step_results[0].message
    assert res.messages[3] == {"role": "tool", "tool_call_id": "id0", "content": "5"}


def test_messages_are_append_only():
    client = FakeClient(
        reply(calls=[("add", '{"a": 1, "b": 1}')]),
        reply(calls=[("add", '{"a": 2, "b": 2}'), ("boom", "{}")]),
        reply("done"),
    )
    run(client)
    for prev, nxt in zip(client.sent, client.sent[1:]):
        assert nxt[: len(prev)] == prev  # each request's messages are a prefix of the next one's


def test_tool_errors_go_back_to_model_and_count_as_steps():
    client = FakeClient(
        reply(calls=[("add", '{"a": 1, "b": ')]),  # invalid JSON
        reply(calls=[("nope", "{}")]),  # unknown tool
        reply(calls=[("boom", "{}")]),  # tool raises
        reply(calls=[("add", '{"a": 1}')]),  # missing argument -> TypeError
        reply("gave up"),
    )
    res = run(client)
    tool_msgs = [m["content"] for m in res.messages if m["role"] == "tool"]
    assert len(tool_msgs) == 4 and all(t.startswith("Error:") for t in tool_msgs)
    assert "not valid JSON" in tool_msgs[0]
    assert "unknown tool 'nope'" in tool_msgs[1]
    assert "kaboom" in tool_msgs[2]
    assert "TypeError" in tool_msgs[3]
    assert (res.status, res.steps) == ("answered", 5)


def test_max_steps_stops_without_running_last_tools():
    client = FakeClient(*[reply(calls=[("add", json.dumps({"a": i, "b": 0}))]) for i in range(3)])
    res = run(client, max_steps=3)
    assert (res.status, res.final_answer, res.steps) == ("max_steps", None, 3)
    assert tool_runs == [(0, 0), (1, 0)]  # step 3's call was not executed
    assert res.messages[-1]["role"] == "assistant"


def test_truncated():
    res = run(FakeClient(reply("la la la la", finish="length")))
    assert (res.status, res.final_answer, res.steps) == ("truncated", "la la la la", 1)


def test_llm_error_ends_task_without_crash():
    client = FakeClient(reply(calls=[("add", '{"a": 1, "b": 1}')]), LLMError("HTTP 400: context exceeded"))
    res = run(client)
    assert (res.status, res.final_answer, res.steps) == ("error", None, 1)
    assert "context exceeded" in res.error
