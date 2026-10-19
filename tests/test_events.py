import asyncio
import json

from agent.events import EventLog, Tracer
from agent.loop import run_agent
from agent.tools import Tool
from tests.test_loop import FakeClient, reply


async def echo(x: str) -> str:
    return x


def test_events_and_headers_for_a_two_step_task(tmp_path):
    path = tmp_path / "events.jsonl"
    log = EventLog(path, "run_test")
    client = FakeClient(reply(calls=[("echo", '{"x": "hi"}'), ("nope", "{}")]), reply("done"))
    tools = [Tool("echo", "", {}, echo, extra_delay_s=0.0)]
    asyncio.run(run_agent("q?", "sys", tools, client, 10, Tracer(log, "agent-7", "task-x")))
    log.close()

    events = [json.loads(line) for line in open(path)]
    assert [(e["event"], e["step"]) for e in events] == [
        ("request_sent", 1), ("first_token", 1), ("request_end", 1),
        ("tool_start", 1), ("tool_end", 1), ("tool_start", 1), ("tool_end", 1),
        ("request_sent", 2), ("first_token", 2), ("request_end", 2),
    ]
    assert all(e["run_id"] == "run_test" and e["agent_id"] == "agent-7" and e["task_id"] == "task-x" for e in events)
    assert "timings" in events[2] and "usage" in events[2]
    assert [e["ok"] for e in events if e["event"] == "tool_end"] == [True, False]  # unknown tool "nope"

    # each request carries the ids in headers, and the request_id matches the events
    sent = [e["request_id"] for e in events if e["event"] == "request_sent"]
    assert [h["X-Request-Id"] for h in client.headers] == sent and len(set(sent)) == 2
    assert [h["X-Step"] for h in client.headers] == ["1", "2"]
    assert client.headers[0]["X-Agent-Id"] == "agent-7" and client.headers[0]["X-Task-Id"] == "task-x"
