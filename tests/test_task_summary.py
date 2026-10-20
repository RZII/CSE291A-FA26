from scripts.task_summary import summarize

CONFIG = {"agents": 4, "tools": {"python_delay_s": 0.0, "search_delay_s": 3.0}}


def ev(ts, event, step=None, **kw):
    return {"ts": ts, "event": event, "run_id": "run_x", "agent_id": "agent-1", "task_id": "t1", "step": step, **kw}


def timings(cache_n, prompt_n, predicted_n):
    return {"cache_n": cache_n, "prompt_n": prompt_n, "predicted_n": predicted_n, "prompt_ms": 100.0, "predicted_ms": 400.0}


# a 2-step hotpotqa task: request 1 -> search tool (3 s) -> request 2
EVENTS = [
    ev(10.0, "task_start", task_type="hotpotqa"),
    ev(10.0, "request_sent", 1, request_id="r1"),
    ev(10.2, "first_token", 1, request_id="r1"),
    ev(11.0, "request_end", 1, request_id="r1", timings=timings(0, 250, 30)),
    ev(11.0, "tool_start", 1, tool="search", call_id="c1"),
    ev(14.0, "tool_end", 1, tool="search", call_id="c1"),
    ev(14.0, "request_sent", 2, request_id="r2"),
    ev(14.5, "first_token", 2, request_id="r2"),
    ev(15.0, "request_end", 2, request_id="r2", timings=timings(279, 600, 20)),
    ev(15.0, "task_end", status="answered", steps=2, correct=True, final_context_tokens=899),
]
SCHED = [
    {"request_id": "r1", "queue_ms": 50.0, "model_request_ms": 900.0},
    {"request_id": "r2", "queue_ms": 300.0, "model_request_ms": 650.0},
    {"request_id": "other-run", "queue_ms": 9999.0, "model_request_ms": 9999.0},
]


def test_row_through_scheduler():
    (row,) = summarize(EVENTS, SCHED, CONFIG, scheduler_k=2)
    assert (row["run_id"], row["task_id"], row["task_type"], row["agent_id"]) == ("run_x", "t1", "hotpotqa", "agent-1")
    assert (row["num_agents"], row["scheduler_k"], row["tool_delay_s"]) == (4, 2, 3.0)
    assert (row["status"], row["is_success"], row["total_steps"], row["final_context_tokens"]) == ("answered", True, 2, 899)
    assert abs(row["total_completion_ms"] - 5000) < 1e-6
    assert row["queue_wait_ms"] == 350.0 and row["model_request_ms"] == 1550.0  # other run's row not counted
    assert row["llama_compute_ms"] == 1000.0
    assert abs(row["tool_wait_ms"] - 3000) < 1e-6
    assert abs(row["first_ttft_ms"] - 200) < 1e-6
    assert row["recompute_tokens"] == (0 + 250 + 30) - 279  # = 1, the normal case


def test_direct_run_has_no_scheduler_fields():
    (row,) = summarize(EVENTS, [], CONFIG, scheduler_k=2)
    assert row["scheduler_k"] is None and row["queue_wait_ms"] is None and row["model_request_ms"] is None
