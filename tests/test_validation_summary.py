import json

from scripts.validation_summary import REQUIRED, aggregate, summarize_run


def event(kind, ts, request=None, step=None, **extra):
    row = {"event": kind, "ts": ts, "task_id": "task", "agent_id": "agent-0",
           "run_id": "test", "step": step, **extra}
    if request:
        row["request_id"] = request
    return row


def success_events():
    return [event("task_start", .1), event("request_sent", .2, "r", 1),
            event("first_token", .3, "r", 1),
            event("request_end", .4, "r", 1, timings={"cache_n": 3, "prompt_n": 7}),
            event("task_end", .5, status="answered", correct=True, final_context_tokens=12)]


def write_run(path, events, runner_end=1):
    path.mkdir()
    experiment = {"finished": 10, "runner_exit_code": 0, "t_runner_start": 0,
                  "t_runner_end": runner_end, "tasks": ["fake:0"],
                  "condition": {"name": "condition", "params": {"k": 1}, "repeat": 0}}
    (path / "experiment.json").write_text(json.dumps(experiment))
    (path / "config.json").write_text("{}\n")
    (path / "events.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    (path / "scheduler.jsonl").write_text(json.dumps({"queue_ms": 1, "model_request_ms": 10, "total_ms": 11}) + "\n")
    (path / "memory.jsonl").write_text(json.dumps({"ts": .1, "kv_tokens": 5,
        "kv_tokens_estimator": "slots.prompt_tokens.v2", "footprint_mb": 100}) + "\n")
    (path / "gpu.jsonl").write_text(json.dumps({"ts": .3, "proc_gpu_mem_mb": 9000, "gpu_util": 50}) + "\n")
    (path / "llama.log").write_text("runtime initialized\n")
    assert all((path / name).stat().st_size for name in REQUIRED)
    return path


def summarize(path):
    return summarize_run("aws_phase2_test", path, {"tasks": ["fake:0"]})


def test_complete_success_remains_valid(tmp_path):
    row = summarize(write_run(tmp_path / "success", success_events()))
    assert row["run_valid"] and row["event_chain_valid"]
    assert row["issues"] == []
    assert row["accuracy"] == row["answered_rate"] == 1


def test_missing_request_sent_cannot_enter_valid_condition_means(tmp_path):
    good = summarize(write_run(tmp_path / "good", success_events()))
    broken = [e for e in success_events() if e["event"] != "request_sent"]
    broken[-1]["correct"] = False
    bad = summarize(write_run(tmp_path / "bad", broken))
    assert bad["task_log_complete"]  # Reproduces the old false-valid case.
    assert not bad["event_chain_valid"] and not bad["run_valid"]
    assert any("missing request_sent" in issue for issue in bad["issues"])
    condition = aggregate([good, bad])[0]
    assert condition["valid_repeats"] == condition["invalid_repeats"] == 1
    assert condition["accuracy"] == 1 and condition["accuracy_n"] == 1


def test_missing_task_boundary_is_invalid_and_unfinished_is_retained(tmp_path):
    row = summarize(write_run(tmp_path / "missing_end", success_events()[:-1]))
    assert not row["task_log_complete"] and not row["event_chain_valid"] and not row["run_valid"]
    assert row["tasks"] == row["unfinished_tasks"] == 1
    assert row["failure_rate"] == 1


def test_timeout_with_inflight_request_and_null_context_remains_valid(tmp_path):
    rows = [event("task_start", .1), event("request_sent", .2, "r", 1),
            event("task_end", 600.1, status="timeout", steps=None, correct=False,
                  final_context_tokens=None)]
    row = summarize(write_run(tmp_path / "timeout", rows, runner_end=601))
    assert row["run_valid"] and row["event_chain_valid"]
    assert row["tasks"] == row["timeout"] == 1
    assert row["accuracy"] == row["answered_rate"] == 0
    assert row["failure_rate"] == 1 and row["unfinished_tasks"] == 0
    assert row["completion_p95_s"] == 600
    assert row["answered_completion_p95_s"] is None and row["ttft_p95_s"] is None
    assert row["cached_frac"] is None


def test_bad_json_retains_run_but_excludes_it_from_valid_means(tmp_path):
    path = write_run(tmp_path / "bad_json", success_events())
    with (path / "events.jsonl").open("a") as stream:
        stream.write("{broken json\n")
    row = summarize(path)
    assert not row["run_valid"] and not row["event_chain_valid"]
    assert row["task_end_count"] == 1
    assert any("invalid JSON" in issue for issue in row["issues"])


def test_null_request_timings_are_invalid_instead_of_crashing_summary(tmp_path):
    events = success_events()
    events[3]["timings"] = None
    row = summarize(write_run(tmp_path / "null_timings", events))
    assert not row["run_valid"] and not row["event_chain_valid"]
    assert row["cached_frac"] is None
    assert any("timings" in issue for issue in row["issues"])


def test_missing_event_file_is_invalid_without_discarding_expected_task_count(tmp_path):
    path = write_run(tmp_path / "missing_events", success_events())
    (path / "events.jsonl").unlink()
    row = summarize(path)
    assert not row["raw_logs_complete"] and not row["event_chain_valid"] and not row["run_valid"]
    assert row["tasks"] == 1 and "events.jsonl" in row["missing_files"]
