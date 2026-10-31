import csv
import json

import pytest

from scripts.validation_audit import audit_run, main


def event(kind, ts, request=None, **extra):
    row = {"event": kind, "ts": ts, "run_id": "run", "task_id": "task",
           "agent_id": "agent-0", "step": 1 if request else None, **extra}
    if request:
        row["request_id"] = request
    return row


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def write_run(path, status="answered", scheduler=True, terminal=True):
    path.mkdir()
    (path / "experiment.json").write_text(json.dumps({
        "tasks": ["fake:0"], "t_runner_start": 0, "t_runner_end": 1,
        "runner_exit_code": 0, "finished": 2}))
    rows = [event("task_start", .1), event("request_sent", .2, "r")]
    if terminal:
        rows.extend([event("first_token", .3, "r"), event("request_end", .4, "r",
                     timings={"cache_n": 1, "prompt_n": 2})])
    rows.append(event("task_end", .5, status=status, correct=status == "answered",
                      final_context_tokens=4 if terminal else None))
    write_jsonl(path / "events.jsonl", rows)
    write_jsonl(path / "scheduler.jsonl", [{
        "run_id": "run", "task_id": "task", "agent_id": "agent-0", "step": "1",
        "request_id": "r", "queue_ms": 3, "model_request_ms": 10, "total_ms": 13,
        "http_status": 200, "response_sent": terminal,
        "error": None if terminal else "stream_interrupted: Broken pipe"}] if scheduler else [])
    write_jsonl(path / "memory.jsonl", [{
        "ts": ts, "kv_tokens": 3, "slots_busy": 1, "slot_tokens": [3],
        "rss_mb": 100, "footprint_mb": 90, "kv_tokens_estimator": "slots.n_prompt_tokens.v2"}
        for ts in (-.1, .4, .9, 1.1)])
    write_jsonl(path / "gpu.jsonl", [{
        "ts": ts, "gpu_mem_used_mb": 200, "proc_gpu_mem_mb": 190, "gpu_util": 50}
        for ts in (-.1, .4, .9, 1.1)])
    (path / "trajectories").mkdir()
    (path / "trajectories" / "task.json").write_text(json.dumps({
        "task_id": "task", "agent_id": "agent-0", "status": status,
        "grade": {"correct": status == "answered"}}))
    return path


def snapshot(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob("*") if p.is_file()}


def test_success_outputs_join_and_never_changes_raw_files(tmp_path):
    path = write_run(tmp_path / "run")
    before = snapshot(path)
    out = tmp_path / "audit"
    assert main([str(path), "--out", str(out)]) == 0
    assert snapshot(path) == before
    result = json.loads((out / "audit.json").read_text())
    row = result["runs"][0]["requests"][0]
    assert result["problem_count"] == 0
    assert result["totals"]["request_sent"] == result["totals"]["matched_requests"] == 1
    assert row["timing_complete"] and not row["censored"]
    assert row["request_elapsed_s"] == pytest.approx(.2)
    assert row["ttft_s"] == pytest.approx(.1)
    assert row["queue_ms"] == row["observed_scheduler_queue_ms"] == 3
    assert result["runs"][0]["memory"]["covers_runner"]
    with (out / "requests.csv").open() as stream:
        assert next(csv.DictReader(stream))["request_id"] == "r"


@pytest.mark.parametrize("status", ["timeout", "crash", "cancelled", "canceled"])
def test_inflight_cancellation_without_scheduler_is_explicitly_censored(tmp_path, status):
    result = audit_run(write_run(tmp_path / "run", status, scheduler=False, terminal=False))
    assert result["problem_count"] == 0
    assert result["counts"]["missing_scheduler"] == result["counts"]["missing_request_terminal"] == 1
    row = result["requests"][0]
    assert row["censored"] and not row["scheduler_present"] and not row["timing_complete"]
    assert "scheduler_terminal_row_unobserved" in row["censored_reason"]
    assert all(row[key] is None for key in ("request_elapsed_s", "ttft_s", "queue_ms", "total_ms"))


def test_cancelled_request_retains_observed_scheduler_timing_without_treating_it_as_complete(tmp_path):
    result = audit_run(write_run(tmp_path / "run", "timeout", terminal=False))
    assert result["problem_count"] == 0
    row = result["requests"][0]
    assert row["censored"] and row["scheduler_present"]
    assert row["model_request_ms"] is None and row["observed_scheduler_model_request_ms"] == 10
    assert row["scheduler_response_sent"] is False
    assert result["counts"]["missing_scheduler"] == 0


@pytest.mark.parametrize("status,terminal", [("answered", False), ("timeout", True)])
def test_missing_scheduler_cannot_be_excused_by_an_answer_or_a_previously_finished_timeout_request(
        tmp_path, status, terminal):
    result = audit_run(write_run(tmp_path / "run", status, scheduler=False, terminal=terminal))
    assert result["problem_count"] > 0
    row = result["requests"][0]
    assert not row["censored"] and not row["timing_complete"]
    assert row["request_elapsed_s"] is None
    assert any("missing scheduler row" in problem for problem in result["problems"])


def test_duplicate_and_identity_mismatch_are_problems_not_silently_joined(tmp_path):
    path = write_run(tmp_path / "run")
    rows = [json.loads(line) for line in (path / "scheduler.jsonl").read_text().splitlines()]
    rows[0]["agent_id"] = "another-agent"
    write_jsonl(path / "scheduler.jsonl", rows * 2)
    result = audit_run(path)
    assert result["counts"]["duplicate_scheduler_rows"] == 1
    assert any("duplicate scheduler" in problem for problem in result["problems"])
    assert any("agent_id identity mismatch" in problem for problem in result["problems"])
    assert result["requests"][0]["queue_ms"] is None


def test_trajectories_must_match_terminal_tasks_and_grades(tmp_path):
    path = write_run(tmp_path / "run")
    (path / "trajectories" / "task.json").unlink()
    (path / "trajectories" / "other.json").write_text(json.dumps({
        "task_id": "other", "status": "answered", "agent_id": "agent-0"}))
    result = audit_run(path)
    assert any("expected one trajectory, got 0" in problem for problem in result["problems"])
    assert any("no unique task_end" in problem for problem in result["problems"])
    assert result["counts"]["trajectories"] == 1  # Equal file count alone is insufficient.


def test_changed_trajectory_grade_is_not_accepted_as_complete_evidence(tmp_path):
    path = write_run(tmp_path / "run")
    trajectory = path / "trajectories" / "task.json"
    row = json.loads(trajectory.read_text())
    row["grade"]["correct"] = False
    trajectory.write_text(json.dumps(row))
    assert any("correct differs" in issue for issue in audit_run(path)["problems"])


def test_kv_nulls_remain_missing_and_valid_sample_gaps_are_reported(tmp_path):
    path = write_run(tmp_path / "run")
    rows = [json.loads(line) for line in (path / "memory.jsonl").read_text().splitlines()]
    rows[1].update(kv_tokens=None, slots_busy=None, slot_tokens=None)
    write_jsonl(path / "memory.jsonl", rows)
    result = audit_run(path)
    assert result["problem_count"] == 0
    metric = result["memory"]["metrics"]["kv_tokens"]
    assert metric["null_samples"] == result["counts"]["kv_null_samples"] == 1
    assert metric["valid_samples"] == 3
    assert metric["max_interval_s"] == 1
    assert result["memory"]["max_interval_s"] == .5
    assert metric["sampled_peak"] == 3


def test_missing_coverage_or_bad_json_is_a_problem_instead_of_a_crash(tmp_path):
    path = write_run(tmp_path / "run")
    write_jsonl(path / "gpu.jsonl", [{"ts": .5, "gpu_mem_used_mb": 200}])
    with (path / "scheduler.jsonl").open("a") as stream:
        stream.write("{bad json\n")
    result = audit_run(path)
    assert any("gpu: samples do not cover" in problem for problem in result["problems"])
    assert any("scheduler.jsonl:2:" in problem for problem in result["problems"])


def test_cli_reports_problem_exit_and_refuses_outputs_inside_raw_run(tmp_path):
    path = write_run(tmp_path / "run", scheduler=False)
    assert main([str(path), "--out", str(tmp_path / "audit")]) == 1
    with pytest.raises(SystemExit):
        main([str(path), "--out", str(path / "audit")])
    assert not (path / "audit").exists()


def test_cross_run_request_id_reuse_is_reported(tmp_path):
    write_run(tmp_path / "run-a")
    write_run(tmp_path / "run-b")
    out = tmp_path / "audit"
    assert main([str(tmp_path), "--out", str(out)]) == 1
    result = json.loads((out / "audit.json").read_text())
    assert all(any("request_id reused" in problem for problem in run["problems"])
               for run in result["runs"])


def test_malformed_task_status_is_reported_even_if_event_checker_cannot_process_it(tmp_path):
    path = write_run(tmp_path / "run", "timeout", scheduler=False, terminal=False)
    rows = [json.loads(line) for line in (path / "events.jsonl").read_text().splitlines()]
    rows[-1]["status"] = ["invalid", "type"]
    write_jsonl(path / "events.jsonl", rows)
    result = audit_run(path)
    assert result["problem_count"] > 0
    assert not result["event_chain_valid"]
    assert not result["requests"][0]["censored"]
