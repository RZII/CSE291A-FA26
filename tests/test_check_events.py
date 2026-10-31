import json
import subprocess
import sys

import pytest

from scripts.check_events import main, pct


def event(event_type, ts, task="task", request=None, step=None, **fields):
    row = {"event": event_type, "ts": ts, "run_id": "test", "agent_id": "agent-0",
           "task_id": task, "step": step, **fields}
    if request is not None:
        row["request_id"] = request
    return row


def write_events(path, rows):
    (path / "events.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_timeout_with_no_completed_stream_is_valid_and_retained(tmp_path, capsys):
    write_events(tmp_path, [event("task_start", 0), event("request_sent", .1, request="r", step=1),
                           event("task_end", 600, status="timeout", steps=None, correct=False,
                                 final_context_tokens=None, error="task exceeded 600s")])
    assert main(str(tmp_path)) == 0
    output = capsys.readouterr().out
    assert "0 problems" in output
    assert "p50=N/A  p95=N/A" in output
    assert "p50=600.0  p95=600.0" in output
    assert "prompt tokens from cache: 0/0 (N/A)" in output
    assert "completed=0 error=0 cancelled=1" in output
    assert "'timeout': 1" in output
    assert "correct 0/1" in output
    assert "mean final context N/A tokens (0/1 observed)" in output


def test_timeout_after_a_completed_request_keeps_available_metrics(tmp_path, capsys):
    write_events(tmp_path, [
        event("task_start", 0), event("request_sent", .1, request="r1", step=1),
        event("first_token", .2, request="r1", step=1),
        event("request_end", .3, request="r1", step=1, timings={"cache_n": 3, "prompt_n": 7}),
        event("request_sent", .4, request="r2", step=2),
        event("task_end", 600, status="timeout", steps=None, correct=False, final_context_tokens=None),
    ])
    assert main(str(tmp_path)) == 0
    output = capsys.readouterr().out
    assert "completed=1 error=0 cancelled=1" in output
    assert "prompt tokens from cache: 3/10 (30%)" in output
    assert "correct 0/1" in output


@pytest.mark.parametrize("status", ["crash", "cancelled", "canceled"])
def test_terminal_cancellation_can_leave_an_inflight_request(tmp_path, capsys, status):
    write_events(tmp_path, [event("task_start", 0), event("request_sent", .1, request="r", step=1),
                           event("task_end", .2, status=status, error="cancelled request")])
    assert main(str(tmp_path)) == 0
    assert "cancelled=1" in capsys.readouterr().out


def test_explicit_request_error_without_successful_stream_has_na_metrics(tmp_path, capsys):
    write_events(tmp_path, [event("task_start", 0), event("request_sent", .1, request="r", step=1),
                           event("request_error", .2, request="r", step=1, kind="context_overflow"),
                           event("task_end", .3, status="error", correct=False, final_context_tokens=0)])
    assert main(str(tmp_path)) == 0
    output = capsys.readouterr().out
    assert "completed=0 error=1 cancelled=0" in output
    assert "prompt tokens from cache: 0/0 (N/A)" in output


@pytest.mark.parametrize("kind", ["request_end", "request_error"])
def test_orphan_request_event_reports_missing_sent_instead_of_keyerror(tmp_path, capsys, kind):
    write_events(tmp_path, [event("task_start", 0), event(kind, .1, request="orphan", step=1),
                           event("task_end", .2, status="timeout", correct=False, final_context_tokens=None)])
    assert main(str(tmp_path)) == 1
    assert "PROBLEM: request orphan: missing request_sent" in capsys.readouterr().out


@pytest.mark.parametrize("rows", [
    [event("task_end", 1, status="crash")],
    [event("task_start", 0), event("request_sent", .1, request="r", step=1)],
    [event("task_start", 0), event("request_sent", .1, request="r", step=1),
     event("task_end", .2, status="answered", correct=True)],
    [event("task_start", 2), event("task_end", 1, status="timeout")],
    [event("task_start", 0), event("task_start", .1), event("task_end", .2, status="timeout")],
    [event("task_start", 0), event("task_end", .1, status=[])],
])
def test_malformed_task_or_request_boundaries_fail_without_crashing(tmp_path, capsys, rows):
    write_events(tmp_path, rows)
    assert main(str(tmp_path)) == 1
    assert "PROBLEM:" in capsys.readouterr().out


def test_empty_log_and_invalid_json_report_problems_with_na_metrics(tmp_path, capsys):
    (tmp_path / "events.jsonl").write_text("{bad json\n")
    assert main(str(tmp_path)) == 1
    output = capsys.readouterr().out
    assert "PROBLEM:" in output
    assert "p50=N/A" in output
    assert pct([], .5) is None


def test_context_mean_uses_only_observed_finite_numbers(tmp_path, capsys):
    rows = []
    for i, context in enumerate([None, "unknown", True, 12, 18]):
        rows += [event("task_start", 0, task=str(i)),
                 event("task_end", 1, task=str(i), status="crash", final_context_tokens=context)]
    write_events(tmp_path, rows)
    assert main(str(tmp_path)) == 0
    assert "mean final context 15 tokens (2/5 observed)" in capsys.readouterr().out


def test_cli_returns_nonzero_for_structural_problems(tmp_path):
    write_events(tmp_path, [event("task_end", 1, status="crash")])
    result = subprocess.run([sys.executable, "-m", "scripts.check_events", str(tmp_path)],
                            capture_output=True, text=True)
    assert result.returncode == 1
    assert "PROBLEM:" in result.stdout
    assert "Traceback" not in result.stderr
