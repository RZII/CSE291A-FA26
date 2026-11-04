"""Read-only joins and sampling audit for downloaded validation runs.

    python -m scripts.validation_audit logs/group_a logs/group_b --out audit

Outputs never replace raw logs. Cancelled in-flight requests remain in both the
JSON and CSV; their comparable durations are null, with observed timestamps and
scheduler timings retained separately. This audits evidence, not task quality or
physical KV allocation. A nonzero exit means an unexplained integrity problem.
"""

import argparse
import contextlib
import csv
import hashlib
import io
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scripts import check_events
from scripts.validation_summary import task_count


REQUEST_EVENTS = ("request_sent", "first_token", "request_end", "request_error")
IDENTITIES = ("run_id", "task_id", "agent_id", "step")
CANCELLED = {"timeout", "crash", "cancelled", "canceled"}
CSV_FIELDS = (
    "run_path", "request_id", *IDENTITIES, "task_status", "task_end_ts",
    "sent_present", "first_token_present", "request_end_present",
    "request_error_present", "scheduler_present", "sent_count",
    "first_token_count", "request_end_count", "request_error_count",
    "scheduler_count", "sent_ts", "first_token_ts", "request_end_ts",
    "request_error_ts", "request_error", "scheduler_http_status",
    "scheduler_response_sent", "scheduler_error", "censored", "censored_reason",
    "timing_complete", "request_elapsed_s", "ttft_s", "queue_ms",
    "model_request_ms", "total_ms", "observed_scheduler_queue_ms",
    "observed_scheduler_model_request_ms", "observed_scheduler_total_ms",
    "problem_count", "problems",
)


def number(value):
    return value if check_events.numeric(value) else None


def read_json(path, problems, hashes):
    try:
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError, UnicodeError) as error:
        problems.append(f"{path.name}: {error}")
        return {}


def read_jsonl(path, problems, hashes):
    try:
        raw = path.read_bytes()
        hashes[str(path)] = hashlib.sha256(raw).hexdigest()
        lines = raw.decode().splitlines()
    except (OSError, UnicodeError) as error:
        problems.append(f"{path.name}: {error}")
        return []
    rows = []
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("expected a JSON object")
            rows.append(row)
        except ValueError as error:
            problems.append(f"{path.name}:{line_number}: {error}")
    return rows


def one(rows):
    return rows[0] if len(rows) == 1 else {}


def samples_audit(rows, start, end, metrics, label, problems):
    stamps = [number(row.get("ts")) for row in rows]
    valid = [ts for ts in stamps if ts is not None]
    if not rows:
        problems.append(f"{label}: no samples")
    if len(valid) != len(rows):
        problems.append(f"{label}: invalid sample timestamps")
    if any(b <= a for a, b in zip(valid, valid[1:])):
        problems.append(f"{label}: non-increasing sample timestamps")

    def coverage(times):
        before = start - min(times) if times and start is not None else None
        after = max(times) - end if times and end is not None else None
        return {"first_ts": min(times) if times else None,
                "last_ts": max(times) if times else None,
                "before_runner_s": before, "after_runner_s": after,
                "covers_runner": before is not None and after is not None
                and before >= 0 and after >= 0,
                "max_interval_s": max((b - a for a, b in zip(times, times[1:])), default=None)}

    result = {"samples": len(rows), "samples_within_runner": sum(
        start <= ts <= end for ts in valid) if start is not None and end is not None else None,
        **coverage(valid), "metrics": {}}
    if start is not None and end is not None and not result["covers_runner"]:
        problems.append(f"{label}: samples do not cover the runner interval")
    for key in metrics:
        values = [(row.get("ts"), number(row.get(key))) for row in rows]
        metric_stamps = [ts for ts, value in values if number(ts) is not None and value is not None]
        invalid = sum(row.get(key) is not None and number(row.get(key)) is None for row in rows)
        if invalid:
            problems.append(f"{label}: {key} has {invalid} invalid non-null values")
        result["metrics"][key] = {
            "valid_samples": len(metric_stamps),
            "null_samples": sum(row.get(key) is None for row in rows),
            "invalid_samples": invalid,
            "sampled_peak": max((value for _, value in values if value is not None), default=None),
            **coverage(metric_stamps),
        }
        if key == metrics[0] and not metric_stamps:
            problems.append(f"{label}: no numeric {key} observations")
    if label == "memory":
        result["slot_fields_null_samples"] = {
            key: sum(row.get(key) is None for row in rows)
            for key in ("slots_busy", "kv_tokens", "slot_tokens")}
        result["kv_estimators"] = dict(Counter(str(row.get("kv_tokens_estimator")) for row in rows))
    return result


def audit_run(run):
    run = Path(run).resolve()
    problems, hashes = [], {}
    meta = read_json(run / "experiment.json", problems, hashes)
    events = read_jsonl(run / "events.jsonl", problems, hashes)
    scheduler = read_jsonl(run / "scheduler.jsonl", problems, hashes)
    with contextlib.redirect_stdout(io.StringIO()) as captured:
        try:
            event_code = check_events.main(str(run))
        except Exception as error:
            event_code = 1
            problems.append(f"events: checker could not validate malformed data: {type(error).__name__}: {error}")
    event_problems = [line.strip() for line in captured.getvalue().splitlines() if "PROBLEM:" in line]
    problems.extend(f"events: {message}" for message in event_problems)
    if event_code and not event_problems:
        problems.append("events: checker returned nonzero")

    starts, ends = defaultdict(list), defaultdict(list)
    by_request = defaultdict(lambda: defaultdict(list))
    for row in events:
        kind, task_id = row.get("event"), row.get("task_id")
        if kind in ("task_start", "task_end") and isinstance(task_id, str):
            (starts if kind == "task_start" else ends)[task_id].append(row)
        if kind in REQUEST_EVENTS:
            rid = row.get("request_id")
            if isinstance(rid, str) and rid:
                by_request[rid][kind].append(row)
    for row in scheduler:
        rid = row.get("request_id")
        if not isinstance(rid, str) or not rid:
            problems.append("scheduler.jsonl: row missing request_id")
        else:
            by_request[rid]["scheduler"].append(row)

    requests = []
    for rid, grouped in sorted(by_request.items()):
        prefix = f"request {rid}: "
        issues = [message.split(prefix, 1)[1] for message in event_problems if prefix in message]
        for kind, records in grouped.items():
            if len(records) > 1:
                issues.append(f"duplicate {kind} ({len(records)})")
        sent, first, done, error, sched = (one(grouped[key]) for key in (*REQUEST_EVENTS, "scheduler"))
        reference = sent or next((row for records in grouped.values() for row in records), {})
        identities = {key: reference.get(key) for key in IDENTITIES}
        for key in ("run_id", "task_id", "agent_id"):
            if not isinstance(identities[key], str) or not identities[key]:
                issues.append(f"missing or invalid {key}")
        if sent and (type(identities["step"]) is not int or identities["step"] < 1):
            issues.append("request_sent: step must be a positive integer")
        condition = meta.get("condition")
        expected_run = condition.get("run") if isinstance(condition, dict) else None
        if expected_run and identities["run_id"] != expected_run:
            issues.append("run_id differs from experiment condition.run")
        for kind, records in grouped.items():
            for row in records:
                for key in IDENTITIES:
                    # Scheduler step is serialized as text; agent steps are integers.
                    if str(row.get(key)) != str(identities[key]):
                        issues.append(f"{kind}: {key} identity mismatch")
        if not grouped["request_sent"]:
            issues.append("missing request_sent")
        task_id = identities["task_id"] if isinstance(identities["task_id"], str) else None
        task = one(ends.get(task_id, []))
        status = task.get("status") if isinstance(task.get("status"), str) else None
        if not task:
            issues.append("no unique terminal task_end")
        for key in ("run_id", "agent_id"):
            if task and str(task.get(key)) != str(identities[key]):
                issues.append(f"task_end: {key} identity mismatch")
        terminal_count = len(grouped["request_end"]) + len(grouped["request_error"])
        pending_cancel = bool(sent and terminal_count == 0 and status in CANCELLED)
        reasons = []
        if terminal_count == 0:
            if pending_cancel:
                reasons.append(f"task_{status}_without_request_terminal")
            else:
                issues.append("missing request terminal without terminal task cancellation")
        if not grouped["scheduler"]:
            if pending_cancel:
                reasons.append("scheduler_terminal_row_unobserved_after_cancellation")
            else:
                issues.append("missing scheduler row without an in-flight task cancellation")
        issues = list(dict.fromkeys(issues))
        timing_complete = bool(sent and terminal_count == 1 and sched and not issues)
        terminal_ts = number((done or error).get("ts"))
        sent_ts, first_ts = number(sent.get("ts")), number(first.get("ts"))
        elapsed = (terminal_ts - sent_ts if timing_complete and terminal_ts is not None
                   and sent_ts is not None else None)
        record = {"run_path": str(run), "request_id": rid, **identities,
                  "task_status": status, "task_end_ts": number(task.get("ts")),
                  "sent_present": bool(grouped["request_sent"]),
                  "first_token_present": bool(grouped["first_token"]),
                  "request_end_present": bool(grouped["request_end"]),
                  "request_error_present": bool(grouped["request_error"]),
                  "scheduler_present": bool(grouped["scheduler"]),
                  "sent_count": len(grouped["request_sent"]),
                  "first_token_count": len(grouped["first_token"]),
                  "request_end_count": len(grouped["request_end"]),
                  "request_error_count": len(grouped["request_error"]),
                  "scheduler_count": len(grouped["scheduler"]),
                  "sent_ts": sent_ts, "first_token_ts": first_ts,
                  "request_end_ts": number(done.get("ts")),
                  "request_error_ts": number(error.get("ts")),
                  "request_error": error.get("error"),
                  "scheduler_http_status": sched.get("http_status"),
                  "scheduler_response_sent": sched.get("response_sent"),
                  "scheduler_error": sched.get("error"),
                  "censored": pending_cancel, "censored_reason": "; ".join(reasons) or None,
                  "timing_complete": timing_complete, "request_elapsed_s": elapsed,
                  "ttft_s": first_ts - sent_ts if timing_complete and first_ts is not None
                  and sent_ts is not None else None,
                  "problem_count": len(issues), "problems": issues}
        for key in ("queue_ms", "model_request_ms", "total_ms"):
            record["observed_scheduler_" + key] = number(sched.get(key))
            record[key] = number(sched.get(key)) if timing_complete else None
        requests.append(record)
        problems.extend(f"request {rid}: {issue}" for issue in issues)

    trajectory_paths = sorted((run / "trajectories").glob("*.json"))
    trajectories = defaultdict(list)
    if not (run / "trajectories").is_dir():
        problems.append("missing trajectories directory")
    for path in trajectory_paths:
        row = read_json(path, problems, hashes)
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            problems.append(f"trajectories/{path.name}: missing task_id")
            continue
        trajectories[task_id].append(row)
        if path.stem != task_id:
            problems.append(f"trajectories/{path.name}: filename differs from task_id")
        task = one(ends.get(task_id, []))
        if not task:
            problems.append(f"trajectory {task_id}: no unique task_end")
        else:
            for key in ("status", "agent_id"):
                if row.get(key) != task.get(key):
                    problems.append(f"trajectory {task_id}: {key} differs from task_end")
            grade = row.get("grade")
            if "correct" in task:
                if not isinstance(grade, dict) or "correct" not in grade:
                    problems.append(f"trajectory {task_id}: missing grade.correct")
                elif grade["correct"] != task["correct"]:
                    problems.append(f"trajectory {task_id}: correct differs from task_end")
    for task_id in ends:
        if len(trajectories[task_id]) != 1:
            problems.append(f"task {task_id}: expected one trajectory, got {len(trajectories[task_id])}")
    for task_id, rows in trajectories.items():
        if len(rows) > 1:
            problems.append(f"trajectory {task_id}: duplicate task_id")

    specs = meta.get("tasks")
    if specs is not None and not isinstance(specs, list):
        problems.append("experiment.json: tasks is not a list")
    expected = task_count(specs) if isinstance(specs, list) else None
    if expected is not None and (sum(map(len, starts.values())) != expected
                                 or sum(map(len, ends.values())) != expected):
        problems.append(f"expected {expected} task starts/ends from experiment task selection")
    start, end = number(meta.get("t_runner_start")), number(meta.get("t_runner_end"))
    if start is None or end is None or end < start:
        problems.append("experiment.json: missing or invalid runner interval")
        start = end = None
    memory = samples_audit(read_jsonl(run / "memory.jsonl", problems, hashes), start, end,
                           ("kv_tokens", "rss_mb", "footprint_mb"), "memory", problems)
    gpu = samples_audit(read_jsonl(run / "gpu.jsonl", problems, hashes), start, end,
                        ("gpu_mem_used_mb", "proc_gpu_mem_mb", "gpu_util"), "gpu", problems)
    counts = {"task_start": sum(map(len, starts.values())), "task_end": sum(map(len, ends.values())),
              "unique_task_start": len(starts), "unique_task_end": len(ends),
              "trajectories": len(trajectory_paths), "expected_tasks": expected,
              "request_sent": sum(r["sent_count"] for r in requests),
              "scheduler_rows": len(scheduler), "unique_requests": len(requests),
              "matched_requests": sum(r["sent_present"] and r["scheduler_present"] for r in requests),
              "missing_scheduler": sum(r["sent_present"] and not r["scheduler_present"] for r in requests),
              "scheduler_without_sent": sum(r["scheduler_present"] and not r["sent_present"] for r in requests),
              "missing_request_terminal": sum(r["sent_present"] and not r["request_end_present"]
                                              and not r["request_error_present"] for r in requests),
              "censored_requests": sum(r["censored"] for r in requests),
              "duplicate_request_sent": sum(max(r["sent_count"] - 1, 0) for r in requests),
              "duplicate_scheduler_rows": sum(max(r["scheduler_count"] - 1, 0) for r in requests),
              "kv_null_samples": memory["metrics"]["kv_tokens"]["null_samples"]}
    return {"run_path": str(run), "counts": counts,
            "task_statuses": dict(Counter(str(row.get("status")) for rows in ends.values() for row in rows)),
            "runner_start": start, "runner_end": end, "runner_exit_code": meta.get("runner_exit_code"),
            "event_chain_valid": event_code == 0, "memory": memory, "gpu": gpu,
            "problem_count": len(problems), "problems": problems,
            "input_sha256": hashes, "requests": requests}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("roots", nargs="+", type=Path, help="run or recursively searched log group directories")
    parser.add_argument("--out", required=True, type=Path, help="separate audit output directory")
    args = parser.parse_args(argv)
    paths = sorted({p.parent.resolve() for root in args.roots
                    for p in ([root / "experiment.json"] if (root / "experiment.json").is_file()
                              else root.rglob("experiment.json"))})
    if not paths:
        parser.error("no experiment.json runs found")
    out = args.out.resolve()
    if any(out == run or run in out.parents for run in paths):
        parser.error("--out must be outside every raw run directory")
    runs = [audit_run(path) for path in paths]
    request_runs = defaultdict(list)
    for run in runs:
        for request in run["requests"]:
            request_runs[request["request_id"]].append(run)
    for rid, owners in request_runs.items():
        if len(owners) > 1:
            for run in owners:
                run["problems"].append(f"request {rid}: request_id reused across {len(owners)} runs")
                run["problem_count"] += 1
    totals = {key: sum(run["counts"][key] for run in runs)
              for key in runs[0]["counts"] if key != "expected_tasks"}
    result = {"schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
              "sources": [str(root.resolve()) for root in args.roots],
              "semantics": {
                  "csv_null": "empty field; booleans are True/False",
                  "censored": "in-flight request lacks request_end/error at a timeout/crash/cancelled task end",
                  "durations": "comparable durations are null for censored, missing or ambiguous joins; observed scheduler fields are retained",
                  "samples": "sampled peaks only; null samples stay missing; row timestamps precede sequential API reads",
                  "kv": "slot-token estimate, not physical allocation or exact peak"},
              "run_count": len(runs), "problem_count": sum(run["problem_count"] for run in runs),
              "totals": totals, "runs": runs}
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    with (out / "requests.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for run in runs:
            for request in run["requests"]:
                writer.writerow({**request, "problems": "; ".join(request["problems"])})
    print(json.dumps({"runs": len(runs), "problems": result["problem_count"],
                      "totals": totals, "out": str(out)}, ensure_ascii=False))
    return 1 if result["problem_count"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
