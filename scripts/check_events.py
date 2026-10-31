"""Check a run's events.jsonl is complete, and compute a few metrics from it alone.

    uv run python -m scripts.check_events logs/run_<time>

Checks: every task has task_start .. task_end; every request has request_sent ->
first_token -> request_end (or request_error) in time order; steps are 1..n per task.
An in-flight request may instead be cancelled by a terminal timeout/crash/cancelled
task. These task outcomes remain in the counts and duration statistics. Missing
observations are reported as N/A rather than filled with zero.
"""

import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def pct(xs: list[float], q: float) -> float | None:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))] if xs else None


def numeric(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def percentiles(values, precision):
    if not values:
        return "p50=N/A  p95=N/A"
    return f"p50={pct(values, .5):.{precision}f}  p95={pct(values, .95):.{precision}f}"


def main(run_dir: str):
    problems = []
    events = []
    try:
        lines = (Path(run_dir) / "events.jsonl").read_text().splitlines()
    except OSError as error:
        print("  PROBLEM:", str(error))
        return 1
    request_events = {"request_sent", "first_token", "request_end", "request_error"}
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except ValueError as error:
            problems.append(f"line {number}: invalid JSON ({error})")
            continue
        if (not isinstance(event, dict) or not isinstance(event.get("task_id"), str)
                or not event["task_id"] or not isinstance(event.get("event"), str)
                or not numeric(event.get("ts"))):
            problems.append(f"line {number}: missing or invalid task_id, event or ts")
            continue
        if event["event"] in request_events and (not isinstance(event.get("request_id"), str)
                                                   or not event["request_id"]):
            problems.append(f"line {number}: request event missing request_id")
        events.append(event)
    if not events:
        problems.append("no usable events")
    by_req, by_task = defaultdict(dict), defaultdict(list)
    for e in events:
        by_task[e["task_id"]].append(e)
        if isinstance(e.get("request_id"), str) and e["request_id"]:
            if e["event"] in by_req[e["request_id"]]:
                problems.append(f"request {e['request_id']}: duplicate {e['event']}")
            by_req[e["request_id"]][e["event"]] = e

    for tid, evs in by_task.items():
        kinds = [e["event"] for e in evs]
        if (kinds[0] != "task_start" or kinds[-1] != "task_end"
                or kinds.count("task_start") != 1 or kinds.count("task_end") != 1):
            problems.append(f"{tid}: does not start with task_start / end with task_end")
        steps = [e.get("step") for e in evs if e["event"] == "request_sent"]
        if not all(type(step) is int for step in steps) or steps != list(range(1, len(steps) + 1)):
            problems.append(f"{tid}: request steps {steps}")
        stamps = [e["ts"] for e in evs]
        if stamps != sorted(stamps):
            problems.append(f"{tid}: task event timestamps out of order")
        for end in (e for e in evs if e["event"] == "task_end"):
            if not isinstance(end.get("status"), str) or not end["status"]:
                problems.append(f"{tid}: task_end missing status")

    task_ends = {e["task_id"]: e for e in events if e["event"] == "task_end"}
    cancelled_statuses = {"timeout", "crash", "cancelled", "canceled"}
    ok_reqs, request_errors, cancelled_requests = [], 0, 0
    for rid, r in by_req.items():
        if "request_sent" not in r:
            problems.append(f"request {rid}: missing request_sent")
            continue
        sent = r["request_sent"]
        if any(e["task_id"] != sent["task_id"] for e in r.values()):
            problems.append(f"request {rid}: task_id changes across request events")
            continue
        terminal = task_ends.get(sent["task_id"], {})
        if "request_error" in r:
            request_errors += 1
            if "request_end" in r:
                problems.append(f"request {rid}: has both request_error and request_end")
            stamps = [sent["ts"]]
            if "first_token" in r:
                stamps.append(r["first_token"]["ts"])
            stamps.append(r["request_error"]["ts"])
            if stamps != sorted(stamps):
                problems.append(f"request {rid}: timestamps out of order")
            continue
        if "request_end" not in r and terminal.get("status") in cancelled_statuses:
            cancelled_requests += 1
            stamps = [sent["ts"]]
            if "first_token" in r:
                stamps.append(r["first_token"]["ts"])
            stamps.append(terminal["ts"])
            if stamps != sorted(stamps):
                problems.append(f"request {rid}: cancellation timestamps out of order")
            continue
        required = {"request_sent", "first_token", "request_end"}
        if not required <= r.keys():
            problems.append(f"request {rid}: missing {sorted(required - r.keys())}")
            continue
        if not sent["ts"] <= r["first_token"]["ts"] <= r["request_end"]["ts"]:
            problems.append(f"request {rid}: timestamps out of order")
            continue
        timings = r["request_end"].get("timings")
        if not isinstance(timings, dict) or not all(
                numeric(timings.get(key)) and timings[key] >= 0 for key in ("cache_n", "prompt_n")):
            problems.append(f"request {rid}: missing or invalid cache_n / prompt_n timings")
            continue
        ok_reqs.append(r)

    ttft = [(r["first_token"]["ts"] - r["request_sent"]["ts"]) * 1000 for r in ok_reqs]
    starts = {e["task_id"]: e["ts"] for e in events if e["event"] == "task_start"}
    ends = [e for e in events if e["event"] == "task_end"]
    durations = []
    for end in ends:
        start = starts.get(end["task_id"])
        if start is not None and end["ts"] >= start:
            durations.append(end["ts"] - start)
        elif start is not None:
            problems.append(f"{end['task_id']}: task_end precedes task_start")
    cached = sum(r["request_end"]["timings"]["cache_n"] for r in ok_reqs)
    total = sum(r["request_end"]["timings"]["cache_n"] + r["request_end"]["timings"]["prompt_n"] for r in ok_reqs)
    contexts = [e["final_context_tokens"] for e in ends if numeric(e.get("final_context_tokens"))]

    print(f"{len(events)} events, {len(by_task)} tasks, {len(by_req)} requests, {len(problems)} problems")
    for p in problems:
        print("  PROBLEM:", p)
    print(f"TTFT ms           {percentiles(ttft, 0)}")
    print(f"task duration s   {percentiles(durations, 1)}")
    ratio = f"{cached / total:.0%}" if total else "N/A"
    print(f"prompt tokens from cache: {cached}/{total} ({ratio})")
    print(f"requests completed={len(ok_reqs)} error={request_errors} cancelled={cancelled_requests}")
    print("task statuses:", dict(Counter(e["status"] if isinstance(e.get("status"), str) else "unknown"
                                        for e in ends)))
    mean_context = f"{statistics.mean(contexts):.0f}" if contexts else "N/A"
    print(f"correct {sum(bool(e.get('correct')) for e in ends)}/{len(ends)}, "
          f"mean final context {mean_context} tokens ({len(contexts)}/{len(ends)} observed)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
