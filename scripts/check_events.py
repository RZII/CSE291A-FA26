"""Check a run's events.jsonl is complete, and compute a few metrics from it alone.

    uv run python -m scripts.check_events logs/run_<time>

Checks: every task has task_start .. task_end; every request has request_sent ->
first_token -> request_end (or request_error) in time order; steps are 1..n per task.
"""

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path


def pct(xs: list[float], q: float) -> float:
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(q * len(xs)))]


def main(run_dir: str):
    events = [json.loads(line) for line in open(Path(run_dir) / "events.jsonl")]
    by_req, by_task = defaultdict(dict), defaultdict(list)
    for e in events:
        by_task[e["task_id"]].append(e)
        if "request_id" in e:
            by_req[e["request_id"]][e["event"]] = e

    problems = []
    for tid, evs in by_task.items():
        kinds = [e["event"] for e in evs]
        if kinds[0] != "task_start" or kinds[-1] != "task_end":
            problems.append(f"{tid}: does not start with task_start / end with task_end")
        steps = [e["step"] for e in evs if e["event"] == "request_sent"]
        if steps != list(range(1, len(steps) + 1)):
            problems.append(f"{tid}: request steps {steps}")
    timed_out = {e["task_id"] for e in events if e["event"] == "task_end" and e["status"] == "timeout"}
    for rid, r in by_req.items():
        if "request_error" in r or ("request_end" not in r and r["request_sent"]["task_id"] in timed_out):
            continue
        if not {"request_sent", "first_token", "request_end"} <= r.keys():
            problems.append(f"request {rid}: missing {sorted({'request_sent', 'first_token', 'request_end'} - r.keys())}")
        elif not r["request_sent"]["ts"] <= r["first_token"]["ts"] <= r["request_end"]["ts"]:
            problems.append(f"request {rid}: timestamps out of order")

    ok_reqs = [r for r in by_req.values() if "request_end" in r and "first_token" in r]
    ttft = [(r["first_token"]["ts"] - r["request_sent"]["ts"]) * 1000 for r in ok_reqs]
    starts = {e["task_id"]: e["ts"] for e in events if e["event"] == "task_start"}
    ends = [e for e in events if e["event"] == "task_end"]
    durations = [e["ts"] - starts[e["task_id"]] for e in ends]
    cached = sum(r["request_end"]["timings"]["cache_n"] for r in ok_reqs)
    total = sum(r["request_end"]["timings"]["cache_n"] + r["request_end"]["timings"]["prompt_n"] for r in ok_reqs)

    print(f"{len(events)} events, {len(by_task)} tasks, {len(by_req)} requests, {len(problems)} problems")
    for p in problems:
        print("  PROBLEM:", p)
    print(f"TTFT ms           p50={pct(ttft, .5):.0f}  p95={pct(ttft, .95):.0f}")
    print(f"task duration s   p50={pct(durations, .5):.1f}  p95={pct(durations, .95):.1f}")
    print(f"prompt tokens from cache: {cached}/{total} ({cached / total:.0%})")
    print(f"correct {sum(bool(e.get('correct')) for e in ends)}/{len(ends)}, "
          f"mean final context {statistics.mean(e.get('final_context_tokens', 0) for e in ends):.0f} tokens")


if __name__ == "__main__":
    main(sys.argv[1])
