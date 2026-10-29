"""One row per task, built from existing logs (the per-task table discussed with 4号).

    uv run python -m scripts.task_summary logs/run_<time> --scheduler-k 2
Writes logs/run_<time>/task_summary.jsonl and prints a short per-type summary.

Where each field comes from:
    run_id, task_id, agent_id, task_type   our events.jsonl (task_start)
    num_agents, tool_delay_s               run_dir/config.json (tool_delay_s = the delay of the
                                           tool this task type uses: python or search)
    scheduler_k                            --scheduler-k (the scheduler does not log K); null for
                                           runs that went straight to llama.cpp
    status, is_success, total_steps,       our task_end event
    final_context_tokens
    total_completion_ms                    task_end.ts - task_start.ts (measured, not a sum)
    queue_wait_ms, model_request_ms        sum over this task's requests of the scheduler log's
                                           queue_ms / model_request_ms, joined on request_id;
                                           null when the run did not go through the scheduler
    llama_compute_ms                       sum of llama.cpp timings prompt_ms + predicted_ms
    tool_wait_ms                           sum of (tool_end.ts - tool_start.ts), extra_delay included
    first_ttft_ms                          step 1: first_token.ts - request_sent.ts
    recompute_tokens                       sum over steps >= 2 of
                                           (previous step's cache_n + prompt_n + predicted_n) - cache_n
                                           (1 per step is normal: the final <|im_end|> is never cached)
peak_memory_mb is a whole-run number (3号's memory sampler), so it is not in the per-task rows.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

DELAY_FIELD = {"gsm8k": "python_delay_s", "hotpotqa": "search_delay_s", "longqa": "search_delay_s"}


def summarize(events: list[dict], scheduler_rows: list[dict], config: dict, scheduler_k: int | None) -> list[dict]:
    by_task = defaultdict(list)
    for e in events:
        by_task[e["task_id"]].append(e)
    sched = {r["request_id"]: r for r in scheduler_rows}
    via_scheduler = any(e.get("request_id") in sched for e in events)

    rows = []
    for task_id, evs in by_task.items():
        start = next(e for e in evs if e["event"] == "task_start")
        end = next((e for e in evs if e["event"] == "task_end"), None)
        if end is None:
            continue  # task never finished (run killed); nothing reliable to report
        requests = defaultdict(dict)  # request_id -> {event: e}
        for e in evs:
            if "request_id" in e:
                requests[e["request_id"]][e["event"]] = e
        reqs = sorted(requests.values(), key=lambda r: r["request_sent"]["step"])
        timings = [r["request_end"]["timings"] for r in reqs if "request_end" in r]

        tool_starts = {e["call_id"]: e["ts"] for e in evs if e["event"] == "tool_start"}
        tool_wait = sum(e["ts"] - tool_starts[e["call_id"]] for e in evs if e["event"] == "tool_end")

        first = reqs[0] if reqs else {}
        ttft = (first["first_token"]["ts"] - first["request_sent"]["ts"]) * 1000 if "first_token" in first else None

        recompute = sum(
            (prev["cache_n"] + prev["prompt_n"] + prev["predicted_n"]) - cur["cache_n"]
            for prev, cur in zip(timings, timings[1:])
        )

        sched_rows = [sched[rid] for rid in requests if rid in sched]
        task_type = start["task_type"]
        rows.append({
            "run_id": start["run_id"],
            "task_id": task_id,
            "task_type": task_type,
            "agent_id": start["agent_id"],
            "num_agents": config.get("agents"),
            "scheduler_k": scheduler_k if via_scheduler else None,
            "tool_delay_s": config["tools"][DELAY_FIELD[task_type]] if "tools" in config else None,
            "status": end["status"],
            "is_success": bool(end.get("correct")),
            "total_steps": end.get("steps"),
            "total_completion_ms": (end["ts"] - start["ts"]) * 1000,
            "queue_wait_ms": sum(r["queue_ms"] for r in sched_rows) if via_scheduler else None,
            "model_request_ms": sum(r["model_request_ms"] for r in sched_rows) if via_scheduler else None,
            "llama_compute_ms": sum(t["prompt_ms"] + t["predicted_ms"] for t in timings),
            "tool_wait_ms": tool_wait * 1000,
            "first_ttft_ms": ttft,
            "recompute_tokens": recompute,
            "final_context_tokens": end.get("final_context_tokens"),
        })
    return sorted(rows, key=lambda r: r["task_id"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--scheduler-log", default="work/requests_minimal.jsonl")
    p.add_argument("--scheduler-k", type=int, help="K the scheduler ran with (it is not in its log)")
    args = p.parse_args()

    run = Path(args.run_dir)
    events = [json.loads(line) for line in open(run / "events.jsonl")]
    sched_path = Path(args.scheduler_log)
    scheduler_rows = [json.loads(line) for line in open(sched_path)] if sched_path.exists() else []
    config_path = run / "config.json"
    if config_path.exists():
        config = json.loads(config_path.read_text())
    else:  # runs from before the runner saved its config
        print("warning: no config.json in this run; num_agents and tool_delay_s are null")
        config = {}
    rows = summarize(events, scheduler_rows, config, args.scheduler_k)
    if rows and rows[0]["queue_wait_ms"] is not None and args.scheduler_k is None:
        print("warning: run went through the scheduler but --scheduler-k was not given; scheduler_k is null")

    out = run / "task_summary.jsonl"
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"{len(rows)} rows -> {out}")
    for kind in sorted({r["task_type"] for r in rows}):
        rs = [r for r in rows if r["task_type"] == kind]
        mean = lambda k: statistics.mean(r[k] for r in rs) if rs[0][k] is not None else float("nan")
        print(f"  {kind:<9} n={len(rs):<3} success={sum(r['is_success'] for r in rs)}/{len(rs)}  "
              f"total={mean('total_completion_ms'):.0f}ms  queue={mean('queue_wait_ms'):.0f}  "
              f"model={mean('model_request_ms'):.0f}  compute={mean('llama_compute_ms'):.0f}  "
              f"tool={mean('tool_wait_ms'):.0f}  ttft1={mean('first_ttft_ms'):.0f}  "
              f"recompute={mean('recompute_tokens'):.1f}  ctx={mean('final_context_tokens'):.0f}")


if __name__ == "__main__":
    main()
