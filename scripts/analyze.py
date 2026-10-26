"""One row per run, then one row per condition (mean and std over repeats), from run_experiment logs.

    uv run python -m scripts.analyze logs/<name>            # every run dir under it
    uv run python -m scripts.analyze logs/a logs/b/k2_a4_r0  # any mix of experiment and run dirs
Writes runs.csv and conditions.csv into the first argument (or --out), and task_summary.jsonl into
each run dir. Run-level fields:

    tasks, answered, timeout, error, ...     task_end status counts
    accuracy                    correct / tasks (unanswered tasks count as wrong)
    failure_rate                (error + crash + timeout) / tasks: the task never produced an answer
    completion_p50/p95/p99_s    task_start -> task_end, all tasks
    gsm8k_/hotpotqa_completion_p50_s    the same per type: are long tasks delayed unfairly?
    long_tasks, short_/long_completion_p50/p95_s    the same by tasks.response_length (long = 500+ word reply)
    queue_p50/p95_s             per request, scheduler queue_ms
    ttft_p50/p95_s              per request, request_sent -> first chunk (includes queueing)
    makespan_s                  first task_start -> last task_end
    throughput_tasks_per_min    answered tasks / makespan
    output_tokens_per_s         generated tokens (predicted_n) / makespan
    footprint_idle/peak_mb      memory the server itself dirtied: before the first request / max
    rss_peak_mb                 includes the mmapped model weights
    kv_tokens_peak, kv_tokens_mean, kv_peak_frac     sum over slots of tokens held; frac = peak / ctx
    prompt_tokens, cached_tokens, cached_frac        llama.cpp timings: prompt_n / cache_n sums
    recompute_tokens            within-task prefix lost between steps (see task_summary.py)
    sched_overhead_ms           mean of total_ms - queue_ms - model_request_ms per request
    max_sample_gap_s            longest gap between memory samples; > 10 s = the machine slept (warning printed)
"""

import argparse
import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from agent.client import CONTEXT_FULL
from scripts.task_summary import summarize

FAILED = ("error", "crash", "timeout")
MAX_GAP_S = 10


def pct(xs: list[float], q: float) -> float | None:
    """Linear-interpolated percentile; None for no data."""
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    i = q * (len(xs) - 1)
    lo, hi = math.floor(i), math.ceil(i)
    return xs[lo] + (xs[hi] - xs[lo]) * (i - lo)


def error_kind(e: dict) -> str:
    """The logged kind, except that runs logged before agent.client recognized in-stream KV-pool overflow
    have it as "stream"."""
    kind = e.get("kind", "other")
    return "context_overflow" if kind == "stream" and CONTEXT_FULL in e.get("error", "") else kind


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path)] if path.exists() else []


def analyze_run(run: Path) -> dict:
    exp = json.loads((run / "experiment.json").read_text())
    cond = exp["condition"]
    params = cond["params"]
    matrix = run.parent / "matrix.json"  # block membership as of now: later blocks may reuse this condition
    blocks = json.loads(matrix.read_text())["conditions"].get(cond["name"], cond)["blocks"] if matrix.exists() \
        else cond["blocks"]
    events = read_jsonl(run / "events.jsonl")
    sched = read_jsonl(run / "scheduler.jsonl")
    mem = read_jsonl(run / "memory.jsonl")
    config = json.loads((run / "config.json").read_text())

    tasks = summarize(events, sched, config, params["k"])
    with open(run / "task_summary.jsonl", "w") as f:
        f.writelines(json.dumps(t) + "\n" for t in tasks)

    ends = [e for e in events if e["event"] == "task_end"]
    starts = {e["task_id"]: e["ts"] for e in events if e["event"] == "task_start"}
    status = Counter(e["status"] for e in ends)
    n = len(ends)
    makespan = max(e["ts"] for e in ends) - min(starts.values()) if ends else None

    by_req = defaultdict(dict)
    for e in events:
        if "request_id" in e:
            by_req[e["request_id"]][e["event"]] = e
    ttft = [r["first_token"]["ts"] - r["request_sent"]["ts"] for r in by_req.values() if "first_token" in r]
    timings = [r["request_end"]["timings"] for r in by_req.values() if "request_end" in r]
    prompt = sum(t["prompt_n"] for t in timings)
    cached = sum(t["cache_n"] for t in timings)
    predicted = sum(t["predicted_n"] for t in timings)

    t0 = exp.get("t_runner_start", math.inf)
    idle = [m["footprint_mb"] for m in mem if m["ts"] < t0]
    kv = [m["kv_tokens"] for m in mem if m.get("kv_tokens") is not None]
    ok_sched = [r for r in sched if r["queue_ms"] is not None and r["model_request_ms"] is not None]
    errors = Counter(error_kind(e) for e in events if e["event"] == "request_error")
    completion = lambda kind=None: [t["total_completion_ms"] / 1000 for t in tasks if kind in (None, t["task_type"])]
    long = {e["task_id"]: e.get("long_answer", False) for e in events if e["event"] == "task_start"}
    by_length = lambda is_long: [t["total_completion_ms"] / 1000 for t in tasks if long[t["task_id"]] == is_long]

    row = {
        "run": run.name, "condition": cond["name"], "blocks": ";".join(blocks), "repeat": cond["repeat"],
        **params,
        "tasks": n, **{s: status.get(s, 0) for s in ("answered", "truncated", "max_steps", *FAILED)},
        **{f"req_error:{k}": v for k, v in sorted(errors.items())},
        "accuracy": sum(bool(e.get("correct")) for e in ends) / n if n else None,
        "failure_rate": sum(status[s] for s in FAILED) / n if n else None,
        "completion_p50_s": pct(completion(), 0.5),
        "completion_p95_s": pct(completion(), 0.95),
        "completion_p99_s": pct(completion(), 0.99),
        **{f"{kind}_completion_{q}_s": pct(completion(kind), v)
           for kind in ("gsm8k", "hotpotqa") for q, v in (("p50", 0.5), ("p95", 0.95))},
        "long_tasks": sum(long.values()),
        **{f"{name}_completion_{q}_s": pct(by_length(is_long), v)
           for name, is_long in (("short", False), ("long", True)) for q, v in (("p50", 0.5), ("p95", 0.95))},
        "queue_p50_s": pct([r["queue_ms"] / 1000 for r in ok_sched], 0.5),
        "queue_p95_s": pct([r["queue_ms"] / 1000 for r in ok_sched], 0.95),
        "ttft_p50_s": pct(ttft, 0.5), "ttft_p95_s": pct(ttft, 0.95),
        "makespan_s": makespan,
        "throughput_tasks_per_min": status["answered"] / makespan * 60 if makespan else None,
        "output_tokens_per_s": predicted / makespan if makespan else None,
        "output_tokens_p50": pct([t["predicted_n"] for t in timings], 0.5),
        "output_tokens_p95": pct([t["predicted_n"] for t in timings], 0.95),
        "output_tokens_max": max((t["predicted_n"] for t in timings), default=None),
        "context_tokens_p95": pct([t["cache_n"] + t["prompt_n"] for t in timings], 0.95),
        "footprint_idle_mb": statistics.median(idle) if idle else None,
        "footprint_peak_mb": max((m["footprint_mb"] for m in mem), default=None),
        "rss_peak_mb": max((m["rss_mb"] for m in mem), default=None),
        "kv_tokens_peak": max(kv, default=None),
        "kv_tokens_mean": statistics.mean(kv) if kv else None,
        "kv_peak_frac": max(kv) / params["ctx"] if kv else None,
        "prompt_tokens": prompt, "cached_tokens": cached,
        "cached_frac": cached / (cached + prompt) if cached + prompt else None,
        "recompute_tokens": sum(t["recompute_tokens"] for t in tasks),
        "sched_overhead_ms": statistics.mean(r["total_ms"] - r["queue_ms"] - r["model_request_ms"] for r in ok_sched)
        if ok_sched else None,
        # longest gap between memory samples: the sampler runs every 0.5 s, so a gap of many seconds means the
        # machine slept (e.g. laptop lid closed) and every timing in this run is invalid
        "max_sample_gap_s": max((b["ts"] - a["ts"] for a, b in zip(mem, mem[1:])), default=None),
    }
    if (row["max_sample_gap_s"] or 0) > MAX_GAP_S:
        print(f"WARNING {run}: {row['max_sample_gap_s']:.0f}s without memory samples; the machine probably slept, "
              f"so its timings are invalid. Move the run dir away and rerun it.")
    return row


def aggregate(rows: list[dict]) -> list[dict]:
    """Mean over repeats for numbers (plus _std for the headline metrics); first value for the rest."""
    groups = defaultdict(list)
    for r in rows:
        groups[r["condition"]].append(r)
    headline = ("accuracy", "failure_rate", "completion_p50_s", "completion_p95_s", "throughput_tasks_per_min",
                "footprint_peak_mb", "kv_tokens_peak")
    out = []
    for name, rs in groups.items():
        agg = {"condition": name, "repeats": len(rs)}
        for key in dict.fromkeys(k for r in rs for k in r):
            if key in ("run", "condition", "repeat"):
                continue
            # request error counts appear only in runs that had that error: missing means 0
            vals = [r.get(key, 0 if key.startswith("req_error:") else None) for r in rs]
            nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
            agg[key] = statistics.mean(nums) if nums and len(nums) == len(vals) else vals[0]
            if key in headline and len(nums) > 1:
                agg[f"{key}_std"] = statistics.stdev(nums)
        out.append(agg)
    return out


def write_csv(path: Path, rows: list[dict]):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: f"{v:.4g}" if isinstance(v, float) else v for k, v in r.items()})


def find_runs(paths: list[str]) -> list[Path]:
    runs = []
    for p in map(Path, paths):
        if (p / "experiment.json").exists():
            runs.append(p)
        else:
            runs += sorted(d.parent for d in p.glob("*/experiment.json"))
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("paths", nargs="+")
    p.add_argument("--out", help="where runs.csv / conditions.csv go (default: the first path)")
    args = p.parse_args()

    runs = [r for r in find_runs(args.paths) if json.loads((r / "experiment.json").read_text()).get("finished")]
    rows = [analyze_run(r) for r in runs]
    out = Path(args.out or args.paths[0])
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "runs.csv", rows)
    conds = aggregate(rows)
    write_csv(out / "conditions.csv", conds)
    print(f"{len(rows)} runs, {len(conds)} conditions -> {out}/runs.csv, conditions.csv")

    # one table per block (a condition shared by several blocks shows up in each); no blocks -> one table
    blocks = sorted({b for c in conds for b in str(c["blocks"]).split(";") if b}) or [None]
    cols = ("accuracy", "failure_rate", "completion_p50_s", "completion_p95_s", "queue_p95_s",
            "throughput_tasks_per_min", "output_tokens_p95", "footprint_peak_mb", "kv_tokens_peak", "cached_frac",
            "recompute_tokens")
    for b in blocks:
        print(f"\n== block {b} ==" if b else "")
        print(f"{'condition':<36}" + "".join(f"{c[:12]:>14}" for c in cols))
        for c in conds:
            if b is None or b in str(c["blocks"]).split(";"):
                print(f"{c['condition']:<36}" + "".join(
                    f"{c[k]:>14.3g}" if isinstance(c.get(k), (int, float)) else f"{'-':>14}" for k in cols))


if __name__ == "__main__":
    main()
