"""Item 8: does the workload look like an agent workload? Per task type, from events.jsonl only.

    uv run python -m scripts.stats logs/run_<time>
"""

import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


def main(run_dir: str):
    events = [json.loads(line) for line in open(Path(run_dir) / "events.jsonl")]
    task_type = {e["task_id"]: e["task_type"] for e in events if e["event"] == "task_start"}
    tool_calls = Counter(e["task_id"] for e in events if e["event"] == "tool_start")
    ends = defaultdict(list)
    for e in events:
        if e["event"] == "task_end":
            ends[task_type[e["task_id"]]].append(e)

    rows = [
        ("tasks", lambda es: f"{len(es)}"),
        ("mean steps", lambda es: f"{statistics.mean(e['steps'] for e in es):.2f}"),
        ("steps distribution", lambda es: " ".join(f"{k}:{v}" for k, v in sorted(Counter(e["steps"] for e in es).items()))),
        ("mean tool calls / task", lambda es: f"{statistics.mean(tool_calls[e['task_id']] for e in es):.2f}"),
        ("tasks that called a tool", lambda es: f"{sum(tool_calls[e['task_id']] > 0 for e in es) / len(es):.0%}"),
        ("hit max_steps", lambda es: f"{sum(e['status'] == 'max_steps' for e in es) / len(es):.0%}"),
        ("status counts", lambda es: str(dict(Counter(e["status"] for e in es)))),
        ("success (correct)", lambda es: f"{sum(bool(e['correct']) for e in es) / len(es):.0%}"),
        ("mean EM / F1", lambda es: f"{statistics.mean(bool(e['em']) for e in es):.2f} / {statistics.mean(e['f1'] or 0 for e in es):.2f}"
            if es[0].get("em") is not None or any(e.get("f1") is not None for e in es) else "-"),
        ("mean final context tokens", lambda es: f"{statistics.mean(e['final_context_tokens'] for e in es):.0f}"),
        ("max final context tokens", lambda es: f"{max(e['final_context_tokens'] for e in es)}"),
    ]
    kinds = sorted(ends)
    print(f"{'':<28}" + "".join(f"{k:>26}" for k in kinds))
    for name, fn in rows:
        print(f"{name:<28}" + "".join(f"{fn(ends[k]):>26}" for k in kinds))


if __name__ == "__main__":
    main(sys.argv[1])
