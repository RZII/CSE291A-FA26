"""Do two runs of the same tasks produce the same trajectories?

Compares every assistant turn (content + tool calls) of each task found in both dirs.

    uv run python -m scripts.compare_runs logs/item5 logs/run_<time>
"""

import json
import sys
from pathlib import Path


def turns(path: Path) -> list:
    msgs = json.loads(path.read_text())["messages"]
    return [
        (m["content"], [(c["function"]["name"], c["function"]["arguments"]) for c in m.get("tool_calls") or []])
        for m in msgs
        if m["role"] == "assistant"
    ]


def trajectory_dir(d: str) -> Path:
    d = Path(d)
    return d / "trajectories" if (d / "trajectories").is_dir() else d  # run dirs vs logs/item5


def main(dir_a: str, dir_b: str):
    a, b = trajectory_dir(dir_a), trajectory_dir(dir_b)
    common = sorted({p.name for p in a.glob("*.json")} & {p.name for p in b.glob("*.json")})
    same = 0
    for name in common:
        ta, tb = turns(a / name), turns(b / name)
        if ta == tb:
            same += 1
            print(f"{name[:-5]:<36} identical ({len(ta)} turns)")
            continue
        k = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), min(len(ta), len(tb)))
        print(f"{name[:-5]:<36} DIFFERS at assistant turn {k + 1} ({len(ta)} vs {len(tb)} turns)")
        for label, t in (("A", ta), ("B", tb)):
            print(f"    {label}: {repr(t[k])[:160] if k < len(t) else '(no turn)'}")
    print(f"\n{same}/{len(common)} tasks identical")


if __name__ == "__main__":
    main(*sys.argv[1:3])
