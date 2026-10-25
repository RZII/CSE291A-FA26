"""Performance curves for one experiment directory (run scripts.analyze on it first).

    uv run --group analysis python -m scripts.plot logs/phase1
Writes logs/<name>/plots/:
    block_<X>.png        headline metrics against the block's factor (two factors: one line per value
                         of the first, e.g. one line per K); error bars = std over repeats
    timeline_<run>.png   KV tokens held, server footprint and busy slots over one run (--timeline;
                         default: each block's first and last condition, repeat 0)
    context_growth.png   prompt context per agent step, by task type (all runs)
    fairness.png         p50 / p95 completion per task type for every condition
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

METRICS = [
    ("throughput_tasks_per_min", "throughput (tasks/min)"),
    ("completion_p95_s", "task completion p95 (s)"),
    ("queue_p95_s", "request queueing p95 (s)"),
    ("kv_tokens_peak", "peak KV tokens held"),
    ("footprint_peak_mb", "peak server footprint (MiB)"),
    ("recompute_tokens", "recomputed prefix tokens"),
    ("failure_rate", "failure rate"),
    ("accuracy", "accuracy"),
]


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def load_conditions(exp: Path) -> dict[str, dict]:
    with open(exp / "conditions.csv") as f:
        return {r["condition"]: r for r in csv.DictReader(f)}


def param(params: dict, key: str):
    """A parameter not set in the matrix means the config.toml default."""
    return params.get(key, "default")


def block_factors(names: list[str], params: dict[str, dict]) -> list[str]:
    keys = dict.fromkeys(k for n in names for k in params[n])
    return [k for k in keys if len({json.dumps(params[n].get(k, "default")) for n in names}) > 1]


def plot_block(block: str, names: list[str], params: dict, conds: dict, out: Path):
    factors = block_factors(names, params)
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5))
    if len(factors) >= 2 and len({json.dumps([param(params[n], f) for f in factors]) for n in names}) == len(names) \
            and all(isinstance(param(params[names[0]], f), (int, float)) for f in factors[:2]):
        line_key, x_key = factors[0], factors[1]  # e.g. A: one line per K, x = agents
        series = defaultdict(list)
        for n in names:
            series[param(params[n], line_key)].append(n)
        for ax, (m, label) in zip(axes.flat, METRICS):
            for value, ns in sorted(series.items()):
                ns = sorted(ns, key=lambda n: param(params[n], x_key))
                xs = [param(params[n], x_key) for n in ns]
                ys = [num(conds[n].get(m)) if n in conds else None for n in ns]
                es = [num(conds[n].get(f"{m}_std")) or 0 if n in conds else 0 for n in ns]
                ax.errorbar(xs, ys, yerr=es, marker="o", capsize=3, label=f"{line_key}={value}")
            ax.set_xlabel(x_key)
            ax.set_xticks(sorted({param(params[n], x_key) for n in names}))
            ax.set_title(label, fontsize=10)
        axes.flat[0].legend()
    else:
        x_key = factors[0] if factors else None
        numeric = x_key and all(isinstance(param(params[n], x_key), (int, float)) and not isinstance(param(params[n], x_key), bool)
                                for n in names)
        ns = sorted(names, key=lambda n: param(params[n], x_key)) if numeric else names
        labels = [f"{param(params[n], x_key):g}" if numeric else n for n in ns]
        for ax, (m, label) in zip(axes.flat, METRICS):
            ys = [num(conds[n].get(m)) if n in conds else None for n in ns]
            es = [num(conds[n].get(f"{m}_std")) or 0 if n in conds else 0 for n in ns]
            ax.bar(range(len(ns)), [y or 0 for y in ys], yerr=es, capsize=3, color="#4c78a8")
            ax.set_xticks(range(len(ns)), labels, rotation=20 if not numeric else 0, fontsize=8)
            ax.set_xlabel(x_key if numeric else "condition")
            ax.set_title(label, fontsize=10)
    fig.suptitle(f"block {block}: " + ", ".join(factors))
    fig.tight_layout()
    fig.savefig(out / f"block_{block}.png", dpi=120)
    plt.close(fig)


def plot_timeline(run: Path, out: Path):
    mem = [json.loads(line) for line in open(run / "memory.jsonl")]
    exp = json.loads((run / "experiment.json").read_text())
    if not mem:
        return
    t0 = exp.get("t_runner_start", mem[0]["ts"])
    ts = [m["ts"] - t0 for m in mem]
    fig, ax = plt.subplots(3, 1, figsize=(11, 7.5), sharex=True)
    ax[0].plot(ts, [m["kv_tokens"] for m in mem], color="#4c78a8")
    ax[0].axhline(exp["condition"]["params"]["ctx"], color="gray", ls="--", label="KV budget (-c)")
    ax[0].set_ylabel("KV tokens held")
    ax[0].legend(loc="upper right")
    ax[1].plot(ts, [m["footprint_mb"] for m in mem], color="#e45756")
    ax[1].set_ylabel("footprint (MiB)")
    ax[2].step(ts, [m["slots_busy"] for m in mem], where="post", color="#54a24b")
    ax[2].set_ylabel("busy slots")
    ax[2].set_xlabel("seconds since agents started")
    fig.suptitle(run.name)
    fig.tight_layout()
    fig.savefig(out / f"timeline_{run.name}.png", dpi=120)
    plt.close(fig)


def plot_context_growth(runs: list[Path], out: Path):
    by = defaultdict(lambda: defaultdict(list))  # task type -> step -> context tokens
    for run in runs:
        kind = {}
        for line in open(run / "events.jsonl"):
            e = json.loads(line)
            if e["event"] == "task_start":
                kind[e["task_id"]] = e["task_type"]
            elif e["event"] == "request_end":
                t = e["timings"]
                by[kind[e["task_id"]]][e["step"]].append(t["cache_n"] + t["prompt_n"] + t["predicted_n"])
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, steps in sorted(by.items()):
        xs = sorted(steps)
        ax.plot(xs, [sum(steps[s]) / len(steps[s]) for s in xs], marker="o", label=f"{k} mean")
        ax.plot(xs, [max(steps[s]) for s in xs], ls=":", label=f"{k} max")
    ax.set_xlabel("agent step (model request #)")
    ax.set_ylabel("context tokens after the step")
    ax.set_title("context growth per agent step (all runs)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "context_growth.png", dpi=120)
    plt.close(fig)


def plot_fairness(conds: dict, out: Path):
    names = list(conds)
    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.6), 5))
    width = 0.2
    for i, (key, label) in enumerate([("gsm8k_completion_p50_s", "GSM8K p50"), ("gsm8k_completion_p95_s", "GSM8K p95"),
                                      ("hotpotqa_completion_p50_s", "HotpotQA p50"),
                                      ("hotpotqa_completion_p95_s", "HotpotQA p95")]):
        ax.bar([x + (i - 1.5) * width for x in range(len(names))], [num(conds[n].get(key)) or 0 for n in names],
               width, label=label)
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("task completion time (s)")
    ax.set_title("completion time by task type: are long tasks delayed unfairly?")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "fairness.png", dpi=120)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("exp_dir")
    p.add_argument("--timeline", nargs="*", help="run dir names to draw timelines for")
    args = p.parse_args()

    exp = Path(args.exp_dir)
    out = exp / "plots"
    out.mkdir(exist_ok=True)
    conds = load_conditions(exp)
    matrix = json.loads((exp / "matrix.json").read_text())["conditions"]
    params = {n: c["params"] for n, c in matrix.items()}
    blocks = defaultdict(list)
    for n, c in matrix.items():
        for b in c["blocks"]:
            blocks[b].append(n)

    for b, names in sorted(blocks.items()):
        plot_block(b, names, params, conds, out)
    runs = sorted(d.parent for d in exp.glob("*/memory.jsonl"))
    timelines = args.timeline if args.timeline is not None else \
        sorted({f"{ns[i]}_r0" for ns in blocks.values() for i in (0, -1)})
    for name in timelines:
        if (exp / name / "memory.jsonl").exists():
            plot_timeline(exp / name, out)
    plot_context_growth(runs, out)
    plot_fairness(conds, out)
    print(f"plots -> {out}")


if __name__ == "__main__":
    main()
