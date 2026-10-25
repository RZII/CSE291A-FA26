"""One command per experiment: for every condition, start a fresh llama-server, the FIFO scheduler
and the memory sampler, run the agents, stop everything.

    uv run python -m scripts.run_experiment --matrix experiments/phase1.toml --dry-run
    uv run python -m scripts.run_experiment --matrix experiments/phase1.toml --blocks A B
    uv run python -m scripts.run_experiment --name k_x_agents --tasks gsm8k:0-9 hotpotqa:0-9 \
        --k 1 2 4 --agents 2 4 8 --set tools.search_delay_s=3

Two ways to define conditions:
  --matrix FILE   blocks of one-factor variations around a shared baseline (format: experiments/phase1.toml).
                  Identical conditions in several blocks run once.
  flags           --k / --agents / --ctx / --np / --cache-ram take lists, --sweep SECTION.FIELD=a,b sweeps
                  any config.toml field, --set fixes one; the full cartesian product runs.
Each condition runs --repeats times; repeat r shuffles the task order with seed r.

Every run gets a fresh llama-server, so no cache (slot KV or the host RAM cache) carries over between
runs. Runs go in shuffled order (--order) with --cooldown seconds between them, so heat build-up on a
laptop does not line up with one factor; `pmset -g therm` is recorded before and after each run (macOS).
Layout:
    logs/<name>/matrix.json            the conditions and which block each belongs to
    logs/<name>/<condition>_r<r>/
        experiment.json    parameters, exact commands, llama.cpp version, model sha256, /props, thermal
        events.jsonl, config.json, trajectories/    agent.runner
        scheduler.jsonl    one line per request (scheduler.py)
        memory.jsonl       scripts/sample_memory.py
        llama.log, scheduler.out, runner.out
A run whose experiment.json has "finished" set is skipped, so an interrupted experiment can just be rerun.
"""

import argparse
import hashlib
import itertools
import json
import os
import random
import shlex
import signal
import socket
import subprocess
import sys
import time
import tomllib
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LLAMA_PORT, SCHEDULER_PORT = 8080, 8000
MODEL = "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:Q8_0"
SERVER_KEYS = {"k": "k", "agents": "a", "ctx": "ctx", "np": "np", "cache_ram": "cram"}  # key -> short name
DEFAULTS = {"k": 2, "agents": 4, "ctx": 32768, "np": 4, "cache_ram": 8192}


def port_open(port: int) -> bool:
    with socket.socket() as s:
        return s.connect_ex(("127.0.0.1", port)) == 0


def wait_until(check, timeout_s: float, what: str, proc: subprocess.Popen):
    deadline = time.monotonic() + timeout_s
    while not check():
        if proc.poll() is not None:
            raise RuntimeError(f"{what} exited with code {proc.returncode}")
        if time.monotonic() > deadline:
            raise RuntimeError(f"{what} not ready after {timeout_s:g}s")
        time.sleep(0.5)


def get_json(url: str):
    try:
        with urllib.request.urlopen(url, timeout=2) as r:
            return json.loads(r.read())
    except (OSError, ValueError):
        return None


def thermal() -> str | None:
    """macOS thermal / CPU speed limit report; None elsewhere."""
    if sys.platform != "darwin":
        return None
    return subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True).stdout.strip()


def model_sha256(path: str) -> str:
    """Hashing 4 GB takes a while, so cache it per (path, size, mtime) in work/."""
    st = os.stat(path)
    cache_file = ROOT / "work" / "model_sha256.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    key = f"{path}|{st.st_size}|{st.st_mtime_ns}"
    if key not in cache:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            while chunk := f.read(1 << 24):
                h.update(chunk)
        cache[key] = h.hexdigest()
        cache_file.parent.mkdir(exist_ok=True)
        cache_file.write_text(json.dumps(cache, indent=2))
    return cache[key]


def stop(proc: subprocess.Popen | None, sig=signal.SIGTERM, timeout_s: float = 30):
    if proc is None or proc.poll() is not None:
        return
    proc.send_signal(sig)
    try:
        proc.wait(timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()


def run_one(run_dir: Path, run: dict, tasks: list[str], args) -> dict:
    p = run["params"]
    run_dir.mkdir(parents=True, exist_ok=True)
    for f in ("events.jsonl", "scheduler.jsonl", "memory.jsonl"):  # leftovers of an interrupted attempt
        (run_dir / f).unlink(missing_ok=True)
    llama_cmd = [*shlex.split(args.llama_cmd), "-hf", args.model, "--jinja", "--port", str(LLAMA_PORT),
                 "-c", str(p["ctx"]), "-np", str(p["np"]), "-kvu", "--metrics", "--cache-ram", str(p["cache_ram"])]
    scheduler_cmd = [sys.executable, str(ROOT / "scheduler.py"), "--k", str(p["k"]),
                     "--queue-capacity", "4096", "--max-connections", "4096",  # never reject: failures must come from the model side
                     "--network-timeout", str(args.network_timeout), "--queue-budget", "86400",
                     "--port", str(SCHEDULER_PORT), "--log", str(run_dir / "scheduler.jsonl")]
    sets = [f"{k}={json.dumps(v)}" for k, v in p.items() if "." in k]  # json literals parse as TOML too
    runner_cmd = [sys.executable, "-m", "agent.runner", *tasks, "--agents", str(p["agents"]),
                  "--seed", str(run["repeat"]), "--out", str(run_dir),
                  "--set", f"llm.base_url=http://127.0.0.1:{SCHEDULER_PORT}",
                  "--set", f"llm.read_timeout_s={args.network_timeout}",
                  *itertools.chain.from_iterable(("--set", s) for s in sets)]
    info = {"condition": run, "tasks": tasks, "llama_cmd": llama_cmd, "scheduler_cmd": scheduler_cmd,
            "runner_cmd": runner_cmd, "platform": sys.platform, "thermal_before": thermal(), "finished": None}
    llama = scheduler = sampler = None
    logs = []
    try:
        logs.append(open(run_dir / "llama.log", "w"))
        llama = subprocess.Popen(llama_cmd, stdout=logs[-1], stderr=subprocess.STDOUT, cwd=ROOT)
        wait_until(lambda: (get_json(f"http://127.0.0.1:{LLAMA_PORT}/health") or {}).get("status") == "ok",
                   600, "llama-server", llama)
        props = get_json(f"http://127.0.0.1:{LLAMA_PORT}/props")
        version = subprocess.run([shlex.split(args.llama_cmd)[0], "--version"], capture_output=True, text=True)
        info |= {"llama_version": (version.stdout + version.stderr).strip(), "llama_pid": llama.pid,
                 "props": {k: props.get(k) for k in ("model_path", "model_alias", "model_ftype", "total_slots")}
                 | {"n_ctx": props["default_generation_settings"]["n_ctx"]},
                 "model_sha256": model_sha256(props["model_path"])}

        logs.append(open(run_dir / "scheduler.out", "w"))
        scheduler = subprocess.Popen(scheduler_cmd, stdout=logs[-1], stderr=subprocess.STDOUT, cwd=ROOT)
        wait_until(lambda: port_open(SCHEDULER_PORT), 30, "scheduler", scheduler)
        sampler = subprocess.Popen([sys.executable, "-m", "scripts.sample_memory", "--pid", str(llama.pid),
                                    "--url", f"http://127.0.0.1:{LLAMA_PORT}", "--out", str(run_dir / "memory.jsonl"),
                                    "--interval", str(args.memory_interval)], cwd=ROOT)
        time.sleep(2 * args.memory_interval)  # idle samples before the first request = the baseline footprint

        info["t_runner_start"] = time.time()
        (run_dir / "experiment.json").write_text(json.dumps(info, indent=2))
        with open(run_dir / "runner.out", "w") as out:
            runner = subprocess.run(runner_cmd, stdout=out, stderr=subprocess.STDOUT, cwd=ROOT)
        info["t_runner_end"] = time.time()
        info["runner_exit_code"] = runner.returncode
        time.sleep(2 * args.memory_interval)
    finally:
        stop(sampler)
        stop(scheduler, signal.SIGINT)  # SIGINT: scheduler drains its queue, then exits
        stop(llama, signal.SIGINT)
        for f in logs:
            f.close()
    info["thermal_after"] = thermal()
    info["finished"] = time.time()
    (run_dir / "experiment.json").write_text(json.dumps(info, indent=2))
    return info


# ---- conditions ----


def short_value(v) -> str:
    return str(int(v)) if isinstance(v, bool) else f"{v:g}" if isinstance(v, float) else str(v)


def condition_name(params: dict, reference: dict, always=()) -> str:
    """Name from the parameters that differ from `reference` (plus `always`); "base" if none do."""
    keys = [k for k in params if params[k] != reference.get(k) or k in always]
    keys.sort(key=lambda k: (k not in SERVER_KEYS, list(SERVER_KEYS).index(k) if k in SERVER_KEYS else 0, k))
    parts = [f"{SERVER_KEYS[k] if k in SERVER_KEYS else k.split('.')[-1]}{short_value(params[k])}" for k in keys]
    return "_".join(parts) or "base"


def expand(block: dict) -> list[dict]:
    """A block's override dicts: the product of `grid`, or the explicit `conditions` list."""
    if "conditions" in block:
        return [dict(c) for c in block["conditions"]]
    grid = block.get("grid", {})
    return [dict(zip(grid, values)) for values in itertools.product(*grid.values())]


def from_matrix(path: Path, only: list[str] | None) -> tuple[dict, list[dict]]:
    spec = tomllib.loads(path.read_text())
    baseline = DEFAULTS | spec["baseline"]
    conds = {}  # name -> {"name", "params", "blocks"}
    for block in spec["block"]:
        if only and block["name"] not in only:
            continue
        for override in expand(block):
            params = baseline | block.get("base", {}) | override
            name = condition_name(params, baseline, always=["k", "agents"])
            conds.setdefault(name, {"name": name, "params": params, "blocks": []})["blocks"].append(block["name"])
    return spec, list(conds.values())


def from_flags(args) -> tuple[dict, list[dict]]:
    axes = {"k": args.k, "agents": args.agents, "ctx": args.ctx, "np": args.np, "cache_ram": args.cache_ram}
    axes |= {k: v.split(",") for k, _, v in (s.partition("=") for s in args.sweep)}
    axes = {k: [tomllib.loads(f"v = {x}")["v"] if isinstance(x, str) else x for x in vs] for k, vs in axes.items()}
    fixed = {k: tomllib.loads(f"v = {v}")["v"] for k, _, v in (s.partition("=") for s in args.set)}
    first = {k: vs[0] for k, vs in axes.items()}
    varied = [k for k, vs in axes.items() if len(vs) > 1]
    conds = []
    for values in itertools.product(*axes.values()):
        params = dict(zip(axes, values)) | fixed
        conds.append({"name": condition_name(params, first, always=["k", "agents", *varied]), "params": params,
                      "blocks": []})
    spec = {"name": args.name, "tasks": args.tasks, "repeats": args.repeats, "baseline": first | fixed}
    return spec, conds


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--matrix", type=Path, help="TOML matrix file (see experiments/phase1.toml)")
    p.add_argument("--blocks", nargs="+", help="with --matrix: run only these blocks")
    p.add_argument("--name", help="experiment name: logs/<name>/ (default: the matrix's name)")
    p.add_argument("--tasks", nargs="+", help="task specs for agent.runner, e.g. gsm8k:0-9 hotpotqa:0-9")
    p.add_argument("--repeats", type=int)
    p.add_argument("--k", type=int, nargs="+", default=[DEFAULTS["k"]], help="scheduler concurrency K")
    p.add_argument("--agents", type=int, nargs="+", default=[DEFAULTS["agents"]])
    p.add_argument("--ctx", type=int, nargs="+", default=[DEFAULTS["ctx"]], help="llama-server -c: the KV budget in tokens")
    p.add_argument("--np", type=int, nargs="+", default=[DEFAULTS["np"]], help="llama-server -np: slots")
    p.add_argument("--cache-ram", type=int, nargs="+", default=[DEFAULTS["cache_ram"]],
                   help="llama-server --cache-ram MiB (host prompt cache)")
    p.add_argument("--sweep", action="append", default=[], metavar="SECTION.FIELD=V1,V2")
    p.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=V")
    p.add_argument("--order", choices=["shuffle", "sequential"], default="shuffle")
    p.add_argument("--cooldown", type=float, default=60, help="seconds of idle between runs")
    p.add_argument("--llama-cmd", default="llama serve", help="how to launch llama-server")
    p.add_argument("--model", default=MODEL)
    p.add_argument("--network-timeout", type=float, default=600, help="scheduler/agent read timeout, seconds")
    p.add_argument("--memory-interval", type=float, default=0.5)
    p.add_argument("--dry-run", action="store_true", help="list the runs without starting anything")
    args = p.parse_args()

    if args.matrix:
        spec, conds = from_matrix(args.matrix, args.blocks)
    else:
        if not (args.name and args.tasks):
            p.error("without --matrix, --name and --tasks are required")
        spec, conds = from_flags(args)
    name = args.name or spec["name"]
    tasks = args.tasks or spec["tasks"]
    repeats = args.repeats or spec.get("repeats", 1)
    runs = [c | {"repeat": r, "run": f"{c['name']}_r{r}"} for c in conds for r in range(repeats)]
    if args.order == "shuffle":
        random.Random(0).shuffle(runs)

    out = ROOT / "logs" / name
    print(f"{len(conds)} conditions x {repeats} repeats = {len(runs)} runs, {len(tasks)} task specs each -> {out}")
    for c in conds:
        blocks = f"  [{','.join(c['blocks'])}]" if c["blocks"] else ""
        print(f"   {c['name']:<36}{blocks}")
    if args.dry_run:
        return
    for port in (LLAMA_PORT, SCHEDULER_PORT):
        if port_open(port):
            sys.exit(f"port {port} is in use: stop the running llama-server / scheduler first")

    out.mkdir(parents=True, exist_ok=True)
    matrix_file = out / "matrix.json"
    known = json.loads(matrix_file.read_text())["conditions"] if matrix_file.exists() else {}
    for c in conds:  # merge, so running blocks one at a time builds up one matrix.json
        known.setdefault(c["name"], {"params": c["params"], "blocks": []})
        known[c["name"]]["blocks"] = sorted(set(known[c["name"]]["blocks"]) | set(c["blocks"]))
    matrix_file.write_text(json.dumps({"spec": spec, "tasks": tasks, "conditions": known}, indent=2, default=str))

    ran = 0
    for i, run in enumerate(runs, 1):
        run_dir = out / run["run"]
        done = run_dir / "experiment.json"
        if done.exists() and json.loads(done.read_text()).get("finished"):
            print(f"[{i}/{len(runs)}] {run['run']}: already finished, skipped")
            continue
        if ran and args.cooldown:
            time.sleep(args.cooldown)
        t0 = time.time()
        print(f"[{i}/{len(runs)}] {run['run']} ...", flush=True)
        info = run_one(run_dir, run, tasks, args)
        ran += 1
        tail = (run_dir / "runner.out").read_text().strip().splitlines()[-1:] or ["(no output)"]
        print(f"    {time.time() - t0:.0f}s, runner exit {info.get('runner_exit_code')}: {tail[0]}", flush=True)


if __name__ == "__main__":
    main()
