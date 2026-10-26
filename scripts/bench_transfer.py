"""Transfer vs recompute microbenchmark: what it costs to move one agent's context out of the slot and back,
against recomputing it, for each GPU / CPU placement.

    uv run python -m scripts.bench_transfer --out logs/bench_transfer                       # Mac: one placement
    uv run python -m scripts.bench_transfer --out logs/bench_transfer --placements gpu kv_cpu layers18 \
        --llama-cmd ~/llama.cpp/build/bin/llama-server                                     # AWS

For each placement a fresh llama-server (-np 1, -lv 4) runs; then for each context length N and repeat:
    1  prefill     erase the slot, send N tokens of HotpotQA text: prefill_ms = recomputing an N-token context
    2  slot API    POST /slots/0?action=save, erase, action=restore: save_api_ms / restore_api_ms move the slot's
                   KV to a file and back (on Linux the file is in /dev/shm, i.e. host RAM, so this is VRAM <-> RAM),
                   then the same N tokens again: after_restore_prompt_ms (should be ~1 token of work)
    3  host cache  send a short other prompt (pushes the N-token context into the --cache-ram host cache),
                   then the N tokens again (pulls it back): host_save_ms / host_load_ms from the log
                   (scripts/cache_transfers.py), host_hit_prompt_ms = that request's remaining prompt work
Every prompt starts with a different header, so no two share a prefix and nothing hits a cache by accident.

Writes <out>/results.jsonl (one row per N x repeat x placement), <out>/meta.json, <out>/server_<placement>.log,
<out>/transfer.png (if matplotlib is installed: uv run --group analysis ...), and prints a table.

Placements (PLACEMENTS): gpu = everything in VRAM; kv_cpu = weights in VRAM, KV pool in host RAM (-nkvo);
layers18 = 18 of 36 layers in VRAM, the rest (and their KV) on the CPU. On Apple Silicon all three share one memory.
"""

import argparse
import json
import os
import shlex
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from scripts.cache_transfers import parse

ROOT = Path(__file__).resolve().parent.parent
MODEL = "bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:Q8_0"
PORT = 8080
PLACEMENTS = {
    "gpu": ["-ngl", "99"],
    "kv_cpu": ["-ngl", "99", "--no-kv-offload"],
    "layers18": ["-ngl", "18"],
}
SHORT_TOKENS = 64  # the "other prompt" that pushes the long context out of the slot


def post(path: str, body: dict | None = None, timeout: float = 3600) -> dict:
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", data=json.dumps(body or {}).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def healthy() -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2) as r:
            return json.loads(r.read()).get("status") == "ok"
    except (OSError, ValueError):
        return False


def complete(tokens: list[int]) -> dict:
    """One request, 1 generated token; returns llama.cpp timings plus client wall time."""
    t0 = time.perf_counter()
    r = post("/completion", {"prompt": tokens, "n_predict": 1, "cache_prompt": True, "temperature": 0})
    return r["timings"] | {"wall_ms": (time.perf_counter() - t0) * 1000}


class LogTail:
    """New prompt cache updates in the server log since the last call."""

    def __init__(self, path: Path):
        self.path, self.pos = path, 0

    def updates(self) -> list[dict]:
        with open(self.path, errors="replace") as f:
            f.seek(self.pos)
            text = f.read()
        cut = text.rfind("\n") + 1  # leave a half-written last line for next time
        self.pos += len(text[:cut].encode())
        return parse(text[:cut].splitlines())


def corpus_tokens() -> list[int]:
    text = "\n\n".join(json.loads(line)["text"] for line in open(ROOT / "data" / "hotpotqa_corpus.jsonl"))
    return post("/tokenize", {"content": text})["tokens"]


def prompt(body: list[int], header: str, n: int) -> list[int]:
    head = post("/tokenize", {"content": header + "\n"})["tokens"]
    return (head + body)[:n]


def bench_placement(name: str, args, out: Path, slot_dir: Path) -> list[dict]:
    log_path = out / f"server_{name}.log"
    cmd = [*shlex.split(args.llama_cmd), "-hf", args.model, "--port", str(PORT), "-c", str(max(args.tokens) + 1024),
           "-np", "1", "-kvu", "--cache-ram", str(args.cache_ram), "--slot-save-path", str(slot_dir),
           "--fit", "off", "-lv", "4", *PLACEMENTS[name]]
    rows = []
    with open(log_path, "w") as log:
        server = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, cwd=ROOT)
        try:
            deadline = time.monotonic() + 600
            while not healthy():
                if server.poll() is not None or time.monotonic() > deadline:
                    raise RuntimeError(f"llama-server ({name}) did not start; see {log_path}")
                time.sleep(1)
            body = corpus_tokens()
            if len(body) < max(args.tokens):
                raise RuntimeError(f"corpus has {len(body)} tokens, fewer than {max(args.tokens)}")
            tail = LogTail(log_path)
            complete(prompt(body, "warm-up", 256))  # first request pays one-off graph setup
            for r in range(args.repeats):
                for n in args.tokens:
                    row = {"placement": name, "n_tokens": n, "repeat": r}
                    long_prompt = prompt(body, f"Document {name}-{n}-{r}.", n)
                    other = prompt(body[::-1], f"Other {name}-{n}-{r}.", SHORT_TOKENS)

                    post("/slots/0?action=erase")
                    tail.updates()
                    t = complete(long_prompt)
                    row |= {"prefill_ms": t["prompt_ms"], "prefill_tokens": t["prompt_n"]}

                    s = post("/slots/0?action=save", {"filename": "bench.bin"})
                    post("/slots/0?action=erase")
                    rs = post("/slots/0?action=restore", {"filename": "bench.bin"})
                    (slot_dir / "bench.bin").unlink(missing_ok=True)
                    t = complete(long_prompt)
                    row |= {"save_api_ms": s["timings"]["save_ms"], "restore_api_ms": rs["timings"]["restore_ms"],
                            "kv_mib": s["n_written"] / 2**20, "after_restore_prompt_ms": t["prompt_ms"],
                            "after_restore_cached": t["cache_n"]}

                    tail.updates()
                    complete(other)  # the long context is saved to the host cache to make room
                    saved = [u for u in tail.updates() if u["save_tokens"]]
                    t = complete(long_prompt)  # ... and loaded back from it
                    loaded = [u for u in tail.updates() if u["load_tokens"]]
                    row |= {"host_save_ms": saved[-1]["save_ms"] if saved else None,
                            "host_save_tokens": saved[-1]["save_tokens"] if saved else None,
                            "host_load_ms": loaded[-1]["load_ms"] if loaded else None,
                            "host_load_tokens": loaded[-1]["load_tokens"] if loaded else None,
                            "host_hit_prompt_ms": t["prompt_ms"], "host_hit_cached": t["cache_n"]}
                    rows.append(row)
                    print(json.dumps(row), flush=True)
        finally:
            server.terminate()
            server.wait(60)
    return rows


def table(rows: list[dict]):
    keys = ("prefill_ms", "host_save_ms", "host_load_ms", "save_api_ms", "restore_api_ms", "kv_mib")
    print(f"\n{'placement':<10}{'tokens':>8}" + "".join(f"{k:>16}" for k in keys) + f"{'prefill/load':>14}")
    groups = {}
    for r in rows:
        groups.setdefault((r["placement"], r["n_tokens"]), []).append(r)
    for (pl, n), rs in groups.items():
        mean = {k: statistics.mean(v) if (v := [r[k] for r in rs if r[k] is not None]) else None for k in keys}
        ratio = mean["prefill_ms"] / mean["host_load_ms"] if mean["host_load_ms"] else None
        print(f"{pl:<10}{n:>8}" + "".join(f"{v:>16.1f}" if v is not None else f"{'-':>16}" for v in mean.values())
              + (f"{ratio:>14.0f}" if ratio else f"{'-':>14}"))


def plot(rows: list[dict], out: Path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(8, 5))
    for pl in dict.fromkeys(r["placement"] for r in rows):
        for key, style, label in (("prefill_ms", "-", "recompute (prefill)"), ("host_load_ms", "--", "host cache load"),
                                  ("host_save_ms", ":", "host cache save")):
            pts = {}
            for r in rows:
                if r["placement"] == pl and r[key] is not None:
                    pts.setdefault(r["n_tokens"], []).append(r[key])
            xs = sorted(pts)
            ax.plot(xs, [statistics.mean(pts[x]) for x in xs], style, marker="o", label=f"{pl}: {label}")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("context tokens")
    ax.set_ylabel("ms")
    ax.set_title("moving a context to host RAM and back vs recomputing it")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "transfer.png", dpi=120)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--placements", nargs="+", default=["gpu"], choices=list(PLACEMENTS))
    p.add_argument("--tokens", type=int, nargs="+", default=[512, 1024, 2048, 4096, 8192, 16384])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--cache-ram", type=int, default=16384, help="MiB; must hold the largest context twice")
    p.add_argument("--llama-cmd", default="llama serve")
    p.add_argument("--model", default=MODEL)
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    # slot files: in RAM where /dev/shm exists, so the slot API measures VRAM <-> host RAM, not the disk
    slot_dir = Path("/dev/shm/bench_slots") if os.path.isdir("/dev/shm") else args.out / "slots"
    slot_dir.mkdir(parents=True, exist_ok=True)
    version = subprocess.run([shlex.split(args.llama_cmd)[0], "--version"], capture_output=True, text=True)
    gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip() if shutil.which("nvidia-smi") else None
    (args.out / "meta.json").write_text(json.dumps({
        "args": {k: str(v) for k, v in vars(args).items()}, "platform": sys.platform, "gpu": gpu,
        "llama_version": (version.stdout + version.stderr).strip(), "slot_dir": str(slot_dir),
        "placements": {k: PLACEMENTS[k] for k in args.placements}}, indent=2))

    rows = []
    with open(args.out / "results.jsonl", "w") as f:
        for name in args.placements:
            for row in bench_placement(name, args, args.out, slot_dir):
                f.write(json.dumps(row) + "\n")
                rows.append(row)
    shutil.rmtree(slot_dir, ignore_errors=True)
    table(rows)
    plot(rows, args.out)


if __name__ == "__main__":
    main()
