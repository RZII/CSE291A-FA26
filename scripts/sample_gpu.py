"""GPU sampler (NVIDIA): one JSON line per interval, next to sample_memory's memory.jsonl.

    uv run python -m scripts.sample_gpu --pid <llama-server pid> --out logs/<run>/gpu.jsonl

On a discrete GPU the model weights and the KV pool sit in VRAM, which the process memory in memory.jsonl does
not include; the host RAM prompt cache (--cache-ram) and a KV pool kept on the CPU (--no-kv-offload) do show up
there. Together the two files give both tiers.

Fields per sample (sizes in MiB):
    ts                  time.time(), lines up with events.jsonl
    gpu_mem_used_mb     VRAM in use on GPU 0 (all processes)
    gpu_mem_total_mb    VRAM size of GPU 0
    gpu_util            GPU 0 utilization, percent
    proc_gpu_mem_mb     VRAM held by the llama-server process (None if nvidia-smi does not list it)

Runs until SIGTERM / SIGINT, or until nvidia-smi fails.
"""

import argparse
import json
import signal
import subprocess
import sys
import time


def query(*args: str) -> list[list[str]]:
    out = subprocess.run(["nvidia-smi", *args, "--format=csv,noheader,nounits"], capture_output=True, text=True,
                         check=True).stdout
    return [[x.strip() for x in line.split(",")] for line in out.strip().splitlines() if line.strip()]


def sample(pid: int) -> dict:
    used, total, util = query("--query-gpu=memory.used,memory.total,utilization.gpu")[0]
    apps = {int(p): float(m) for p, m in query("--query-compute-apps=pid,used_memory")}
    return {"ts": time.time(), "gpu_mem_used_mb": float(used), "gpu_mem_total_mb": float(total),
            "gpu_util": float(util), "proc_gpu_mem_mb": apps.get(pid)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pid", type=int, required=True, help="llama-server process id")
    p.add_argument("--out", required=True)
    p.add_argument("--interval", type=float, default=0.5)
    args = p.parse_args()

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    with open(args.out, "a", encoding="utf-8") as f:
        try:
            while True:
                t0 = time.monotonic()
                try:
                    row = sample(args.pid)
                except (subprocess.CalledProcessError, FileNotFoundError):
                    return  # no NVIDIA driver / GPU
                f.write(json.dumps(row) + "\n")
                f.flush()
                time.sleep(max(0.0, args.interval - (time.monotonic() - t0)))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
