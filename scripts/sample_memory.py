"""Memory sampler: one JSON line per interval for the llama-server process and its KV slots.

    uv run python -m scripts.sample_memory --pid <llama-server pid> --out logs/<run>/memory.jsonl

Fields per sample (sizes in MiB):
    ts                   time.time(), lines up with events.jsonl
    rss_mb               resident set; includes the mmapped model file (clean, file-backed pages)
    footprint_mb         macOS phys_footprint / Linux RssAnon: memory the process itself dirtied
                         (KV cache, compute buffers, host prompt cache). This is what grows with load.
    sys_available_mb     memory the OS could still hand out
    slots_busy           slots currently processing a request
    kv_tokens            sum over slots of n_prompt_tokens + n_decoded: tokens each slot holds,
                         idle slots included (they keep their KV until it is reused or evicted).
                         An estimate: /slots does not report the KV cells actually allocated.
    slot_tokens          the same per slot, so retention of idle slots is visible
    requests_processing, requests_deferred, n_tokens_max    raw from /metrics (needs --metrics)

Runs until SIGTERM / SIGINT, or until the process exits.
"""

import argparse
import ctypes
import json
import re
import signal
import subprocess
import sys
import time
import urllib.request

MIB = 1024 * 1024


class RusageInfoV2(ctypes.Structure):
    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [(name, ctypes.c_uint64) for name in (
        "user_time", "system_time", "pkg_idle_wkups", "interrupt_wkups", "pageins", "wired_size",
        "resident_size", "phys_footprint", "proc_start_abstime", "proc_exit_abstime", "child_user_time",
        "child_system_time", "child_pkg_idle_wkups", "child_interrupt_wkups", "child_pageins",
        "child_elapsed_abstime", "diskio_bytesread", "diskio_byteswritten",
    )]


def process_memory_darwin(pid: int) -> tuple[float, float]:
    libc = ctypes.CDLL("libc.dylib", use_errno=True)
    info = RusageInfoV2()
    if libc.proc_pid_rusage(pid, 2, ctypes.byref(info)) != 0:  # 2 = RUSAGE_INFO_V2
        raise ProcessLookupError(pid)
    return info.resident_size / MIB, info.phys_footprint / MIB


def process_memory_linux(pid: int) -> tuple[float, float]:
    fields = {}
    with open(f"/proc/{pid}/status") as f:
        for line in f:
            key, _, value = line.partition(":")
            fields[key] = value
    kib = lambda k: int(fields[k].split()[0]) / 1024
    return kib("VmRSS"), kib("RssAnon")


def available_darwin() -> float:
    out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", out).group(1))
    pages = {m.group(1): int(m.group(2)) for m in re.finditer(r"^(.+?):\s+(\d+)\.", out, re.M)}
    return sum(pages.get(k, 0) for k in ("Pages free", "Pages inactive", "Pages speculative")) * page / MIB


def available_linux() -> float:
    with open("/proc/meminfo") as f:
        return next(int(line.split()[1]) / 1024 for line in f if line.startswith("MemAvailable:"))


process_memory = process_memory_darwin if sys.platform == "darwin" else process_memory_linux
available = available_darwin if sys.platform == "darwin" else available_linux


def get(url: str, timeout: float = 2.0) -> str | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.read().decode()
    except OSError:
        return None  # server busy or not up yet: leave the fields null rather than stall the sampler


def slot_fields(base_url: str) -> dict:
    text = get(base_url + "/slots")
    if text is None:
        return {"slots_busy": None, "kv_tokens": None, "slot_tokens": None}
    slots = json.loads(text)
    tokens = [s.get("n_prompt_tokens", 0) + sum(t.get("n_decoded", 0) for t in s.get("next_token", [])) for s in slots]
    return {"slots_busy": sum(bool(s.get("is_processing")) for s in slots), "kv_tokens": sum(tokens), "slot_tokens": tokens}


METRICS = ("requests_processing", "requests_deferred", "n_tokens_max")


def metric_fields(base_url: str) -> dict:
    text = get(base_url + "/metrics") or ""
    found = dict(re.findall(r"^llamacpp:(\w+) ([\d.e+-]+)$", text, re.M))
    return {k: float(found[k]) if k in found else None for k in METRICS}


def sample(pid: int, base_url: str) -> dict:
    rss, footprint = process_memory(pid)
    return {
        "ts": time.time(), "rss_mb": round(rss, 1), "footprint_mb": round(footprint, 1),
        "sys_available_mb": round(available(), 1), **slot_fields(base_url), **metric_fields(base_url),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pid", type=int, required=True, help="llama-server process id")
    p.add_argument("--url", default="http://127.0.0.1:8080", help="llama-server base URL (for /slots, /metrics)")
    p.add_argument("--out", required=True)
    p.add_argument("--interval", type=float, default=0.5)
    args = p.parse_args()

    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    with open(args.out, "a", encoding="utf-8") as f:
        try:
            while True:
                t0 = time.monotonic()
                try:
                    row = sample(args.pid, args.url)
                except (ProcessLookupError, FileNotFoundError):
                    return  # server gone
                f.write(json.dumps(row) + "\n")
                f.flush()
                time.sleep(max(0.0, args.interval - (time.monotonic() - t0)))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
