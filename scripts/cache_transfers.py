"""Host prompt cache transfers, parsed from llama.log (needs llama-server -lv 4; level 3 does not log them).

    uv run python -m scripts.cache_transfers logs/<name>/<run>      # prints the transfers and their totals

With --cache-ram, when a slot is reused for a different prompt, llama-server first copies the slot's KV out to
the host RAM cache ("save"), then looks for a cached prompt sharing a prefix with the new one and copies it back
into the slot ("load"). On a discrete GPU both are VRAM <-> host RAM copies over PCIe; on Apple Silicon they stay
in the one shared memory. One "updating prompt cache" block in the log:

    get_availabl: updating prompt cache
    prompt_save:  - saving prompt with length 1804, total state size = 253.709 MiB    <- save starts
    load:  - looking for better prompt, ...                                           <- save done
    load:    - prompt with length    3004, lcp =    3001, f_keep = 0.999, f_sim = 1.000
    load:  - found better prompt with f_keep = 0.999, f_sim = 1.000                     <- load starts
    update:  - cache state: ...                                                       <- load done
    get_availabl: prompt cache update took 53.29 ms

Save and load times are the gaps between those log timestamps (ms resolution is plenty: 100s of MiB take 10s of
ms). Loaded MiB = loaded tokens x the KV bytes per token seen in the saves.
"""

import json
import re
import statistics
import sys
from collections.abc import Iterable
from pathlib import Path

TS = re.compile(r"^(\d+)\.(\d+)\.(\d+)\.(\d+) ")
SAVE = re.compile(r"prompt_save:\s+- saving prompt with length (\d+), total state size = ([\d.]+) MiB")
CANDIDATE = re.compile(r"load:\s+- prompt with length\s+(\d+), lcp =\s+(\d+), f_keep = ([\d.]+), f_sim = ([\d.]+)")
FOUND = re.compile(r"load:\s+- found better prompt with f_keep = ([\d.]+), f_sim = ([\d.]+)")
TOOK = re.compile(r"prompt cache update took ([\d.]+) ms")


def seconds(line: str) -> float | None:
    """llama.cpp log time "M.SS.mmm.uuu" since server start -> seconds."""
    m = TS.match(line)
    if not m:
        return None
    minutes, secs, ms, us = map(int, m.groups())
    return minutes * 60 + secs + ms / 1e3 + us / 1e6


def parse(lines: Iterable[str]) -> list[dict]:
    """llama.log lines -> one dict per prompt cache update: t (s since server start), save_tokens / save_mib / save_ms,
    load_tokens / load_lcp / load_ms (None when nothing was saved / loaded), total_ms."""
    updates, cur = [], None
    pending = None  # ("save" | "load", start time): ends at the next timestamped line of the block
    candidates = []
    for line in lines:
        t = seconds(line)
        if t is None:
            continue
        if pending and cur is not None and "load:    - prompt with length" not in line:
            kind, t0 = pending
            cur[f"{kind}_ms"] = (t - t0) * 1000
            pending = None
        if "updating prompt cache" in line:
            cur = {"t": t, "save_tokens": None, "save_mib": None, "save_ms": None,
                   "load_tokens": None, "load_lcp": None, "load_ms": None, "total_ms": None}
            candidates = []
        elif cur is None:
            continue
        elif m := SAVE.search(line):
            cur["save_tokens"], cur["save_mib"] = int(m[1]), float(m[2])
            pending = ("save", t)
        elif m := CANDIDATE.search(line):
            candidates.append((int(m[1]), int(m[2]), m[3], m[4]))
        elif m := FOUND.search(line):
            # the chosen candidate is the one whose f_keep / f_sim the "found" line repeats
            chosen = [c for c in candidates if (c[2], c[3]) == (m[1], m[2])]
            if chosen:
                cur["load_tokens"], cur["load_lcp"] = chosen[0][0], chosen[0][1]
            pending = ("load", t)
        elif m := TOOK.search(line):
            cur["total_ms"] = float(m[1])
            updates.append(cur)
            cur, pending = None, None
    return updates


def summarize(updates: list[dict]) -> dict:
    """Run-level totals for scripts.analyze."""
    saves = [u for u in updates if u["save_tokens"]]
    loads = [u for u in updates if u["load_tokens"]]
    mib_per_token = statistics.median(u["save_mib"] / u["save_tokens"] for u in saves) if saves else None
    load_mib = sum(u["load_tokens"] for u in loads) * mib_per_token if mib_per_token else None
    save_ms = sum(u["save_ms"] or 0 for u in saves)
    load_ms = sum(u["load_ms"] or 0 for u in loads)
    return {
        "cache_saves": len(saves),
        "cache_save_mib": sum(u["save_mib"] for u in saves),
        "cache_save_ms": save_ms,
        "cache_loads": len(loads),
        "cache_load_tokens": sum(u["load_tokens"] for u in loads),
        "cache_load_mib": load_mib,
        "cache_load_ms": load_ms,
        "cache_update_ms": sum(u["total_ms"] or 0 for u in updates),
        # effective copy bandwidth, GiB/s: on a discrete GPU this is the PCIe path VRAM <-> host RAM
        "cache_save_gib_s": sum(u["save_mib"] for u in saves) / 1024 / (save_ms / 1000) if save_ms else None,
        "cache_load_gib_s": load_mib / 1024 / (load_ms / 1000) if load_mib and load_ms else None,
    }


def main():
    run = Path(sys.argv[1])
    log = run / "llama.log" if run.is_dir() else run
    updates = parse(open(log, errors="replace"))
    for u in updates:
        print(json.dumps(u))
    print(json.dumps(summarize(updates), indent=2))


if __name__ == "__main__":
    main()
