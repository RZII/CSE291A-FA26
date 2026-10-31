"""Probe the pinned llama.cpp runtime without modifying benchmark requests.

Read-only capability and token-count probes:
    uv run python -m scripts.probe_runtime --url http://127.0.0.1:8081 --out logs/runtime_probe.json

An optional isolated cache test issues one short completion and save/erase/restore
actions. Use a dedicated server with --slot-save-path /tmp/cse291a-slot-probe/;
never run --exercise-cache alongside an experiment.
"""

import argparse
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path


class Probe:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.calls = []

    def call(self, path: str, body=None):
        started = time.monotonic()
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base_url + path, data=data,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                status, text = response.status, response.read().decode()
        except urllib.error.HTTPError as exc:
            status, text = exc.code, exc.read().decode()
        elapsed_ms = (time.monotonic() - started) * 1000
        self.calls.append({"method": "GET" if body is None else "POST", "path": path,
                           "http_status": status, "elapsed_ms": round(elapsed_ms, 3)})
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            value = text
        return status, value

    def require(self, path: str, body=None):
        status, value = self.call(path, body)
        if status != 200:
            raise RuntimeError(f"{path}: HTTP {status}: {str(value)[:500]}")
        return value


def slot_summary(slots):
    return [{k: s.get(k) for k in ("id", "n_ctx", "is_processing", "n_prompt_tokens",
                                  "n_prompt_tokens_processed", "n_prompt_tokens_cache", "next_token")}
            for s in slots]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8081")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--exercise-cache", action="store_true",
                        help="mutate slot 0 on a dedicated idle probe server only")
    args = parser.parse_args()
    probe = Probe(args.url)
    report = {"url": args.url, "timestamp_unix_s": time.time(), "passed": False,
              "sources": [
                  "https://github.com/ggml-org/llama.cpp/blob/620d2225c/tools/server/server-context.cpp",
                  "https://github.com/ggml-org/llama.cpp/blob/620d2225c/tools/server/server-task.cpp"],
              "semantics": {
                  "slot_tokens": "n_prompt_tokens already includes generated context tokens; approximate, not allocated KV cells",
                  "n_tokens_max": "largest observed sequence length; not shared-pool occupancy or free space",
                  "slot_save": "disk serialization; save alone does not erase the source slot",
                  "host_cache": "cache-ram is a distinct automatic RAM prompt cache; no direct RAM offload/prefetch API established",
              }}
    try:
        report["health"] = probe.require("/health")
        props = probe.require("/props")
        report["props"] = {k: props.get(k) for k in ("model_path", "model_alias", "model_ftype", "total_slots")}
        report["props"]["default_generation_n_ctx"] = props.get("default_generation_settings", {}).get("n_ctx")
        slots = probe.require("/slots")
        assert isinstance(slots, list)
        report["slots_before"] = slot_summary(slots)
        status, available = probe.call("/slots?fail_on_no_slot=1")
        report["slot_available_http_status"] = status
        assert status in (200, 503)
        metrics_status, metrics = probe.call("/metrics")
        report["metrics_http_status"] = metrics_status
        if metrics_status == 200:
            report["metrics"] = {k: float(v) for k, v in re.findall(
                r"^llamacpp:(\w+) ([\d.e+\-]+)$", metrics, re.M)}

        body = {
            "model": props.get("model_alias", "probe"),
            "messages": [{"role": "user", "content": "Use identity to return the number seven."}],
            "tools": [{"type": "function", "function": {
                "name": "identity", "description": "Return the supplied integer.",
                "parameters": {"type": "object", "properties": {"value": {"type": "integer"}},
                               "required": ["value"]}}}],
            "temperature": 0, "max_tokens": 8,
        }
        counted = probe.require("/v1/chat/completions/input_tokens", body)
        assert isinstance(counted.get("input_tokens"), int) and counted["input_tokens"] > 0
        rendered = probe.require("/apply-template", body)["prompt"]
        tokenized = probe.require("/tokenize", {"content": rendered, "add_special": True,
                                              "parse_special": True})["tokens"]
        assert len(tokenized) == counted["input_tokens"], "chat token counter disagrees with rendered prompt"
        report["token_count"] = {"input_tokens": counted["input_tokens"],
                                 "rendered_tokenize_tokens": len(tokenized),
                                 "match": True, "includes_tools": True}

        if args.exercise_cache:
            assert slots and not any(s.get("is_processing") for s in slots), "Probe server must be idle"
            assert any(s.get("id") == 0 for s in slots)
            native = {"prompt": "Count slowly from one to twenty, separated by commas:",
                      "id_slot": 0, "n_predict": 8, "temperature": 0,
                      "seed": 0, "cache_prompt": True, "stream": False}
            completion = probe.require("/completion", native)
            before = next(s for s in probe.require("/slots") if s["id"] == 0)
            filename = f"cse291a-probe-{os.getpid()}.bin"
            saved = probe.require("/slots/0?action=save", {"filename": filename})
            after_save = next(s for s in probe.require("/slots") if s["id"] == 0)
            erased = probe.require("/slots/0?action=erase", {})
            after_erase = next(s for s in probe.require("/slots") if s["id"] == 0)
            restored = probe.require("/slots/0?action=restore", {"filename": filename})
            after_restore = next(s for s in probe.require("/slots") if s["id"] == 0)
            assert saved["n_saved"] > 0 and saved["n_saved"] == restored["n_restored"]
            assert before["n_prompt_tokens"] == after_save["n_prompt_tokens"] == saved["n_saved"]
            assert after_erase.get("n_prompt_tokens", 0) == 0
            assert after_restore["n_prompt_tokens"] == restored["n_restored"]
            report["cache_probe"] = {
                "completion_tokens_evaluated": completion.get("tokens_evaluated"),
                "completion_tokens_predicted": completion.get("tokens_predicted"),
                "save": saved, "erase": erased, "restore": restored,
                "slots": {"before_save": slot_summary([before]), "after_save": slot_summary([after_save]),
                          "after_erase": slot_summary([after_erase]), "after_restore": slot_summary([after_restore])},
                "save_preserves_source_slot": True, "erase_clears_reported_context": True,
                "restore_recovers_saved_token_count": True,
            }
            probe.require("/slots/0?action=erase", {})
        report["passed"] = True
    except Exception as exc:
        report["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        report["calls"] = probe.calls
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["passed"] else 1)


if __name__ == "__main__":
    main()
