"""Bounded AWS reproduction, runtime probe, and frozen development/test comparison.

    uv run python -m scripts.run_validation --dry-run
    uv run --group analysis python -m scripts.run_validation \
        --llama-cmd ~/llama.cpp/build/bin/llama-server \
        --deadline-epoch <OS-shutdown-epoch-minus-1800>

Already complete, valid representative runs are reused. A saved deadline is never
extended by resuming. The parent wrapper owns the OS shutdown and result archive.
Individual task timeouts/errors are experimental outcomes, not orchestration errors.
"""

import argparse
import hashlib
import json
import os
import re
import shlex
import signal
import socket
import subprocess
import sys
import time
import tomllib
import urllib.request
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scripts.run_experiment import MODEL, from_matrix


ROOT = Path(__file__).resolve().parent.parent
MODEL_SHA256 = "260b5b5b6ad73e44df81a43ea1f5c11c37007b6bac18eb3cd2016e8667c19662"
MATRIX_PATHS = [ROOT / "experiments" / name for name in
                ("representative_short.toml", "representative_long.toml", "phase2_dev.toml")]
CORE_PATHS = [ROOT / name for name in
              ("scheduler.py", "scripts/run_experiment.py", "scripts/sample_memory.py", "config.toml", "uv.lock")]
SELECTION_RULE = ["maximize answered/tasks", "then maximize correct/tasks",
                  "then maximize pooled correct tasks per task-makespan minute", "then smallest K"]


class DeadlineReached(RuntimeError):
    pass


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def task_ids(specs):
    ids = []
    for spec in specs:
        kind, selection = spec.split(":", 1)
        rows = read_jsonl(ROOT / "data" / f"{kind}_50.jsonl")
        if "-" in selection:
            first, last = map(int, selection.split("-", 1))
            chosen = rows[first:last + 1]
            assert len(chosen) == last - first + 1
        else:
            chosen = [rows[int(selection)]]
        ids.extend(row["id"] for row in chosen)
    assert len(set(ids)) == len(ids), "Task IDs must be unique"
    return set(ids)


def expected_runs(matrix):
    spec, conditions = from_matrix(Path(matrix), None)
    return spec, [(condition, repeat, ROOT / "logs" / spec["name"] / f"{condition['name']}_r{repeat}")
                  for condition in conditions for repeat in range(spec.get("repeats", 1))]


def inspect_run(path, expected_ids, params, core_hashes):
    """Structural validation with explicit terminal outcomes for every selected task."""
    row = {"run_dir": str(path), "valid": False, "issues": [], "exists": path.exists()}
    try:
        info = json.loads((path / "experiment.json").read_text())
        row.update(finished=bool(info.get("finished")), runner_exit_code=info.get("runner_exit_code"),
                   k=params["k"], repeat=info["condition"]["repeat"])
        if not row["finished"] or row["runner_exit_code"] != 0:
            row["issues"].append("experiment did not finish with runner exit 0")
        if info.get("model_sha256") != MODEL_SHA256:
            row["issues"].append("model SHA256 differs from the pinned baseline")
        if "build 11669" not in info.get("llama_version", "") or "commit 620d2225c" not in info.get("llama_version", ""):
            row["issues"].append("llama.cpp version differs from the pinned baseline")
        if info.get("source_sha256") != core_hashes:
            row["issues"].append("core source hashes differ from the frozen validation source")
        if info["condition"]["params"] != params:
            row["issues"].append("condition parameters differ from the matrix")
        for name in ("config.json", "events.jsonl", "scheduler.jsonl", "llama.log", "memory.jsonl", "gpu.jsonl"):
            if not (path / name).is_file() or not (path / name).stat().st_size:
                row["issues"].append("missing or empty " + name)
        events = read_jsonl(path / "events.jsonl")
        starts = [e for e in events if e["event"] == "task_start"]
        ends = [e for e in events if e["event"] == "task_end"]
        if len(starts) != len(expected_ids) or {e["task_id"] for e in starts} != expected_ids:
            row["issues"].append("task starts do not match the selected task set")
        if len(ends) != len(expected_ids) or {e["task_id"] for e in ends} != expected_ids:
            row["issues"].append("task ends do not match the selected task set")
        by_task = defaultdict(list)
        by_request = defaultdict(dict)
        for event in events:
            by_task[event["task_id"]].append(event)
            if event.get("request_id"):
                by_request[event["request_id"]][event["event"]] = event
        terminal = {e["task_id"]: e for e in ends}
        for tid, sequence in by_task.items():
            if sequence[0]["event"] != "task_start" or sequence[-1]["event"] != "task_end":
                row["issues"].append(f"task {tid} has incomplete event boundaries")
            steps = [e["step"] for e in sequence if e["event"] == "request_sent"]
            if steps != list(range(1, len(steps) + 1)):
                row["issues"].append(f"task {tid} request steps are not consecutive")
        dangling = 0
        for rid, events_by_kind in by_request.items():
            sent = events_by_kind.get("request_sent")
            if not sent:
                row["issues"].append(f"request {rid} has no request_sent")
                continue
            if "request_error" in events_by_kind:
                continue
            if "request_end" not in events_by_kind and terminal.get(sent["task_id"], {}).get("status") in (
                    "timeout", "crash", "cancelled", "canceled"):
                dangling += 1
                continue
            first, end = events_by_kind.get("first_token"), events_by_kind.get("request_end")
            if first is None or end is None or not sent["ts"] <= first["ts"] <= end["ts"]:
                row["issues"].append(f"request {rid} has missing/out-of-order terminal events")
        status = Counter(e["status"] for e in ends)
        start_times = {e["task_id"]: e["ts"] for e in starts}
        if any(e["ts"] < start_times.get(e["task_id"], float("inf")) for e in ends):
            row["issues"].append("task end precedes task start")
        makespan = max((e["ts"] for e in ends), default=0) - min(start_times.values(), default=0)
        row.update(tasks=len(expected_ids), task_start_count=len(starts), task_end_count=len(ends),
                   answered=status["answered"], correct=sum(bool(e.get("correct")) for e in ends),
                   statuses=dict(status), makespan_s=makespan, requests=len(by_request),
                   completed_requests=sum("request_end" in ev for ev in by_request.values()),
                   expected_interrupted_requests=dangling)
        row["valid"] = not row["issues"]
    except (OSError, ValueError, KeyError, TypeError) as error:
        row["issues"].append(f"{type(error).__name__}: {error}")
    return row


def select_fixed_k(rows):
    """Development data only; lexicographic reliability/quality/throughput rule."""
    groups = defaultdict(list)
    for row in rows:
        if not row.get("valid"):
            raise ValueError("Invalid development run cannot select K")
        groups[int(row["k"])].append(row)
    if set(groups) != {1, 2, 4}:
        raise ValueError("Development candidates must be K=1,2,4")
    candidates = []
    for k, repeats in sorted(groups.items()):
        if len(repeats) != 3 or {r["repeat"] for r in repeats} != {0, 1, 2}:
            raise ValueError("Every development K requires the three predeclared repeats")
        tasks = sum(r["tasks"] for r in repeats)
        answered = sum(r["answered"] for r in repeats)
        correct = sum(r["correct"] for r in repeats)
        seconds = sum(r["makespan_s"] for r in repeats)
        candidates.append({"k": k, "runs": len(repeats), "tasks": tasks, "answered": answered,
                           "correct": correct, "answered_rate": answered / tasks,
                           "accuracy": correct / tasks,
                           "correct_tasks_per_min_makespan": correct / seconds * 60 if seconds > 0 else 0})
    winner = max(candidates, key=lambda c: (c["answered_rate"], c["accuracy"],
                                           c["correct_tasks_per_min_makespan"], -c["k"]))
    return {"selected_fixed_k": winner["k"], "selection_rule": SELECTION_RULE, "candidates": candidates}


def heldout_toml(best_k):
    return f'''# Frozen after development selection and before any held-out model request.
name = "aws_phase2_test"
tasks = ["longqa:20-39"]
repeats = 3

[baseline]
k = 4
agents = 16
ctx = 65536
np = 8
cache_ram = 8192
ngl = 99
kv_offload = true
fit = "off"
lv = 4
policy = "fifo"
kv_margin_tokens = 256
kv_headroom = 0.9
"tools.python_delay_s" = 3
"tools.search_delay_s" = 3
"llm.cache_prompt" = true
"agent.max_steps" = 16
"agent.task_timeout_s" = 600
"tasks.response_length" = "short"

[[block]]
name = "HELDOUT"
factor = "same-budget frozen comparison of FIFO K4, development-selected fixed K, and KV admission"
conditions = [
    {{k = 4, policy = "fifo"}},
    {{k = {best_k}, policy = "fifo"}},
    {{k = 4, policy = "kv-admission"}},
]
'''


def port_open(port):
    with socket.socket() as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


class Validation:
    def __init__(self, args):
        self.args, self.output = args, args.out.resolve()
        self.output.mkdir(parents=True, exist_ok=True)
        self.state_path = self.output / "state.json"
        self.state = json.loads(self.state_path.read_text()) if self.state_path.exists() else {
            "created_at_utc": utc_now(), "created_epoch": time.time(), "stages": {}}
        if self.state.get("failure") or self.state.get("summary_error"):
            self.state.setdefault("resume_history", []).append({
                "resumed_at_utc": utc_now(), "failure": self.state.pop("failure", None),
                "summary_error": self.state.pop("summary_error", None)})
        candidates = [time.time() + args.max_hours * 3600]
        if args.deadline_epoch:
            candidates.append(args.deadline_epoch)
        if self.state.get("deadline_epoch"):
            candidates.append(self.state["deadline_epoch"])
        self.state["deadline_epoch"] = min(candidates)
        self.deadline = time.monotonic() + max(0, self.state["deadline_epoch"] - time.time())
        self.core_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in CORE_PATHS}
        self.groups = [ROOT / "logs" / tomllib.loads(p.read_text())["name"] for p in MATRIX_PATHS]
        self.groups.append(ROOT / "logs/aws_phase2_test")
        self.current_stage = "initializing"
        self.freeze_protocol()

    def remaining(self):
        return self.deadline - time.monotonic()

    def status(self, stage, state="running", **details):
        self.current_stage = stage
        value = {"stage": stage, "state": state, "updated_at_utc": utc_now(),
                 "remaining_s": max(0, round(self.remaining(), 1)),
                 "deadline_epoch": self.state["deadline_epoch"], **details}
        write_json(self.output / "status.json", value)
        self.state["last_status"] = value
        write_json(self.state_path, self.state)
        print(json.dumps(value, ensure_ascii=False), flush=True)

    def freeze_protocol(self):
        protocol = {
            "version": 1, "model": self.args.model, "model_sha256": MODEL_SHA256,
            "llama_cpp_build": 11669, "llama_cpp_commit": "620d2225c",
            "matrix_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in MATRIX_PATHS},
            "source_sha256": self.core_hashes,
            "development_tasks": ["longqa:0-9"], "heldout_tasks": ["longqa:20-39"],
            "heldout_task_ids": sorted(task_ids(["longqa:20-39"])),
            "development_task_ids": sorted(task_ids(["longqa:0-9"])),
            "dataset_sha256": sha256(ROOT / "data/longqa_50.jsonl"),
            "selection_rule": SELECTION_RULE, "fixed_k_candidates": [1, 2, 4], "repeats": 3,
            "policy": {"name": "kv-admission", "max_k": 4, "headroom": 0.9, "margin_tokens": 256},
            "matched_test_resources": {"ctx": 65536, "np": 8, "agents": 16, "cache_ram": 8192,
                                       "task_timeout_s": 600, "max_steps": 16, "tool_delay_s": 3},
            "scope": "Closed-loop held-out task comparison; no open-loop arrival claim.",
        }
        assert not set(protocol["development_task_ids"]) & set(protocol["heldout_task_ids"])
        path = self.output / "protocol.json"
        if path.exists():
            old = json.loads(path.read_text())
            if {k: v for k, v in old.items() if k != "frozen_at_utc"} != protocol:
                raise RuntimeError("Frozen protocol/source changed; preserve this run and use a new validation session")
        else:
            write_json(path, {**protocol, "frozen_at_utc": utc_now()})

    def run_command(self, stage, command, max_seconds=None, allow_failure=False, reserve_seconds=90):
        available = self.remaining() - reserve_seconds
        if available <= 5:
            raise DeadlineReached("Insufficient time remains inside the global budget")
        timeout = min(available, max_seconds) if max_seconds is not None else available
        path = self.output / "stages" / (re.sub(r"[^a-zA-Z0-9_.-]", "_", stage) + ".log")
        path.parent.mkdir(exist_ok=True)
        self.status(stage, command=command, log=str(path))
        with path.open("a") as stream:
            offset = stream.tell()
            stream.write(f"\n[{utc_now()}] {shlex.join(command)}\n")
            stream.flush()
            proc = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=True)
            started, last_progress = time.monotonic(), None
            last_progress_at = float("-inf")
            try:
                while proc.poll() is None:
                    if time.monotonic() - started >= timeout:
                        raise DeadlineReached(f"{stage} exceeded its command/global deadline")
                    if stage.startswith("run_") and time.monotonic() - last_progress_at >= 10:
                        last_progress_at = time.monotonic()
                        progress = self.group_progress()
                        if progress != last_progress:
                            self.status(stage, pid=proc.pid, progress=progress, log=str(path))
                            last_progress = progress
                    time.sleep(2)
            except BaseException:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGINT)
                    try:
                        proc.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait(timeout=5)
                raise
        with path.open("rb") as stream:
            stream.seek(offset)
            output = stream.read().decode(errors="replace")
        result = {"returncode": proc.returncode, "elapsed_s": round(time.monotonic() - started, 3),
                  "log": str(path), "output": output}
        if proc.returncode and not allow_failure:
            raise RuntimeError(f"{stage} exited {proc.returncode}; see {path}")
        return result

    def group_progress(self):
        finished, observed, active = 0, 0, []
        for group in self.groups:
            for path in group.glob("*/experiment.json"):
                try:
                    info = json.loads(path.read_text())
                    observed += 1
                    if info.get("finished") and info.get("runner_exit_code") == 0:
                        finished += 1
                    elif not info.get("finished"):
                        events = read_jsonl(path.parent / "events.jsonl") if (path.parent / "events.jsonl").exists() else []
                        active.append({"group": group.name, "run": path.parent.name,
                                       "task_starts": sum(e["event"] == "task_start" for e in events),
                                       "task_ends": sum(e["event"] == "task_end" for e in events)})
                except (OSError, ValueError):
                    pass  # atomically written parent status tolerates an in-flight child write
        return {"finished_runs": finished, "observed_runs": observed, "active": active}

    def run_group(self, matrix):
        spec, runs = expected_runs(matrix)
        expected = task_ids(spec["tasks"])
        group = ROOT / "logs" / spec["name"]
        records = [inspect_run(path, expected, cond["params"], self.core_hashes) for cond, _, path in runs]
        # A runner can exit 0 despite an interrupted/incomplete log; never silently reuse it.
        for record, (_, _, path) in zip(records, runs):
            if record.get("finished") and record.get("runner_exit_code") == 0 and not record["valid"]:
                archived = group / "attempts" / f"{path.name}_validation_rejected_{time.time_ns()}"
                archived.parent.mkdir(parents=True, exist_ok=True)
                path.rename(archived)
                write_json(archived / "validation-rejection.json", record)
                print(json.dumps({"retained_invalid_attempt": str(archived), "issues": record["issues"]}), flush=True)
        if not all(record["valid"] for record in records):
            self.run_command("run_" + spec["name"], [sys.executable, "-m", "scripts.run_experiment",
                "--matrix", str(matrix), "--llama-cmd", self.args.llama_cmd, "--model", self.args.model,
                "--cooldown", str(self.args.cooldown), "--network-timeout", "600",
                "--run-timeout-s", str(self.args.run_timeout_s)])
        records = [inspect_run(path, expected, cond["params"], self.core_hashes) for cond, _, path in runs]
        write_json(self.output / (spec["name"] + "_checks.json"), records)
        invalid = [record for record in records if not record["valid"]]
        if invalid:
            raise RuntimeError(f"{spec['name']}: {len(invalid)} incomplete/invalid experiments; see group checks")
        self.run_command("analyze_" + spec["name"], [sys.executable, "-m", "scripts.analyze", str(group)], max_seconds=120)
        for record, (_, _, path) in zip(records, runs):
            checked = self.run_command("check_" + spec["name"] + "_" + path.name,
                [sys.executable, "-m", "scripts.check_events", str(path)], max_seconds=60)
            text = checked["output"]
            if "PROBLEM:" in text:
                raise RuntimeError(f"Event checker failed for {path}; see {checked['log']}")
        self.state["stages"][spec["name"]] = {"completed_at_utc": utc_now(), "runs": len(records),
                                                "tasks_per_run": len(expected), "statuses": [r["statuses"] for r in records]}
        self.status(spec["name"], "completed", runs=len(records))
        return records

    def runtime_probe(self):
        output = self.output / "runtime_probe.json"
        if output.exists() and json.loads(output.read_text()).get("passed"):
            self.status("runtime_probe", "completed", reused=True)
            return
        if any(port_open(port) for port in (8000, 8080, 8081)):
            raise RuntimeError("Stop experiment/probe servers on ports 8000/8080/8081 before runtime probing")
        cache = self.output / "slot_cache"
        cache.mkdir(exist_ok=True)
        command = shlex.split(self.args.llama_cmd)
        command[0] = os.path.expanduser(command[0])
        command += ["-hf", self.args.model, "--jinja", "--port", "8081", "-c", "32768", "-np", "4",
                    "-kvu", "--metrics", "--cache-ram", "8192", "-ngl", "99", "--fit", "off",
                    "--slot-save-path", str(cache) + os.sep]
        self.status("runtime_probe_start", command=command)
        with (self.output / "runtime_probe_server.log").open("a") as stream:
            proc = subprocess.Popen(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                started = time.monotonic()
                while True:
                    if self.remaining() < 120 or time.monotonic() - started > 90:
                        raise DeadlineReached("Probe server readiness deadline")
                    if proc.poll() is not None:
                        raise RuntimeError(f"Probe server exited {proc.returncode}")
                    try:
                        with urllib.request.urlopen("http://127.0.0.1:8081/health", timeout=2) as response:
                            if json.load(response).get("status") == "ok":
                                break
                    except OSError:
                        pass
                    time.sleep(1)
                self.run_command("runtime_probe", [sys.executable, "-m", "scripts.probe_runtime", "--url",
                    "http://127.0.0.1:8081", "--exercise-cache", "--out", str(output)], max_seconds=180)
            finally:
                if proc.poll() is None:
                    os.killpg(proc.pid, signal.SIGINT)
                    try:
                        proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        os.killpg(proc.pid, signal.SIGKILL)
                        proc.wait(timeout=5)
        self.state["stages"]["runtime_probe"] = {"completed_at_utc": utc_now(), "report": str(output)}
        self.status("runtime_probe", "completed")

    def freeze_selection(self, rows):
        path = self.output / "selection.json"
        matrix = self.output / "heldout.toml"
        evidence = {str(path.relative_to(ROOT)): sha256(path)
                    for run in rows for path in (Path(run["run_dir"]) / "experiment.json", Path(run["run_dir"]) / "events.jsonl")}
        if path.exists():
            selection = json.loads(path.read_text())
            if selection["development_evidence_sha256"] != evidence:
                raise RuntimeError("Development evidence changed after selection was frozen")
            if not matrix.is_file() or sha256(matrix) != selection["heldout_matrix_sha256"]:
                raise RuntimeError("Frozen held-out matrix changed")
            return matrix
        if any(self.groups[-1].glob("*/events.jsonl")):
            raise RuntimeError("Held-out requests exist before a frozen development selection")
        selection = select_fixed_k(rows)
        matrix.write_text(heldout_toml(selection["selected_fixed_k"]))
        selection.update(frozen_at_utc=utc_now(), development_tasks=["longqa:0-9"],
                         heldout_tasks=["longqa:20-39"], policy={"max_k": 4, "headroom": 0.9, "margin_tokens": 256},
                         development_evidence_sha256=evidence, heldout_matrix_sha256=sha256(matrix),
                         protocol_sha256=sha256(self.output / "protocol.json"),
                         development_rows=rows)
        write_json(path, selection)
        self.status("selection", "frozen", selected_fixed_k=selection["selected_fixed_k"])
        return matrix

    def summarize(self, plot):
        if self.remaining() < 10:
            raise DeadlineReached("Global deadline leaves no time for summary; raw logs retained")
        command = [sys.executable, "-m", "scripts.validation_summary", "--validation-root", str(self.output),
                   "--groups", *map(str, self.groups)]
        if plot:
            command.append("--plot")
        self.run_command("summary", command, reserve_seconds=10)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-cmd", default=str(Path.home() / "llama.cpp/build/bin/llama-server"))
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--out", type=Path, default=ROOT / "logs/aws_validation")
    parser.add_argument("--max-hours", type=float, default=5.5)
    parser.add_argument("--deadline-epoch", type=float, help="absolute global deadline; never extended on resume")
    parser.add_argument("--run-timeout-s", type=float, default=1800)
    parser.add_argument("--cooldown", type=float, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not 0 < args.max_hours <= 5.5 or not 0 < args.run_timeout_s <= 1800 or args.cooldown < 0:
        parser.error("Use 0 < max-hours <= 5.5, 0 < run-timeout-s <= 1800, and cooldown >= 0")
    if args.dry_run:
        groups = []
        for matrix in MATRIX_PATHS:
            spec, runs = expected_runs(matrix)
            groups.append({"matrix": str(matrix), "group": spec["name"], "runs": len(runs),
                           "tasks_per_run": len(task_ids(spec["tasks"]))})
        print(json.dumps({"groups": groups, "isolated_probe_port": 8081,
                          "heldout": {"tasks": "longqa:20-39", "repeats": 3, "runs": "6 if selected K=4, otherwise 9",
                                      "ctx": 65536, "policy_headroom": 0.9, "policy_margin_tokens": 256},
                          "selection_rule": SELECTION_RULE, "max_hours": args.max_hours,
                          "deadline_epoch": args.deadline_epoch}, indent=2))
        return
    def interrupted(signum, frame):
        raise DeadlineReached(f"Received signal {signum}; retaining partial artifacts")
    signal.signal(signal.SIGTERM, interrupted)
    validation, failure = None, None
    try:
        validation = Validation(args)
        for matrix in MATRIX_PATHS[:2]:
            validation.run_group(matrix)
        validation.runtime_probe()
        development = validation.run_group(MATRIX_PATHS[2])
        heldout = validation.freeze_selection(development)
        validation.run_group(heldout)
        validation.summarize(plot=True)
        validation.state["completed_at_utc"] = utc_now()
        validation.status("complete", "completed", groups=[str(p) for p in validation.groups])
    except (Exception, KeyboardInterrupt) as error:
        failure = error
        if validation is not None:
            failed_stage = validation.current_stage
            validation.state["failure"] = {"stage": failed_stage, "type": type(error).__name__, "message": str(error)}
            try:
                validation.summarize(plot=False)
            except Exception as summary_error:
                validation.state["summary_error"] = str(summary_error)
            validation.status(failed_stage, "deadline" if isinstance(error, DeadlineReached) else "failed",
                              error=f"{type(error).__name__}: {error}")
        else:
            print(f"Validation initialization failed: {type(error).__name__}: {error}", flush=True)
    if failure is not None:
        raise SystemExit(2 if isinstance(failure, DeadlineReached) else 1)


if __name__ == "__main__":
    main()
