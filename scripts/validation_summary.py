"""Summarize AWS validation without discarding failed or incomplete runs.

    python -m scripts.validation_summary --validation-root logs/aws_validation \
        --groups logs/aws_representative_short logs/aws_representative_long \
                 logs/aws_phase2_dev logs/aws_phase2_test --plot

This module uses only the standard library unless --plot is requested. Raw logs
are read without modification; the manifest hashes portable group/run paths.
"""

import argparse
import contextlib
import csv
import hashlib
import io
import json
import math
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scripts import check_events


REQUIRED = ("experiment.json", "config.json", "events.jsonl", "scheduler.jsonl",
            "memory.jsonl", "gpu.jsonl", "llama.log")
RAW_FILES = (*REQUIRED, "runner.out", "scheduler.out")
METRICS = (
    "tasks", "task_start_count", "task_end_count", "request_count",
    "accuracy", "answered_rate", "failure_rate", "correct", "answered", "error", "crash", "timeout",
    "max_steps", "truncated",
    "unfinished_tasks", "runner_wall_s", "task_makespan_s", "completion_p50_s", "completion_p95_s",
    "answered_completion_p50_s", "answered_completion_p95_s", "correct_tasks_per_min",
    "correct_tasks_per_min_runner", "correct_tasks_per_min_makespan", "answered_tasks_per_min_runner",
    "queue_p50_s", "queue_p95_s", "ttft_p50_s", "ttft_p95_s", "policy_estimate_ms_mean",
    "policy_estimate_ms_p95", "policy_wait_ms_p95", "policy_estimate_errors", "policy_oversize_single",
    "queue_expired", "request_errors", "kv_tokens_peak", "kv_tokens_legacy_peak", "kv_full_events",
    "vram_peak_mb", "footprint_peak_mb", "gpu_util_mean", "cached_frac",
)


def number(value):
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) else None


def percentile(values, quantile):
    values = sorted(v for v in values if number(v) is not None)
    if not values:
        return None
    position = (len(values) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    return values[low] + (values[high] - values[low]) * (position - low)


def mean(values):
    values = [v for v in values if number(v) is not None]
    return statistics.mean(values) if values else None


def read_json(path, issues):
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text())
        if not isinstance(value, dict):
            raise ValueError("expected a JSON object")
        return value
    except (OSError, ValueError) as error:
        issues.append(f"{path.name}: {error}")
        return {}


def read_jsonl(path, issues):
    if not path.is_file():
        return []
    rows, bad = [], []
    try:
        with path.open() as stream:
            for line_number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    if not isinstance(row, dict):
                        raise ValueError("expected a JSON object")
                    rows.append(row)
                except ValueError as error:
                    bad.append(f"line {line_number}: {error}")
    except (OSError, UnicodeError) as error:
        issues.append(f"{path.name}: {error}")
    if bad:
        issues.append(f"{path.name}: {len(bad)} malformed rows; " + "; ".join(bad[:3]))
    return rows


def task_count(specs):
    """Count explicit task ranges only; unknown selection syntax stays unknown."""
    total = 0
    for spec in specs or []:
        _, separator, selection = str(spec).partition(":")
        if not separator:
            return None
        try:
            if "-" in selection:
                first, last = map(int, selection.split("-", 1))
                if first < 0 or last < first:
                    return None
                total += last - first + 1
            else:
                if int(selection) < 0:
                    return None
                total += 1
        except ValueError:
            return None
    return total or None


def summarize_run(group, run, matrix):
    issues = []
    missing = [name for name in REQUIRED if not (run / name).is_file() or not (run / name).stat().st_size]
    exp = read_json(run / "experiment.json", issues)
    read_json(run / "config.json", issues)
    condition = exp.get("condition", {})
    name = condition.get("name") or run.name.rsplit("_r", 1)[0]
    planned = matrix.get("conditions", {}).get(name, {})
    params = condition.get("params", planned.get("params", {}))
    if not isinstance(params, dict):
        issues.append("condition.params is not an object")
        params = {}
    expected = task_count(exp.get("tasks") or matrix.get("tasks") or matrix.get("spec", {}).get("tasks"))
    events = read_jsonl(run / "events.jsonl", issues)
    # Use the same structural validator as orchestration. A task can have both
    # boundaries while its request chain is incomplete; such logs must never
    # enter the valid-repeat means or comparison plot after a pipeline failure.
    checker_output = io.StringIO()
    try:
        with contextlib.redirect_stdout(checker_output):
            event_chain_valid = check_events.main(str(run)) == 0
    except Exception as error:
        event_chain_valid = False
        issues.append(f"event chain checker: {type(error).__name__}: {error}")
    if not event_chain_valid:
        details = [line.strip() for line in checker_output.getvalue().splitlines() if "PROBLEM:" in line]
        issues.extend("event chain: " + line for line in details)
        if not details:
            issues.append("event chain checker failed")
    scheduler = read_jsonl(run / "scheduler.jsonl", issues)
    memory = read_jsonl(run / "memory.jsonl", issues)
    gpu = read_jsonl(run / "gpu.jsonl", issues)
    starts, ends = {}, {}
    for event in events:
        kind, task_id = event.get("event"), event.get("task_id")
        if kind in ("task_start", "task_end"):
            target = starts if kind == "task_start" else ends
            if task_id in target:
                issues.append(f"duplicate {kind}: {task_id}")
            target[task_id] = event
    statuses = Counter(str(end.get("status", "unknown")) for end in ends.values())
    unknown_statuses = set(statuses) - {"answered", "error", "crash", "timeout", "max_steps", "truncated"}
    if unknown_statuses:
        issues.append("unknown task statuses: " + ", ".join(sorted(unknown_statuses)))
    denominator = max(expected or 0, len(starts), len(ends)) or None
    durations, answered_durations = [], []
    for task_id, end in ends.items():
        start = starts.get(task_id)
        if not start or number(start.get("ts")) is None or number(end.get("ts")) is None:
            issues.append(f"task without valid paired timestamps: {task_id}")
            continue
        duration = end["ts"] - start["ts"]
        if duration < 0:
            issues.append(f"negative task duration: {task_id}")
            continue
        durations.append(duration)
        if end.get("status") == "answered":
            answered_durations.append(duration)
    start_times = [event["ts"] for event in starts.values() if number(event.get("ts")) is not None]
    end_times = [event["ts"] for event in ends.values() if number(event.get("ts")) is not None]
    makespan = max(end_times) - min(start_times) if start_times and end_times else None
    runner_start, runner_end = exp.get("t_runner_start"), exp.get("t_runner_end")
    wall = runner_end - runner_start if number(runner_start) is not None and number(runner_end) is not None else None
    if wall is not None and wall <= 0:
        issues.append("nonpositive runner wall time")
        wall = None
    sent, first_tokens = {}, {}
    request_ids = set()
    for event in events:
        request_id = event.get("request_id")
        if request_id:
            request_ids.add(request_id)
            if event.get("event") == "request_sent":
                sent[request_id] = event
            if event.get("event") == "first_token":
                first_tokens[request_id] = event
    ttft = [first_tokens[key]["ts"] - event["ts"] for key, event in sent.items()
            if key in first_tokens and number(event.get("ts")) is not None
            and number(first_tokens[key].get("ts")) is not None and first_tokens[key]["ts"] >= event["ts"]]
    correct = sum(bool(event.get("correct")) for event in ends.values()) if events else None
    answered = statuses["answered"] if events else None
    unfinished = max(0, (denominator or 0) - len(ends)) if events else None
    task_log_complete = bool(starts) and set(starts) == set(ends) and (expected is None or len(ends) == expected)
    completed = bool(exp.get("finished")) and exp.get("runner_exit_code") == 0
    timings = [event["timings"] for event in events if event.get("event") == "request_end"
               and isinstance(event.get("timings"), dict)]
    cached = sum(number(timing.get("cache_n")) or 0 for timing in timings)
    prompt = sum(number(timing.get("prompt_n")) or 0 for timing in timings)
    log_path = run / "llama.log"
    kv_full = None
    if log_path.is_file():
        with log_path.open(errors="replace") as stream:
            kv_full = sum("failed to find free space in the KV cache" in line for line in stream)
    gpu_busy = [row.get("gpu_util") for row in gpu if number(row.get("ts")) is not None
                and number(runner_start) is not None and number(runner_end) is not None
                and runner_start <= row["ts"] <= runner_end]
    estimators = sorted({str(row.get("kv_tokens_estimator", "legacy_unlabelled")) for row in memory})
    legacy_values = [r.get("kv_tokens_legacy_prompt_plus_decoded", r.get("kv_tokens_legacy")) for r in memory]
    row = {
        "group": group, "condition": name, "run": run.name, "run_dir": str(run),
        "repeat": condition.get("repeat"), "params": params,
        "experiment_finished": bool(exp.get("finished")), "runner_exit_code": exp.get("runner_exit_code"),
        "experiment_completed": completed, "task_log_complete": task_log_complete,
        "event_chain_valid": event_chain_valid,
        "raw_logs_complete": not missing,
        "run_valid": completed and task_log_complete and event_chain_valid and not missing and not issues,
        "expected_tasks": expected, "tasks": denominator, "task_start_count": len(starts),
        "task_end_count": len(ends), "task_statuses": dict(sorted(statuses.items())),
        "answered": answered, "correct": correct,
        "accuracy": correct / denominator if correct is not None and denominator else None,
        "answered_rate": answered / denominator if answered is not None and denominator else None,
        "failure_rate": (sum(statuses[key] for key in ("error", "crash", "timeout")) + unfinished) / denominator
        if unfinished is not None and denominator else None,
        **{key: statuses[key] if events else None for key in ("error", "crash", "timeout", "max_steps", "truncated")},
        "unfinished_tasks": unfinished, "request_count": len(request_ids),
        "request_errors": sum(event.get("event") == "request_error" for event in events) if events else None,
        "runner_wall_s": wall, "task_makespan_s": makespan,
        "completion_p50_s": percentile(durations, 0.5), "completion_p95_s": percentile(durations, 0.95),
        "answered_completion_p50_s": percentile(answered_durations, 0.5),
        "answered_completion_p95_s": percentile(answered_durations, 0.95),
        "correct_tasks_per_min": correct / wall * 60 if correct is not None and wall else None,
        "correct_tasks_per_min_runner": correct / wall * 60 if correct is not None and wall else None,
        "correct_tasks_per_min_makespan": correct / makespan * 60 if correct is not None and makespan and makespan > 0 else None,
        "answered_tasks_per_min_runner": answered / wall * 60 if answered is not None and wall else None,
        "queue_p50_s": percentile([r.get("queue_ms", 0) / 1000 for r in scheduler if number(r.get("queue_ms")) is not None], 0.5),
        "queue_p95_s": percentile([r.get("queue_ms", 0) / 1000 for r in scheduler if number(r.get("queue_ms")) is not None], 0.95),
        "ttft_p50_s": percentile(ttft, 0.5), "ttft_p95_s": percentile(ttft, 0.95),
        "policy_estimate_ms_mean": mean([r.get("policy_estimate_ms") for r in scheduler]),
        "policy_estimate_ms_p95": percentile([r.get("policy_estimate_ms") for r in scheduler], 0.95),
        "policy_wait_ms_p95": percentile([r.get("policy_wait_ms") for r in scheduler], 0.95),
        "policy_estimate_errors": sum(bool(r.get("policy_estimate_error")) for r in scheduler) if scheduler else None,
        "policy_oversize_single": sum("oversize_single" in str(r.get("policy_reason", "")) for r in scheduler) if scheduler else None,
        "queue_expired": sum(r.get("error") == "queue_expired" for r in scheduler) if scheduler else None,
        "kv_tokens_estimator": ";".join(estimators),
        "kv_tokens_peak": max((r["kv_tokens"] for r in memory if number(r.get("kv_tokens")) is not None), default=None),
        "kv_tokens_legacy_peak": max((value for value in legacy_values if number(value) is not None), default=None),
        "kv_full_events": kv_full,
        "vram_peak_mb": max((r["proc_gpu_mem_mb"] for r in gpu if number(r.get("proc_gpu_mem_mb")) is not None), default=None),
        "footprint_peak_mb": max((r["footprint_mb"] for r in memory if number(r.get("footprint_mb")) is not None), default=None),
        "gpu_util_mean": mean(gpu_busy), "cached_frac": cached / (cached + prompt) if cached + prompt else None,
        "model_sha256": exp.get("model_sha256"), "llama_version": exp.get("llama_version"),
        "missing_files": missing, "issues": issues,
    }
    return row


def discover_group(group):
    issues = []
    if not group.is_dir():
        return {}, [], [f"missing group: {group}"]
    if (group / "experiment.json").is_file() or (group / "events.jsonl").is_file():
        return {}, [group], issues
    matrix = read_json(group / "matrix.json", issues)
    runs = {path for path in group.iterdir() if path.is_dir()
            and (any((path / name).exists() for name in RAW_FILES) or (path / "trajectories").is_dir())}
    repeats = matrix.get("spec", {}).get("repeats", 1)
    if isinstance(repeats, int) and repeats > 0:
        for condition in matrix.get("conditions", {}):
            runs.update(group / f"{condition}_r{repeat}" for repeat in range(repeats))
    return matrix, sorted(runs), issues


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["group"], row["condition"], json.dumps(row["params"], sort_keys=True))
        grouped[key].append(row)
    conditions = []
    for (group, condition, _), repeats in sorted(grouped.items()):
        valid = [row for row in repeats if row["run_valid"]]
        result = {"group": group, "condition": condition, "params": repeats[0]["params"],
                  "repeats": len(repeats), "valid_repeats": len(valid),
                  "experiment_completed_repeats": sum(row["experiment_completed"] for row in repeats),
                  "invalid_repeats": len(repeats) - len(valid),
                  "kv_tokens_estimator": ";".join(sorted({row["kv_tokens_estimator"] for row in valid}))}
        for key in METRICS:
            values = [row[key] for row in valid if number(row.get(key)) is not None]
            result[key] = statistics.mean(values) if values else None
            result[key + "_std"] = statistics.stdev(values) if len(values) > 1 else None
            result[key + "_n"] = len(values)
        conditions.append(result)
    return conditions


def write_csv(path, rows):
    columns = list(dict.fromkeys(key for row in rows for key in row)) or ["group", "condition", "run"]
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True)
                             if isinstance(value, (dict, list)) else value for key, value in row.items()})


def archived_attempts(groups):
    inventory = []
    for group in groups:
        base = group / "attempts"
        paths = {path.parent for name in RAW_FILES for path in base.rglob(name)} if base.is_dir() else set()
        for path in sorted(paths):
            issues = []
            experiment = read_json(path / "experiment.json", issues)
            inventory.append({
                "group": group.name, "path": str(path),
                "archive_path": group.name + "/attempts/" + str(path.relative_to(base)),
                "experiment_finished": bool(experiment.get("finished")),
                "runner_exit_code": experiment.get("runner_exit_code"),
                "experiment_completed": bool(experiment.get("finished")) and experiment.get("runner_exit_code") == 0,
                "missing_files": [name for name in REQUIRED if not (path / name).is_file()],
                "issues": issues,
            })
    return inventory


def raw_manifest(groups, runs, validation_root):
    files = {}
    for group in groups:
        path = group / "matrix.json"
        if path.is_file():
            files[group.name + "/matrix.json"] = path
    for row in runs:
        run = Path(row["run_dir"])
        paths = [run / name for name in RAW_FILES]
        paths.extend(sorted((run / "trajectories").glob("*.json")))
        for path in paths:
            if path.is_file():
                files[f"{row['group']}/{run.name}/{path.relative_to(run)}"] = path
    for group in groups:
        base = group / "attempts"
        if base.is_dir():
            for path in base.rglob("*"):
                if path.is_file() and (path.name in RAW_FILES or path.suffix == ".json" and "trajectories" in path.parts):
                    files[group.name + "/" + str(path.relative_to(group))] = path
    artifacts = {}
    for name in ("protocol.json", "selection.json", "heldout.toml", "runtime_probe.json"):
        path = validation_root / name
        if path.is_file():
            artifacts[validation_root.name + "/" + name] = path
    for pattern in ("*.log", "*.out"):
        for path in validation_root.rglob(pattern):
            if path.is_file() and "probe" in str(path.relative_to(validation_root)).lower():
                artifacts[validation_root.name + "/" + str(path.relative_to(validation_root))] = path

    def hash_entry(name, path):
        before = path.stat()
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        after = path.stat()
        return {"path": name, "size_bytes": after.st_size, "sha256": digest.hexdigest(),
                "changed_during_hash": before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns}

    entries = [hash_entry(name, path) for name, path in sorted(files.items())]
    artifact_entries = [hash_entry(name, path) for name, path in sorted(artifacts.items())]
    return {"algorithm": "sha256", "path_base": "archive parent containing group directories",
            "files": entries, "artifacts": artifact_entries,
            "raw_file_count": len(entries), "artifact_file_count": len(artifact_entries),
            "file_count": len(entries) + len(artifact_entries)}


def plot_conditions(conditions, output):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = [row for row in conditions if "phase2_test" in row["group"] or "heldout" in row["group"]]
    if not rows:
        return []
    labels = [row["condition"] + f"\n(n={row['valid_repeats']})" for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for axis, metric, title in zip(axes, ("answered_rate", "correct_tasks_per_min", "completion_p95_s"),
                                   ("Answered and correct task fractions", "Correct tasks per runner minute", "Task completion p95 seconds")):
        values = [row.get(metric) if row.get(metric) is not None else float("nan") for row in rows]
        errors = [row.get(metric + "_std") or 0 for row in rows]
        axis.bar(range(len(rows)), values, yerr=errors, capsize=4, color="#4c78a8", label="answered" if metric == "answered_rate" else None)
        axis.set_xticks(range(len(rows)), labels, rotation=20, ha="right", fontsize=8)
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.2)
    axes[0].plot(range(len(rows)), [row.get("accuracy") if row.get("accuracy") is not None else float("nan") for row in rows],
                 "k^", label="correct")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend()
    axes[2].plot(range(len(rows)), [row.get("answered_completion_p95_s") if row.get("answered_completion_p95_s") is not None else float("nan") for row in rows],
                 "k^", label="answered only")
    axes[2].legend()
    fig.suptitle("Held-out task comparison; error bars are sample SD across completed, valid repeats")
    fig.tight_layout()
    path = output / "heldout_comparison.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return [path.name]


def summarize_groups(validation_root, groups, plot=False):
    output, groups = Path(validation_root), [Path(group) for group in groups]
    output.mkdir(parents=True, exist_ok=True)
    rows, warnings = [], []
    seen = set()
    for group in groups:
        matrix, paths, problems = discover_group(group)
        warnings.extend(problems)
        for path in paths:
            if path.resolve() in seen:
                continue
            seen.add(path.resolve())
            label = group.parent.name if path == group else group.name
            rows.append(summarize_run(label, path, matrix))
    conditions = aggregate(rows)
    attempts = archived_attempts(groups)
    manifest = raw_manifest(groups, rows, output)
    for entry in manifest["files"] + manifest["artifacts"]:
        if entry["changed_during_hash"]:
            warnings.append("raw file changed during hashing: " + entry["path"])
    plots = []
    if plot:
        try:
            plots = plot_conditions(conditions, output)
            if not plots:
                warnings.append("No held-out group is available for plotting.")
        except Exception as error:
            warnings.append(f"Plot unavailable: {type(error).__name__}: {error}")
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Representative validation and focused held-out policy comparison; open-loop arrivals are outside this run.",
        "metric_definitions": {
            "accuracy": "correct task ends / max(expected tasks, observed unique task starts, task ends)",
            "answered_rate": "answered task ends / the same task denominator",
            "failure_rate": "error + crash + timeout + unfinished tasks / the same task denominator",
            "correct_tasks_per_min": "correct task ends / (t_runner_end - t_runner_start) * 60",
            "correct_tasks_per_min_makespan": "correct task ends / (last task end - first task start) * 60",
            "answered_completion_p95_s": "linear-interpolated p95 of completed status=answered tasks only",
            "policy_estimate_ms_mean": "Mean scheduler wall milliseconds spent on request estimation; missing for logs without this instrumentation.",
            "condition_statistics": "Means and sample SD from run_valid repeats only; invalid runs remain in runs and counts.",
            "kv_tokens_peak": "Runtime slot-token estimate; compare only matching kv_tokens_estimator labels.",
        },
        "groups": [str(group) for group in groups], "run_count": len(rows),
        "valid_run_count": sum(row["run_valid"] for row in rows),
        "experiment_completed_count": sum(row["experiment_completed"] for row in rows),
        "invalid_run_count": sum(not row["run_valid"] for row in rows),
        "archived_attempt_count": len(attempts), "archived_attempts": attempts,
        "warnings": warnings, "plots": plots,
        "manifest_path": "raw_log_manifest.json", "runs": rows, "conditions": conditions,
    }
    write_csv(output / "summary_runs.csv", rows)
    write_csv(output / "summary_conditions.csv", conditions)
    (output / "raw_log_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validation-root", type=Path, default=Path("logs/aws_validation"))
    parser.add_argument("--groups", type=Path, nargs="+", required=True)
    parser.add_argument("--plot", action="store_true")
    arguments = parser.parse_args()
    summary = summarize_groups(arguments.validation_root, arguments.groups, arguments.plot)
    print(json.dumps({key: summary[key] for key in ("run_count", "valid_run_count", "invalid_run_count", "warnings", "plots")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
