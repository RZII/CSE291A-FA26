"""Event log: one JSON object per line, one file per run (events.jsonl).

Every event has: ts (time.time()), event, run_id, agent_id, task_id, step (null for
task-level events). Event types and their extra fields:

    task_start    task_type
    request_sent  request_id
    first_token   request_id, ts_first_delta   (ts = first SSE chunk = server's 1st token)
    request_end   request_id, finish_reason, timings (raw llama.cpp), usage (raw)
    request_error request_id, error
    tool_start    tool, call_id
    tool_end      tool, call_id, ok, output_chars, delay_s
    task_end      status, steps, correct, pred, em, f1, final_context_tokens, error

Each request also carries X-Agent-Id / X-Task-Id / X-Step / X-Request-Id headers, so a
proxy in front of llama.cpp can log the same request_id. (Not yet aligned with 2号 / 4号.)
"""

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path


class EventLog:
    def __init__(self, path: Path, run_id: str):
        self.run_id = run_id
        self.f = open(path, "a", encoding="utf-8")

    def write(self, event: str, ts: float | None = None, **fields):
        row = {"ts": ts if ts is not None else time.time(), "event": event, "run_id": self.run_id, **fields}
        self.f.write(json.dumps(row, ensure_ascii=False) + "\n")
        self.f.flush()  # a crash leaves only complete lines; `tail -f` works

    def close(self):
        self.f.close()


@dataclass
class Tracer:
    """One task's view of the log: fills in agent_id / task_id. log=None -> events are dropped."""

    log: EventLog | None
    agent_id: str
    task_id: str

    def emit(self, event: str, step: int | None = None, ts: float | None = None, **fields):
        if self.log:
            self.log.write(event, ts, agent_id=self.agent_id, task_id=self.task_id, step=step, **fields)

    def new_request(self, step: int) -> tuple[str, dict]:
        request_id = uuid.uuid4().hex
        headers = {"X-Agent-Id": self.agent_id, "X-Task-Id": self.task_id, "X-Step": str(step), "X-Request-Id": request_id}
        return request_id, headers


NO_TRACE = Tracer(None, "-", "-")
