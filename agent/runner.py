"""Run many agents at once.

run_task(task, agent_id) is the entry point for other code (e.g. an experiment script).
It builds its own LLM client and tools, so agents share no state; the BM25 index is
the one shared object, and it is read-only.

run_batch() runs a task list with N agents pulling from one shared queue (closed loop:
an agent takes the next task as soon as it finishes one, so N agents stay busy).

    uv run python -m agent.runner --agents 4 gsm8k:0-4 hotpotqa:0-4
    uv run python -m agent.runner hotpotqa:0-49 --set tasks.hotpotqa_prompt=search_again
Writes logs/run_<time>/events.jsonl (format: agent/events.py), config.json (the config
actually used, overrides applied) and trajectories/<task_id>.json (full messages, for debugging).
"""

import argparse
import asyncio
import functools
import json
import random
import time
import traceback
from collections import Counter
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from agent.client import LLMClient
from agent.config import DEFAULT_PATH, Config, load_config
from agent.events import EventLog, Tracer
from agent.loop import run_agent
from agent.tasks import grade, make_tools, select_tasks, system_prompt

LOGS = Path(__file__).resolve().parent.parent / "logs"


async def run_task(task: dict, agent_id: str, cfg: Config | None = None, log: EventLog | None = None) -> dict:
    cfg = cfg or load_config()
    tracer = Tracer(log, agent_id, task["id"])
    client = LLMClient(cfg.llm)
    t_start = time.time()
    tracer.emit("task_start", ts=t_start, task_type=task["type"])
    try:
        res = await run_agent(
            task["question"], system_prompt(task["type"], cfg), make_tools(task["type"], cfg), client,
            cfg.agent.max_steps, tracer,
        )
    except Exception:
        tracer.emit("task_end", status="crash", error=traceback.format_exc())
        raise
    finally:
        await client.aclose()
    g = grade(task, res)
    last = res.step_results[-1].timings if res.step_results else None
    tracer.emit(
        "task_end", status=res.status, steps=res.steps, correct=g["correct"], pred=g["pred"],
        em=g.get("em"), f1=g.get("f1"), error=res.error,
        final_context_tokens=last["cache_n"] + last["prompt_n"] + last["predicted_n"] if last else 0,
    )
    return {
        "task_id": task["id"],
        "agent_id": agent_id,
        "status": res.status,
        "steps": res.steps,
        "grade": g,
        "error": res.error,
        "t_start": t_start,
        "t_end": time.time(),
        "task": task,
        "messages": res.messages,
        "requests": [{k: v for k, v in asdict(r).items() if k != "message"} for r in res.step_results],
    }


async def run_batch(tasks: list[dict], n_agents: int, run_fn=run_task, on_result=None) -> list[dict]:
    queue = list(reversed(tasks))  # pop() from the end = original order
    results = []

    async def agent(agent_id: str):
        while queue:  # no await between this check and pop(), so no two agents get the same task
            task = queue.pop()
            try:
                r = await run_fn(task, agent_id)
            except Exception:  # a bug in one agent must not take down the batch
                r = {"task_id": task["id"], "agent_id": agent_id, "status": "crash", "error": traceback.format_exc()}
            results.append(r)
            if on_result:
                on_result(r)

    await asyncio.gather(*(agent(f"agent-{i}") for i in range(n_agents)))
    return results


def main():
    p = argparse.ArgumentParser()
    p.add_argument("specs", nargs="+", help="e.g. gsm8k:0-49 hotpotqa:0-49")
    p.add_argument("--agents", type=int, default=4)
    p.add_argument("--seed", type=int, default=0, help="shuffle seed for the task order")
    p.add_argument("--config", default=str(DEFAULT_PATH))
    p.add_argument("--set", action="append", default=[], metavar="SECTION.FIELD=VALUE", help="override one config field")
    args = p.parse_args()

    cfg = load_config(args.config, args.set)
    tasks = select_tasks(args.specs)
    random.Random(args.seed).shuffle(tasks)  # interleave short and long tasks
    run_id = f"run_{datetime.now():%Y%m%d-%H%M%S}"
    out = LOGS / run_id
    (out / "trajectories").mkdir(parents=True)
    log = EventLog(out / "events.jsonl", run_id)
    (out / "config.json").write_text(json.dumps({"agents": args.agents, "seed": args.seed, **asdict(cfg)}, indent=2))

    def save(r: dict):
        (out / "trajectories" / f"{r['task_id']}.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
        correct = r.get("grade", {}).get("correct")
        print(f"{r['agent_id']:<8} {r['task_id']:<36} status={r['status']:<9} steps={r.get('steps', '-')} correct={correct}", flush=True)

    t0 = time.time()
    results = asyncio.run(run_batch(tasks, args.agents, functools.partial(run_task, cfg=cfg, log=log), save))
    log.close()
    n_correct = sum(r.get("grade", {}).get("correct", False) for r in results)
    print(f"\n{len(results)} tasks, {args.agents} agents, {time.time() - t0:.1f}s wall | "
          f"status {dict(Counter(r['status'] for r in results))} | correct {n_correct}/{len(results)} | -> {out}")


if __name__ == "__main__":
    main()
