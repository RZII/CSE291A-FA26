import asyncio
import random

from agent.runner import run_batch
from agent.tasks import select_tasks


def fake_tasks(n):
    return [{"id": f"t{i}"} for i in range(n)]


def test_select_tasks():
    tasks = select_tasks(["gsm8k:0-2", "hotpotqa:13"])
    assert [t["type"] for t in tasks] == ["gsm8k"] * 3 + ["hotpotqa"]


def test_every_task_runs_once_with_n_agents_busy():
    active, peak, rng = 0, 0, random.Random(0)

    async def fake_run(task, agent_id):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(rng.uniform(0.001, 0.02))
        active -= 1
        return {"task_id": task["id"], "agent_id": agent_id, "status": "answered"}

    results = asyncio.run(run_batch(fake_tasks(20), 4, fake_run))
    assert sorted(r["task_id"] for r in results) == sorted(f"t{i}" for i in range(20))
    assert peak == 4  # never more than N at once, and all N actually used
    assert {r["agent_id"] for r in results} == {f"agent-{i}" for i in range(4)}


def test_one_crashing_agent_does_not_stop_the_batch():
    async def fake_run(task, agent_id):
        if task["id"] == "t3":
            raise KeyError("bug in agent code")
        return {"task_id": task["id"], "agent_id": agent_id, "status": "answered"}

    results = {r["task_id"]: r for r in asyncio.run(run_batch(fake_tasks(8), 4, fake_run))}
    assert len(results) == 8
    assert results["t3"]["status"] == "crash" and "KeyError" in results["t3"]["error"]
    assert all(r["status"] == "answered" for k, r in results.items() if k != "t3")
