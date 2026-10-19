"""Item 5: does llama.cpp's prefix cache survive each agent step?

Runs tasks one after another with a single agent. For every request it compares the
server's cache_n with what the previous request left in the cache:

    prev_total = cache_n + prompt_n + predicted_n   of the previous request
    lost       = prev_total - cache_n              tokens computed before, recomputed now

Within a task, `at_break` shows the token of this step's prompt sitting at position
cache_n, i.e. the first token the server could not reuse. Across tasks (step 1), it
shows the common token prefix with the previous task's first prompt (max possible reuse).

    uv run python -m scripts.cache_check gsm8k:0-4 hotpotqa:0-4 hotpotqa:13 hotpotqa:15
Raw results go to logs/item5/<task_id>.json.
"""

import asyncio
import json
import sys
from dataclasses import asdict
from pathlib import Path

import httpx

from agent.client import LLMClient
from agent.config import load_config
from agent.tasks import grade, make_tools, select_tasks, system_prompt
from agent.loop import run_agent

OUT = Path(__file__).resolve().parent.parent / "logs" / "item5"


def prompt_tokens(http: httpx.Client, messages: list[dict], tools: list[dict]) -> list[dict]:
    """Token list ({id, piece}) of the prompt the server builds for these messages."""
    prompt = http.post("/apply-template", json={"messages": messages, "tools": tools}).json()["prompt"]
    return http.post("/tokenize", json={"content": prompt, "with_pieces": True}).json()["tokens"]


def common_prefix(a: list[dict], b: list[dict]) -> int:
    n = 0
    while n < min(len(a), len(b)) and a[n]["id"] == b[n]["id"]:
        n += 1
    return n


async def main(specs: list[str]):
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    http = httpx.Client(base_url=cfg.llm.base_url, timeout=30)
    client = LLMClient(cfg.llm)
    rows, prev_total, prev_first_prompt = [], None, None

    for task in select_tasks(specs):
        kind = task["type"]
        tools = make_tools(kind, cfg)
        res = await run_agent(task["question"], system_prompt(kind, cfg), tools, client, cfg.agent.max_steps)
        g = grade(task, res)
        schemas = [t.schema() for t in tools]
        assistant_at = [i for i, m in enumerate(res.messages) if m["role"] == "assistant"]

        for step, r in enumerate(res.step_results, 1):
            t = r.timings
            toks = prompt_tokens(http, res.messages[: assistant_at[step - 1]], schemas)
            assert len(toks) == t["cache_n"] + t["prompt_n"], "local render != server prompt"
            row = {"task": task["id"], "step": step, **{k: t[k] for k in ("cache_n", "prompt_n", "predicted_n")}}
            if step == 1:
                shared = common_prefix(toks, prev_first_prompt) if prev_first_prompt else 0
                row["note"] = f"shared prefix with prev task's 1st prompt = {shared}"
                prev_first_prompt = toks
            else:
                row["lost"] = prev_total - t["cache_n"]
                row["ratio"] = t["cache_n"] / prev_total
                c = t["cache_n"]
                row["note"] = "at_break: " + "".join(x["piece"] for x in toks[max(0, c - 4) : c]) + " ‖ " + "".join(
                    x["piece"] for x in toks[c : c + 6]
                )
            prev_total = t["cache_n"] + t["prompt_n"] + t["predicted_n"]
            rows.append(row)

        (OUT / f"{task['id']}.json").write_text(json.dumps({
            "task": task, "status": res.status, "grade": g, "messages": res.messages,
            "steps": [{k: v for k, v in asdict(r).items() if k != "message"} for r in res.step_results],
        }, indent=2, ensure_ascii=False))
        print(f"{task['id']:<36} status={res.status:<9} steps={res.steps:<2} correct={g['correct']}", flush=True)

    await client.aclose()
    print(f"\n{'task':<36} {'step':>4} {'cache_n':>7} {'prompt_n':>8} {'pred_n':>6} {'lost':>5} {'ratio':>6}  note")
    for r in rows:
        lost = r.get("lost", "")
        ratio = f"{r['ratio']:.4f}" if "ratio" in r else ""
        note = r["note"].replace("\n", "\\n")
        print(f"{r['task']:<36} {r['step']:>4} {r['cache_n']:>7} {r['prompt_n']:>8} {r['predicted_n']:>6} {lost:>5} {ratio:>6}  {note}")

    within = [r for r in rows if "lost" in r]
    print(f"\nwithin-task steps: {len(within)}, lost values: {sorted({r['lost'] for r in within})}, "
          f"min ratio: {min(r['ratio'] for r in within):.4f}")


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
