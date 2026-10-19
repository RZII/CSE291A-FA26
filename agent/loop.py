"""Agent loop: ask the model, run the tools it calls, feed the results back, repeat.

`messages` is append-only. Each step appends the assistant message exactly as
the client reassembled it, then one tool message per tool call. Nothing earlier
is ever edited, so the messages sent at step k are a prefix of those sent at
step k+1 (that is what lets llama.cpp reuse its prefix cache).

A step is one LLM request. The task ends when:
  answered   - the reply has no tool_calls; its content is the final answer
  truncated  - finish_reason == "length" (max_tokens hit); content kept for grading
  max_steps  - the model still wants tools at step max_steps; those calls are not run
  error      - the LLM request failed (HTTP error, context overflow, timeout)
Tool failures (bad JSON, unknown tool, exception) never end the task: the error
text goes back to the model as the tool message, and the loop continues.
"""

import asyncio
import json
import sys
from dataclasses import dataclass

import httpx

from agent.client import ChatResult, LLMClient, LLMError
from agent.config import load_config
from agent.events import NO_TRACE, Tracer
from agent.tools import Tool
from agent.tools.python import make_python_tool


@dataclass
class AgentResult:
    status: str  # "answered" | "truncated" | "max_steps" | "error"
    final_answer: str | None
    messages: list[dict]
    step_results: list[ChatResult]  # one per completed LLM request
    error: str | None = None

    @property
    def steps(self) -> int:
        return len(self.step_results)


async def run_tool(tools: dict[str, Tool], call: dict) -> tuple[str, bool]:
    """Execute one tool call -> (output, ok). Any failure becomes an "Error: ..." string for the model.

    ok=False means the call itself failed (unknown tool, bad arguments, exception incl. timeout).
    Code that runs but raises (a traceback in python's output) still counts as ok.
    """
    name = call["function"]["name"]
    if name not in tools:
        return f"Error: unknown tool '{name}'. Available tools: {', '.join(tools)}.", False
    try:
        args = json.loads(call["function"]["arguments"])
    except json.JSONDecodeError as e:
        return f"Error: arguments are not valid JSON ({e}).", False
    if not isinstance(args, dict):
        return "Error: arguments must be a JSON object.", False
    try:
        output = await tools[name].run(**args)
    except Exception as e:
        return f"Error: {type(e).__name__}: {e}", False
    return output, True


async def run_agent(
    question: str,
    system_prompt: str | None,
    tools: list[Tool],
    client: LLMClient,
    max_steps: int,
    tracer: Tracer = NO_TRACE,
) -> AgentResult:
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    tool_map = {t.name: t for t in tools}
    schemas = [t.schema() for t in tools]
    step_results = []

    for step in range(1, max_steps + 1):
        request_id, headers = tracer.new_request(step)
        tracer.emit("request_sent", step, request_id=request_id)
        try:
            r = await client.chat(messages, schemas, headers=headers)
        except (LLMError, httpx.HTTPError) as e:
            error = f"{type(e).__name__}: {e}"
            tracer.emit("request_error", step, request_id=request_id, error=error)
            return AgentResult("error", None, messages, step_results, error=error)
        tracer.emit("first_token", step, ts=r.t_first_chunk, request_id=request_id, ts_first_delta=r.t_first_delta)
        tracer.emit("request_end", step, ts=r.t_end, request_id=request_id,
                    finish_reason=r.finish_reason, timings=r.timings, usage=r.usage)
        step_results.append(r)
        messages.append(r.message)

        if r.finish_reason == "length":
            return AgentResult("truncated", r.message["content"], messages, step_results)
        calls = r.message.get("tool_calls")
        if not calls:
            return AgentResult("answered", r.message["content"], messages, step_results)
        if step == max_steps:
            break  # hit the limit: don't run tools whose results nobody will read

        for call in calls:
            name = call["function"]["name"]
            tracer.emit("tool_start", step, tool=name, call_id=call["id"])
            output, ok = await run_tool(tool_map, call)
            delay = tool_map[name].extra_delay_s if name in tool_map else 0.0
            tracer.emit("tool_end", step, tool=name, call_id=call["id"], ok=ok, output_chars=len(output), delay_s=delay)
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})

    return AgentResult("max_steps", None, messages, step_results)


def print_result(res: AgentResult):
    print("=== messages actually sent/received ===")
    print(json.dumps(res.messages, indent=2, ensure_ascii=False))
    print("\n=== per step ===")
    for i, r in enumerate(res.step_results, 1):
        t = r.timings
        print(f"step {i}: finish={r.finish_reason:<10} cache_n={t['cache_n']:<5} prompt_n={t['prompt_n']:<5} predicted_n={t['predicted_n']}")
    print(f"\nstatus={res.status} steps={res.steps} final_answer={res.final_answer!r}")


# ---- demo: python -m agent.loop [config.toml] ----


DEMO_SYSTEM = "You are a helpful assistant. Use the python tool for any calculation. When you are done, reply with the final answer."
DEMO_QUESTION = (
    "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. "
    "How many clips did Natalia sell altogether in April and May?"
)


async def _demo(config_path: str | None):
    cfg = load_config(config_path) if config_path else load_config()
    client = LLMClient(cfg.llm)
    try:
        tools = [make_python_tool(cfg.tools)]
        res = await run_agent(DEMO_QUESTION, DEMO_SYSTEM, tools, client, cfg.agent.max_steps)
    finally:
        await client.aclose()
    print_result(res)


if __name__ == "__main__":
    asyncio.run(_demo(sys.argv[1] if len(sys.argv) > 1 else None))
