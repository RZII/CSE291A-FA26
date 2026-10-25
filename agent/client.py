"""Async streaming client for llama.cpp's /v1/chat/completions.

One LLMClient per agent. chat() streams the response, reassembles the
assistant message from the SSE deltas, and returns it together with the
server's raw timings/usage and client-side wall-clock timestamps.

SSE stream as observed on llama.cpp b11540 (stream_options.include_usage on):
    data: {choices:[{delta:{role:"assistant", content:null}}]}       <- sent with the 1st generated token
    data: {choices:[{delta:{tool_calls:[{index, id, function:{name, arguments:"{"}}]}}]}
    data: {choices:[{delta:{tool_calls:[{index, function:{arguments:"\"code"}}]}}]}   <- argument fragments
    ...
    data: {choices:[{delta:{}, finish_reason:"tool_calls"}]}
    data: {choices:[], timings:{...}, usage:{...}}                    <- only chunk with timings/usage
    data: [DONE]
"""

import asyncio
import json
import sys
import time
from dataclasses import asdict, dataclass

import httpx

from agent.config import LLMConfig, load_config


class LLMError(Exception):
    """Server returned a non-200 status or an error inside the stream."""

    def __init__(self, message: str, status: int | None = None):
        super().__init__(message)
        self.status = status


def error_kind(e: Exception) -> str:
    """Coarse failure class for the event log, so failure rates can be split by cause."""
    if isinstance(e, httpx.TimeoutException):
        return "timeout"
    if isinstance(e, httpx.HTTPError):
        return "connection"
    status = getattr(e, "status", None)
    if status == 400 and "exceed_context_size" in str(e):
        return "context_overflow"
    return {429: "queue_full", 503: "unavailable", 504: "queue_expired"}.get(status, "http" if status else "stream")


@dataclass
class ChatResult:
    message: dict  # assistant message, same shape as a non-streaming response
    finish_reason: str  # "stop" | "tool_calls" | "length"
    timings: dict  # llama.cpp timings, raw (cache_n, prompt_n, predicted_n, ...)
    usage: dict  # OpenAI usage, raw
    t_send: float  # all t_* are time.time(), so they line up with other processes' logs
    t_first_chunk: float  # first SSE chunk (role-only) = server produced its 1st token
    t_first_delta: float | None  # first non-empty content / reasoning / tool_call fragment
    t_end: float  # [DONE] received


class LLMClient:
    def __init__(self, cfg: LLMConfig):
        self.cfg = cfg
        self.http = httpx.AsyncClient(
            base_url=cfg.base_url,
            timeout=httpx.Timeout(connect=5.0, read=cfg.read_timeout_s, write=10.0, pool=None),
        )

    async def aclose(self):
        await self.http.aclose()

    def build_body(self, messages: list[dict], tools: list[dict] | None = None) -> dict:
        body = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature,
            "seed": self.cfg.seed,
            "cache_prompt": self.cfg.cache_prompt,
            "max_tokens": self.cfg.max_tokens,
            "chat_template_kwargs": {"enable_thinking": self.cfg.enable_thinking},
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if tools:
            body["tools"] = tools
        return body

    async def chat(self, messages: list[dict], tools: list[dict] | None = None, headers: dict | None = None) -> ChatResult:
        body = self.build_body(messages, tools)
        content, reasoning = [], []
        tool_calls = {}  # index -> {"id", "type", "function": {"name", "arguments"}}
        finish_reason = timings = usage = None
        t_first_chunk = t_first_delta = None
        done = False

        t_send = time.time()
        async with self.http.stream("POST", "/v1/chat/completions", json=body, headers=headers) as resp:
            if resp.status_code != 200:
                await resp.aread()
                raise LLMError(f"HTTP {resp.status_code}: {resp.text}", resp.status_code)

            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue  # blank separator lines
                now = time.time()
                if t_first_chunk is None:
                    t_first_chunk = now
                data = line.removeprefix("data: ")
                if data == "[DONE]":
                    done = True
                    break

                chunk = json.loads(data)
                if "error" in chunk:
                    raise LLMError(f"error in stream: {chunk['error']}")
                timings = chunk.get("timings", timings)
                usage = chunk.get("usage", usage)

                for choice in chunk["choices"]:  # empty list in the final usage chunk
                    delta = choice["delta"]
                    if delta.get("content"):
                        content.append(delta["content"])
                    if delta.get("reasoning_content"):
                        reasoning.append(delta["reasoning_content"])
                    for tc in delta.get("tool_calls") or []:
                        call = tool_calls.setdefault(
                            tc["index"],
                            {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
                        )
                        if tc.get("id"):
                            call["id"] = tc["id"]
                        fn = tc.get("function") or {}
                        call["function"]["name"] += fn.get("name") or ""
                        call["function"]["arguments"] += fn.get("arguments") or ""
                    if t_first_delta is None and (
                        delta.get("content") or delta.get("reasoning_content") or delta.get("tool_calls")
                    ):
                        t_first_delta = now
                    if choice.get("finish_reason"):
                        finish_reason = choice["finish_reason"]
        t_end = time.time()
        if not done:  # e.g. a proxy in between timed out mid-answer: never treat half an answer as final
            raise LLMError("stream ended before [DONE]")

        message = {"role": "assistant", "content": "".join(content)}
        if reasoning:
            message["reasoning_content"] = "".join(reasoning)
        if tool_calls:
            message["tool_calls"] = [tool_calls[i] for i in sorted(tool_calls)]
        return ChatResult(message, finish_reason, timings, usage, t_send, t_first_chunk, t_first_delta, t_end)


DEMO_MESSAGES = [{"role": "user", "content": "What is 37 * 89 + 15? Use the python tool."}]
DEMO_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "python",
            "description": "Run Python code and return stdout",
            "parameters": {"type": "object", "properties": {"code": {"type": "string"}}, "required": ["code"]},
        },
    }
]


async def _demo(config_path: str | None):
    cfg = load_config(config_path) if config_path else load_config()
    client = LLMClient(cfg.llm)
    try:
        r = await client.chat(DEMO_MESSAGES, DEMO_TOOLS)
    finally:
        await client.aclose()
    print(json.dumps(asdict(r), indent=2))
    ms = lambda t: f"{(t - r.t_send) * 1000:.1f}ms"
    print(f"\nTTFT first_chunk={ms(r.t_first_chunk)}  first_delta={ms(r.t_first_delta)}  total={ms(r.t_end)}")


if __name__ == "__main__":
    asyncio.run(_demo(sys.argv[1] if len(sys.argv) > 1 else None))
