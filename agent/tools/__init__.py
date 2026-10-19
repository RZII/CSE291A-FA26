import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict  # JSON schema of the arguments the model must pass
    fn: Callable[..., Awaitable[str]]  # called as await fn(**arguments)
    extra_delay_s: float = 0.0  # config-side only; not part of schema(), so the model never sees it

    def schema(self) -> dict:
        return {
            "type": "function",
            "function": {"name": self.name, "description": self.description, "parameters": self.parameters},
        }

    async def run(self, **arguments) -> str:
        if self.extra_delay_s:
            await asyncio.sleep(self.extra_delay_s)  # non-blocking: other agents keep running meanwhile
        return await self.fn(**arguments)
