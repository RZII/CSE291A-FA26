"""Load config.toml into frozen dataclasses.

A missing or misspelled field raises TypeError at load time instead of
silently falling back to a default. Overrides like "tasks.hotpotqa_prompt=search_again"
replace single fields (the value is read as TOML, else as a plain string).
"""

import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config.toml"


@dataclass(frozen=True)
class LLMConfig:
    base_url: str
    model: str
    temperature: float
    seed: int
    max_tokens: int
    enable_thinking: bool
    cache_prompt: bool
    read_timeout_s: float


@dataclass(frozen=True)
class AgentConfig:
    max_steps: int
    task_timeout_s: float


@dataclass(frozen=True)
class ToolsConfig:
    python_delay_s: float
    python_timeout_s: float
    python_max_output_chars: int
    search_delay_s: float
    search_top_k: int


@dataclass(frozen=True)
class TasksConfig:
    hotpotqa_prompt: str
    hotpotqa_corpus: str


@dataclass(frozen=True)
class Config:
    llm: LLMConfig
    agent: AgentConfig
    tools: ToolsConfig
    tasks: TasksConfig


def apply_override(raw: dict, override: str):
    key, _, value = override.partition("=")
    section, field = key.split(".")
    if field not in raw[section]:
        raise TypeError(f"unknown config field {key!r}")
    try:
        raw[section][field] = tomllib.loads(f"v = {value}")["v"]
    except tomllib.TOMLDecodeError:
        raw[section][field] = value


def load_config(path: str | Path = DEFAULT_PATH, overrides: list[str] = ()) -> Config:
    with open(path, "rb") as f:
        raw = tomllib.load(f)
    for o in overrides:
        apply_override(raw, o)
    return Config(
        llm=LLMConfig(**raw["llm"]), agent=AgentConfig(**raw["agent"]),
        tools=ToolsConfig(**raw["tools"]), tasks=TasksConfig(**raw["tasks"]),
    )


if __name__ == "__main__":
    print(load_config(*sys.argv[1:]))
