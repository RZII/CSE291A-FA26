# Agent + task set

> This README covers only my part (agent + task set): the llama.cpp server it talks to, and how to run the agent.

## Server

```bash
llama serve -hf bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:Q8_0 --jinja --port 8080 -c 32768 -np 4 -kvu
```

- Model: Qwen3-4B-Instruct-2507 (non-thinking). Qwen publishes no official GGUF for it; this is bartowski's.
  Its embedded chat template is Qwen's Aug-2025 version (Qwen edited it on 2025-09-17); for prompts that end
  in a user/tool message, which is all the agent sends, both versions render the same text.
- `:Q8_0` pins the precision. Without it, `-hf` picks Q4_K_M if the repo has one.
- `-c 32768` must be explicit: the default (`-c 0` = model's 256K) plus `--fit on` sizes the context
  from whatever memory is free at launch, so it would differ run to run.
- `-np 4 -kvu`: 4 slots sharing one 32K KV pool. An explicit `-np` turns unified KV off unless `-kvu` is given.
- `--jinja` renders the model's own chat template; needed for `tools` / tool calls.
- Tested with llama.cpp build `b11540`. Health check: `curl localhost:8080/health`.

## Agent

Requires [uv](https://docs.astral.sh/uv/). Python 3.14 and dependencies are pinned in `.python-version` / `uv.lock`.

```bash
uv sync          # create .venv from uv.lock
uv run pytest    # run tests (live ones are skipped if the server is down)
uv run python -m agent.client   # one streaming request; prints message, timings, TTFT
uv run python -m agent.tasks gsm8k 0      # run one task end to end; prints trajectory + grade
uv run python -m agent.runner --agents 4 gsm8k:0-49 hotpotqa:0-49   # N concurrent agents, closed loop
uv run python -m scripts.check_events logs/run_<time>                # validate a run's event log
uv run python -m scripts.task_summary logs/run_<time> --scheduler-k 2   # one row per task -> task_summary.jsonl
uv run python scheduler.py   # 2号's FIFO scheduler on :8000; then add --set llm.base_url=http://127.0.0.1:8000 to the runner
```

Each run writes `logs/run_<time>/events.jsonl` (one event per line; schema in `agent/events.py`) and
`trajectories/<task_id>.json`. Requests carry `X-Run-ID` / `X-Task-ID` / `X-Agent-ID` / `X-Request-ID` / `X-Step-ID` headers
(the names the scheduler in `scheduler.py` requires).

Task sets (committed): `data/gsm8k_50.jsonl`, `data/hotpotqa_50.jsonl`, `data/hotpotqa_corpus.jsonl`.
GSM8K test and HotpotQA dev-distractor, 50 each, seed 0, sources pinned by commit. Rebuild with
`uv run --group build python scripts/build_tasks.py`.

All settings (server URL, `max_steps`, `temperature`, `seed`, `max_tokens`, tool timeouts / top-k /
per-tool `*_delay_s`) live in `config.toml`.
