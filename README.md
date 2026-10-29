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

All settings (server URL, `max_steps`, `task_timeout_s`, `temperature`, `seed`, `max_tokens`, `cache_prompt`,
tool timeouts / top-k / per-tool `*_delay_s`) live in `config.toml`.

## Experiments

`scripts/run_experiment.py` runs a whole sweep with one command. Per run it starts a fresh llama-server
(cold caches), `scheduler.py` with the run's K, the memory sampler and `agent.runner`, then stops all of
them; everything lands in `logs/<name>/<condition>_r<repeat>/`. Stop any llama-server / scheduler on
ports 8080 / 8000 first.

```bash
uv run python -m scripts.run_experiment --matrix experiments/phase1.toml --dry-run   # list the runs
uv run python -m scripts.run_experiment --matrix experiments/phase1.toml --blocks A  # run block A
uv run python -m scripts.analyze logs/phase1                  # -> runs.csv, conditions.csv, a table per block
uv run --group analysis python -m scripts.plot logs/phase1    # -> logs/phase1/plots/*.png
```

- `experiments/phase1.toml` is the Phase 1 matrix: a baseline and blocks A–E, each varying one factor
  (concurrency, KV budget, context length, tool idle time, context retention). Conditions shared by
  several blocks run once.
- Runs go in shuffled order with a 180 s cooldown between them (`--order`, `--cooldown`), and
  `pmset -g therm` is saved before and after each run, so laptop heat does not line up with one factor.
- A finished run is skipped, so an interrupted experiment (Ctrl+C) can be rerun with the same command.
- Ad-hoc sweeps without a matrix file: `--name x --tasks ... --k 1 2 4 --agents 2 8 --sweep SECTION.FIELD=a,b
  --set SECTION.FIELD=v` (full cartesian product).
- Memory budget: llama-server allocates the KV pool for `-c` tokens at startup (Qwen3-4B: 144 KiB/token,
  so 32K = 4.5 GiB). Contexts pushed out of a slot are copied to a host-RAM prompt cache (`--cache-ram`,
  default 8 GiB) and restored when the agent returns. `memory.jsonl` tracks both: `footprint_mb` (memory
  the server dirtied: KV pool + RAM cache + buffers; RSS also counts the mmapped weights) and `kv_tokens`
  (tokens held by the slots, from `/slots`).
- GPU / CPU placement (discrete GPU): `experiments/gpu.toml` adds the keys `ngl` (layers in VRAM), `kv_offload`
  (false = KV pool in host RAM) and `fit`. Where `nvidia-smi` exists, `gpu.jsonl` records VRAM and GPU
  utilization (`scripts/sample_gpu.py`), and with `lv = 4` the analysis reports every host prompt cache transfer
  (VRAM <-> host RAM; `scripts/cache_transfers.py`). `scripts/bench_transfer.py` measures transfer vs recompute
  time per context length and placement. Set up an AWS g6.2xlarge instance (NVIDIA L4) with `scripts/aws_setup.sh`.
- Long-context supplement: `experiments/longctx.toml` runs task type `longqa` (`data/longqa_50.jsonl`, built by
  `scripts/build_tasks.py longqa` from LongBench): the agent reads a ~10K-word document page by page with the
  `read_page` tool, so contexts grow to ~18K tokens.
- Phase 1 results on AWS (the ones the report uses): `results/aws/{gpu,bench_transfer,longctx}`, analysis in
  `docs/phase1_aws_report.md`, figures in `docs/phase1_aws_figs/`. The earlier Mac runs (`results/phase1/`,
  `docs/phase1_analysis*.md`) are kept for reference only.
- The scheduler now takes `--k`, `--queue-capacity`, `--log`, ... (`python scheduler.py -h`); the
  experiment script gives it a queue large enough that it never rejects, so failures come from the model side.
  Failed requests are logged with a `kind` (`context_overflow`, `timeout`, `queue_full`, ...).
