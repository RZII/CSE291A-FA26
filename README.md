# Memory-Aware Agent Edge Serving

CSE 291A (Fall 2026), Option 1. Tool-using LLM agents share one llama.cpp server on a single GPU; a scheduler
between the agents and the server decides which model requests run. Phase 1 measures a FIFO scheduler with
fixed concurrency K and characterizes how concurrency, context growth, idle time between agent steps and
context retention affect memory pressure, latency, throughput and failures.

## Phase 1 results

Everything the Phase 1 report uses is in [`phase1/`](phase1/); start with [`phase1/README.md`](phase1/README.md).

| | |
|---|---|
| Analysis | [`phase1/report.md`](phase1/report.md) |
| Figures | [`phase1/figs/`](phase1/figs/) |
| Per-condition summaries | [`phase1/tables/`](phase1/tables/) |
| Raw runs | [`phase1/data/`](phase1/data/): 129 short-workload runs, 39 long-context runs, transfer microbenchmark |

All results come from AWS g6.2xlarge (NVIDIA L4 24 GB), Qwen3-4B-Instruct-2507 Q8_0 and llama.cpp build 11669
(commit 620d2225c).

## AWS reproduction and validation

The completed AWS validation is published in [`results/aws-validation-20261030/`](results/aws-validation-20261030/README.md):
27 runs, 450 task executions, raw logs, summaries, plots, a request audit, runtime probes, and cost/stop evidence.
See the [runbook](docs/aws_runbook.md) for setup and reproduction, and the [results report](docs/aws_validation_results.md)
for the representative, development, and held-out comparisons. The candidate (`scheduler.py --policy kv-admission`)
improved on FIFO K=4 in this sample, but did not beat the development-selected fixed K=1. Each condition has three
repeats; no significance claim is made.

Extracted raw logs use `work/aws-validation-20261030/logs/`. Absolute paths inside the evidence JSON files describe
the original run; readers do not need to recreate those paths. New memory samples label the corrected slot-token
estimator as `slots.n_prompt_tokens.v2`; compare historical KV peaks (including those in `phase1/`) using the
separately retained legacy estimate rather than mixing the two definitions.

## Repository layout

| Path | Contents |
|---|---|
| `scheduler.py` | FIFO scheduler (default): HTTP proxy with a fixed number K of concurrent model requests; `--policy kv-admission` adds optional KV-budget admission |
| `agent/` | Agent loop, tools (Python, BM25 retrieval, `read_page`), task sets, event logging |
| `data/` | Task sets: GSM8K, HotpotQA (+ retrieval corpus), long documents from LongBench |
| `experiments/` | Experiment matrices: `gpu.toml` (short workload, GPU/CPU placement), `longctx.toml` (long contexts); `phase1.toml` is the original matrix `gpu.toml` builds on; `representative_*.toml`, `phase2_dev.toml`, `aws_smoke.toml` for the AWS validation |
| `scripts/` | Experiment runner, memory/GPU samplers, analysis, plots, transfer benchmark, AWS setup; validation runner, audit and summary (`run_validation.py`, `validation_audit.py`, `validation_summary.py`), runtime probe |
| `results/` | Frozen AWS validation evidence (byte-preserved; see `.gitattributes`) |
| `tests/` | Unit tests (`uv run pytest`; never while an experiment is running, since live tests hit port 8080) |
| `config.toml` | Agent, model and tool settings |
| `docs/` | Usage notes ([`docs/agent_and_experiments.md`](docs/agent_and_experiments.md)), AWS [runbook](docs/aws_runbook.md) and [validation results](docs/aws_validation_results.md), project description, division of work |
| `调度器文件说明.md`, `给1号和3号的接口与日志说明.md` | Scheduler notes and the agent/scheduler interface and log format |

## Reproducing Phase 1

On an AWS GPU instance (we used g6.2xlarge with the Deep Learning Base AMI with Single CUDA, Ubuntu 24.04),
from the repository root:

```bash
bash scripts/aws_setup.sh     # build llama.cpp with CUDA, install uv and dependencies, download the model

# short workload: blocks A-I and GPU/CPU placement T1-T3 (129 runs)
uv run python -m scripts.run_experiment --matrix experiments/gpu.toml \
    --llama-cmd ~/llama.cpp/build/bin/llama-server --cooldown 10
# long-context workload: blocks L1-L3 (39 runs)
uv run python -m scripts.run_experiment --matrix experiments/longctx.toml \
    --llama-cmd ~/llama.cpp/build/bin/llama-server --cooldown 10
# transfer vs re-prefill microbenchmark
uv run --group analysis python -m scripts.bench_transfer --out logs/bench_transfer \
    --placements gpu kv_cpu layers18 --llama-cmd ~/llama.cpp/build/bin/llama-server

uv run python -m scripts.analyze logs/gpu                  # runs.csv, conditions.csv, a table per block
uv run --group analysis python -m scripts.plot logs/gpu    # logs/gpu/plots/*.png
```

Each run starts a fresh llama-server, the scheduler with the run's K, the memory and GPU samplers and the
agent runner, and writes its logs to `logs/<matrix>/<condition>_r<repeat>/`. A run is skipped only after it exited
successfully, so an interrupted matrix can be resumed with the same command; failed or interrupted attempts are kept
under `logs/<matrix>/attempts/`. `--dry-run` lists the runs, `--blocks` selects blocks.

To run the agent or scheduler on their own, see [`docs/agent_and_experiments.md`](docs/agent_and_experiments.md).
