# Phase 1 资料（AWS）

Phase 1 报告需要的实验、数据和分析都在这个文件夹里。所有数据来自 AWS g6.2xlarge（NVIDIA L4）；早先在 Mac 上跑的结果
不再使用，已移到 `archive/mac_phase1/`，可以忽略。

## 先看什么

1. **`report.md`**：完整的实验说明和分析。写报告主要用这一份，里面的数字都来自下面的表。
2. **`figs/`**：报告中各组实验对应的图，可以直接放进报告。
3. **`tables/`**：按条件汇总的数字（CSV），需要查具体数值时看这里。
4. `data/`：每一次运行的原始日志，一般用不到，需要核对或重新分析时再看。

## 文件夹结构

```
phase1/
├── README.md                      本文件
├── report.md                      实验设置 + 结果 + 分析
├── figs/                          图
├── tables/
│   ├── baseline_conditions.csv    基线负载（短任务）43 个条件的汇总
│   ├── longctx_conditions.csv     长上下文 13 个条件的汇总
│   └── transfer_summary.csv       搬运 vs 重算基准（每个长度 3 次的中位数）
└── data/
    ├── gpu/                       基线负载 129 次运行（A–I 组 + T1–T3 组）
    ├── bench_transfer/            搬运基准原始数据（results.jsonl）和服务器日志
    └── longctx/                   长上下文 39 次运行（L1–L3 组）
```

## 图和报告章节的对应

| 图 | 内容 | 报告章节 |
|---|---|---|
| `gpu_block_A.png` | 并发：K × agent 数 | 2.1 |
| `gpu_block_B.png` | KV 预算 4K–32K | 2.2 |
| `gpu_block_C.png` | 检索段数（上下文长度） | 2.3 |
| `gpu_block_D.png` | 工具延迟 0–10 s | 2.4 |
| `gpu_block_E.png` | 主机缓存 / 前缀复用 | 2.5 |
| `gpu_block_F.png` | 回复长度 短 / 混合 / 长 | 2.6 |
| `gpu_block_G.png`、`gpu_block_H.png` | 小预算下的长回复和并发 | 2.7 |
| `gpu_block_I.png` | 小预算下的工具延迟 | 2.8 |
| `gpu_block_T1.png`–`T3.png` | GPU / CPU 放置 | 2.9 |
| `transfer.png` | 搬运 vs 重算 | 3 |
| `longctx_block_L1.png` | 长上下文：并发 × KV 预算 | 4.1 |
| `longctx_timeline_k4_a16_r0.png` | 65K 预算下一次运行的时间线 | 4.2 |
| `longctx_block_L2.png` | 长上下文：主机缓存 / 前缀复用 | 4.3 |
| `longctx_block_L3.png` | 长上下文：工具延迟 | 4.4 |
| `longctx_context_growth.png` | 长上下文任务的上下文增长 | 1.3 |

## 条件名怎么读

条件名（也是 `data/` 下的目录名）只写出和基线不同的参数，最后的 `_r0/_r1/_r2` 是第几次重复：

| 片段 | 含义 |
|---|---|
| `k4` | 调度器并发 K=4 |
| `a8` | 8 个 agent |
| `ctx8192` | KV 预算 8192 token |
| `cram0` | 关闭主机缓存（cache_ram = 0） |
| `cache_prompt0` | 关闭前缀复用 |
| `kvo0` | KV 池放主机内存（--no-kv-offload） |
| `ngl18` | 只有 18 层放显存 |
| `python_delay_s10_search_delay_s10` | 工具延迟 10 s |
| `search_top_k20` | 检索返回 20 段 |
| `response_lengthmixed` / `long` | 混合 / 长回复 |

两部分实验的基线不同：基线负载为 K=2、4 agent、32K；长上下文为 K=4、16 agent、65K。完整定义见
`experiments/gpu.toml` 和 `experiments/longctx.toml`。

## 汇总表中常用的列

`accuracy`（正确率）、`failure_rate`（失败率）、`error` / `timeout`（每次运行平均的溢出失败数 / 超时数）、
`throughput_tasks_per_min`（吞吐量）、`completion_p50_s` / `completion_p95_s`（任务完成时间）、`queue_p95_s`（排队时间）、
`kv_tokens_peak`（KV 峰值 token 数）、`kv_full_events`（KV 池满次数）、`cached_frac`（缓存命中比例）、
`recompute_tokens`（需要重算的 token 数）、`vram_peak_mb`（显存峰值）、`gpu_util_mean`（GPU 利用率）、
`deadline_miss_60s` / `120s`（截止未达率）。带 `_std` 的列是 3 次重复的标准差。每一列的精确定义见 `scripts/analyze.py` 开头。

## 每次运行目录里的文件

| 文件 | 内容 |
|---|---|
| `config.json` | 这次运行的完整配置 |
| `events.jsonl` | agent 事件：任务开始/结束、每次请求、工具调用 |
| `scheduler.jsonl` | 调度器日志：每个请求的排队、转发、完成时间 |
| `llama.log` | llama-server 日志：prefill 耗时、KV 池满、主机缓存存取 |
| `memory.jsonl` / `gpu.jsonl` | 内存和显存采样 |
| `task_summary.jsonl` | 每个任务一行的汇总 |
| `trajectories/` | 每个任务的完整对话 |

## 相关代码（不在本文件夹）

- 实验矩阵：`experiments/gpu.toml`、`experiments/longctx.toml`
- 运行与分析：`scripts/run_experiment.py`、`scripts/analyze.py`、`scripts/plot.py`、`scripts/bench_transfer.py`
- 长上下文任务：`scripts/build_tasks.py`（生成 `data/longqa_50.jsonl`）、`agent/tools/read.py`
- AWS 环境：`scripts/aws_setup.sh`
