# 2026-10-30 AWS 复现与策略验证

本目录保存第 3 号负责的运行环境验证、三个代表条件复现，以及一组开发／留出任务的调度比较。日期采用洛杉矶时间；原始日志中的 UTC 日期为 2026-10-31。

共 **27 轮、450 次任务执行**，27 个运行均通过结构与文件验收。任务错误和超时作为实验结果保留，不表示所有任务都成功或答对。代码基于 `phase1-real-tasks` 的 `bfcfefd3bd8785a4f5c3ac1834bfefaef8a82008` 加本分支改动；实测的 5 个核心源码哈希记录在 `protocol.json` 和每轮 `experiment.json`，与提交前工作树一致。原有 `phase1/` 数据没有改写。

## 先看这些材料

- [运行说明](../../docs/aws_runbook.md)：固定环境、参数、实验命令、日志检查与停机步骤。
- [完整验证报告](../../docs/aws_validation_results.md)：逐轮结果、与原 Phase 1 对照、接口证据、局限及费用估算。
- [逐条件汇总](summary_conditions.csv)、[逐轮汇总](summary_runs.csv) 和 [比较图](heldout_comparison.png)。
- [逐请求审计](audit/requests.csv)：请求关联、取消删失及缺失计时。

## 主要结果

三个代表条件各运行 3 次、每次 20 题：

| 条件 | 作答 / 60 | 答对 / 60 | error | timeout |
|---|---:|---:|---:|---:|
| 短任务，K=2 / N=4 / ctx=32768 | 60 | 51 | 0 | 0 |
| 长任务，K=4 / N=16 / ctx=65536 | 43 | 28 | 5 | 12 |
| 长任务，K=4 / N=16 / ctx=98304 | 60 | 35 | 0 | 0 |

开发集 `longqa:0-9` 为 K=1、2、4 各 3 次；按事先声明的作答率、准确率、合并答对吞吐、较小 K 的顺序选择 K=1。K=1 与 K=2 前两项相同，吞吐点估计仅相差约 1.2%。配置于 UTC 02:26:14.062513 冻结后，才开始不重叠的 `longqa:20-39` 留出集请求。

留出集资源固定为 ctx=65536、8 slots、16 agents、8192 MiB 主机缓存，每种方案各运行 3 次：

| 方案 | 作答 / 60 | 答对 / 60 | timeout | 答对任务 / runner 分钟 | 全任务 p95 秒 |
|---|---:|---:|---:|---:|---:|
| FIFO K=4 | 56 | 25 | 4 | 0.901 ± 0.190 | 554.304 ± 41.084 |
| 开发集选择的 FIFO K=1 | 60 | 30 | 0 | 1.625 ± 0.070 | 357.508 ± 22.567 |
| KV admission，K 上限 4 | 60 | 29 | 0 | 1.408 ± 0.055 | 403.952 ± 73.250 |

表中为每轮指标的均值 ± 样本标准差，非置信区间。候选策略相对 FIFO K=4 的答对吞吐点估计高 56.2%，但比调优 K=1 低 13.4%，没有胜过最强固定基线。本次只有闭环任务比较，尚未覆盖持续到达率或短长任务混合负载，也没有显著性检验。

## 数据文件与证据

| 材料 | 文件 |
|---|---|
| 冻结的完整服务器日志 | `raw-logs.tar.gz`，约 22 MB；展开约 83 MB |
| 结果与指标定义 | `summary.json`、`summary_runs.csv`、`summary_conditions.csv` |
| 原 Phase 1 对照 | `historical_comparison.json` |
| 协议、开发选择与留出矩阵 | `protocol.json`、`selection.json`、`heldout.toml` |
| 真实模型接口探针 | `runtime_probe.json` |
| 每组验收记录 | `checks/*_checks.json` |
| 原始文件清单 | `raw_log_manifest.json`，703 个清单文件 |
| 全量只读请求与采样审计 | `audit/audit.json`、`audit/requests.csv` |
| 候选预留预算核对 | `policy-validation.json` |
| 部署、下载、费用与停机证据 | `provenance/` |
| 本目录文件的 SHA256 | `SHA256SUMS` |

完整原始归档保持下载时的字节内容，SHA256 为 `c25c956ed5c1041ca11c450312b8a9628f7d110939470ae9cc2b8cf2407c63a5`。JSON 中的远端路径、本机路径和历史 IP 是原始观测记录，保留用于来源核对；它们不是队友运行脚本所需的固定路径或当前连接地址。

审计覆盖 450 个任务终态及 450 份轨迹。3036 个请求对应 3034 条调度记录；15 个在途请求因任务超时取消，其中 2 个没有调度耗时。缺失值明确标记，不补 0。另有 86 行 KV 空值，有效 KV 最大采样间隔为 4.140 秒，所有峰值均为采样峰值。新口径 `slots.n_prompt_tokens.v2` 与旧版重复计数口径分开保存，不能直接混用。

## 校验并重新审计

在本目录执行文件校验（macOS）：

```bash
shasum -a 256 -c SHA256SUMS
```

Linux 可使用 `sha256sum -c SHA256SUMS`。随后从仓库根目录展开到被 Git 忽略的 `work/`，并重新执行只读审计：

```bash
mkdir -p work/aws-validation-20261030
tar -xzf results/aws-validation-20261030/raw-logs.tar.gz -C work/aws-validation-20261030
uv run python -m scripts.validation_audit \
  work/aws-validation-20261030/logs/aws_representative_short \
  work/aws-validation-20261030/logs/aws_representative_long \
  work/aws-validation-20261030/logs/aws_phase2_dev \
  work/aws-validation-20261030/logs/aws_phase2_test \
  --out work/aws-validation-20261030/audit
```

预期为 27 runs、0 problems。审计输出与原始运行目录分开；不要覆盖归档内原始文件。继续运行 AWS 实验前，按运行说明设置新的绝对截止和预算守护，不复用本次已经过去的 epoch。

## 费用和运行状态

实例已于洛杉矶时间 10 月 30 日 20:48:19 经 API 核验为 stopped，环境和磁盘保留。本轮计算费估算约 2.68 美元；加此前课程账面 2.30 美元及本次其他费用预留 0.20 美元，保守累计约 5.18 美元，低于本次 10 美元上限。此为会话预算估算，非最终账单；保留磁盘的未来费用另计。详细口径及正常停机证据见 `provenance/cost-estimate.json` 与 `provenance/instance-stopped.json`。
