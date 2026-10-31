# AWS 实验运行与复现说明

这套环境用于第 3 号负责的 llama.cpp 服务、实验执行和原始数据交接。固定模型、运行时、KV 预算和任务配置后，由 `scripts/run_experiment.py` 启停模型、FIFO 调度器、agents 和采样进程。正式比较同时检查任务质量、失败和完整日志，不能只看程序是否退出。

本轮已完成代表 9 轮、开发 9 轮、留出 9 轮，共 450 次任务执行及隔离接口探针，27 轮均通过结构与文件验收。[仓库结果入口](../results/aws-validation-20261030/README.md)包含原始归档、汇总、审计和溯源材料，完整解读见 [AWS 验证报告](aws_validation_results.md)。保守累计费用估算约 5.18 美元，正常 EC2 Stop 后已由 API 确认实例 stopped。

## 从仓库查看与展开原始结果

在仓库根目录执行，原始归档的 SHA256 应为 `c25c956ed5c1041ca11c450312b8a9628f7d110939470ae9cc2b8cf2407c63a5`：

```bash
shasum -a 256 results/aws-validation-20261030/raw-logs.tar.gz
mkdir -p work/aws-validation-20261030
tar -xzf results/aws-validation-20261030/raw-logs.tar.gz \
  -C work/aws-validation-20261030
```

展开后的四组原始运行目录统一位于 `work/aws-validation-20261030/logs/`。已发布的汇总、图和审计无需解压即可从结果入口阅读。JSON、CSV 和原始日志中的路径字段保留原运行记录以维持证据哈希，它们是溯源信息，不要求读者复建原作者本机或服务器的绝对路径。重新分析请输出到 `work/` 的独立目录，保留原始日志不变。

## 已验证的环境

| 项目 | 配置 |
|---|---|
| 课程入口 | <https://awsed.ucsd.edu/>，选择 `CSE291A_FA26_003`，进入 Individual Login Page |
| 实例 | `i-01d734efb54dbd071`，`g6.2xlarge`，`us-west-2a`，Oregon |
| CPU 和主机内存 | 8 vCPU，32 GiB RAM |
| GPU | NVIDIA L4，`nvidia-smi` 报告 23034 MiB，驱动 `595.91.07` |
| 系统镜像 | `ami-06dbbcfbeac7b1d33`，Deep Learning Base AMI with Single CUDA Ubuntu 24.04 `20261016` |
| 系统和 CUDA | Ubuntu 24.04.5 LTS，CUDA 13.2，nvcc `13.2.51` |
| llama.cpp | build `11669`，commit `620d2225c`，GNU 13.3.0，CUDA native 架构编译 |
| 模型 | `bartowski/Qwen_Qwen3-4B-Instruct-2507-GGUF:Q8_0` |
| 模型 SHA256 | `260b5b5b6ad73e44df81a43ea1f5c11c37007b6bac18eb3cd2016e8667c19662` |
| 初始项目版本 | `bfcfefd3bd8785a4f5c3ac1834bfefaef8a82008`，来自 `phase1-real-tasks` |
| 根盘 | 100 GiB 加密 gp3；代码、模型和日志存于 EBS |
| 服务器项目目录 | `/home/ubuntu/cse291a-first-smoke/repo` |
| 模型程序 | `/home/ubuntu/llama.cpp/build/bin/llama-server` |
| SSH 登录 | 使用自己已获授权的密钥，路径由 `SSH_KEY` 指定；私钥不入库、不共享 |

实际版本以每次 `experiment.json`、环境记录和模型哈希为准。仓库首页较早的 `b11540` 说明不能代替这里的 AWS 版本。

首次 4 题试运行于洛杉矶时间 10 月 30 日 17:11:21 完成，原始 UTC 时间为 `2026-10-31T00:11:21Z`。4 个任务均 `answered` 且评分正确，9 次模型请求、2 次 Python 调用和 3 次搜索，事件检查为 0 个问题。日志确认模型全部放入 GPU，CUDA KV 缓冲区为 4608 MiB，模型监听 `127.0.0.1:8080`。

这 4 题的最终答案与原始 20 任务基线一致；其中一道 HotpotQA 本次少检索一步，因此请求和工具次数与原记录的 10 次、6 次不同。它验证了运行链路和文件版本，尚不构成完整实验条件的统计复现。

## 参数与内存预算

| 参数 | 作用 |
|---|---|
| `-c` 或矩阵 `ctx` | 与 `-kvu` 配合时，是所有 slots 共享的 KV 总 token 预算；启动时预分配 |
| `-np` 或 `--parallel` | llama-server 可同时接纳请求的 slots 数；空闲 slot 也可能保留上下文 |
| 调度器 `K` | 最多同时转发到模型的请求数；应不大于可用 slots 数 |
| agents 数 `N` | 闭环任务执行者数量；执行工具时暂时不发模型请求 |
| `--cache-ram` | 主机 prompt cache 容量，单位 MiB；保存被替换出的上下文 |
| 请求 `cache_prompt` | 是否复用已有前缀；关闭主机缓存与关闭前缀复用是两种不同实验 |
| `-ngl 99` | 本模型全部层放入 GPU |
| `--no-kv-offload` | 将整个 KV 池放到主机；它不等同于空闲上下文的选择性保存与恢复 |
| `--fit off` | 保持所给配置，不按空闲显存自动缩小预算 |
| `--metrics` 和 `-lv 4` | 开放运行指标，记录主机缓存保存和加载日志 |

本模型的 KV 预算按每 token 144 KiB 估算：32768 token 为 4.5 GiB，65536 为 9 GiB，98304 为 13.5 GiB。显存还需容纳模型权重和计算缓冲区。`/slots` 中 token 数的汇总是占用估计，不能当作精确的空闲 KV 单元数；池满事件需结合 `llama.log` 核对。

短任务基线为 `K=2, N=4, np=4, ctx=32768`；长任务代表条件为 `K=4, N=16, np=8, ctx=65536/98304`。各条件均使用 8192 MiB 主机缓存、GPU KV、temperature 0、seed 0 和最多 1024 个输出 token。长任务最多 16 步，单任务墙钟上限 600 秒。

## 启动已有实例

1. 在课程入口进入 AWS，区域选择 Oregon，打开上表中的实例并选择启动。使用同一实例可保留模型和编译产物。
2. 等实例进入 running 后，从控制台读取这次分配的公有 IPv4；停止再启动可能改变地址。
3. 在终端填入当前地址及自己已获授权的私钥路径并连接：

```bash
EC2_HOST='填入当前公有IPv4'
SSH_KEY='/path/to/your/authorized-key.pem'
ssh -o IdentitiesOnly=yes -i "$SSH_KEY" "ubuntu@$EC2_HOST"
```

本轮课程角色未开放 `ec2-instance-connect:SendSSHPublicKey`，验证使用 SSH 密钥连接。仓库不包含私钥，也不新增任何实例访问授权；读者使用自己已获授权的课程角色和密钥。课程角色、权限和实例应从课程入口获取，无需在此环境保存长期 AWS API 密钥。

在实例内检查环境并设置本轮六小时关机兜底。之前的 smoke 使用两小时保护；定时关机不跨重启保留，每次启动都需重新设置。本轮编排进程另有 5.5 小时执行上限，为收集日志和停机留出余量：

```bash
sudo shutdown -h +360 'CSE291A runtime budget guard'
sudo shutdown --show
export PATH="$HOME/.local/bin:/usr/local/cuda/bin:$PATH"
cd /home/ubuntu/cse291a-first-smoke/repo
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
~/llama.cpp/build/bin/llama-server --version
uv run python --version
ss -ltnp '( sport = :8000 or sport = :8080 )'
```

已有实例无需重新执行 `scripts/aws_setup.sh`。本次所有实验在同一实例内通过 `agent → 127.0.0.1:8000 → 127.0.0.1:8080` 连接；启动实验前确认两个端口没有遗留服务。实验脚本会自行启动和结束这些服务。

## 部署固定版本代码

仓库需要 GitHub 登录，云端直接匿名 clone 会失败。首次部署使用本机已认证 checkout 生成 `git archive`，通过 SSH 上传；归档不包含 `.git` 和本机凭据。初始归档 SHA256 为 `b8c39b83ac4dda3067d909a65efe89c93cb0b6b4108cadd1de19917db1c480f4`，远端 `source-provenance.txt` 记录对应 commit 和校验值。

更新时为本轮源码生成新归档、记录 SHA256，再上传至新目录，保留上一轮代码和日志。新增但未提交的矩阵或脚本不会进入 `git archive HEAD`；应将实际部署文件纳入归档并记录补丁哈希，或单独上传并记录校验值。不要把本机 GitHub 凭据放进源代码包或云端启动参数。

新机器的基础安装入口仍为 `bash scripts/aws_setup.sh`，它固定 llama.cpp commit，安装 uv、依赖并预下载模型。已完成安装的机器只同步本轮代码和配置。每次源码、运行时或模型变更均需重新记录版本和哈希。

## 三个代表条件

| 条件 | 任务 | K | agents | slots | KV token 预算 |
|---|---|---:|---:|---:|---:|
| 短任务基线 | GSM8K 10 题和 HotpotQA 10 题 | 2 | 4 | 4 | 32768 |
| 长任务压力条件 | LongQA 前 20 题 | 4 | 16 | 8 | 65536 |
| 长任务较充裕条件 | 同一组 LongQA 20 题 | 4 | 16 | 8 | 98304 |

本轮每条件运行 3 次，共 9 轮、180 次任务执行，检查失败类型和缓存行为。矩阵继承 `experiments/gpu.toml`、`experiments/longctx.toml` 的相应基线参数；工具延迟均为额外 3 秒。先用 `--dry-run` 确认短任务为 1 个条件、3 轮，长任务为 2 个条件、6 轮。

短任务条件：

```bash
uv run python -m scripts.run_experiment \
  --matrix experiments/representative_short.toml \
  --llama-cmd "$HOME/llama.cpp/build/bin/llama-server" --cooldown 10
```

两个长任务条件：

```bash
uv run python -m scripts.run_experiment \
  --matrix experiments/representative_long.toml \
  --llama-cmd "$HOME/llama.cpp/build/bin/llama-server" --cooldown 10
```

结果分别写入 `logs/aws_representative_short/` 和 `logs/aws_representative_long/`。`repeat r` 用 seed r 打乱任务顺序。每次条件启动新的 llama 进程，因此进程内 slot KV 和主机 prompt cache 不跨轮保留；没有主动清除 OS 文件页缓存。

### 本轮自动执行顺序与截止

本轮自动流程先等待已经启动的代表条件进程结束并检查退出码为 0，再由 `scripts/run_validation.py` 检查各轮结构和冻结的源码哈希。已经完成且通过验收的代表轮次会复用；不会与现有模型进程同时启动另一套实验。之后依次执行隔离接口探针、开发集固定 K 选择、冻结留出集矩阵、留出集策略对比和统一汇总。

本轮执行绝对截止为 **2026 年 10 月 31 日 06:27:58 UTC**，即洛杉矶时间 **10 月 30 日 23:27:58**，epoch 为 `1793428078`。它早于原定 OS 守护关机截止，为日志收集和停机预留时间。以下入口仅在现有代表条件进程退出后使用；本轮外层守护流程负责完成这一等待：

```bash
uv run --group analysis python -m scripts.run_validation \
  --llama-cmd "$HOME/llama.cpp/build/bin/llama-server" \
  --deadline-epoch 1793428078
```

已保存的截止时间在恢复时只能保留或提前，不能因重启命令、创建新状态文件或重设关机定时器而延长。到截止仍未完成时，保留已完成轮次、未完成日志及归档尝试，在汇总中标明剩余工作，不缩短任务、输出或单任务超时来赶工。

日志归档成功后，服务器安排 15 分钟后提前关机；操作端下载并校验归档后会更早调用 EC2 Stop。无论流程成功、失败还是到达截止，原有绝对守护截止都不能延长。

### 本轮代表条件结果

三个代表条件均已完成 3 次重复，共 9 轮、180 次任务执行；编排器的逐轮结构、源码和模型哈希检查及事件检查全部通过。这里的完成指每个任务都有终态，包含实验中保留的溢出和超时。180 个任务终态与轨迹完整匹配，但有 2 个超时收尾请求没有调度日志，另有请求级计时删失；不能把它表述成所有请求耗时均完整。

| 条件 | 各轮 answered | 各轮 correct | 各轮 error | 各轮 timeout | runner 秒 均值 ± 样本 SD |
|---|---|---|---|---|---|
| 短任务 32K | 20 / 20 / 20 | 17 / 17 / 17 | 0 / 0 / 0 | 0 / 0 / 0 | 32.886 ± 0.780 |
| 长任务 65K | 14 / 16 / 13 | 9 / 10 / 9 | 1 / 4 / 0 | 5 / 0 / 7 | 603.574 ± 10.069 |
| 长任务 98K | 20 / 20 / 20 | 12 / 12 / 11 | 0 / 0 / 0 | 0 / 0 / 0 | 372.370 ± 45.281 |

短任务准确率为 85%，长任务 98K 为 35/60 正确且无 error 或 timeout；这两项与原 Phase 1 的质量和失败趋势一致。长任务 65K 有 5 次 `context_overflow` 和 12 次 600 秒超时，原 Phase 1 相同条件为 19 次超时、无 overflow，失败构成存在差异。本轮 5 次错误都在流式响应内部报告 code 500，代理外层 HTTP 仍为 200，不能只按 HTTP 状态统计模型成功。

请求审计识别到 998 个唯一 request_sent 和 996 条 scheduler 记录。12 个超时任务的最后请求没有 request_end 或 request_error，其中 10 个有 Broken pipe 响应失败记录，2 个未写出 scheduler 行且无 first_token；这 2 条排队和模型耗时保留为缺失，不补 0。内存日志中 35 条 KV 字段为 null，RSS 仍存在；报告峰值均指采样峰值。

代表原始日志位于[已发布归档](../results/aws-validation-20261030/raw-logs.tar.gz)，解压后在 `work/aws-validation-20261030/logs/`；[历史对照 JSON](../results/aws-validation-20261030/historical_comparison.json)可直接查看。新旧逐条件对照、错误证据、API 探针和 KV 口径说明见 [AWS 验证结果](aws_validation_results.md)。每条件仅 3 次重复，表中为样本统计，不构成显著性检验或性能优化收益的结论。

## 日志验收与交接

每个运行目录至少保留：

| 文件 | 用途 |
|---|---|
| `experiment.json` 和 `config.json` | 精确启动命令、条件、模型哈希、版本和退出状态 |
| `events.jsonl` 和 `trajectories/` | 任务状态、请求、工具调用、质量评分和完整轨迹 |
| `scheduler.jsonl` | 请求身份标签、排队与转发耗时、HTTP 状态 |
| `llama.log` | GPU 放置、KV 池满、prefill、缓存保存与加载 |
| `memory.jsonl` 和 `gpu.jsonl` | 主机内存、slot token 估计、显存与 GPU 利用率 |
| `runner.out` 和 `scheduler.out` | 运行诊断和异常 |
| `task_summary.jsonl` | 每个完整任务一行的汇总 |

对解压后的具体运行目录执行，`RUN_DIR` 应替换为实际目录：

```bash
RUN_DIR='work/aws-validation-20261030/logs/aws_representative_short/k2_a4_r0'
uv run python -m scripts.check_events "$RUN_DIR"
```

代表和留出条件每轮应有 20 个 `task_start` 与 20 个 `task_end`，开发条件每轮为 10 个。检查事件和调度日志按 `request_id` 可关联，采样文件非空且覆盖运行时段，`runner_exit_code` 为 0。任务本身的溢出、超时、错误答案都需要保留；它们与实验进程崩溃分开统计。本轮新版 `check_events.py` 遇到结构错误时打印 `PROBLEM` 并以退出码 1 结束；通过时仍应阅读其任务数、问题数和统计输出。预期超时导致的请求中断按事件检查器规则处理。`task_summary` 和 `scripts.analyze` 会写入运行目录，应仅用于新实验或另建的分析副本；本轮已生成的表可直接从结果入口查看。

下载后还可用只读审计器检查跨日志身份、轨迹和采样覆盖。输出必须放在原始运行目录之外：

```bash
uv run python -m scripts.validation_audit \
  work/aws-validation-20261030/logs/aws_representative_short \
  work/aws-validation-20261030/logs/aws_representative_long \
  work/aws-validation-20261030/logs/aws_phase2_dev \
  work/aws-validation-20261030/logs/aws_phase2_test \
  --out work/aws-validation-20261030/audit
```

`audit.json` 和 `requests.csv` 保留每个请求的关联情况。超时取消造成的未完成请求标为 `censored`，缺失阶段耗时保留为空；原有调度观测单独保存，不补零。未解释的缺失、重复、身份或轨迹不符会返回非零退出码。内存峰值均为采样峰值，KV 空值及有效观测间隔同时报告。

`finished` 非空只表示实验脚本到达结束记录，不能代替退出码和日志检查。恢复时应只跳过已完成且退出码为 0、任务日志完整的轮次。对需要重新测量的条件使用新的 `--name`，保留失败原始目录；中断后的 GPU 和主机采样不得与重试混合。旧版脚本只按 `finished` 判断跳过，且未清理重试目录中的 `gpu.jsonl`，部署时需核对本轮修复版本。

`scripts.analyze` 的吞吐是 answered 数除以 makespan，不是答对数除以 makespan。超时会影响尾延迟；报告完成时间时同时保留质量、失败率和超时数，不把提前失败造成的短时长解释为性能提升。

未来实验交接时，在服务器打包整个实验目录，包括顶层 `matrix.json`、汇总表与验证记录；下载并确认压缩包可读后再停机。以下为采集示例，不覆盖仓库已发布的归档：

```bash
tar -czf /home/ubuntu/new-raw-logs.tar.gz \
  logs/aws_representative_short logs/aws_representative_long \
  logs/aws_phase2_dev logs/aws_phase2_test logs/aws_validation
```

```bash
mkdir -p work/aws-validation-20261030
scp -o IdentitiesOnly=yes -i "$SSH_KEY" \
  "ubuntu@$EC2_HOST:/home/ubuntu/new-raw-logs.tar.gz" \
  work/aws-validation-20261030/new-raw-logs.tar.gz
tar -tzf work/aws-validation-20261030/new-raw-logs.tar.gz
```

首次 4 题 smoke 的详细文件仅为作者本地可选诊断资料，未纳入本轮仓库结果；本轮 27 轮的完整原始证据不依赖这些文件。

本轮所有组可从展开的原始日志重新汇总，输出置于独立分析目录：

```bash
uv run --group analysis python -m scripts.validation_summary \
  --validation-root work/aws-validation-20261030/analysis \
  --groups work/aws-validation-20261030/logs/aws_representative_short \
    work/aws-validation-20261030/logs/aws_representative_long \
    work/aws-validation-20261030/logs/aws_phase2_dev \
    work/aws-validation-20261030/logs/aws_phase2_test --plot
```

输出 `summary_runs.csv`、`summary_conditions.csv`、`summary.json` 和 `raw_log_manifest.json`；有留出集结果及 matplotlib 时生成 `heldout_comparison.png`。失败或未完成轮次保留在逐轮表，条件均值只使用日志完整且进程成功完成的重复，并列出有效重复数和样本标准差。`attempts/` 中的历史失败尝试单独列出并参与哈希，不混入正式重复的均值。

统一汇总中的 `correct_tasks_per_min` 和 `correct_tasks_per_min_runner` 使用 runner 墙钟时间作分母，`correct_tasks_per_min_makespan` 使用首个任务开始至最后任务结束的时间。原有 `scripts.analyze` 的同名吞吐指标以 task makespan 为分母；对比表中必须使用一致口径。`answered_completion_p95_s` 只统计 status 为 answered 的任务，完整任务 p95 和 timeout 数仍需同时保留。

哈希清单覆盖原始配置、实验与采样日志、任务轨迹、归档尝试，以及验证协议、固定 K 选择、留出集矩阵和接口探针资料。日志打包后另记录整个压缩包的 SHA256，下载后再次核对。

本轮[全量归档](../results/aws-validation-20261030/raw-logs.tar.gz)的 SHA256 为 `c25c956ed5c1041ca11c450312b8a9628f7d110939470ae9cc2b8cf2407c63a5`。下载后验证 703 个清单文件哈希；只读审计对整个下载目录的 789 个文件作前后哈希比较，均未改变，两者覆盖范围不同。汇总为 27 valid、0 invalid、0 warnings，450 个任务与轨迹匹配。3036 个唯一请求有 3034 条 scheduler 记录，15 条在途请求因超时删失，其中只有前述代表条件 2 条缺 scheduler。全量 86 行 KV null 保留缺失，有效 KV 最大间隔 4.140 秒、GPU 最大 0.560 秒；所有峰值仍为采样峰值。逐请求覆盖和删失见已发布的 [requests.csv](../results/aws-validation-20261030/audit/requests.csv) 与 [audit.json](../results/aws-validation-20261030/audit/audit.json)。

发布前执行完整命令 `.venv/bin/python -m pytest -q -rs`，结果为 86 passed、2 skipped，7.92 秒；跳过的 `tests/test_client.py` 两项需要本机 llama 服务。AWS 真实接口探针和 450 次任务链路另行验收。

## 运行时接口与 Phase 2 实验

先验证本版本实际提供的 `/health`、`/props`、`/slots` 和 `/metrics`。只读探针可在服务器启动后调用：

```bash
curl -fsS http://127.0.0.1:8080/health
curl -fsS http://127.0.0.1:8080/props
curl -fsS http://127.0.0.1:8080/slots
curl -fsS http://127.0.0.1:8080/metrics
```

`/health` 表示进程就绪，`/props` 用于核对模型和 slots 配置，`/slots` 与 `/metrics` 支撑占用和忙碌状态采样。固定版本中 `n_tokens_max` 是已观察到的最大单序列长度，不是共享池占用或空闲量。`/slots.n_prompt_tokens` 的上下文向量已经包含生成 token；本轮采样标记 `slots.n_prompt_tokens.v2`，不再额外叠加 `n_decoded`，同时保留旧公式的审计字段。旧版与新版 KV 峰值不能直接混用。控制策略必须明确采样间隔、缺失值处理和估计误差；不能仅凭 nvidia-smi 的预分配显存推断可接纳的新请求数。

`POST /v1/chat/completions/input_tokens` 使用完整聊天请求体，包括 tools，经与生成相同的解析器计算输入 token，可用于准入预算估计。

缓存保存、恢复或删除等写接口，应先在独立小探针中验证所需启动参数、请求格式、缓存文件位置、成功返回和实际复用效果。本轮探针在代表实验退出后使用单独的 `8081` 端口，设置 `--slot-save-path` 指向项目的 `logs/aws_validation/slot_cache/`；正式实验的 `8080` 服务不开放这个路径。固定版本的 slot 保存写入磁盘，不等于主机 RAM 缓存，也不会自动清空原 slot；erase 清除逻辑上下文，不释放启动时预分配的显存池。忙碌 slot 的写操作等待其空闲后执行。

`--cache-ram` 的自动主机缓存不证明调度器能主动按任务搬运或预取。本轮不宣称有精确空闲 KV 指标、活动请求搬运、预取或运行中修改 `-c`、`-np` 的能力。主动控制能力与保存和工具等待重叠的效果，需要各自证据。

### 本轮接口验证与策略对比

隔离接口探针已通过：health、props、slots、metrics 和含 tools 的输入 token 计数接口均返回 200；计数为 152，与模板渲染后 tokenize 的 152 一致。slot 0 保存前为 18 token，save 后仍为 18，erase 后为 0，restore 后恢复为 18。save 报告写入 2,655,412 字节、耗时 2.572 ms；restore 为 1.395 ms。这是一次小上下文功能检查，不能外推 18K 上下文延迟、PCIe 带宽或整体收益。

实测证据为已发布的 [runtime_probe.json](../results/aws-validation-20261030/runtime_probe.json)，服务器探针日志展开后位于 `work/aws-validation-20261030/logs/aws_validation/runtime_probe_server.log`。开发集固定 K 选择与留出集策略比较均已完成。接口存在、接口返回成功和端到端有收益是三个不同结论。

策略对比保持模型、精度、KV 预算、slots、主机缓存、任务、输出上限、工具延迟和超时一致。固定 K FIFO、调优后的固定 K 基线与新策略分别使用新的日志目录和相同重复 seed。测试任务不能参与策略阈值选择；只有开发集试运行应标注为 pilot，不能作为冻结测试集的最终收益。

本轮开发任务为 `longqa:0-9`，固定 K=1、2、4 各跑 3 次，9 轮均已完成并通过结构验收。预声明选择规则依次最大化作答率、准确率、合并答对吞吐，最后在完全相同时选择较小 K。这里合并答对吞吐为三轮 correct 总数除以三轮 task makespan 总和，再换算为每分钟，与前文使用 runner 时间的代表对照表分母不同。

| 开发集固定 K | answered / 30 | correct / 30 | error / timeout | 合并答对任务每分钟（makespan） |
|---|---:|---:|---:|---:|
| 1 | 30 | 15 | 0 / 0 | 1.84429 |
| 2 | 30 | 15 | 0 / 0 | 1.82214 |
| 4 | 22 | 13 | 8 / 0 | 1.48150 |

K=1 和 K=2 的前两项相同，按第三项点估计选择 K=1；二者吞吐只相差约 1.2%，不构成显著优势的证据。K=4 的 8 个 error 均记录为 `context_overflow`。选择于 **2026-10-31 02:26:14.062513 UTC**（洛杉矶 10 月 30 日 19:26:14）冻结，证据为 [selection.json](../results/aws-validation-20261030/selection.json)，其中记录开发事件与实验元数据的 18 个 SHA256。[冻结矩阵](../results/aws-validation-20261030/heldout.toml)固定比较 FIFO K=4、FIFO K=1 和 KV admission K 上限 4，每种 3 次；留出任务均为 `longqa:20-39`。开发集与测试集任务 ID 不重叠，代表条件使用的 `longqa:0-19` 不替代这个测试集。开发原始日志展开后位于 `work/aws-validation-20261030/logs/aws_phase2_dev/`，完整哈希见验证报告。

KV admission 使用输入 token 数、最多输出 token 数及 256 token 增长余量估算单个请求的预算；同时活跃请求的预算和不超过 `floor(ctx × 0.9)`，并保留 K 上限。它不以空闲 slots 的历史 KV token 估计直接阻塞准入，也不保证每次 KV 分配都成功。策略开销、失败、质量和长任务延迟应与基线一起报告；收益需由本轮数据决定。这是一个聚焦候选策略的留出集对比，尚未覆盖开环到达率或短长任务混合到达，不能作为整个课程 Phase 2 已完成的结论。

留出集三种方案各 3 次已全部完成，以下为每轮指标均值 ± 样本标准差；吞吐使用 runner 时间，与开发集选择的 pooled makespan 吞吐分开：

| 方案 | answered / 60 | correct / 60 | timeout | 答对任务每分钟 | 全任务 p95 秒 |
|---|---:|---:|---:|---:|---:|
| FIFO K=4 | 56 | 25 | 4 | 0.901 ± 0.190 | 554.304 ± 41.084 |
| FIFO K=1 | 60 | 30 | 0 | 1.625 ± 0.070 | 357.508 ± 22.567 |
| KV admission，K 上限 4 | 60 | 29 | 0 | 1.408 ± 0.055 | 403.952 ± 73.250 |

候选相对 FIFO K=4 的点估计为吞吐增加 56.2%、任务 p95 降低 27.1%；相对调优固定 K=1，吞吐低 13.4%、p95 高 13.0%。因此不能声称它胜过最佳固定基线，3 次重复也不支持显著性结论。详细质量、answered p95、TTFT、排队和内存采样表见验证报告与最终 `heldout_comparison.png`。

448 个候选请求的预留均符合 `prompt_tokens + 1024 + 256`，最大活跃数 4，活跃预留峰值 58891 ≤ 58982，无输入估计错误或已知预留越界；辅助 `/slots` 观测有 8 次 timeout，不影响已得到的 reservation。证据见 [策略生效检查](../results/aws-validation-20261030/policy-validation.json)。估计耗时各轮均值的均值为 144.981 ± 47.209 ms，p95 为 1172.135 ± 467.682 ms，包含输入计数和观测接口的负载等待；它与小探针的 slot 保存耗时不同。

正式实验使用 `--queue-capacity 4096 --max-connections 4096 --queue-budget 86400`，流式网络间隔超时 600 秒；任务外层 600 秒超时仍生效，不能按调度器默认的 300 秒队列预算解释结果。

## 费用与停机

课程政策为每组总额度 200 美元、每日 20 美元；本轮用户授权的累计费用上限为 10 美元，包含前面的首次试运行。账单展示可能延迟，不能以门户显示的旧总额作为本次剩余预算的唯一依据。

本实例已核验的按需计算单价为 0.9776 美元每小时，磁盘和公有 IPv4 另计。费用控制使用累计运行时间的保守上界并留出附加费用余量。本轮启动前课程门户显示团队累计 2.30 美元，其中本人实例前次试运行约 0.33 美元；统计有延迟。本轮按完整团队累计数 2.30 美元作保守起点，六小时最多约 5.87 美元计算费，再预留 0.20 美元附加费用，规划上界约 8.37 美元。不能在保护到期后直接延长而不重新核对累计预算。运行前设置关机保护、完成后立即停止，自动定时关机是运行时间保护，不是 AWS 硬性金额预算。

实际本轮从 UTC `2026-10-31 00:56:34` 启动至 `03:41:06.162764` 正常 Stop 请求为 9872.162764 秒，计算费估算为 2.68084 美元；加此前整个课程账面快照 2.30 美元及本次磁盘／IPv4 等预留 0.20 美元，保守累计 **5.18084 美元 < 10 美元**。这是有延迟账单下的预算估算，不是最终收费。正常非休眠停止的实例从 stopping 起不收实例使用费，EBS 仍收费，见 [AWS 实例生命周期说明](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-lifecycle.html)。100 GiB 根盘继续保留，附加费用预留不覆盖其长期保留费用。

API 于 UTC **2026-10-31 03:48:19.054262**（洛杉矶 10 月 30 日 **20:48:19**）观测到本实例 `i-01d734efb54dbd071` 为 stopped，正常 Stop 完成，未强制停止。这是核验时间，不是精确状态切换时刻；未操作队友实例。证据见 [费用估算](../results/aws-validation-20261030/provenance/cost-estimate.json) 与 [停机观测](../results/aws-validation-20261030/provenance/instance-stopped.json)。本地截图和额外交接 zip 未作为仓库复现前提，所有必要材料以[仓库结果入口](../results/aws-validation-20261030/README.md)为准。

停止实例保留 EBS 数据并停止计算计费，磁盘仍按容量计费；终止实例会删除这台机器的根盘。实验交接使用 Stop，完成数据备份并确认资源归属后再单独处理不需要的资源。

正常结束后先停止实验服务、同步数据，再使用控制台 Stop 或服务器 `sudo shutdown -h now`。到控制台确认最终状态为 stopped，并记录时间。若长期停在 stopping，按 AWS 文档检查底层宿主机问题；强制停止后的下一次启动应核对文件系统和环境，保留所有失败与修复记录。首次试运行最终确认 stopped 的时间是洛杉矶 10 月 30 日 17:19:35。

## 资料入口

- [AWS 上的原始 Phase 1 报告](../phase1/report.md)
- [短任务实验矩阵](../experiments/gpu.toml)
- [长任务实验矩阵](../experiments/longctx.toml)
- [一键实验入口](../scripts/run_experiment.py)
- [分析脚本与字段定义](../scripts/analyze.py)
- [课程 AWS 最新额度政策](https://piazza.com/class/muhr8yxvobhn9/post/46)
- [UCSD AWS Educate 入口与连接说明](https://support.ucsd.edu/its?id=kb_article&sysparm_article=KB0032513)
- [AWS 停机故障排查及 stopping 计费说明](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/TroubleshootingInstancesStopping.html)
