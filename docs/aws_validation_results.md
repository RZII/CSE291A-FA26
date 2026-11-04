# AWS 复现、运行时与留出集策略验证结果

三个代表条件已完成每条件 3 次重复，共 9 轮、180 次任务执行，原始日志验收和隔离接口探针均通过。短任务的 85% 准确率与长任务 98K 下的零运行失败趋势，与原 Phase 1 相符。长任务 65K 仍表现出明显压力，但本轮出现上下文溢出与超时两类失败，历史相同条件只有超时。

本轮现已完成代表条件 9 轮、开发集 9 轮和冻结留出集 9 轮，共 27 轮、450 次任务执行。最终汇总为 27 个有效运行、0 个无效运行、0 条汇总警告，隔离接口探针通过。KV admission 相比原 FIFO K=4 的作答与尾延迟点估计更好，但未胜过开发集选出的固定 K=1。本轮是一个闭环留出集候选比较，没有覆盖开环到达率或短长混合到达，不能作为整个课程 Phase 2 已完成的结论。保守累计费用估算约 5.18 美元，低于 10 美元；正常 Stop 后已由 API 确认实例 stopped。

所有必要证据均从[仓库结果入口](../results/aws-validation-20261030/README.md)获取。原始归档解压到仓库根目录的 `work/aws-validation-20261030/` 后，日志统一位于 `work/aws-validation-20261030/logs/`；命令见[运行说明](aws_runbook.md)。证据 JSON、CSV 和日志中的绝对路径保持原样以维持哈希，它们描述原运行位置，不是读取或复现实验时必须使用的路径。

## 环境与实验控制

| 项目 | 本轮配置 |
|---|---|
| 实例和区域 | `i-01d734efb54dbd071`，g6.2xlarge，Oregon us-west-2a |
| 硬件 | 8 vCPU，32 GiB RAM，NVIDIA L4 23034 MiB |
| 系统镜像 | `ami-06dbbcfbeac7b1d33`，Ubuntu 24.04 CUDA DLAMI `20261016` |
| 驱动和 CUDA | NVIDIA 595.91.07，CUDA 13.2，nvcc 13.2.51 |
| llama.cpp | build 11669，commit `620d2225c`，CUDA native 编译 |
| 模型 | Qwen3-4B-Instruct-2507，Q8_0 |
| 模型 SHA256 | `260b5b5b6ad73e44df81a43ea1f5c11c37007b6bac18eb3cd2016e8667c19662` |
| 解码和预算 | temperature 0，seed 0，最多 1024 输出 token，单任务上限 600 秒 |
| 缓存和放置 | 全层 GPU，GPU KV，统一共享池，8192 MiB 主机 prompt cache，前缀复用开启，fit off |
| 重复 | 3 次，repeat 0/1/2 分别用同号 seed 打乱任务顺序 |

短任务使用 GSM8K 和 HotpotQA 各 10 题，K=2、4 agents、4 slots、32768 token KV。两个长任务条件都使用 `longqa:0-19`，K=4、16 agents、8 slots、最多 16 步，KV 分别为 65536 和 98304 token。工具额外等待均为 3 秒。

本轮代表条件的运行记录从洛杉矶 2026 年 10 月 30 日 18:00:40 至 18:53:34；原始时间为 UTC `2026-10-31T01:00:40Z` 至 `01:53:34Z`。每个条件启动新的 llama 进程，进程内 KV 和主机 prompt cache 不跨轮保留；没有主动清除 OS 文件页缓存。模型、运行时、任务和主要资源参数与历史基线对齐，实验框架增加了恢复与验收功能，并修正 KV 采样口径，不能把两轮当作全部源码逐字相同的实验。

## 逐轮任务结果

每轮都选定 20 个任务，下表按 repeat 0、1、2 排列。`answered` 表示给出答案，`correct` 表示评分正确；error 和 timeout 是任务终态，不等同于整个实验进程崩溃。

| 条件 | repeat | answered | correct | error | timeout | runner 秒 |
|---|---:|---:|---:|---:|---:|---:|
| 短任务 32K | 0 | 20 | 17 | 0 | 0 | 33.621 |
| 短任务 32K | 1 | 20 | 17 | 0 | 0 | 32.970 |
| 短任务 32K | 2 | 20 | 17 | 0 | 0 | 32.067 |
| 长任务 65K | 0 | 14 | 9 | 1 | 5 | 603.374 |
| 长任务 65K | 1 | 16 | 10 | 4 | 0 | 593.606 |
| 长任务 65K | 2 | 13 | 9 | 0 | 7 | 613.741 |
| 长任务 98K | 0 | 20 | 12 | 0 | 0 | 345.776 |
| 长任务 98K | 1 | 20 | 12 | 0 | 0 | 346.679 |
| 长任务 98K | 2 | 20 | 11 | 0 | 0 | 424.653 |

九轮均有 20 个任务开始和 20 个任务终态、`runner_exit_code=0`、匹配的模型版本与哈希，以及所需日志和非空采样文件，编排器结构检查和事件检查通过。独立只读汇总也得到 9 个有效运行，没有必需日志文件缺失或格式错误。180 个任务终态与 180 条轨迹完整匹配；请求级计时存在下述删失，不能把结构验收通过解释为全部请求计时完整。三类条件均未出现 crash、max_steps 或 truncated 终态。

## 与原 Phase 1 对照

下面各值为 3 次独立运行的均值 ± 样本标准差；不是置信区间，也没有进行显著性检验。runner 时间是 `t_runner_end - t_runner_start`，不含编译和服务启动。答对吞吐是每轮 correct 除以该轮 runner 时间，再换算为每分钟，表中再对三轮取均值。任务 p95 从该轮任务的开始到终态时间作线性插值计算；answered p95 仅取 status 为 answered 的任务。

| 条件 | 来源 | runner 秒 | 答对任务每分钟 | 全任务 p95 秒 | answered p95 秒 | 采样显存峰值 MiB |
|---|---|---:|---:|---:|---:|---:|
| 短任务 32K | 原 Phase 1 | 35.108 ± 2.775 | 29.178 ± 2.361 | 9.757 ± 1.030 | 9.757 ± 1.030 | 9028.0 ± 0.0 |
| 短任务 32K | 本轮 | 32.886 ± 0.780 | 31.028 ± 0.739 | 8.931 ± 0.342 | 8.931 ± 0.342 | 9022.7 ± 4.6 |
| 长任务 65K | 原 Phase 1 | 615.136 ± 7.239 | 0.943 ± 0.057 | 600.003 ± 0.002 | 160.242 ± 61.448 | 13668.0 ± 0.0 |
| 长任务 65K | 本轮 | 603.574 ± 10.069 | 0.929 ± 0.072 | 590.900 ± 15.765 | 299.942 ± 245.681 | 13668.0 ± 0.0 |
| 长任务 98K | 原 Phase 1 | 374.660 ± 47.200 | 1.882 ± 0.176 | 363.996 ± 47.984 | 363.996 ± 47.984 | 18308.0 ± 0.0 |
| 长任务 98K | 本轮 | 372.370 ± 45.281 | 1.904 ± 0.303 | 366.265 ± 50.245 | 366.265 ± 50.245 | 18308.0 ± 0.0 |

短任务两轮均为每轮 17/20 正确。长任务 98K 的两轮总正确数均为 35/60，且全部 60 个任务均作答，均无 error 或 timeout。运行时间落在相近范围，支持这两个代表条件的总体趋势得到复现；这里没有策略优化对照，不能把时间上的小幅变化解释为优化收益。

长任务 65K 本轮为 43/60 作答、28/60 正确、17/60 失败；原 Phase 1 为 41/60 作答、29/60 正确、19/60 失败。总体仍显示相同预算下的失败压力，但任务轨迹、正确数和失败类型没有逐轮复现。600 秒超时会把尾部时间钉在截止附近；本轮 repeat 1 的 4 个任务提前因溢出失败、没有 timeout，也不能因此宣称更快。answered p95 的大幅波动还受到哪些长任务成功进入该子集的影响。

## 65K 条件的错误和超时证据

| repeat | context_overflow | timeout | KV 池满日志次数 |
|---|---:|---:|---:|
| 0 | 1 | 5 | 43 |
| 1 | 4 | 0 | 51 |
| 2 | 0 | 7 | 0 |

五个 error 在 agent 的 `request_error` 中均为 `kind=context_overflow`。repeat 0 发生在任务第 12 步，repeat 1 发生在第 14、14、15、15 步。模型在流式响应内部返回：

```text
code: 500
message: Context size has been exceeded.
type: server_error
```

这些请求的 `scheduler.jsonl` 外层 `http_status` 均为 200，`error` 为空且 `response_sent=true`。因此 HTTP 传输成功不能代替模型请求成功；失败应以 agent 解析的流式错误与任务终态计算。十二个 timeout 的任务错误原文均为 `task exceeded 600s`，与上下文溢出分开计数。

前两轮同时出现 KV 空间分配失败日志，与上下文压力一致；第三轮有超时但没有该类池满日志。这里没有独立隔离 allocator 碎片、上下文替换、请求交错或重复 prefill 等机制，不能把它们指定为唯一根因，也不能把这五个错误写成物理显存 OOM。

## 请求计时删失与采样覆盖

独立请求审计得到 998 个唯一 `request_sent` 和 996 条 scheduler 记录，没有重复请求 ID 或身份错配。12 个没有 `request_end` 或 `request_error` 的请求全部属于 65K 的超时任务，分别为 repeat 0 的 5 个和 repeat 2 的 7 个。

其中 10 个保留了 Broken pipe 响应失败的 scheduler 记录；另 2 个来自 repeat 2，未写出 scheduler 行，也没有 `first_token`。该轮收尾耗时 35.50 秒，日志停在开始排空队列，没有排空完成记录；结合实验脚本的限时进程回收逻辑，符合收尾取消删失的情形。这 2 条的 queue 和 model 时间无法从已有数据恢复，应在请求清单中标为删失，不填 0，也不从总体分母中静默移除。仅基于匹配记录计算的排队和模型耗时统计，应同时报告该覆盖范围。

任务级开始与终态记录仍覆盖全部 180 次执行，因此上表任务质量、runner 时间和任务完成时间可从完整任务边界计算。它们不能替代删失请求的精细阶段耗时。

内存采样有 35 行 KV 字段为 null，但 RSS 仍有记录；有效 KV 样本的最大间隔为 4.079 秒，GPU 样本最大间隔为 0.560 秒。所有内存、显存和 KV 峰值均为采样峰值，缺失 KV 不按 0 处理。

最终全量审计扩展至 27 轮：450 个任务开始、450 个终态和 450 个轨迹完整匹配，审计结构问题为 0。3036 个唯一 `request_sent` 对应 3034 条 scheduler 记录，无重复请求、重复调度行或无来源调度行；缺失的仍为上述代表条件的 2 条。共 15 条没有请求终态的在途请求标记为超时删失，其中代表条件 12 条、留出 FIFO K=4 repeat 2 有 3 条。全实验有 16 个任务 timeout，任务超时数与当时仍在途的请求数不是同一指标，不能混用。

全量日志有 86 行 KV null（代表 35、开发 26、留出 25），有效 KV 样本最大间隔为 4.140 秒，GPU 为 0.560 秒。[requests.csv](../results/aws-validation-20261030/audit/requests.csv)为逐请求清单，删失请求的可比较阶段耗时保留为空；[audit.json](../results/aws-validation-20261030/audit/audit.json)保留原始可观察字段和覆盖统计，不补齐不存在的时长。

## KV 采样口径修正

本版本 `/slots.n_prompt_tokens` 的上下文向量已包含生成 token。旧采样器再叠加 `next_token.n_decoded`，会重复计数。本轮使用 `slots.n_prompt_tokens.v2`，并保留 `kv_tokens_legacy_prompt_plus_decoded` 审计旧公式。

| 条件 | 原 Phase 1 旧 v1 采样峰值均值 | 本轮 v2 采样峰值均值 | 本轮旧公式采样峰值均值 |
|---|---:|---:|---:|
| 短任务 32K | 3379.0 | 2995.3 | 3012.0 |
| 长任务 65K | 65118.0 | 66355.3 | 66411.0 |
| 长任务 98K | 70347.7 | 69396.0 | 69481.7 |

原 Phase 1 的 v1 与本轮 v2 不能直接用于判断占用增减。即使修正重复计数，slot 上下文向量仍不是精确已分配 KV 单元；例如本轮 65K repeat 1 的估计峰值为 67522，超过配置的 65536。这不表示 GPU 物理内存多分配了这些 token，也不能据此计算精确空闲 KV。应同时保留估计器标签、启动预算、显存峰值和运行时错误信号。

`/metrics` 的 `n_tokens_max` 是已观察到的最大单序列长度，也不能当作整个共享池的占用或余量。本轮候选准入策略使用请求输入和输出预留估计，不以空闲 slot 历史上下文数直接充当精确空闲池指标。

## 实测运行时接口

代表条件退出后，探针在独立端口 `8081`、32K 共享 KV、4 slots 上运行，启用 `--metrics` 和 `--slot-save-path`，缓存文件位于项目的 `logs/aws_validation/slot_cache/`。它不与正式 GPU 条件并行。

| 检查 | 观测结果 |
|---|---|
| `GET /health` | HTTP 200，status 为 ok |
| `GET /props` | Q8_0、4 slots、上下文配置 32768，与预期模型路径一致 |
| `GET /slots` | HTTP 200，可观察 slot 状态；初始空 slot 的上下文字段未报告 |
| `GET /slots?fail_on_no_slot=1` | 空闲状态下 HTTP 200；没有据此测试满载行为 |
| `GET /metrics` | HTTP 200，可读处理请求和历史计数；没有精确空闲共享池指标 |
| 含 tools 的 `input_tokens` | 152，与 apply-template 后 tokenize 的 152 一致 |
| slot save erase restore | 18 token → save 后仍 18 → erase 后 0 → restore 后 18，最后再次 erase |

slot save 报告 `n_saved=18`、`n_written=2655412`、`save_ms=2.572`；restore 报告 `n_restored=18`、`n_read=2655412`、`restore_ms=1.395`。save 不清空来源 slot，erase 清空逻辑上下文，restore 恢复了所保存的上下文长度。这是一次约 2.6 MB 的小上下文功能性往返检查，没有测量 18K token 的延迟、持久化写盘带宽、独立 PCIe 成本、并发收益或保存与工具等待的重叠。

这些 slot 接口的保存是磁盘序列化，与 `--cache-ram` 的自动主机内存 prompt cache 不同；erase 也不会释放启动时预分配的 CUDA KV 缓冲区。没有据此验证活动请求的主动 offload、prefetch，或运行中更改 `-c`、`-np`。

## 开发集固定 K 选择与冻结证据

开发任务为 `longqa:0-9`，K=1、2、4 各运行 3 次，共 9 轮、90 次任务执行。资源固定为 65536 token KV、8 slots、16 agents、8192 MiB 主机缓存、最多 16 步、工具等待 3 秒和单任务超时 600 秒；模型与代表条件相同。九轮均有 10 个任务开始与终态，`runner_exit_code=0`，冻结选择记录中的结构检查全部有效。

| 固定 K | 各轮 answered | 各轮 correct | 各轮 error | timeout 总数 | 合并答对任务每分钟（makespan） |
|---|---|---|---|---:|---:|
| 1 | 10 / 10 / 10 | 5 / 5 / 5 | 0 / 0 / 0 | 0 | 1.84429 |
| 2 | 10 / 10 / 10 | 5 / 5 / 5 | 0 / 0 / 0 | 0 | 1.82214 |
| 4 | 6 / 6 / 10 | 4 / 4 / 5 | 4 / 4 / 0 | 0 | 1.48150 |

K=1 与 K=2 均为 30/30 作答、15/30 正确；K=4 为 22/30 作答、13/30 正确，8 个 error 均在事件中记录为 `context_overflow`。本表吞吐按三轮正确任务总数除以三轮 task makespan 总和，再换算为每分钟，不是前文代表条件表的 runner 时间分母，也不是三轮吞吐的简单平均。

预声明选择规则按作答率、准确率、合并答对吞吐的顺序降序比较，完全相同时选较小 K。K=1 和 K=2 的前两项相同，K=1 依第三项点估计被选择，二者吞吐仅相差约 1.2%。这是小开发集上按规则得到的配置选择，没有显著性检验，不能解释为 K=1 具有稳定或普遍优势，更不能代替留出集上的策略结论。

实验协议于 **2026-10-31 01:53:39.040427 UTC** 冻结，固定 K 选择于 **02:26:14.062513 UTC**（洛杉矶 10 月 30 日 19:26:14）冻结，在留出模型请求开始前生成 `heldout.toml`。该矩阵固定使用 `longqa:20-39`，比较 FIFO K=4、开发集选出的 FIFO K=1、KV admission K 上限 4，各 3 次；候选策略的 headroom=0.9、margin=256 token。开发任务 ID 与留出任务 ID 不重叠，不能再根据留出结果改变阈值或固定 K。

| 冻结材料 | SHA256 |
|---|---|
| `protocol.json` | `40c7a2a205f11081b023ace30554ef14be22f4d2b8e6501a43321667fbc64618` |
| `selection.json` | `266373d94dd5608d24535773f4cbfd0e39258380efb2f4bbc412a65243f96ff1` |
| `heldout.toml` | `0adac2a42d68f1a2e3069410a3d4d0b034d0a62fd4781b357ca2b6a7e4e66cad` |

本机只读核对了 `selection.json` 引用的 18 个开发事件／实验元数据文件哈希，并确认协议及留出矩阵哈希与选择记录一致。最终下载核验也确认选择冻结早于所有留出模型请求。

## 冻结留出集结果

留出任务为 `longqa:20-39`，三种方案各 3 次重复，每轮 20 题，共 180 次执行。资源、任务、工具延迟、输出上限和超时保持一致，只改变调度策略或其 K。各条件均完整完成 3 次，error、crash、max_steps 和 truncated 均为 0。

| 方案 | 各轮 answered | 各轮 correct | 各轮 timeout | 作答率 | 准确率 |
|---|---|---|---|---:|---:|
| FIFO K=4 | 20 / 20 / 16 | 9 / 9 / 7 | 0 / 0 / 4 | 56/60，93.3% | 25/60，41.7% |
| 开发集选出的 FIFO K=1 | 20 / 20 / 20 | 10 / 10 / 10 | 0 / 0 / 0 | 60/60，100% | 30/60，50.0% |
| KV admission，K 上限 4 | 20 / 20 / 20 | 9 / 11 / 9 | 0 / 0 / 0 | 60/60，100% | 29/60，48.3% |

下表为各轮指标的均值 ± 样本标准差，答对吞吐以 runner 时间作分母；p95 的统计方法与代表条件表相同。任务 p95 包含超时任务，answered p95 单独排除未作答终态。

| 方案 | runner 秒 | 答对任务每分钟 | 全任务 p50 秒 | 全任务 p95 秒 | answered p95 秒 |
|---|---:|---:|---:|---:|---:|
| FIFO K=4 | 561.518 ± 47.768 | 0.901 ± 0.190 | 91.818 ± 16.241 | 554.304 ± 41.084 | 535.737 ± 13.292 |
| FIFO K=1 | 369.709 ± 16.309 | 1.625 ± 0.070 | 114.103 ± 3.457 | 357.508 ± 22.567 | 357.508 ± 22.567 |
| KV admission | 413.645 ± 66.163 | 1.408 ± 0.055 | 113.938 ± 7.370 | 403.952 ± 73.250 | 403.952 ± 73.250 |

相对 FIFO K=4，候选策略本次均值点估计为答对吞吐增加 56.2%、全任务 p95 降低 27.1%、runner 时间降低 26.3%，并避免该基线的 4 次超时。相对经开发集选择的 FIFO K=1，候选策略答对吞吐低 13.4%、全任务 p95 高 13.0%、runner 时间高 11.9%，总正确数还少 1。因此当前证据支持“候选优于原 K=4 的点估计”，不支持“候选优于调优固定基线”。每种仅 3 次重复，未做显著性检验；这里的百分比不是稳定收益保证。

| 方案 | 请求 TTFT p95 秒 | 请求排队 p95 秒 | v2 KV 采样峰值均值 ± SD | 显存采样峰值 MiB |
|---|---:|---:|---:|---:|
| FIFO K=4 | 57.269 ± 13.193 | 26.516 ± 10.843 | 64013.0 ± 999.1 | 13668 |
| FIFO K=1 | 33.470 ± 1.634 | 29.275 ± 1.401 | 18132.0 ± 0.0 | 13668 |
| KV admission | 37.516 ± 10.622 | 26.869 ± 4.784 | 53565.7 ± 362.6 | 13668 |

请求阶段耗时由具有相应记录的请求计算，审计清单另行标记了 FIFO K=4 repeat 2 的 3 条在途超时删失。上表汇总读取既有阶段记录，不能替代逐请求审计中的删失处理。三种方案显存采样峰值相同，因为配置的 CUDA KV 缓冲区在启动时预分配；slot token 估计较低不代表 CUDA 已归还同等显存。留出三条件的 `llama.log` 池满事件计数都为 0，FIFO K=4 的超时不能据此单独归因为物理显存 OOM 或已证实的 allocator 故障。

## 候选策略实际生效与开销

三轮候选共有 448 条 scheduler 记录，全部标记 `policy=kv-admission`、`policy_reason=within_budget`。448 次输入 token 估计均成功，预留值逐条符合 `prompt_tokens + 1024 + 256`，无 oversize solo；同时活跃请求最高 4，各轮活跃预留 token 峰值为 58229、58891、58773，均不超过 `floor(65536 × 0.9)=58982`。[策略生效检查](../results/aws-validation-20261030/policy-validation.json)通过、无已知预留越界。辅助 `/slots` 观测另有 1、2、5 次 timeout，共 8 次；它们不改变已经成功算出的 reservation，应与输入计数错误分开报告，不能写成所有 API 调用均成功。

这些记录验证软件预留约束生效，不能当作精确 KV allocator 空闲量或保证未来请求不溢出。日志未记录独立的准入／释放事件时间，因此不能从最终行独立重建全部活动区间；策略证明的范围是各条所记录的准入状态与预算公式。

| 开销指标 | repeat 0 | repeat 1 | repeat 2 | 三轮均值 ± 样本 SD |
|---|---:|---:|---:|---:|
| 请求估计耗时均值，ms | 122.864 | 112.891 | 199.189 | 144.981 ± 47.209 |
| 请求估计耗时 p95，ms | 1211.124 | 686.179 | 1619.101 | 1172.135 ± 467.682 |
| 预留准入等待 p95，ms | 3011.222 | 4208.677 | 1924.976 | 3048.292 ± 1142.301 |

`policy_estimate_ms` 包含完整请求 token 计数接口和 `/slots` 观测的墙钟时间，受到真实服务负载影响，不是纯 CPU 计算时间；448 请求合并平均值为 144.932 ms。`policy_wait_ms` 是估计完成后等待预留准入的时间，已属于请求排队过程，不能与端到端时长再重复相加。小探针中约 2.6 ms 的 slot 保存耗时与这里的估计开销是不同操作，不能互相替代或外推。

正式实验的实际 scheduler 启动参数为队列容量 4096、最大连接数 4096、`--queue-budget 86400`、流式网络间隔超时 600 秒，单任务外层 600 秒超时仍生效。不能用调度器默认的 300 秒队列预算解释这些实验。测试任务之间仍为闭环分配，未测开环吞吐极限。

## 验证与测试范围

发布前在仓库根目录执行完整测试 `.venv/bin/python -m pytest -q -rs`，结果为 **86 passed、2 skipped，7.92 秒**。跳过项均为 `tests/test_client.py` 中依赖本机 llama 服务的两项。AWS 上另以真实模型验证接口探针和 450 次任务的 agent—调度器—模型—工具链路；本机跳过的集成项不能写成已在本机通过。

下载后逐项核验 703 个清单文件的 SHA256，27 个运行均有效，0 个无效，汇总警告为 0；全部运行记录中的 5 个核心源码哈希与本机冻结部署版本一致。全量只读审计前后比较了下载目录中的 789 个文件，所有哈希均未改变；789 是目录核对范围，与 703 个原始清单文件范围不同。原始数据压缩包 SHA256 为 `c25c956ed5c1041ca11c450312b8a9628f7d110939470ae9cc2b8cf2407c63a5`。最终模型任务于 UTC `2026-10-31T03:39:35Z` 结束，整条流程于 `03:40:35Z` 以退出码 0 结束，早于预设的绝对截止。结构与文件验收通过仍不消除前述请求计时删失。

## 原始数据与后续交接

| 材料 | 位置 |
|---|---|
| 统一入口与解压说明 | [结果 README](../results/aws-validation-20261030/README.md) |
| 本轮短任务原始日志 | 归档展开后 `work/aws-validation-20261030/logs/aws_representative_short/` |
| 本轮长任务原始日志 | 归档展开后 `work/aws-validation-20261030/logs/aws_representative_long/` |
| 开发集原始日志 | 归档展开后 `work/aws-validation-20261030/logs/aws_phase2_dev/` |
| 留出集原始日志 | 归档展开后 `work/aws-validation-20261030/logs/aws_phase2_test/` |
| 探针、协议、固定 K 选择与留出矩阵 | [runtime_probe.json](../results/aws-validation-20261030/runtime_probe.json)、[protocol.json](../results/aws-validation-20261030/protocol.json)、[selection.json](../results/aws-validation-20261030/selection.json)、[heldout.toml](../results/aws-validation-20261030/heldout.toml) |
| 最终汇总与科学绘图 | [summary_runs.csv](../results/aws-validation-20261030/summary_runs.csv)、[summary_conditions.csv](../results/aws-validation-20261030/summary_conditions.csv)、[summary.json](../results/aws-validation-20261030/summary.json)、[heldout_comparison.png](../results/aws-validation-20261030/heldout_comparison.png) |
| 新旧机器可读对照 | [historical_comparison.json](../results/aws-validation-20261030/historical_comparison.json) |
| 原始文件 SHA256 | [raw_log_manifest.json](../results/aws-validation-20261030/raw_log_manifest.json) |
| 下载核验 | [download-verification.json](../results/aws-validation-20261030/provenance/download-verification.json) |
| 全量只读审计交接 | [audit.json](../results/aws-validation-20261030/audit/audit.json)、[requests.csv](../results/aws-validation-20261030/audit/requests.csv) |
| 逐组验收记录 | [checks/](../results/aws-validation-20261030/checks/) |
| 候选策略生效证明 | [policy-validation.json](../results/aws-validation-20261030/policy-validation.json) |
| 完整归档 | [raw-logs.tar.gz](../results/aws-validation-20261030/raw-logs.tar.gz) 与 [SHA256](../results/aws-validation-20261030/raw-logs.tar.gz.sha256) |
| 部署溯源 | [source-deployment.json](../results/aws-validation-20261030/provenance/source-deployment.json)、[helper-deployment.json](../results/aws-validation-20261030/provenance/helper-deployment.json)、[helper-v2-deployment.json](../results/aws-validation-20261030/provenance/helper-v2-deployment.json) |
| 费用估算和停机记录 | [cost-estimate.json](../results/aws-validation-20261030/provenance/cost-estimate.json)、[instance-stopped.json](../results/aws-validation-20261030/provenance/instance-stopped.json) |
| 原 Phase 1 短任务对照 | `phase1/data/gpu/k2_a4_r0`、`r1`、`r2` |
| 原 Phase 1 长任务对照 | `phase1/data/longctx/k4_a16_r0`、`r1`、`r2` 及 `k4_a16_ctx98304_r0`、`r1`、`r2` |

以上仓库结果目录作为团队交接入口。本地额外 zip、截图和早期 smoke 文件未作为仓库复现前提；完整的 27 轮原始证据均在归档中。只读分析不改写原始文件；历史对照使用相同统计函数重读原 Phase 1 日志。绘图的误差线为三次重复的样本标准差，不是置信区间。完整服务器运行方法、固定参数和预算守护见 [运行说明](aws_runbook.md)。

## 费用估算与停机

本轮实例于 UTC `2026-10-31 00:56:34` 启动，于 `03:41:06.162764` 发起正常、非休眠的 EC2 Stop，启动至停止请求间隔 9872.162764 秒。按已核验单价 0.9776 美元／小时计算，本轮计算费估算为 **2.68084 美元**。正常停止进入 stopping 后不再收实例使用费，EBS 继续计费；相关状态规则见 [AWS 实例生命周期说明](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/ec2-instance-lifecycle.html)。

| 估算组成 | 美元 |
|---|---:|
| 此前课程账面快照，包含旧试运行和队友费用 | 2.30 |
| 本轮启动至正常停止请求的计算费估算 | 2.68084 |
| 本次磁盘、公有 IPv4 等附加费用预留 | 0.20 |
| 保守累计估算 | **5.18084** |
| 用户累计上限 | **10.00** |

此为本轮预算核对估算，不是最终账单；课程账面有延迟。为保守计算，起点采用整个团队已显示的 2.30 美元，已经包含先前试运行，未重复加该次费用。保留的根盘为 100 GiB EBS，停止后仍持续计费；0.20 美元仅预留本次会话附加费用，不覆盖长期保留磁盘的未来费用。

API 于 UTC **2026-10-31 03:48:19.054262**（洛杉矶 10 月 30 日 **20:48:19**）观测到本实例 `i-01d734efb54dbd071` 为 stopped，正常 Stop 完成，未强制停止。该时间是状态核验时间，不是精确状态切换时刻；未操作队友实例。最终 API 证据见 [instance-stopped.json](../results/aws-validation-20261030/provenance/instance-stopped.json)。
