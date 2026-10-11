1 号：Agent + 任务集

写 agent 循环：发请求，解析 tool call，执行 tool，再把结果拼回去
写 tool，处理 tool 等待时间怎么来
挑任务集（比如数学 + python），设最大步数，记录每个任务成功没有

2 号：调度层

写一个夹在 agent 和 llama.cpp 之间的代理，所有请求都经过它
baseline 先做先来先服务，固定并发数
每个请求要带上是哪个 agent、第几步，为 Phase 2 的策略留好接口

3 号：llama.cpp + 实验运行

装好 llama.cpp，搞清 -c、--parallel、cache 复用这些参数怎么用，把内存预算定下来
写一键实验脚本：启动 server，起 N 个 agent，跑完自动存日志
正式数据在你的 Mac 上跑，所以这个角色你来做比较顺

4 号：Eval

定日志格式（今天就要定，其他三个人都按这个格式写）
写分析脚本，从日志算出完成时间的 p50/p95、TTFT、吞吐、重算量、内存峰值、成功率
画图，设计实验矩阵（并发数 × tool 等待时间 × agent 数量），负责 Phase 1 report 的结果部分