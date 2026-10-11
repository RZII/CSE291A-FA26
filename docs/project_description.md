Option 1: Memory-Aware Agent Edge Serving
Project description
Agent workloads generate multiple dependent model requests, use tools between requests, and retain context across steps. On a memory-constrained machine, admitting too much work can increase latency or exhaust memory; admitting too little can leave resources idle. This project investigates how a serving system should manage concurrent agent tasks and their context under a limited memory budget.
The core question: How can admission and scheduling decisions improve completed agent work while keeping memory use bounded?
Hardware and software environment
Use small AWS instances (or limit large ones usable memory) or your laptops with a supported local inference runtime, an agent workload runner, and a queue or scheduler under your control. Keep the model and precision fixed in the main comparisons. Begin with workloads that fit comfortably on the selected machine, then vary concurrency and context size. Any cache-aware extension must use cache controls and measurements that the runtime actually exposes.
Phase 1: Baseline implementation and measurement
Implement a FIFO queue with fixed concurrency. Create a mixture of short and long contexts, short and long responses, and multi-step tasks with controlled tool delays. Instrument both individual model requests and complete agent tasks.
Measure task completion time, queueing time, tail latency, throughput, peak memory, and failures or timeouts. Determine how concurrency, context growth, and idle periods between agent steps affect memory pressure and completed work. Characterize whether retained contexts or reusable prefixes materially affect the chosen runtime.
Deliver a reproducible baseline, workload definitions, performance curves, and an explanation of the dominant bottleneck.
Phase 2: Memory-aware scheduling
Build a policy motivated by Phase 1. Possible mechanisms include estimating the memory needed by incoming requests, adapting concurrency to observed pressure, prioritizing ready agent steps, or choosing which reusable context to retain when supported by the runtime.
Select a focused mechanism and compare it against FIFO and the best fixed-concurrency baseline selected on development workloads. A cache-aware design should also compare against a simple fixed cache policy. Evaluate changing arrival rates and workload mixes, including cases where the policy provides little benefit.
Evaluation and deliverables
Report completed tasks, median and tail completion latency, throughput, peak memory, and failure rates. Check that performance gains do not come from skipping tasks, shortening outputs, or reducing task quality. Account for policy overhead and test whether long tasks are unfairly delayed.
Submit the scheduler, workload runner, raw measurements, analysis scripts, and a report connecting observed behavior to memory-management decisions.
Detailed assessment criteria
Strong submissions explain the memory bottleneck, justify the policy from measurements, compare against tuned simple baselines, and demonstrate reproducible end-to-end behavior. Implementing many policies is less valuable than understanding one well.
Starting resources
llama.cpp

