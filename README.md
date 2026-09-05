# Agent Cost Control

Bounds AI agent execution cost before it runs: pre-execution estimation, model routing, semantic caching, and a hard budget ceiling. Measured on cost variance, not mean.

Agent workloads are not just expensive, they are unpredictable. The same task costs a dollar most of the time and ten dollars occasionally, and nothing tells you which until it finishes.

```bash
pip install -e ".[dev]"
agent-cost-control bench     # the table below
agent-cost-control sweep     # cost against solve rate as the ceiling moves
pytest -q                    # 39 tests
```

Runs offline and deterministically. No API keys.

## Result

280 measured tasks after 120 warm-up, ceiling $0.15, 25% repeat rate.

| Configuration | Total | p50 exec | p99 exec | p99/p50 | Solved | Solved (exec) | $/solved | Aborted |
|---|---|---|---|---|---|---|---|---|
| baseline | $20.08 | $0.0249 | $0.4711 | 18.9× | 94.6% | 94.6% | $0.0758 | 0 |
| + routing | $15.96 | $0.0118 | $0.4711 | **39.8×** | 90.0% | 90.0% | $0.0633 | 0 |
| + escalation | $17.82 | $0.0118 | $0.4777 | 40.4× | 94.3% | 94.3% | $0.0675 | 0 |
| + ceiling | $10.08 | $0.0118 | **$0.1486** | 12.6× | 87.9% | 87.9% | $0.0410 | 34 |
| **+ cache (all)** | **$8.57** | $0.0169 | $0.1488 | **8.8×** | 90.0% | 86.9% | **$0.0340** | 28 |

**Cost per solved task 2.2× lower. Tail cost 3.2× lower. Solve rate 94.6% → 90.0%.**

The 4.6 points are the price of the ceiling, and they are a setting rather than a finding: at $0.25 the solve rate is 93.6% and cost per solved task is still 1.9× better. `Solved (exec)` excludes cache hits, which are answers an earlier run already paid for, so the cache cannot lift it.

Two controls look worse alone, which is why they are ablated separately. Routing halves the median and doubles the variance ratio, because the tail is expensive for reasons no model choice fixes. Escalation costs more in isolation and buys back the 4 points of solve rate routing gave away. The ceiling is the only control that touches the tail; the cache is the only one that lowers cost without touching quality.

## Why cost is unpredictable

Step count is data-dependent, and every step resends the transcript, so input tokens at step *k* grow with *k*. Cost is roughly **quadratic in step count**: 2 steps versus 12 is not 6× but closer to 36×.

With no controls, per model:

| Model | Solve rate | p50 | p99 | p99/p50 |
|---|---|---|---|---|
| small | 56.4% | $0.0170 | $0.0314 | 1.8× |
| mid | 80.0% | $0.0169 | $0.1257 | 7.4× |
| frontier | 94.6% | $0.0249 | $0.4711 | 18.9× |

`small` has the best variance ratio and the worst outcomes, because it gives up quickly. **Variance can always be improved by failing faster**, so every table reports solve rate beside cost and the primary metric is cost per *solved* task.

Capability is held per category, not as one number: the cheap tier handles mechanical lookups nearly as well as the frontier tier and collapses on migrations. With a single number there would be one correct model everywhere and nothing for a router to decide.

## The ceiling is a dial

| Ceiling | Total | p99 exec | Solve rate | $/solved | Aborted |
|---|---|---|---|---|---|
| $0.05 | $5.27 | $0.0479 | 78.2% | $0.0241 | 61 |
| $0.08 | $6.35 | $0.0704 | 82.5% | $0.0275 | 49 |
| $0.10 | $7.24 | $0.0957 | 87.5% | $0.0296 | 35 |
| $0.15 | $8.57 | $0.1488 | 90.0% | $0.0340 | 28 |
| $0.25 | $10.54 | $0.2303 | 93.6% | $0.0402 | 17 |
| $0.50 | $13.23 | $0.4733 | 96.8% | $0.0488 | 1 |

Monotonic in both directions. No setting is simply correct; the choice is how much unsolved work is acceptable.

## One decision

```
$ agent-cost-control explain task_0150

task task_0150  [lookup]
  prompt: find where the auth middleware is configured in the gateway

routing
  -> small     solve rate 100%         (15 observations)
     mid       solve rate no evidence  (0 observations)
     frontier  solve rate no evidence  (0 observations)

estimate on small
  expected $0.0010 over 1.4 steps
  interval $0.0007 - $0.0027   confident
  ceiling  $0.15  ->  admitted

outcome: success
  models  small
  steps   1
  actual  $0.0007
  inside the predicted interval
```

Estimates are intervals. A point estimate reproduces the problem it was built to solve. Interval coverage is reported by `bench`: 85% of runs land inside the predicted band against a nominal 80%.

## Configuration

Prices and capability profiles are data, in `src/agent_cost_control/model_catalog.json`. Repricing, adding a tier, or pointing the benchmark at another vendor is a JSON edit, and a test re-runs the whole ablation against a second table to confirm the findings turn on the ratio between tiers rather than any particular numbers.

```bash
agent-cost-control bench --catalog my_prices.json
agent-cost-control bench --repeat-rate 0.0     # cache disabled by the workload
```

`--repeat-rate` is the share of tasks that restate an earlier request. It is an input rather than an accident of the corpus, so the cache row can be falsified: at `0.0` the cache saves nothing, and the routing and ceiling rows are unchanged.

## Design

```
src/agent_cost_control/
  models.py     tasks, model specs, attempts, estimates
  catalog.py    model table loading and the task corpus
  executor.py   the agent loop (simulated | live provider)
  estimate.py   cost prediction with intervals, learned from history
  route.py      model selection with bounded exploration
  cache.py      semantic cache
  budget.py     pre-flight refusal and mid-flight abort
  control.py    composition; every mechanism independently ablatable
  bench.py      metrics and the ablation runner
  cli.py        bench | sweep | explain
```

Estimation conditions on task category and on the prompt sizes nearest the task, falling back to the category alone until there is enough history to spare. The gain from scope is small, and that is the finding: cost varies mostly because step count is stochastic, not because scope went unmeasured.

The controller is real. Estimation, routing, caching, and ceiling enforcement operate on token counts and prices, and run unchanged against a live provider through `ProviderExecutor`. The executor is simulated, because no live provider gives a reproducible cost distribution. Each task seeds from its id, so every configuration meets identical luck; without common random numbers, comparing arms would mostly measure which tasks happened to resolve early.

## License

MIT — see [LICENSE](LICENSE).
