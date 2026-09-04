# Agent Cost Control

Bounds AI agent execution cost before it runs: pre-execution estimation, model routing, semantic caching, and a hard budget ceiling. Measured on cost variance, not mean.

Agent workloads are not just expensive, they are unpredictable. The same task costs a dollar most of the time and ten dollars occasionally, and nothing tells you which until it finishes.

```bash
pip install -e ".[dev]"
agent-cost-control bench     # the table below
agent-cost-control sweep     # cost against solve rate as the ceiling moves
pytest -q                    # 29 tests
```

Runs offline and deterministically. No API keys.

## Result

280 measured tasks after 120 warm-up, ceiling $0.15.

| Configuration | Total | p50 exec | p99 exec | p99/p50 | Solve rate | $/solved | Aborted |
|---|---|---|---|---|---|---|---|
| baseline | $17.50 | $0.0246 | $0.4532 | 18.4× | 96.1% | $0.0651 | 0 |
| + routing | $14.65 | $0.0118 | $0.4525 | **38.4×** | 88.6% | $0.0591 | 0 |
| + escalation | $17.82 | $0.0139 | $0.4813 | 34.5× | 94.6% | $0.0673 | 0 |
| + ceiling | $11.62 | $0.0139 | **$0.1489** | 10.7× | 87.9% | $0.0472 | 31 |
| **+ cache (all)** | **$3.64** | $0.0145 | $0.1462 | 10.1× | **97.1%** | **$0.0134** | 8 |

**Cost per solved task 4.9× lower. Tail cost 3.1× lower. Solve rate 96.1% → 97.1%.**

Two controls look worse alone, which is why they are ablated separately. Routing halves the median and doubles the variance ratio, because the tail is expensive for reasons no model choice fixes. Escalation costs more in isolation and buys back the 6 points of solve rate routing gave away. The ceiling is the only control that touches the tail; the cache is the only one that moves total cost.

## Why cost is unpredictable

Step count is data-dependent, and every step resends the transcript, so input tokens at step *k* grow with *k*. Cost is roughly **quadratic in step count**: 2 steps versus 12 is not 6× but closer to 36×.

With no controls, per model:

| Model | Solve rate | p50 | p99 | p99/p50 |
|---|---|---|---|---|
| small | 62.1% | $0.0141 | $0.0307 | 2.2× |
| mid | 83.6% | $0.0121 | $0.1222 | 10.1× |
| frontier | 96.1% | $0.0246 | $0.4532 | 18.4× |

`small` has the best variance ratio and the worst outcomes, because it gives up quickly. **Variance can always be improved by failing faster**, so every table reports solve rate beside cost and the primary metric is cost per *solved* task.

## The ceiling is a dial

| Ceiling | Total | p99 exec | Solve rate | $/solved | Aborted |
|---|---|---|---|---|---|
| $0.05 | $2.79 | $0.0448 | 88.9% | $0.0112 | 31 |
| $0.08 | $3.18 | $0.0662 | 92.1% | $0.0123 | 22 |
| $0.10 | $3.64 | $0.0908 | 93.2% | $0.0139 | 19 |
| $0.15 | $3.64 | $0.1462 | 97.1% | $0.0134 | 8 |
| $0.25 | $4.12 | $0.2212 | 97.5% | $0.0151 | 7 |
| $0.50 | $5.39 | $0.4582 | 98.6% | $0.0195 | 1 |

Monotonic in both directions. No setting is simply correct; the choice is how much unsolved work is acceptable.

## One decision

```
$ agent-cost-control explain task_0150

task task_0150  [lookup]
  prompt: find where the migration runner is configured in this service

routing
  -> small     solve rate 100%         (15 observations)
     mid       solve rate no evidence  (0 observations)
     frontier  solve rate no evidence  (0 observations)

estimate on small
  expected $0.0016 over 2.0 steps
  interval $0.0007 - $0.0027   confident
  ceiling  $0.15  ->  admitted

outcome: success
  models  small
  steps   3
  actual  $0.0027
  inside the predicted interval
```

Estimates are intervals. A point estimate reproduces the problem it was built to solve.

## Design

```
src/agent_cost_control/
  models.py     tasks, model specs, attempts, estimates
  catalog.py    model prices and the task corpus
  executor.py   the agent loop (simulated | live provider)
  estimate.py   cost prediction with intervals, learned from history
  route.py      model selection with bounded exploration
  cache.py      semantic cache
  budget.py     pre-flight refusal and mid-flight abort
  control.py    composition; every mechanism independently ablatable
  bench.py      metrics and the ablation runner
  cli.py        bench | sweep | explain
```

The controller is real: estimation, routing, caching, and ceiling enforcement operate on token counts and prices, and run unchanged against a live provider through `ProviderExecutor`. The executor is simulated, because no live provider gives a reproducible cost distribution. Each task seeds from its id, so every configuration meets identical luck; without common random numbers, comparing arms would mostly measure which tasks happened to resolve early.

## License

MIT — see [LICENSE](LICENSE).
