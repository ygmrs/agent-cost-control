"""Domain model.

The premise of this project is a single observation about where agent cost
variance comes from, and it is worth stating precisely because every control
here follows from it.

A one-shot LLM call is cheap to predict: tokens in, bounded tokens out. An
*agent* task is a loop, and two things compound inside it:

1. the number of steps is data-dependent -- the agent stops when it succeeds,
   and how quickly it succeeds depends on the task and the model;
2. every step resends the accumulated transcript, so the input cost of step
   ``k`` grows with ``k``.

Total cost is therefore roughly quadratic in step count. A task that resolves
in 2 steps and the same task flailing for 12 do not differ by 6x, they differ
by something closer to 36x. That is the mechanism behind reports of identical
operations costing a dollar one time and ten dollars the next, and it is why
the metric that matters is the *spread* of cost rather than its mean.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Outcome(str, Enum):
    """How an attempt ended.

    ``ABORTED`` and ``REFUSED`` are separate on purpose. Aborted means the loop
    started and was stopped mid-flight when projected spend crossed the ceiling;
    refused means the estimator rejected it before spending anything. They cost
    very different amounts and conflating them hides whether pre-execution
    estimation is doing any work.
    """
    SUCCESS = "success"
    EXHAUSTED = "exhausted"      # hit the step limit without solving
    ABORTED = "aborted"          # stopped mid-flight by the budget ceiling
    REFUSED = "refused"          # rejected before execution by the estimator
    CACHED = "cached"            # served from the semantic cache


@dataclass(frozen=True)
class ModelSpec:
    """A model's price and its capability profile.

    Capability is held per task category rather than as one number, because a
    model is not uniformly strong: the cheap tier handles mechanical lookups
    nearly as well as the frontier tier and collapses on migrations, so the gap
    between tiers is different in every category. A scalar would average those
    apart and make the router's job look easier than it is -- with a single
    number there is one correct ordering of models everywhere, and choosing per
    category would be pointless.

    ``capability`` is the fallback for a category with no entry, so adding a
    category does not require editing every model.
    """
    name: str
    capability: float
    input_usd_per_mtok: float
    output_usd_per_mtok: float
    capability_by_category: dict[str, float] = field(default_factory=dict)

    def capability_for(self, category: str) -> float:
        return self.capability_by_category.get(category, self.capability)

    def step_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (input_tokens * self.input_usd_per_mtok / 1e6
                + output_tokens * self.output_usd_per_mtok / 1e6)


@dataclass(frozen=True)
class Task:
    """A unit of agent work.

    ``difficulty`` is ground truth used only by the simulated executor and by
    reporting. No control may read it -- an estimator with access to difficulty
    would be clairvoyant, and the benchmark would measure nothing. A test
    enforces this by inspecting the control modules for the attribute.
    """
    id: str
    category: str
    prompt: str
    difficulty: float
    prompt_tokens: int


@dataclass
class StepRecord:
    index: int
    model: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class Attempt:
    """The full record of one task execution."""
    task_id: str
    outcome: Outcome
    steps: list[StepRecord] = field(default_factory=list)
    models_used: list[str] = field(default_factory=list)
    estimated_usd: float = 0.0
    estimate_interval: tuple[float, float] = (0.0, 0.0)
    cache_hit: bool = False

    @property
    def cost_usd(self) -> float:
        return sum(s.cost_usd for s in self.steps)

    @property
    def step_count(self) -> int:
        return len(self.steps)

    @property
    def solved(self) -> bool:
        return self.outcome in (Outcome.SUCCESS, Outcome.CACHED)


@dataclass
class Estimate:
    """A predicted cost with an interval.

    A point estimate is the wrong shape for this problem. The failure teams
    report is not that the average is high, it is that the tail is unpredictable
    -- so an estimator that returns one number reproduces the problem it was
    meant to solve. ``p90`` is what the ceiling logic reserves against.
    """
    expected_usd: float
    p10_usd: float
    p90_usd: float
    expected_steps: float
    confident: bool = True

    @property
    def interval(self) -> tuple[float, float]:
        return (self.p10_usd, self.p90_usd)
