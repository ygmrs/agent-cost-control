"""Predicting what a task will cost before running it.

The estimator learns from outcomes it has already seen. It is given the task
category, the prompt size, and the model under consideration -- never the
task's difficulty, which exists only inside the simulation. An estimator with
access to difficulty would be perfect, and the benchmark would be measuring
nothing. A test asserts this module never reads that attribute.

It conditions on category *and* prompt size, backing off to category alone
where the finer split has too little evidence to be worth trusting. Size is a
weak signal on its own -- a long prompt describing a lookup is still a lookup --
but within a category it separates a narrow change from a sprawling one, which
is the difference between four steps and twelve. The backoff matters as much as
the split: partitioning a category three ways divides its evidence three ways,
and an estimator that trusted a two-observation bucket would be confidently
wrong exactly where the ceiling depends on it.

It returns an interval rather than a number. The reported failure is not that
mean cost is high, it is that the tail is unpredictable, so a point estimate
reproduces the problem it was built to solve. The ceiling logic reserves
against ``p90``, not against the mean.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

from .catalog import MODELS
from .executor import (OUTPUT_TOKENS_PER_STEP, SYSTEM_TOKENS,
                       TOOL_RESULT_TOKENS)
from .models import Attempt, Estimate, ModelSpec, Task

# Steps assumed for a category with no history. Deliberately pessimistic: a
# fresh estimator that guesses low would let unbounded tasks through, and the
# whole point is to fail safe until evidence exists.
COLD_START_STEPS = 6.0
MIN_OBSERVATIONS = 5
# Neighbours required before conditioning on prompt size at all, and the share
# of a category's history kept once there are enough. Half is a compromise: a
# tighter band is more specific about this task and has less evidence behind
# it, and the whole reason for the threshold is that specificity without
# evidence is how an estimator becomes confidently wrong.
MIN_NEIGHBOURS = 12
NEIGHBOUR_FRACTION = 0.5


def nearest_by_size(samples: list[tuple[int, int]], prompt_tokens: int,
                    fraction: float = NEIGHBOUR_FRACTION) -> list[int]:
    """Step counts of the observations closest in prompt size.

    Fixed size bands were the obvious approach and they do nothing here: task
    scope drives both prompt length and category, so a band boundary drawn in
    absolute tokens lands between categories rather than inside one, and the
    bucket ends up being the category under another name. A neighbourhood
    measured against *this* task adapts to wherever the category happens to
    sit.
    """
    keep = max(MIN_NEIGHBOURS, int(len(samples) * fraction))
    nearest = sorted(samples, key=lambda pair: abs(pair[0] - prompt_tokens))[:keep]
    return [steps for _size, steps in nearest]


def cost_of_steps(model: ModelSpec, prompt_tokens: int, steps: float) -> float:
    """Closed-form cost for a run of ``steps`` steps.

    Input tokens at step k are ``base + k * growth``, so summing over k gives a
    term in k(k-1)/2 -- the quadratic that turns a moderate increase in steps
    into a large increase in dollars. Computing it directly rather than
    simulating keeps estimation cheap enough to run before every task.
    """
    if steps <= 0:
        return 0.0
    base = SYSTEM_TOKENS + prompt_tokens
    growth = OUTPUT_TOKENS_PER_STEP + TOOL_RESULT_TOKENS
    whole = int(steps)
    frac = steps - whole

    total_input = whole * base + growth * (whole * (whole - 1) / 2)
    total_output = whole * OUTPUT_TOKENS_PER_STEP
    if frac > 0:
        total_input += frac * (base + growth * whole)
        total_output += frac * OUTPUT_TOKENS_PER_STEP
    return model.step_cost(int(total_input), int(total_output))


def first_step_cost(model: ModelSpec, prompt_tokens: int) -> float:
    """Cost of the opening step of a run.

    Shared by the ceiling and by escalation, both of which need the price of a
    step that has not been taken yet.
    """
    return model.step_cost(SYSTEM_TOKENS + prompt_tokens, OUTPUT_TOKENS_PER_STEP)


@dataclass
class History:
    """Observed step counts and outcomes.

    ``steps``/``solved`` are keyed by (category, model) and are what routing
    reads and what the estimator falls back to. ``samples`` additionally keeps
    the prompt size beside each step count, so the estimator can condition on
    scope once a category has enough history for that to mean anything.
    """
    steps: dict[tuple[str, str], list[int]] = field(
        default_factory=lambda: defaultdict(list))
    solved: dict[tuple[str, str], list[bool]] = field(
        default_factory=lambda: defaultdict(list))
    samples: dict[tuple[str, str], list[tuple[int, int]]] = field(
        default_factory=lambda: defaultdict(list))

    def record(self, task: Task, model_name: str, attempt: Attempt) -> None:
        steps = max(1, attempt.step_count)
        self.steps[(task.category, model_name)].append(steps)
        self.solved[(task.category, model_name)].append(attempt.solved)
        self.samples[(task.category, model_name)].append(
            (task.prompt_tokens, steps))

    def observations(self, category: str, model_name: str) -> int:
        # ``.get`` rather than indexing: these are defaultdicts, and indexing
        # them inserts an empty entry, so a read would quietly mutate history.
        return len(self.steps.get((category, model_name), ()))

    def solve_rate(self, category: str, model_name: str) -> float | None:
        outcomes = self.solved.get((category, model_name), ())
        if len(outcomes) < MIN_OBSERVATIONS:
            return None
        return sum(outcomes) / len(outcomes)


class Estimator:
    """Cost prediction with intervals.

    ``size_aware`` exists so the benchmark can measure what the second feature
    buys instead of asserting it. Turning it off falls back to category-only
    estimation, which is what the comparison test runs against.
    """

    def __init__(self, history: History | None = None, size_aware: bool = True):
        self.history = history or History()
        self.size_aware = size_aware

    def _observed_steps(self, task: Task, model_name: str) -> list[int]:
        """Step counts to estimate from, at the finest resolution earned."""
        if self.size_aware:
            samples = self.history.samples.get((task.category, model_name), ())
            if len(samples) >= 2 * MIN_NEIGHBOURS:
                return nearest_by_size(list(samples), task.prompt_tokens)
        return list(self.history.steps.get((task.category, model_name), ()))

    def _step_quantiles(self, task: Task, model_name: str
                        ) -> tuple[float, float, float, bool]:
        """(p10, expected, p90, confident) step counts."""
        observed = sorted(self._observed_steps(task, model_name))
        if len(observed) < MIN_OBSERVATIONS:
            return (1.0, COLD_START_STEPS, COLD_START_STEPS * 2, False)

        def quantile(q: float) -> float:
            idx = min(len(observed) - 1, max(0, int(math.ceil(q * len(observed)) - 1)))
            return float(observed[idx])

        mean = sum(observed) / len(observed)
        return (quantile(0.10), mean, quantile(0.90), True)

    def estimate(self, task: Task, model_name: str) -> Estimate:
        model = MODELS[model_name]
        p10_steps, expected_steps, p90_steps, confident = self._step_quantiles(
            task, model_name)
        return Estimate(
            expected_usd=cost_of_steps(model, task.prompt_tokens, expected_steps),
            p10_usd=cost_of_steps(model, task.prompt_tokens, p10_steps),
            p90_usd=cost_of_steps(model, task.prompt_tokens, p90_steps),
            expected_steps=expected_steps,
            confident=confident,
        )

    def project_final_cost(self, task: Task, model_name: str,
                           attempt: Attempt) -> float:
        """Projected total for a run already in flight.

        Used by the ceiling to decide mid-task whether to stop. Remaining steps
        are estimated as whatever the p90 allows beyond the current position, so
        a task that has already exceeded its typical length projects high and
        gets cut rather than being allowed to run to the step limit.
        """
        model = MODELS[model_name]
        _p10, _expected, p90_steps, _confident = self._step_quantiles(
            task, model_name)
        remaining = max(0.0, p90_steps - attempt.step_count)
        base = SYSTEM_TOKENS + task.prompt_tokens
        growth = OUTPUT_TOKENS_PER_STEP + TOOL_RESULT_TOKENS
        projected = attempt.cost_usd
        for offset in range(int(math.ceil(remaining))):
            step_index = attempt.step_count + offset
            projected += model.step_cost(base + growth * step_index,
                                         OUTPUT_TOKENS_PER_STEP)
        return projected
