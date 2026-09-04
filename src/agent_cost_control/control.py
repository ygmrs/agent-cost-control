"""The controller.

Composes the four mechanisms in the only order that makes sense:

    cache      -- cheapest possible outcome, so check first
    route      -- pick the model before estimating against it
    admit      -- refuse before spending anything
    execute    -- with a mid-flight guard
    escalate   -- if a cheap model gave up and budget remains

Each mechanism can be switched off independently, because the benchmark's job
is to attribute the result to specific controls rather than to the bundle. A
configuration where every control is off is the naive baseline: always the
strongest model, no ceiling, no reuse.

Escalation deserves a note. When a cheap model exhausts its steps without
solving, retrying on a stronger one looks wasteful and usually is not: the
strong model resolves in few steps, and because cost is quadratic in steps, a
short run on an expensive model frequently costs less than a long run on a cheap
one. The benchmark shows where that holds.
"""
from __future__ import annotations

from dataclasses import dataclass

from .budget import CeilingPolicy
from .cache import SemanticCache
from .catalog import DEFAULT_MODEL, MODELS
from .estimate import Estimator, History, first_step_cost
from .executor import Executor, SimulatedExecutor
from .models import Attempt, Outcome, Task
from .route import Router


@dataclass
class ControlConfig:
    """Which mechanisms are active. Every field is independently ablatable."""
    name: str = "all"
    caching: bool = True
    routing: bool = True
    ceiling: bool = True
    escalation: bool = True
    ceiling_usd: float = 0.15
    target_solve_rate: float = 0.80


class Controller:
    def __init__(self, config: ControlConfig, executor: Executor | None = None,
                 history: History | None = None):
        self.config = config
        self.executor = executor or SimulatedExecutor()
        self.estimator = Estimator(history=history or History())
        self.router = Router(estimator=self.estimator,
                             target_solve_rate=config.target_solve_rate)
        self.cache = SemanticCache()
        self.policy = CeilingPolicy(estimator=self.estimator,
                                    ceiling_usd=config.ceiling_usd)

    # ------------------------------------------------------------------ run
    def run(self, task: Task) -> Attempt:
        if self.config.caching:
            cached = self.cache.lookup(task)
            if cached is not None:
                # A hit is recorded with no steps: it consumed no tokens, and
                # attributing the original run's cost to it a second time would
                # overstate what the cache saved.
                return Attempt(task_id=task.id, outcome=Outcome.CACHED,
                               models_used=list(cached.models_used),
                               cache_hit=True)

        model_name = self.router.select(task) if self.config.routing else DEFAULT_MODEL

        if self.config.ceiling:
            admitted, estimate = self.policy.admit(task, model_name)
            if not admitted:
                return Attempt(task_id=task.id, outcome=Outcome.REFUSED,
                               models_used=[model_name],
                               estimated_usd=estimate.expected_usd,
                               estimate_interval=estimate.interval)
        else:
            estimate = self.estimator.estimate(task, model_name)

        guard = self.policy.step_guard(task, model_name) if self.config.ceiling else None
        attempt = self.executor.run(task, model_name, on_step=guard)
        attempt.estimated_usd = estimate.expected_usd
        attempt.estimate_interval = estimate.interval

        # Learn from what actually happened, regardless of outcome. Recording
        # only successes would bias the estimator toward optimism precisely on
        # the categories that run long.
        self.estimator.history.record(task, model_name, attempt)

        if (self.config.escalation and not attempt.solved
                and attempt.outcome is Outcome.EXHAUSTED):
            attempt = self._escalate(task, model_name, attempt)

        if self.config.caching:
            self.cache.store(task, attempt)
        return attempt

    def _escalate(self, task: Task, model_name: str, first: Attempt) -> Attempt:
        stronger = self.router.escalate(model_name)
        if stronger is None:
            return first

        spent = first.cost_usd
        if self.config.ceiling:
            remaining = self.policy.ceiling_usd - spent
            # The retry's first step is unavoidable -- nothing can be learned
            # about a run without starting it -- so if the remaining budget
            # cannot cover that step, escalating would breach the ceiling.
            # Declining here is what keeps the bound true end to end.
            if remaining < first_step_cost(MODELS[stronger], task.prompt_tokens):
                return first
            retry_policy = CeilingPolicy(estimator=self.estimator,
                                         ceiling_usd=remaining)
            guard = retry_policy.step_guard(task, stronger)
        else:
            guard = None

        second = self.executor.run(task, stronger, on_step=guard)
        self.estimator.history.record(task, stronger, second)

        # Merge: the caller is billed for both attempts, because both happened.
        # A retry stopped by the ceiling is reported as ABORTED rather than
        # inheriting the first attempt's outcome -- otherwise the ceiling's
        # intervention disappears from the numbers.
        if second.solved:
            outcome = second.outcome
        elif second.outcome is Outcome.ABORTED:
            outcome = Outcome.ABORTED
        else:
            outcome = first.outcome

        return Attempt(
            task_id=task.id,
            outcome=outcome,
            steps=first.steps + second.steps,
            models_used=[model_name, stronger],
            estimated_usd=first.estimated_usd,
            estimate_interval=first.estimate_interval,
        )

    # ---------------------------------------------------------------- warmup
    def warm(self, tasks: list[Task]) -> None:
        """Populate history before measurement.

        The estimator, the router and the cache all learn from outcomes, so
        scoring them on the tasks they learned from would report memorization as
        prediction. Warm-up runs unmeasured and its cost is excluded.
        """
        for task in tasks:
            self.run(task)
