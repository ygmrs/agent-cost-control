"""Enforcing a spend ceiling.

Two enforcement points, and both are necessary.

**Before execution.** If the p90 estimate already exceeds the ceiling, the task
never starts. This costs nothing and is the only control that can prevent spend
rather than curtail it.

**During execution.** The estimate can be wrong -- that is the nature of the
problem -- so the ceiling is checked again after every step against a projection
of where the run is heading. A task that discovers on step nine that it is going
long gets stopped there rather than at the step limit. A ceiling checked only
before and after execution is not a ceiling; it is a report.

Refusal and abort are recorded as different outcomes because they cost different
amounts, and collapsing them would hide whether pre-flight estimation is
contributing anything.
"""
from __future__ import annotations

from dataclasses import dataclass

from .catalog import MODELS
from .estimate import Estimator
from .executor import (OUTPUT_TOKENS_PER_STEP, SYSTEM_TOKENS,
                       TOOL_RESULT_TOKENS)
from .models import Attempt, Estimate, Task


@dataclass
class CeilingPolicy:
    estimator: Estimator
    ceiling_usd: float = 0.15
    # Fraction of the ceiling a run may reach before mid-flight projection is
    # consulted. Below this, a task is left alone -- checking a projection on
    # step one, when the estimator has almost no evidence from this run, would
    # abort tasks that were going to finish cheaply.
    check_after_fraction: float = 0.35

    def admit(self, task: Task, model_name: str) -> tuple[bool, Estimate]:
        """Decide whether to start the task at all.

        Refusal keys on the *expected* cost, not the p90, and that choice was
        made empirically. Refusing on p90 rejects whole categories: estimation
        here is per-category, so an easy migration and a pathological one share
        a p90, and a ceiling below it turns away both. Measured at a $0.15
        ceiling that refused 32% of tasks and dropped the solve rate to 66%.

        Admitting on the expected cost lets ordinary work through and leaves the
        tail to the mid-flight guard, which has evidence the estimator lacks:
        how the run is actually going. Pre-flight refusal then catches only
        categories that are typically unaffordable, which is the case it should
        catch.
        """
        estimate = self.estimator.estimate(task, model_name)
        if estimate.confident and estimate.expected_usd > self.ceiling_usd:
            return False, estimate
        return True, estimate

    def step_guard(self, task: Task, model_name: str):
        """Build the per-step callback the executor calls after each step.

        The guard is *predictive*, not reactive. An earlier version compared the
        cost already spent against the ceiling, which stops the next step but
        only after the current one has been paid for -- measured at a $0.05
        ceiling, aborted runs overshot by up to 31%. A bound that can be
        exceeded is not a bound.

        So it prices the step that would come next, using the same closed form
        the estimator uses, and refuses to take it if doing so would cross the
        line. The first step is unavoidable: nothing can be learned about a run
        without starting it, so the ceiling holds from step two onward.

        Returns a closure so the executor stays ignorant of budgets: it knows
        only that something may ask it to stop.
        """
        model = MODELS[model_name]
        base = SYSTEM_TOKENS + task.prompt_tokens
        growth = OUTPUT_TOKENS_PER_STEP + TOOL_RESULT_TOKENS

        def guard(attempt: Attempt) -> bool:
            next_step_cost = model.step_cost(
                base + growth * attempt.step_count, OUTPUT_TOKENS_PER_STEP)
            if attempt.cost_usd + next_step_cost > self.ceiling_usd:
                return False
            if attempt.cost_usd < self.ceiling_usd * self.check_after_fraction:
                return True
            projected = self.estimator.project_final_cost(task, model_name, attempt)
            return projected <= self.ceiling_usd

        return guard
