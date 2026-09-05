"""Choosing which model runs a task.

The naive policy is "always use the best model", and it is what produces the
bill. The opposite, "always use the cheapest", is worse than it looks: a weaker
model needs more steps, steps cost superlinearly, and it fails more often. Being
cheaper per token does not make it cheaper per task.

So the router picks the cheapest model whose *observed* solve rate for this
category clears a target, and escalates when nothing does. It reads history, not
task difficulty -- the same constraint the estimator works under.

The threshold is the honest dial in this project. Lower it and cost falls while
some tasks stop getting solved. The benchmark reports solve rate beside cost for
exactly that reason: variance can always be reduced by giving up faster, and a
cost number without a solve rate beside it is not a result.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import catalog
from .estimate import MIN_OBSERVATIONS, Estimator
from .models import Task


@dataclass
class Router:
    estimator: Estimator
    # Minimum observed solve rate a model must show for a category to be chosen.
    target_solve_rate: float = 0.80
    # Observations required before a model's solve rate is trusted for a
    # category. Below this the router deliberately probes.
    min_observations: int = MIN_OBSERVATIONS

    def select(self, task: Task) -> str:
        """Cheapest model meeting the target, exploring where evidence is thin.

        The obvious implementation -- pick the cheapest model whose observed
        solve rate clears the target -- silently never works. With no history
        every model's rate is unknown, so it falls back to the strongest one,
        which then accumulates all the evidence, which keeps its rate the only
        one known. The cheap models are never tried and therefore never
        chosen, and the router degrades into "always use frontier" while
        appearing to be a policy.

        So it explores first: any cheaper model lacking evidence for this
        category gets probed. Exploration is bounded by construction, because
        the models it probes are the cheap ones, and by the ceiling, which stops
        a probe that runs long. Once evidence exists the policy exploits it.
        """
        for name in catalog.LADDER:
            if self.estimator.history.observations(task.category, name) < self.min_observations:
                return name
            rate = self.estimator.history.solve_rate(task.category, name)
            if rate is not None and rate >= self.target_solve_rate:
                return name
        return catalog.DEFAULT_MODEL

    def escalate(self, current: str) -> str | None:
        """Next model up, or None at the top.

        Used after a cheap model exhausts its step budget without solving. One
        retry on a stronger model is usually cheaper than letting the weak model
        grind, because the strong model resolves in few steps and the quadratic
        works in its favour.
        """
        ladder = catalog.LADDER
        index = ladder.index(current)
        return ladder[index + 1] if index + 1 < len(ladder) else None
