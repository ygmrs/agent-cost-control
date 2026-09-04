"""Measurement.

The headline number is the ratio between the expensive tail and the typical
case, not the mean. Teams report the failure as unpredictability -- the same
operation costing a dollar one time and ten the next -- and a mean hides exactly
that.

Solve rate is reported beside every cost figure, without exception. Cost
variance can be driven to nearly zero by refusing everything, so a cost result
without a solve rate next to it is not a result. Every table here carries both.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .catalog import build_tasks, split_tasks
from .control import ControlConfig, Controller
from .models import Attempt, Outcome


@dataclass
class Report:
    config: str
    attempts: list[Attempt] = field(default_factory=list)
    cache_hit_rate: float = 0.0

    # ------------------------------------------------------------------ cost
    def _costs(self) -> np.ndarray:
        return np.array([a.cost_usd for a in self.attempts]) if self.attempts \
            else np.array([0.0])

    @property
    def total_usd(self) -> float:
        return float(self._costs().sum())

    @property
    def mean_usd(self) -> float:
        return float(self._costs().mean())

    @property
    def p50_usd(self) -> float:
        return float(np.percentile(self._costs(), 50))

    @property
    def p99_usd(self) -> float:
        return float(np.percentile(self._costs(), 99))

    def _executed_costs(self) -> np.ndarray:
        """Costs of tasks that actually ran.

        Cached and refused tasks cost nothing, so including them drags the
        median to zero and makes a p99/p50 ratio meaningless -- an arm would
        look infinitely unpredictable precisely because it avoided most of the
        work. Predictability is a property of what executes.
        """
        costs = [a.cost_usd for a in self.attempts if a.steps]
        return np.array(costs) if costs else np.array([0.0])

    @property
    def executed_p50_usd(self) -> float:
        return float(np.percentile(self._executed_costs(), 50))

    @property
    def executed_p99_usd(self) -> float:
        return float(np.percentile(self._executed_costs(), 99))

    @property
    def variance_ratio(self) -> float:
        """p99 / p50 over executed tasks -- the unpredictability being bounded."""
        p50 = self.executed_p50_usd
        return self.executed_p99_usd / p50 if p50 > 0 else float("inf")

    # --------------------------------------------------------------- quality
    @property
    def solve_rate(self) -> float:
        if not self.attempts:
            return 0.0
        return sum(1 for a in self.attempts if a.solved) / len(self.attempts)

    @property
    def refused(self) -> int:
        return sum(1 for a in self.attempts if a.outcome is Outcome.REFUSED)

    @property
    def aborted(self) -> int:
        return sum(1 for a in self.attempts if a.outcome is Outcome.ABORTED)

    @property
    def cost_per_solved(self) -> float:
        """Total spend divided by tasks actually solved.

        The metric that resists gaming. Refusing work lowers total cost and
        raises this, so an arm cannot look good by simply doing less.
        """
        solved = sum(1 for a in self.attempts if a.solved)
        return self.total_usd / solved if solved else float("inf")

    # -------------------------------------------------------------- accuracy
    @property
    def estimate_coverage(self) -> float:
        """Share of executed tasks whose actual cost fell inside the p10-p90 band.

        A well-calibrated interval should cover about 80%. Much higher means the
        band is uselessly wide; much lower means the ceiling is reserving against
        a bound that does not hold.
        """
        executed = [a for a in self.attempts
                    if a.steps and a.estimate_interval != (0.0, 0.0)]
        if not executed:
            return 0.0
        inside = sum(1 for a in executed
                     if a.estimate_interval[0] <= a.cost_usd <= a.estimate_interval[1])
        return inside / len(executed)


def configurations(ceiling_usd: float = 0.15) -> list[ControlConfig]:
    """The ablation. Each arm adds exactly one mechanism to the previous."""
    return [
        ControlConfig(name="baseline", caching=False, routing=False,
                      ceiling=False, escalation=False),
        ControlConfig(name="+ routing", caching=False, routing=True,
                      ceiling=False, escalation=False),
        ControlConfig(name="+ escalation", caching=False, routing=True,
                      ceiling=False, escalation=True),
        ControlConfig(name="+ ceiling", caching=False, routing=True,
                      ceiling=True, escalation=True, ceiling_usd=ceiling_usd),
        ControlConfig(name="+ cache (all)", caching=True, routing=True,
                      ceiling=True, escalation=True, ceiling_usd=ceiling_usd),
    ]


def run_benchmark(task_count: int = 400, warmup: int = 120,
                  ceiling_usd: float = 0.15) -> list[Report]:
    """Score every configuration over the same tasks.

    Each arm gets a fresh controller so learned history does not leak between
    them, and every arm sees the identical task sequence. Because the executor
    seeds from the task id, an arm also meets the identical luck -- differences
    are attributable to the controls rather than to which tasks happened to
    resolve early.
    """
    tasks = build_tasks(count=task_count)
    warm_tasks, measured = split_tasks(tasks, warmup=warmup)

    reports = []
    for config in configurations(ceiling_usd=ceiling_usd):
        controller = Controller(config)
        controller.warm(warm_tasks)
        # Cache statistics are measured on the scored set only; warm-up hits
        # would inflate the rate with tasks nobody was charged for.
        controller.cache.hits = controller.cache.misses = 0

        report = Report(config=config.name)
        for task in measured:
            report.attempts.append(controller.run(task))
        report.cache_hit_rate = controller.cache.hit_rate
        reports.append(report)
    return reports
