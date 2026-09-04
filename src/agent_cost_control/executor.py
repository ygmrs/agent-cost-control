"""Executing an agent task.

The controller in this repository is real code. The executor is a simulation,
and the boundary matters: no live provider will produce a reproducible cost
distribution, and a benchmark nobody can rerun is not a benchmark. What is
simulated is *how many steps a task takes and what each step costs*. What is
real is every decision made about that -- estimation, routing, caching, and
ceiling enforcement all run unchanged against a live provider.

Two properties of the simulation are worth stating because the results depend
on them:

**Context accumulates.** Step ``k`` resends the transcript built by steps
``0..k-1``, so input tokens grow linearly and total cost grows roughly
quadratically in step count. This is not a modelling flourish; it is the
mechanism that turns a 6x difference in steps into a 36x difference in dollars.

**Configurations share random numbers.** Each task carries a seed derived from
its id, so the same task meets the same sequence of draws under every
configuration. Without this, comparing arms would mostly measure luck: an arm
could look 30% cheaper because its expensive tasks happened to resolve early.
Common random numbers is the standard variance-reduction technique for exactly
this, and it is why differences in the benchmark are attributable to the
controls.
"""
from __future__ import annotations

import hashlib
import math
import random
from typing import Protocol

from .catalog import MODELS
from .models import Attempt, ModelSpec, Outcome, StepRecord, Task

MAX_STEPS = 14
# Tokens the agent emits per step, and the tool result it reads back. Together
# these set how fast the transcript grows.
OUTPUT_TOKENS_PER_STEP = 320
TOOL_RESULT_TOKENS = 540
SYSTEM_TOKENS = 700
# Slope of the success curve. Higher makes capability a sharper threshold;
# this value gives a capable model roughly 80% per-step success on a task well
# within its range and roughly 20% on one beyond it.
SUCCESS_SLOPE = 5.0
# Baseline friction. Without it a capable model one-shots almost everything and
# the cost distribution is flat -- but real agent loops explore, act and verify
# even on easy work, so the floor is two to three steps rather than one.
SUCCESS_OFFSET = 1.6


def task_seed(task: Task) -> int:
    """Stable per-task seed. Same task, same luck, under every configuration.

    Hashed with blake2b rather than the builtin ``hash``. Python randomizes
    string hashing per process unless PYTHONHASHSEED is pinned, so the obvious
    implementation produces different results on every run -- the benchmark
    reproduced perfectly inside one process and disagreed with itself across
    two, which is worse than being wrong because it looks stable under testing.
    """
    digest = hashlib.blake2b(task.id.encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**31)


def step_success_probability(model: ModelSpec, task: Task) -> float:
    """Per-step probability that the agent resolves the task.

    A logistic in (capability - difficulty). The shape matters more than the
    constants: capability above difficulty resolves quickly, capability below it
    grinds, and the grinding is what costs money.
    """
    exponent = SUCCESS_SLOPE * (model.capability - task.difficulty) - SUCCESS_OFFSET
    return 1.0 / (1.0 + math.exp(-exponent))


class Executor(Protocol):
    def run(self, task: Task, model_name: str, on_step=None) -> Attempt: ...


class SimulatedExecutor:
    """Deterministic agent loop.

    ``on_step`` is the hook the budget controller uses to intervene mid-flight.
    It receives the running cost after each step and returns True to continue or
    False to abort. Enforcement has to live here rather than in a wrapper: a
    ceiling checked only before and after execution cannot stop a task that
    discovers it is expensive on step nine.
    """

    name = "simulated"

    def run(self, task: Task, model_name: str, on_step=None) -> Attempt:
        model = MODELS[model_name]
        rng = random.Random(task_seed(task))
        attempt = Attempt(task_id=task.id, outcome=Outcome.EXHAUSTED,
                          models_used=[model_name])

        transcript_tokens = 0
        p_success = step_success_probability(model, task)

        for index in range(MAX_STEPS):
            input_tokens = SYSTEM_TOKENS + task.prompt_tokens + transcript_tokens
            output_tokens = OUTPUT_TOKENS_PER_STEP
            cost = model.step_cost(input_tokens, output_tokens)

            attempt.steps.append(StepRecord(index=index, model=model_name,
                                            input_tokens=input_tokens,
                                            output_tokens=output_tokens,
                                            cost_usd=cost))

            # The transcript carries forward what the agent said and what the
            # tool returned, so the next step pays for both.
            transcript_tokens += output_tokens + TOOL_RESULT_TOKENS

            if rng.random() < p_success:
                attempt.outcome = Outcome.SUCCESS
                return attempt

            # Only consult the guard when a further step is actually possible.
            # Calling it after the last step relabels ordinary step-limit
            # exhaustion as a budget abort, which overstates what the ceiling
            # did and understates how often the loop simply ran out of room.
            if index < MAX_STEPS - 1 and on_step is not None and not on_step(attempt):
                attempt.outcome = Outcome.ABORTED
                return attempt

        return attempt


class ProviderExecutor:  # pragma: no cover - requires an API key
    """Live execution against a real provider.

    Present to make the boundary concrete: the controller calls this exactly as
    it calls the simulator, because every control operates on token counts and
    prices rather than on anything the simulation invents. Token accounting
    comes from the provider's usage fields; the loop, the ceiling hook, and the
    cost arithmetic are the same code.
    """

    name = "provider"

    def __init__(self, model_map: dict[str, str] | None = None):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model_map = model_map or {
            "small": "claude-haiku-4-5",
            "mid": "claude-sonnet-5",
            "frontier": "claude-opus-5",
        }

    def run(self, task: Task, model_name: str, on_step=None) -> Attempt:
        spec = MODELS[model_name]
        attempt = Attempt(task_id=task.id, outcome=Outcome.EXHAUSTED,
                          models_used=[model_name])
        messages = [{"role": "user", "content": task.prompt}]

        for index in range(MAX_STEPS):
            response = self.client.messages.create(
                model=self.model_map[model_name], max_tokens=1024,
                messages=messages)
            usage = response.usage
            cost = spec.step_cost(usage.input_tokens, usage.output_tokens)
            attempt.steps.append(StepRecord(index=index, model=model_name,
                                            input_tokens=usage.input_tokens,
                                            output_tokens=usage.output_tokens,
                                            cost_usd=cost))

            text = "".join(b.text for b in response.content if b.type == "text")
            if "TASK_COMPLETE" in text:
                attempt.outcome = Outcome.SUCCESS
                return attempt

            messages.append({"role": "assistant", "content": text or "(no output)"})
            messages.append({"role": "user", "content": "continue"})

            if index < MAX_STEPS - 1 and on_step is not None and not on_step(attempt):
                attempt.outcome = Outcome.ABORTED
                return attempt

        return attempt
