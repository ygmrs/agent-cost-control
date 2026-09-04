"""Model catalog and task corpus.

**On the prices.** These are a dated snapshot in the shape of 2026 tiered
pricing, not a live price list, and they are configurable. What the benchmark
depends on is the *ratio* between tiers -- roughly 15x from small to frontier --
which has been stable across vendors and generations even as absolute prices
fell. Substituting today's exact numbers moves the dollar figures and leaves the
conclusions intact.

**On the corpus.** Difficulty is drawn from a right-skewed distribution rather
than a uniform one, because that is what produces the reported failure: most
tasks are ordinary and a minority are pathological, and the pathological
minority is where the spend goes. A uniform corpus would show a tidy cost
distribution and prove nothing.
"""
from __future__ import annotations

import random

from .models import ModelSpec, Task

# Capability is on the same [0, 1] scale as task difficulty, so a model with
# capability 0.55 resolves a difficulty-0.55 task at even odds per step.
MODELS: dict[str, ModelSpec] = {
    "small": ModelSpec("small", capability=0.40,
                       input_usd_per_mtok=0.25, output_usd_per_mtok=1.25),
    "mid": ModelSpec("mid", capability=0.62,
                     input_usd_per_mtok=1.00, output_usd_per_mtok=5.00),
    "frontier": ModelSpec("frontier", capability=0.82,
                          input_usd_per_mtok=3.75, output_usd_per_mtok=18.75),
}

DEFAULT_MODEL = "frontier"
LADDER = ["small", "mid", "frontier"]   # ascending capability and price


# Task families. Each has a difficulty centre; the spread is what creates the
# tail. Prompts are short because prompt length is a poor difficulty signal --
# and the estimator having to work without one is part of the problem.
CATEGORIES = [
    ("lookup", 0.22, "find where {thing} is configured in this service"),
    ("edit", 0.34, "update {thing} to use the new interface"),
    ("bugfix", 0.52, "fix the failing test in {thing} without changing behaviour"),
    ("refactor", 0.58, "extract {thing} into a separate module and update callers"),
    ("integration", 0.68, "wire {thing} through the request path end to end"),
    ("migration", 0.76, "migrate {thing} off the deprecated client"),
]

SUBJECTS = [
    "the retry policy", "the auth middleware", "the pagination helper",
    "the rate limiter", "the cache layer", "the webhook dispatcher",
    "the session store", "the audit logger", "the schema validator",
    "the connection pool", "the feature flag client", "the metrics exporter",
    "the token refresh loop", "the idempotency key store", "the outbox worker",
    "the request tracer", "the config loader", "the batch scheduler",
    "the dead letter queue", "the health probe", "the tenant resolver",
    "the signature verifier", "the migration runner", "the backoff calculator",
    "the payload serializer", "the quota tracker", "the event deduplicator",
    "the cursor encoder", "the permission cache", "the retry budget",
    "the shard router", "the checkpoint store", "the fan-out publisher",
    "the timeout controller", "the credential rotator", "the index builder",
]

SEED = 20260819


def build_tasks(count: int = 400, seed: int = SEED) -> list[Task]:
    """Generate the task corpus deterministically.

    Difficulty is the category centre plus a right-skewed perturbation: most
    tasks land near the centre, a few land far above it. Those few are the ones
    that produce the expensive tail, and every control in this project is judged
    on what it does about them.
    """
    rng = random.Random(seed)
    tasks: list[Task] = []
    for i in range(count):
        category, centre, template = CATEGORIES[i % len(CATEGORIES)]
        subject = rng.choice(SUBJECTS)
        # Lognormal-ish skew: symmetric noise would give a symmetric cost
        # distribution and hide the tail the project exists to bound.
        skew = rng.lognormvariate(mu=-2.0, sigma=0.70)
        difficulty = min(0.97, max(0.05, centre + skew - 0.10))
        prompt = template.format(thing=subject)
        tasks.append(Task(
            id=f"task_{i:04d}",
            category=category,
            prompt=prompt,
            difficulty=difficulty,
            # Prompt size varies little and correlates weakly with difficulty,
            # which is exactly why estimating from prompt length alone fails.
            prompt_tokens=rng.randint(320, 900),
        ))
    return tasks


def split_tasks(tasks: list[Task], warmup: int = 120) -> tuple[list[Task], list[Task]]:
    """Split into a warm-up prefix and the measured set.

    The estimator and cache learn from history, so measuring them on the same
    tasks they learned from would report memorization as prediction. The warm-up
    prefix populates history; every number reported comes from the tasks that
    follow it.
    """
    return tasks[:warmup], tasks[warmup:]
