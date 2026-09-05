"""Model catalog and task corpus.

**Models are data.** Prices and capability profiles live in
``model_catalog.json`` and are loaded at import; ``--catalog FILE`` overrides
it. Repricing, adding a tier, or pointing the benchmark at a different vendor
is an edit to a JSON file rather than a code change, and
``test_conclusions_survive_a_different_price_table`` runs the whole ablation
against a second table to check the findings turn on the *ratio* between tiers
rather than on any particular numbers.

**The corpus is parameterised.** Difficulty is right-skewed, because that is
what produces the reported failure: most tasks are ordinary, a minority are
pathological, and the pathological minority is where the spend goes. Prompt
size scales with scope plus noise, the way a larger ask is written at greater
length. Repeats are injected at an explicit ``repeat_rate`` so the cache hit
rate is an input to the benchmark and not an accident of how many subject and
template combinations happened to collide.
"""
from __future__ import annotations

import json
import random
from importlib import resources
from pathlib import Path

from .models import ModelSpec, Task

CATALOG_FILE = "model_catalog.json"

# Populated by ``load_catalog`` at import. ``MODELS`` is mutated in place rather
# than rebound, so modules that imported the dict keep seeing current prices.
MODELS: dict[str, ModelSpec] = {}
LADDER: list[str] = []          # ascending capability and price
DEFAULT_MODEL: str = ""


def read_catalog(path: str | Path | None = None) -> dict:
    """Read a catalog document from ``path``, or the packaged default."""
    if path is not None:
        return json.loads(Path(path).read_text())
    return json.loads(
        resources.files(__package__).joinpath(CATALOG_FILE).read_text())


def load_catalog(path: str | Path | None = None) -> dict[str, ModelSpec]:
    """Install a catalog as the active one.

    Validates the ladder against the model table, because a ladder naming a
    model that does not exist would fail later inside routing, at a point far
    from the mistake.
    """
    global DEFAULT_MODEL

    document = read_catalog(path)
    models = {
        name: ModelSpec(
            name=name,
            capability=float(spec["capability"]),
            input_usd_per_mtok=float(spec["input_usd_per_mtok"]),
            output_usd_per_mtok=float(spec["output_usd_per_mtok"]),
            capability_by_category={
                k: float(v)
                for k, v in spec.get("capability_by_category", {}).items()},
        )
        for name, spec in document["models"].items()
    }

    ladder = list(document["ladder"])
    unknown = [name for name in ladder if name not in models]
    if unknown:
        raise ValueError(f"catalog ladder names unknown models: {unknown}")
    default = document["default"]
    if default not in models:
        raise ValueError(f"catalog default names an unknown model: {default}")

    prices = [models[name].input_usd_per_mtok for name in ladder]
    if prices != sorted(prices):
        raise ValueError("catalog ladder must be ordered cheapest first")

    MODELS.clear()
    MODELS.update(models)
    LADDER[:] = ladder
    DEFAULT_MODEL = default
    return MODELS


load_catalog()


# Task families. Each has a difficulty centre; the spread is what creates the
# tail.
CATEGORIES = [
    ("lookup", 0.22, "find where {thing} is configured in {place}"),
    ("edit", 0.34, "update {thing} in {place} to use the new interface"),
    ("bugfix", 0.52, "fix the failing test for {thing} in {place}"),
    ("refactor", 0.58, "extract {thing} in {place} into a module and update callers"),
    ("integration", 0.68, "wire {thing} through the request path in {place}"),
    ("migration", 0.76, "migrate {thing} in {place} off the deprecated client"),
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

# A second axis so that two tasks colliding word for word is a repeat someone
# actually issued twice, not an artefact of a small subject list.
PLACES = [
    "the billing service", "the ingestion path", "the admin API",
    "the notification worker", "the search indexer", "the reporting job",
    "the gateway", "the sync daemon", "the export pipeline",
    "the onboarding flow", "the webhook receiver", "the ledger service",
]

SEED = 20260819
DEFAULT_REPEAT_RATE = 0.25

# Prompt size as a function of scope. Bigger jobs get described at greater
# length, but weakly enough that size alone is a poor predictor -- the
# estimator has to combine it with the category to get anything out of it.
PROMPT_TOKENS_BASE = 260
PROMPT_TOKENS_PER_DIFFICULTY = 700
PROMPT_TOKENS_NOISE = 120


def build_tasks(count: int = 400, seed: int = SEED,
                repeat_rate: float = DEFAULT_REPEAT_RATE) -> list[Task]:
    """Generate the task corpus deterministically.

    ``repeat_rate`` is the share of tasks that restate an earlier request
    verbatim. It is an explicit input because the cache's value depends
    entirely on it, and a hit rate that emerged from the corpus by accident
    would make the cache row unfalsifiable: nobody could tell whether the
    saving was a property of the control or of the fixture.
    """
    if not 0.0 <= repeat_rate < 1.0:
        raise ValueError("repeat_rate must be in [0, 1)")

    rng = random.Random(seed)
    tasks: list[Task] = []
    for i in range(count):
        # Repeats can only refer backwards, and need a prefix to refer to.
        if tasks and i >= 20 and rng.random() < repeat_rate:
            source = tasks[rng.randrange(max(0, i - 80), i)]
            tasks.append(Task(id=f"task_{i:04d}", category=source.category,
                              prompt=source.prompt, difficulty=source.difficulty,
                              prompt_tokens=source.prompt_tokens))
            continue

        category, centre, template = CATEGORIES[i % len(CATEGORIES)]
        # Lognormal-ish skew: symmetric noise would give a symmetric cost
        # distribution and hide the tail the project exists to bound.
        skew = rng.lognormvariate(mu=-2.0, sigma=0.70)
        difficulty = min(0.97, max(0.05, centre + skew - 0.10))
        prompt_tokens = int(PROMPT_TOKENS_BASE
                            + PROMPT_TOKENS_PER_DIFFICULTY * difficulty
                            + rng.gauss(0.0, PROMPT_TOKENS_NOISE))
        tasks.append(Task(
            id=f"task_{i:04d}",
            category=category,
            prompt=template.format(thing=rng.choice(SUBJECTS),
                                   place=rng.choice(PLACES)),
            difficulty=difficulty,
            prompt_tokens=min(1200, max(200, prompt_tokens)),
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
