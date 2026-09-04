"""Reusing work already paid for.

Agent workloads repeat. The same category of task arrives against the same
subject, phrased slightly differently, and re-running it costs full price for an
answer already bought.

Matching is lexical -- normalized tokens, Jaccard overlap above a threshold.
An embedding model would generalize further, and the interface allows one, but
lexical matching is enough to demonstrate the effect and it keeps the benchmark
reproducible without a provider. What matters for the result is the hit rate and
what a hit saves, not how similarity is computed.

The threshold is set high on purpose. A cache that returns a near-miss is worse
than no cache: it answers the wrong question at low cost and with full
confidence, and nothing downstream knows.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .models import Attempt, Task

TOKEN = re.compile(r"[a-z0-9]+")
STOPWORDS = frozenset("the a an to in of for this that and or is are be".split())


def normalize(text: str) -> frozenset[str]:
    return frozenset(t for t in TOKEN.findall(text.lower()) if t not in STOPWORDS)


def jaccard(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class SemanticCache:
    threshold: float = 0.85
    # (category, token set) -> the attempt that answered it
    entries: list[tuple[str, frozenset[str], Attempt]] = field(default_factory=list)
    hits: int = 0
    misses: int = 0

    def lookup(self, task: Task) -> Attempt | None:
        """Return a prior answer for an equivalent task, or None.

        Category must match exactly. Two tasks can share most of their wording
        and mean entirely different work -- "find where the retry policy is
        configured" and "migrate the retry policy off the deprecated client"
        overlap heavily in tokens and not at all in effort.
        """
        signature = normalize(task.prompt)
        for category, cached_signature, attempt in self.entries:
            if category != task.category:
                continue
            if jaccard(signature, cached_signature) >= self.threshold:
                self.hits += 1
                return attempt
        self.misses += 1
        return None

    def store(self, task: Task, attempt: Attempt) -> None:
        """Cache a result. Only successes are stored.

        Caching a failure would make the system permanently give up on a task
        that a later, stronger model could solve -- turning a transient outcome
        into a durable one.
        """
        if not attempt.solved:
            return
        self.entries.append((task.category, normalize(task.prompt), attempt))

    @property
    def hit_rate(self) -> float:
        total = self.hits + self.misses
        return self.hits / total if total else 0.0
