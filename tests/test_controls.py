"""Tests.

Two matter more than the rest.

``test_controls_never_read_task_difficulty`` pins the invariant the whole
benchmark rests on: difficulty exists only inside the simulated executor, and a
control that read it would be clairvoyant.

``test_cost_is_superlinear_in_steps`` pins the mechanism the project is about.
If cost were linear in steps there would be no tail worth bounding and no
reason for any of this to exist.
"""
from __future__ import annotations

import ast
import inspect

import pytest

from agent_cost_control.budget import CeilingPolicy
from agent_cost_control.cache import SemanticCache, jaccard, normalize
from agent_cost_control.catalog import (CATEGORIES, MODELS, build_tasks,
                                        split_tasks)
from agent_cost_control.control import ControlConfig, Controller
from agent_cost_control.estimate import Estimator, History, cost_of_steps
from agent_cost_control.executor import SimulatedExecutor, step_success_probability
from agent_cost_control.models import Attempt, Outcome, Task
from agent_cost_control.route import Router


@pytest.fixture(scope="module")
def tasks():
    return build_tasks()


# ------------------------------------------------------------ the invariant
def test_controls_never_read_task_difficulty():
    """Difficulty is ground truth. Only the simulated executor may see it.

    Checked on the AST rather than by grepping, because these modules discuss
    difficulty in prose while explaining that they must not use it.
    """
    from agent_cost_control import budget, cache, control, estimate, route

    for module in (estimate, route, cache, budget, control):
        tree = ast.parse(inspect.getsource(module))
        attributes = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        assert "difficulty" not in attributes, (
            f"{module.__name__} reads task difficulty, which is ground truth")


# ---------------------------------------------------------- cost mechanics
def test_cost_is_superlinear_in_steps():
    """Doubling steps must more than double cost.

    The transcript is resent every step, so input tokens grow with step index.
    This is the mechanism behind the tail; without it there is nothing to bound.
    """
    model = MODELS["frontier"]
    two = cost_of_steps(model, prompt_tokens=500, steps=2)
    four = cost_of_steps(model, prompt_tokens=500, steps=4)
    eight = cost_of_steps(model, prompt_tokens=500, steps=8)

    assert four > 2 * two
    assert eight > 2 * four
    # And the effect compounds: 4x the steps is well over 4x the cost.
    assert eight / two > 4.0


def test_closed_form_cost_matches_the_executor():
    """The estimator's formula must agree with what execution actually charges.

    They are separate implementations -- one closed-form, one accumulated in a
    loop -- and a divergence would make every estimate quietly wrong.
    """
    task = Task(id="t", category="edit", prompt="x", difficulty=0.99,
                prompt_tokens=500)
    attempt = SimulatedExecutor().run(task, "small")
    predicted = cost_of_steps(MODELS["small"], task.prompt_tokens, attempt.step_count)
    assert attempt.cost_usd == pytest.approx(predicted, rel=1e-9)


def test_harder_tasks_take_more_steps():
    """Measured across many tasks, not one pair.

    Step count is stochastic: an easy task and a hard one can both resolve on
    step one, and asserting on a single pair tests luck rather than the model.
    The relationship is a property of the distribution, so it has to be measured
    over a sample.
    """
    executor = SimulatedExecutor()
    easy = [Task(f"e{i}", "lookup", "x", difficulty=0.15, prompt_tokens=400)
            for i in range(60)]
    hard = [Task(f"h{i}", "migration", "x", difficulty=0.90, prompt_tokens=400)
            for i in range(60)]

    easy_steps = sum(executor.run(t, "frontier").step_count for t in easy) / len(easy)
    hard_steps = sum(executor.run(t, "frontier").step_count for t in hard) / len(hard)
    assert hard_steps > easy_steps * 2, f"easy={easy_steps:.1f} hard={hard_steps:.1f}"


def test_stronger_models_succeed_more_often_per_step():
    task = Task("t", "bugfix", "x", difficulty=0.60, prompt_tokens=400)
    probabilities = [step_success_probability(MODELS[m], task)
                     for m in ("small", "mid", "frontier")]
    assert probabilities == sorted(probabilities)


def test_execution_is_deterministic_for_a_task():
    """Common random numbers: same task, same luck, every configuration.

    Without this, comparing arms would mostly measure which tasks happened to
    resolve early.
    """
    task = build_tasks(count=10)[3]
    executor = SimulatedExecutor()
    first = executor.run(task, "mid")
    second = executor.run(task, "mid")
    assert first.step_count == second.step_count
    assert first.cost_usd == second.cost_usd


# ----------------------------------------------------------------- routing
def test_router_explores_before_it_exploits():
    """The bug this guards against made routing a no-op.

    Selecting only on observed solve rate means an empty history falls back to
    the strongest model, which then collects all the evidence, so cheaper models
    are never tried and never chosen. Routing silently degrades to "always
    frontier" while looking like a policy.
    """
    router = Router(estimator=Estimator(History()))
    task = Task("t", "lookup", "x", difficulty=0.3, prompt_tokens=400)
    assert router.select(task) == "small", "must probe the cheapest model first"


def test_router_exploits_once_evidence_exists():
    history = History()
    task = Task("t", "lookup", "find the thing", difficulty=0.3, prompt_tokens=400)
    # small is reliable here; mid and frontier are never needed.
    for _ in range(10):
        history.record(task, "small", Attempt("t", Outcome.SUCCESS,
                                              steps=[], models_used=["small"]))
    for key in (("lookup", "mid"), ("lookup", "frontier")):
        history.steps[key] = [3] * 10
        history.solved[key] = [True] * 10

    router = Router(estimator=Estimator(history))
    assert router.select(task) == "small"


def test_router_escalates_up_the_ladder_and_stops_at_the_top():
    router = Router(estimator=Estimator(History()))
    assert router.escalate("small") == "mid"
    assert router.escalate("mid") == "frontier"
    assert router.escalate("frontier") is None


# ------------------------------------------------------------------- cache
def test_cache_matches_only_within_a_category():
    """Near-identical wording can mean entirely different work.

    "find where the retry policy is configured" and "migrate the retry policy
    off the deprecated client" share most of their tokens and none of their
    effort.
    """
    cache = SemanticCache()
    lookup = Task("a", "lookup", "find where the retry policy is configured",
                  0.3, 400)
    migration = Task("b", "migration", "find where the retry policy is configured",
                     0.8, 400)
    cache.store(lookup, Attempt("a", Outcome.SUCCESS, models_used=["small"]))
    assert cache.lookup(migration) is None


def test_cache_does_not_store_failures():
    """Caching a failure makes a transient outcome permanent."""
    cache = SemanticCache()
    task = Task("a", "lookup", "find the thing", 0.3, 400)
    cache.store(task, Attempt("a", Outcome.EXHAUSTED, models_used=["small"]))
    assert cache.lookup(task) is None


def test_cache_returns_a_hit_for_an_identical_task():
    cache = SemanticCache()
    task = Task("a", "lookup", "find where the retry policy is configured", 0.3, 400)
    cache.store(task, Attempt("a", Outcome.SUCCESS, models_used=["small"]))
    assert cache.lookup(task) is not None


def test_similarity_is_symmetric_and_bounded():
    a, b = normalize("fix the failing test"), normalize("fix the broken test")
    assert jaccard(a, b) == jaccard(b, a)
    assert 0.0 <= jaccard(a, b) <= 1.0
    assert jaccard(a, a) == 1.0


# ----------------------------------------------------------------- ceiling
def test_ceiling_refuses_a_category_it_cannot_afford():
    """Pre-flight refusal fires when typical cost exceeds the ceiling."""
    history = History()
    task = Task("t", "migration", "migrate the thing", 0.9, 800)
    history.steps[("migration", "frontier")] = [12] * 10
    history.solved[("migration", "frontier")] = [True] * 10

    policy = CeilingPolicy(estimator=Estimator(history), ceiling_usd=0.001)
    admitted, estimate = policy.admit(task, "frontier")
    assert not admitted
    assert estimate.expected_usd > 0.001


def test_ceiling_admits_what_it_can_afford():
    history = History()
    task = Task("t", "lookup", "find the thing", 0.2, 400)
    history.steps[("lookup", "small")] = [2] * 10
    history.solved[("lookup", "small")] = [True] * 10

    policy = CeilingPolicy(estimator=Estimator(history), ceiling_usd=1.0)
    admitted, _ = policy.admit(task, "small")
    assert admitted


def test_ceiling_aborts_a_run_that_goes_long():
    """Mid-flight enforcement. A ceiling checked only before and after is a report."""
    history = History()
    task = Task("t", "migration", "migrate the thing", difficulty=0.97,
                prompt_tokens=800)
    history.steps[("migration", "frontier")] = [3] * 10
    history.solved[("migration", "frontier")] = [True] * 10

    policy = CeilingPolicy(estimator=Estimator(history), ceiling_usd=0.02)
    attempt = SimulatedExecutor().run(task, "frontier",
                                      on_step=policy.step_guard(task, "frontier"))
    assert attempt.outcome is Outcome.ABORTED
    unbounded = SimulatedExecutor().run(task, "frontier")
    assert attempt.step_count < unbounded.step_count


# ------------------------------------------------------------- integration
def test_warmup_is_excluded_from_measurement(tasks):
    warm, measured = split_tasks(tasks, warmup=120)
    assert len(warm) == 120
    assert len(measured) == len(tasks) - 120
    assert not ({t.id for t in warm} & {t.id for t in measured})


def test_full_controls_beat_baseline_on_cost_per_solved():
    from agent_cost_control.bench import run_benchmark
    reports = run_benchmark(task_count=240, warmup=80)
    baseline, full = reports[0], reports[-1]
    assert full.cost_per_solved < baseline.cost_per_solved
    # And the tail, which is the point.
    assert full.executed_p99_usd < baseline.executed_p99_usd


def test_benchmark_is_reproducible():
    from agent_cost_control.bench import run_benchmark
    first = run_benchmark(task_count=240, warmup=80)
    second = run_benchmark(task_count=240, warmup=80)
    for a, b in zip(first, second):
        assert a.total_usd == pytest.approx(b.total_usd, rel=1e-12)
        assert a.solve_rate == b.solve_rate


def test_refusing_everything_is_visible_as_a_collapsed_solve_rate():
    """Cost can always be driven down by doing less; the metrics must show it."""
    from agent_cost_control.bench import Report

    controller = Controller(ControlConfig(ceiling_usd=1e-9))
    warm, measured = split_tasks(build_tasks(count=200), warmup=60)
    controller.warm(warm)
    report = Report(config="starved")
    for task in measured:
        report.attempts.append(controller.run(task))

    assert report.refused > 0
    assert report.solve_rate < 0.5
    assert report.cost_per_solved > 0


def test_task_seed_is_process_independent():
    """Regression: the seed used Python's builtin hash().

    String hashing is randomized per process unless PYTHONHASHSEED is pinned, so
    the benchmark reproduced perfectly within one process and disagreed with
    itself across two -- the worst failure mode, because in-process tests said
    it was stable. Verified here against a known-good digest so a change to the
    hashing scheme has to be deliberate.
    """
    import hashlib
    import subprocess
    import sys

    from agent_cost_control.executor import task_seed

    task = Task("task_0042", "lookup", "x", 0.3, 400)
    expected = int.from_bytes(
        hashlib.blake2b(b"task_0042", digest_size=8).digest(), "big") % (2**31)
    assert task_seed(task) == expected

    # And confirm it holds in a fresh interpreter with a different hash seed.
    out = subprocess.run(
        [sys.executable, "-c",
         "from agent_cost_control.executor import task_seed;"
         "from agent_cost_control.models import Task;"
         "print(task_seed(Task('task_0042','lookup','x',0.3,400)))"],
        capture_output=True, text=True, env={"PYTHONHASHSEED": "9999",
                                             "PYTHONPATH": "src",
                                             "PATH": "/usr/bin:/bin"})
    assert out.stdout.strip() == str(expected), out.stderr


def test_benchmark_reproduces_across_processes():
    """The numbers in the README must reproduce on someone else's machine."""
    import subprocess
    import sys

    script = ("from agent_cost_control.bench import run_benchmark;"
              "rs = run_benchmark(task_count=200, warmup=60);"
              "print(round(rs[-1].total_usd, 6))")
    runs = []
    for seed in ("0", "12345"):
        out = subprocess.run([sys.executable, "-c", script], capture_output=True,
                             text=True, env={"PYTHONHASHSEED": seed,
                                             "PYTHONPATH": "src",
                                             "PATH": "/usr/bin:/bin"})
        assert out.returncode == 0, out.stderr
        runs.append(out.stdout.strip())
    assert runs[0] == runs[1], f"benchmark differs across processes: {runs}"


def test_ceiling_is_never_exceeded_including_escalation():
    """Regression: aborted runs used to overshoot the ceiling by up to 31%.

    The guard compared cost already spent against the ceiling, which stops the
    next step only after the current one has been paid for. It now prices the
    next step before taking it. Escalation is included because a retry bills a
    second run against the same ceiling.
    """
    warm, measured = split_tasks(build_tasks(), warmup=120)
    for ceiling in (0.03, 0.05, 0.15):
        controller = Controller(ControlConfig(caching=False, ceiling_usd=ceiling))
        controller.warm(warm)
        for task in measured:
            attempt = controller.run(task)
            assert attempt.cost_usd <= ceiling + 1e-12, (
                f"{attempt.task_id} spent ${attempt.cost_usd:.4f} "
                f"against a ${ceiling:.2f} ceiling")


def test_escalated_attempt_bills_every_step_it_took():
    """A merged attempt must carry the steps of both runs, not just the retry."""
    warm, measured = split_tasks(build_tasks(), warmup=120)
    controller = Controller(ControlConfig(caching=False, ceiling=False))
    controller.warm(warm)

    escalated = 0
    for task in measured:
        attempt = controller.run(task)
        if len(attempt.models_used) > 1:
            escalated += 1
            assert {s.model for s in attempt.steps} == set(attempt.models_used)
            assert attempt.cost_usd == pytest.approx(
                sum(s.cost_usd for s in attempt.steps), rel=1e-12)
    assert escalated > 0, "no escalation occurred; the test proved nothing"


# -------------------------------------------------------- regression tests
def test_step_limit_exhaustion_is_not_reported_as_a_budget_abort():
    """Regression: the guard ran after the final step.

    The executor called the budget hook once more after the last step it was
    ever going to take, so a run that simply hit the step limit was recorded as
    ABORTED. That credited the ceiling with work it did not do -- visible in the
    published $0.50 sweep row, where the ceiling was never the binding
    constraint at all.
    """
    from agent_cost_control.executor import MAX_STEPS

    history = History()
    # This id seeds a run that exhausts rather than resolving; step count is
    # stochastic, so the case has to be pinned rather than assumed.
    task = Task("z0", "migration", "migrate the thing", difficulty=0.99,
                prompt_tokens=400)
    assert SimulatedExecutor().run(task, "small").step_count == MAX_STEPS
    history.steps[("migration", "small")] = [2] * 10
    history.solved[("migration", "small")] = [True] * 10

    # A ceiling comfortably above the cost of a full-length run: nothing here
    # should ever be stopped by budget.
    ceiling = cost_of_steps(MODELS["small"], 400, MAX_STEPS) * 2
    policy = CeilingPolicy(estimator=Estimator(history), ceiling_usd=ceiling)
    attempt = SimulatedExecutor().run(task, "small",
                                      on_step=policy.step_guard(task, "small"))

    assert attempt.step_count == MAX_STEPS
    assert attempt.outcome is Outcome.EXHAUSTED


def test_escalation_respects_the_remaining_budget():
    """Regression: the retry's first step could breach the ceiling.

    A step cannot be priced without taking it, so the opening step of a retry is
    unavoidable. When the first attempt had consumed nearly the whole ceiling,
    escalating spent past it. The controller now declines to escalate at all
    unless the remainder covers that step.
    """
    from agent_cost_control.executor import MAX_STEPS

    history = History()
    for model in ("small", "mid", "frontier"):
        history.steps[("lookup", model)] = [2] * 10
        history.solved[("lookup", model)] = [True] * 10

    # Ceiling sits a fraction above a full-length run on the cheap model, so
    # the first attempt exhausts and almost nothing is left for a retry.
    ceiling = cost_of_steps(MODELS["small"], 400, MAX_STEPS) + 1e-4
    controller = Controller(ControlConfig(caching=False, ceiling_usd=ceiling),
                            history=history)
    task = Task("z0", "lookup", "find the thing", difficulty=0.99,
                prompt_tokens=400)

    attempt = controller.run(task)
    assert attempt.models_used == ["small"], "escalated with no budget for it"
    assert attempt.cost_usd <= ceiling + 1e-12


def test_a_retry_stopped_by_the_ceiling_is_reported_as_aborted():
    """Regression: a merged attempt inherited the first run's outcome.

    When the retry was cut short by the ceiling, the merged record reported
    EXHAUSTED -- so the ceiling's intervention vanished from the numbers.
    """
    warm, measured = split_tasks(build_tasks(), warmup=120)
    controller = Controller(ControlConfig(caching=False, ceiling_usd=0.15))
    controller.warm(warm)

    aborted_retries = sum(
        1 for task in measured
        if (a := controller.run(task)).outcome is Outcome.ABORTED
        and len(a.models_used) > 1)
    assert aborted_retries > 0, "no retry was stopped by the ceiling"


def test_reading_history_does_not_create_entries():
    """Reads went through a defaultdict and silently inserted empty keys."""
    history = History()
    assert history.observations("lookup", "small") == 0
    assert history.solve_rate("lookup", "small") is None
    assert not history.steps and not history.solved


def test_table_renders_with_no_rows():
    """Regression: max() over a single int raised TypeError on an empty table."""
    from agent_cost_control.cli import _table
    assert "ceiling" in _table(["ceiling", "total"], [])


# ------------------------------------------------- catalog is data, not code
def test_catalog_rejects_an_incoherent_table():
    """A ladder out of price order or naming a missing model fails at load.

    Validated where the table is read rather than deep inside routing, which is
    a long way from the typo that caused it.
    """
    import json
    import tempfile

    from agent_cost_control.catalog import load_catalog, read_catalog

    document = read_catalog()
    document["ladder"] = ["frontier", "small"]          # not cheapest first
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(document, fh)
        path = fh.name
    with pytest.raises(ValueError):
        load_catalog(path)
    load_catalog()                                       # restore the default


def test_conclusions_survive_a_different_price_table():
    """Prices are an input, and the findings must not depend on the snapshot.

    The whole ablation is re-run against a table with different absolute prices
    but a comparable spread between tiers. Dollar figures move; the ordering
    the project reports must not.
    """
    import json
    import tempfile

    from agent_cost_control.bench import run_benchmark
    from agent_cost_control.catalog import load_catalog, read_catalog

    document = read_catalog()
    for spec in document["models"].values():             # halve every price
        spec["input_usd_per_mtok"] /= 2
        spec["output_usd_per_mtok"] /= 2
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
        json.dump(document, fh)
        path = fh.name

    # The ceiling is an absolute dollar amount, so it has to be halved too --
    # otherwise cheaper tokens buy more steps and the arms are not comparable.
    try:
        load_catalog(path)
        cheap = run_benchmark(task_count=240, warmup=80, ceiling_usd=0.075)
    finally:
        load_catalog()
    default = run_benchmark(task_count=240, warmup=80, ceiling_usd=0.15)

    for arm in (cheap, default):
        assert arm[-1].cost_per_solved < arm[0].cost_per_solved
        assert arm[-1].executed_p99_usd < arm[0].executed_p99_usd

    for halved, full in zip(cheap, default):
        # Identical decisions, so identical quality: only the bill changed.
        assert halved.solve_rate == full.solve_rate
        assert halved.total_usd == pytest.approx(full.total_usd / 2, rel=1e-9)


# --------------------------------------------- capability is per-category
def test_capability_is_not_a_single_ordering():
    """The cheap model is nearly adequate somewhere and hopeless elsewhere.

    With scalar capability there is one correct model for every task and
    routing per category is pointless. The gap between tiers has to differ by
    category for the router to have a decision to make.
    """
    gaps = {}
    for category, _centre, _template in CATEGORIES:
        small = MODELS["small"].capability_for(category)
        frontier = MODELS["frontier"].capability_for(category)
        gaps[category] = frontier - small

    assert min(gaps.values()) < 0.35 < max(gaps.values()), gaps
    # And no model is a scalar in disguise.
    assert len({round(MODELS["small"].capability_for(c), 3) for c in gaps}) > 1


def test_routing_chooses_different_models_for_different_categories():
    """The payoff of per-category capability, measured end to end."""
    warm, measured = split_tasks(build_tasks(), warmup=120)
    controller = Controller(ControlConfig(caching=False))
    controller.warm(warm)

    chosen = {}
    for task in measured:
        chosen.setdefault(task.category, set()).add(controller.router.select(task))
    picked = {next(iter(v)) for v in chosen.values() if len(v) == 1}
    assert len(picked) > 1, f"one model won every category: {chosen}"


# ------------------------------------------------ cache accounting is honest
def test_cache_hit_rate_tracks_the_configured_repeat_rate():
    """The hit rate is an input, not a property of the fixture.

    If it emerged from how many template and subject combinations happened to
    collide, the cache row would be unfalsifiable: nobody could separate the
    control's value from the corpus's luck.
    """
    from agent_cost_control.bench import run_benchmark

    for repeat_rate in (0.0, 0.25, 0.50):
        report = run_benchmark(task_count=320, warmup=100,
                               repeat_rate=repeat_rate)[-1]
        assert abs(report.cache_hit_rate - repeat_rate) < 0.12, (
            f"repeat_rate={repeat_rate} gave {report.cache_hit_rate:.2f}")


def test_cache_hits_do_not_inflate_the_reported_solve_rate():
    """``solve_rate_executed`` is the figure a cache cannot lift.

    A hit returns a real answer, so it counts as solved -- but it was solved by
    an earlier run, and without the second figure any arm could raise its
    headline quality by repeating itself.
    """
    from agent_cost_control.bench import run_benchmark

    report = run_benchmark(task_count=320, warmup=100, repeat_rate=0.50)[-1]
    assert report.cache_hit_rate > 0.3
    assert report.solve_rate_executed < report.solve_rate
    assert report.solve_rate_executed > 0.5


# --------------------------------------- estimation uses more than category
def test_prompt_size_carries_signal_but_not_on_its_own():
    """Longer asks describe bigger jobs, weakly enough to need the category.

    A perfect correlation would make the category redundant; none would make
    the size split noise. The estimator's backoff exists because it is neither.
    """
    import statistics

    tasks = build_tasks(count=600, repeat_rate=0.0)
    sizes = [t.prompt_tokens for t in tasks]
    hardness = [t.difficulty for t in tasks]
    correlation = statistics.correlation(sizes, hardness)
    assert 0.4 < correlation < 0.95, correlation


def test_conditioning_on_scope_earns_its_place():
    """The second feature is measured rather than assumed.

    Both estimators see identical history. The gain is real but small, and the
    reason is the point of the project: cost varies mostly because step count
    is stochastic, not because scope was unmeasured. Nominal interval coverage
    is 80%, so moving toward it is the improvement, not away from it.
    """
    from agent_cost_control.bench import Report

    def score(size_aware: bool) -> tuple[float, float]:
        warm, measured = split_tasks(build_tasks(), warmup=120)
        controller = Controller(ControlConfig(caching=False, ceiling=False),
                                size_aware=size_aware)
        controller.warm(warm)
        report = Report(config=str(size_aware))
        for task in measured:
            report.attempts.append(controller.run(task))
        executed = [a for a in report.attempts if a.steps]
        error = sum(abs(a.estimated_usd - a.cost_usd) for a in executed) / len(executed)
        return error, report.estimate_coverage

    coarse_error, coarse_coverage = score(False)
    fine_error, fine_coverage = score(True)

    assert fine_error < coarse_error, (fine_error, coarse_error)
    assert abs(fine_coverage - 0.80) <= abs(coarse_coverage - 0.80), (
        fine_coverage, coarse_coverage)


def test_thin_history_falls_back_to_the_whole_category():
    """Conditioning on scope divides a category's evidence.

    An estimator that trusted a handful of neighbours would be confidently
    wrong exactly where the ceiling depends on it, so the neighbourhood is only
    used once the category has enough history to spare.
    """
    from agent_cost_control.estimate import MIN_NEIGHBOURS

    task = Task("t", "lookup", "find the thing", 0.3, prompt_tokens=300)
    history = History()
    history.steps[("lookup", "small")] = [3] * 40
    history.samples[("lookup", "small")] = [(300, 11)] * (2 * MIN_NEIGHBOURS - 1)

    estimator = Estimator(history)
    assert estimator.estimate(task, "small").expected_steps == 3.0

    history.samples[("lookup", "small")] = [(300, 11)] * (2 * MIN_NEIGHBOURS)
    assert estimator.estimate(task, "small").expected_steps == 11.0


def test_the_neighbourhood_tracks_the_task_rather_than_a_fixed_boundary():
    """Nearest neighbours, not absolute size bands.

    Scope drives prompt length and category together, so a boundary drawn in
    absolute tokens separates categories instead of splitting one, and the
    band degenerates into the category under another name.
    """
    from agent_cost_control.estimate import MIN_NEIGHBOURS, nearest_by_size

    samples = [(200, 2)] * MIN_NEIGHBOURS + [(1000, 12)] * MIN_NEIGHBOURS
    assert set(nearest_by_size(samples, 210)) == {2}
    assert set(nearest_by_size(samples, 990)) == {12}
