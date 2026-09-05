"""Command line interface."""
from __future__ import annotations

import argparse
import json
import os
import sys

from .bench import Report, run_benchmark
from .catalog import (DEFAULT_REPEAT_RATE, MODELS, build_tasks,
                      load_catalog, split_tasks)
from .control import ControlConfig, Controller


def _table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [max([len(h)] + [len(r[i]) for r in rows])
              for i, h in enumerate(headers)]
    out = ["| " + " | ".join(h.ljust(w) for h, w in zip(headers, widths)) + " |",
           "|" + "|".join("-" * (w + 2) for w in widths) + "|"]
    out += ["| " + " | ".join(c.ljust(w) for c, w in zip(r, widths)) + " |" for r in rows]
    return "\n".join(out)


def _ratio(report: Report) -> str:
    value = report.variance_ratio
    return f"{value:.1f}x" if value != float("inf") else "n/a"


def cmd_bench(args) -> int:
    if args.warmup >= args.tasks:
        print("--warmup must be smaller than --tasks")
        return 1
    reports = run_benchmark(task_count=args.tasks, warmup=args.warmup,
                            ceiling_usd=args.ceiling,
                            repeat_rate=args.repeat_rate)
    rows = [[
        r.config,
        f"${r.total_usd:.2f}",
        f"${r.executed_p50_usd:.4f}",
        f"${r.executed_p99_usd:.4f}",
        _ratio(r),
        f"{r.solve_rate:.1%}",
        f"{r.solve_rate_executed:.1%}",
        f"${r.cost_per_solved:.4f}",
        f"{r.estimate_coverage:.0%}",
        str(r.aborted),
    ] for r in reports]

    measured = args.tasks - args.warmup
    print(f"\n{measured} measured tasks after {args.warmup} warm-up · "
          f"ceiling ${args.ceiling:.2f} · repeats {args.repeat_rate:.0%} · "
          f"models {'/'.join(MODELS)}\n")
    print(_table(["configuration", "total", "p50 exec", "p99 exec", "p99/p50",
                  "solved", "solved (exec)", "$/solved", "coverage",
                  "aborted"], rows))

    base, final = reports[0], reports[-1]
    print(f"\ncost per solved task: ${base.cost_per_solved:.4f} -> "
          f"${final.cost_per_solved:.4f} "
          f"({base.cost_per_solved / final.cost_per_solved:.1f}x better)")
    print(f"tail cost (p99):      ${base.executed_p99_usd:.4f} -> "
          f"${final.executed_p99_usd:.4f} "
          f"({base.executed_p99_usd / final.executed_p99_usd:.1f}x lower)")
    print(f"solve rate:           {base.solve_rate:.1%} -> {final.solve_rate:.1%} "
          f"({(final.solve_rate - base.solve_rate) * 100:+.1f} points)")
    print("\np50/p99 are over executed tasks; cached and refused tasks cost "
          "nothing and would drag the median to zero.")
    print("'solved (exec)' excludes cache hits, which are answers an earlier "
          "run already paid for.")
    print("'coverage' is the share of runs landing inside the predicted "
          "p10-p90 interval; well-calibrated is ~80%.\n")

    if args.json:
        payload = {r.config: {
            "total_usd": r.total_usd, "mean_usd": r.mean_usd,
            "executed_p50_usd": r.executed_p50_usd,
            "executed_p99_usd": r.executed_p99_usd,
            "variance_ratio": r.variance_ratio,
            "solve_rate": r.solve_rate,
            "solve_rate_executed": r.solve_rate_executed,
            "cost_per_solved": r.cost_per_solved,
            "cache_hit_rate": r.cache_hit_rate,
            "refused": r.refused, "aborted": r.aborted,
            "estimate_coverage": r.estimate_coverage,
        } for r in reports}
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"raw results written to {args.json}\n")
    return 0


def cmd_sweep(args) -> int:
    """Show the ceiling as the dial it is: cost against solve rate."""
    if args.warmup >= args.tasks:
        print("--warmup must be smaller than --tasks")
        return 1
    rows = []
    for ceiling in (0.05, 0.08, 0.10, 0.15, 0.25, 0.50):
        report = run_benchmark(task_count=args.tasks, warmup=args.warmup,
                               ceiling_usd=ceiling,
                               repeat_rate=args.repeat_rate)[-1]
        rows.append([
            f"${ceiling:.2f}", f"${report.total_usd:.2f}",
            f"${report.executed_p99_usd:.4f}", f"{report.solve_rate:.1%}",
            f"${report.cost_per_solved:.4f}",
            str(report.refused), str(report.aborted),
        ])
    print("\nceiling sweep, all controls active\n")
    print(_table(["ceiling", "total", "p99 exec", "solve rate", "$/solved",
                  "refused", "aborted"], rows))
    print("\nA lower ceiling always looks cheaper. Read it against solve rate: "
          "spend can be driven to zero by refusing everything.\n")
    return 0


def cmd_explain(args) -> int:
    """Trace the decisions made for one task."""
    tasks = build_tasks(repeat_rate=args.repeat_rate)
    warm, measured = split_tasks(tasks)
    controller = Controller(ControlConfig(ceiling_usd=args.ceiling))
    controller.warm(warm)

    task = next((t for t in measured if t.id == args.task_id), None)
    if task is None:
        print(f"unknown task: {args.task_id}  (try {measured[0].id})")
        return 1

    # Snapshot the evidence the router actually decided on. Running the task
    # adds its own outcome to history, so reading rates afterwards would show
    # the decision being justified by its own result.
    chosen = controller.router.select(task)
    estimate = controller.estimator.estimate(task, chosen)
    admitted, _ = controller.policy.admit(task, chosen)
    evidence = {name: (controller.estimator.history.solve_rate(task.category, name),
                       controller.estimator.history.observations(task.category, name))
                for name in MODELS}
    attempt = controller.run(task)

    print(f"\ntask {task.id}  [{task.category}]")
    print(f"  prompt: {task.prompt}")
    print("\nrouting")
    for name, (rate, seen) in evidence.items():
        marker = "->" if name == chosen else "  "
        shown = f"{rate:.0%}" if rate is not None else "no evidence"
        print(f"  {marker} {name:<9} solve rate {shown:<12} ({seen} observations)")
    print(f"\nestimate on {chosen}")
    print(f"  expected ${estimate.expected_usd:.4f} over {estimate.expected_steps:.1f} steps")
    print(f"  interval ${estimate.p10_usd:.4f} - ${estimate.p90_usd:.4f}"
          f"   {'confident' if estimate.confident else 'cold start'}")
    print(f"  ceiling  ${args.ceiling:.2f}  ->  {'admitted' if admitted else 'REFUSED'}")
    print(f"\noutcome: {attempt.outcome.value}")
    if attempt.cache_hit:
        print("  served from cache; no tokens were spent")
    print(f"  models  {' -> '.join(attempt.models_used)}")
    print(f"  steps   {attempt.step_count}")
    print(f"  actual  ${attempt.cost_usd:.4f}")
    if attempt.steps:
        inside = estimate.p10_usd <= attempt.cost_usd <= estimate.p90_usd
        print(f"  {'inside' if inside else 'OUTSIDE'} the predicted interval")
    print()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agent-cost-control",
        description="Bound agent execution cost before it runs.")
    sub = parser.add_subparsers(dest="command", required=True)

    def corpus_options(sub_parser, tasks: bool = True):
        if tasks:
            sub_parser.add_argument("--tasks", type=int, default=400)
            sub_parser.add_argument("--warmup", type=int, default=120)
        sub_parser.add_argument(
            "--repeat-rate", type=float, default=DEFAULT_REPEAT_RATE,
            dest="repeat_rate",
            help="share of tasks that restate an earlier request "
                 f"(default {DEFAULT_REPEAT_RATE})")
        sub_parser.add_argument(
            "--catalog", help="model price and capability table (JSON)")
        return sub_parser

    bench = corpus_options(sub.add_parser(
        "bench", help="ablation across the four controls"))
    bench.add_argument("--ceiling", type=float, default=0.15)
    bench.add_argument("--json", help="write raw results here")
    bench.set_defaults(func=cmd_bench)

    sweep = corpus_options(sub.add_parser(
        "sweep", help="cost against solve rate as the ceiling moves"))
    sweep.set_defaults(func=cmd_sweep)

    explain = corpus_options(sub.add_parser(
        "explain", help="trace the decisions for one task"), tasks=False)
    explain.add_argument("task_id")
    explain.add_argument("--ceiling", type=float, default=0.15)
    explain.set_defaults(func=cmd_explain)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "catalog", None):
        load_catalog(args.catalog)
    try:
        return args.func(args)
    except BrokenPipeError:
        # `agent-cost-control bench | head` closes stdout mid-write. Redirect
        # the remaining writes to devnull so Python does not report the failure
        # again while flushing at exit.
        os.dup2(os.open(os.devnull, os.O_WRONLY), sys.stdout.fileno())
        return 0
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
