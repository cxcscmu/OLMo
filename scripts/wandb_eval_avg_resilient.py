#!/usr/bin/env python3
"""
Fetch eval/downstream metrics from a W&B run without full-history scans.

This script avoids the common "stuck on scanning history" pattern by:
1) normalizing run paths like entity/project/runs/<id> -> entity/project/<id>
2) discovering downstream metric keys from run.summary
3) scanning each metric key independently with scan_history(keys=["_step", key])
"""

from __future__ import annotations

import argparse
import time
from collections import defaultdict

import wandb
from prettytable import PrettyTable
from tqdm import tqdm


DOWNSTREAM_PREFIX = "eval/downstream/"
BOLD = "\033[1m"
RESET = "\033[0m"


def format_task_name(task: str) -> str:
    for suffix in ("_len_norm", "_acc"):
        if task.endswith(suffix):
            return task[: -len(suffix)]
    return task


def bold(text: str) -> str:
    return f"{BOLD}{text}{RESET}"


def normalize_run_path(run_path: str) -> str:
    parts = [p for p in run_path.strip().split("/") if p]
    if len(parts) == 4 and parts[2] == "runs":
        return f"{parts[0]}/{parts[1]}/{parts[3]}"
    if len(parts) == 3:
        return run_path.strip("/")
    raise ValueError(
        f"Invalid run path '{run_path}'. Expected 'entity/project/run_id' or 'entity/project/runs/run_id'."
    )


def discover_downstream_keys(run, prefix: str, excluded: set[str]) -> list[str]:
    summary_keys = [k for k in run.summary.keys() if k.startswith(prefix)]
    keys = sorted(k for k in summary_keys if k not in excluded)
    return keys


def fetch_metric_by_step(
    run,
    metric_key: str,
    eval_interval: int,
    page_size: int,
    min_step: int | None,
    max_step: int | None,
) -> dict[int, float]:
    values_by_step: dict[int, float] = {}
    history = run.scan_history(
        keys=["_step", metric_key],
        page_size=page_size,
        min_step=min_step,
        max_step=max_step,
    )
    for row in history:
        step = row.get("_step")
        value = row.get(metric_key)
        if not isinstance(step, int) or step <= 0:
            continue
        if eval_interval > 0 and step % eval_interval != 0:
            continue
        if value is None:
            continue
        values_by_step[step] = value
    return values_by_step


def get_eval_downstream_metrics(
    run_path: str,
    eval_interval: int,
    page_size: int,
    min_step: int | None,
    max_step: int | None,
    excluded: set[str],
) -> dict[int, dict[str, float]]:
    api = wandb.Api()
    normalized = normalize_run_path(run_path)
    run = api.run(normalized)

    keys = discover_downstream_keys(run, DOWNSTREAM_PREFIX, excluded)
    if not keys:
        return {}

    metrics_by_step: dict[int, dict[str, float]] = defaultdict(dict)
    start = time.time()

    for metric_key in tqdm(keys, desc="Scanning metrics", unit="metric"):
        per_metric = fetch_metric_by_step(
            run=run,
            metric_key=metric_key,
            eval_interval=eval_interval,
            page_size=page_size,
            min_step=min_step,
            max_step=max_step,
        )
        short_name = metric_key.replace(DOWNSTREAM_PREFIX, "")
        for step, value in per_metric.items():
            metrics_by_step[step][short_name] = value

    elapsed = time.time() - start
    print(f"Scanned {len(keys)} metrics in {elapsed:.2f}s.")
    return metrics_by_step


def probe_metric_steps(run, metric_key: str, page_size: int, limit: int = 20) -> list[int]:
    steps: list[int] = []
    history = run.scan_history(keys=["_step", metric_key], page_size=page_size)
    for row in history:
        step = row.get("_step")
        if isinstance(step, int) and step > 0:
            steps.append(step)
            if len(steps) >= limit:
                break
    return steps


def display_results(metrics_by_step: dict[int, dict[str, float]], title: str):
    if not metrics_by_step:
        print("No eval/downstream metrics found.")
        return

    all_tasks = sorted({task for tasks in metrics_by_step.values() for task in tasks})
    best_avg = -float("inf")
    best_avg_steps: set[int] = set()
    best_task_values: dict[str, float] = {}

    for step in sorted(metrics_by_step):
        step_tasks = metrics_by_step[step]
        if not step_tasks:
            continue
        avg = sum(step_tasks.values()) / len(step_tasks)
        if avg > best_avg:
            best_avg = avg
            best_avg_steps = {step}
        elif avg == best_avg:
            best_avg_steps.add(step)

    for task in all_tasks:
        values = [tasks[task] for tasks in metrics_by_step.values() if task in tasks]
        if values:
            best_task_values[task] = max(values)

    display_names = {task: format_task_name(task) for task in all_tasks}
    table = PrettyTable()
    table.field_names = ["Step", "Average", *[display_names[task] for task in all_tasks]]
    table.align["Step"] = "r"
    table.align["Average"] = "r"
    for task in all_tasks:
        table.align[display_names[task]] = "r"

    for step in sorted(metrics_by_step):
        step_tasks = metrics_by_step[step]
        scores = list(step_tasks.values())
        if not scores:
            continue
        avg = sum(scores) / len(scores)
        avg_str = f"{avg:.4f}"
        if step in best_avg_steps:
            avg_str = bold(avg_str)
        row = [step, avg_str]
        for task in all_tasks:
            value = step_tasks.get(task)
            if value is None:
                row.append("-")
                continue
            value_str = f"{value:.4f}"
            if task in best_task_values and value == best_task_values[task]:
                value_str = bold(value_str)
            row.append(value_str)
        table.add_row(row)

    print(f"\n=== {title} ===")
    print(table)


def display_best_step(metrics_by_step: dict[int, dict[str, float]]):
    best_step = None
    best_avg = -float("inf")
    for step, tasks in metrics_by_step.items():
        if not tasks:
            continue
        avg = sum(tasks.values()) / len(tasks)
        if avg > best_avg:
            best_avg = avg
            best_step = step

    if best_step is None:
        return

    print(f"\n=== Best Step: {best_step} (Average: {best_avg:.4f}) ===")
    table = PrettyTable()
    table.field_names = ["Task", "Score"]
    table.align["Task"] = "l"
    table.align["Score"] = "r"
    for task in sorted(metrics_by_step[best_step]):
        table.add_row([format_task_name(task), f"{metrics_by_step[best_step][task]:.4f}"])
    print(table)


def main():
    parser = argparse.ArgumentParser(
        description="Compute per-step eval/downstream averages from a W&B run with efficient history scanning."
    )
    parser.add_argument("--run-path", required=True, help="Run path: entity/project/run_id or entity/project/runs/run_id")
    parser.add_argument("--eval-interval", type=int, required=True, help="Only keep steps where step %% eval_interval == 0")
    parser.add_argument("--page-size", type=int, default=2000, help="W&B history scan page size")
    parser.add_argument("--min-step", type=int, default=None, help="Minimum step (inclusive)")
    parser.add_argument("--max-step", type=int, default=None, help="Maximum step (exclusive)")
    parser.add_argument(
        "--exclude-key",
        action="append",
        default=[f"{DOWNSTREAM_PREFIX}boolq_acc"],
        help="Metric key to exclude; can be passed multiple times",
    )
    args = parser.parse_args()

    print(f"Run path: {args.run_path}")
    print(f"Eval interval: {args.eval_interval}")
    if args.min_step is not None or args.max_step is not None:
        print(f"Step window: [{args.min_step}, {args.max_step})")

    normalized = normalize_run_path(args.run_path)
    metrics_by_step = get_eval_downstream_metrics(
        run_path=args.run_path,
        eval_interval=args.eval_interval,
        page_size=args.page_size,
        min_step=args.min_step,
        max_step=args.max_step,
        excluded=set(args.exclude_key),
    )
    display_results(metrics_by_step, title="Eval/Downstream Results by Step")
    display_best_step(metrics_by_step)

    if not metrics_by_step:
        api = wandb.Api(timeout=30)
        run = api.run(normalized)
        keys = discover_downstream_keys(run, DOWNSTREAM_PREFIX, set(args.exclude_key))
        if keys:
            steps = probe_metric_steps(run, keys[0], page_size=args.page_size, limit=12)
            if len(steps) >= 2:
                deltas = [b - a for a, b in zip(steps, steps[1:])]
                common = max(set(deltas), key=deltas.count)
                print(
                    f"\nHint: no rows matched eval_interval={args.eval_interval}. "
                    f"Observed steps for {keys[0]}: {steps[:6]} ... Suggested interval: {common}"
                )


if __name__ == "__main__":
    main()
