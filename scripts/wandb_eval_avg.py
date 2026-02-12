#!/usr/bin/env python3
"""
Gather eval/downstream metrics from a W&B run and calculate averages per step.
"""
import time
import argparse
from collections import defaultdict

import wandb
from tqdm import tqdm
from prettytable import PrettyTable


def get_eval_downstream_metrics(run_path: str, eval_interval: int) -> dict:
    """
    Fetch eval/downstream metrics from a W&B run at steps that are multiples of eval_interval.·

    Args:
        run_path: Full run path in format "entity/project/run_id"
        eval_interval: Fetch metrics at steps that are multiples of this value

    Returns:
        Dictionary mapping step -> {metric_name: value}
    """
    api = wandb.Api()
    run = api.run(run_path)

    # step -> {metric_name: value}
    metrics_by_step = defaultdict(dict)

    start_time = time.time()
    history = run.beta_scan_history()
    print(f"Fetched history in {time.time() - start_time:.2f} seconds.")

    for row in tqdm(history):
        step = row.get("_step", 0)
        if step == 0 or step % eval_interval != 0:
            continue
        for key, value in row.items():
            if key.startswith("eval/downstream/") and value is not None:
                if key == "eval/downstream/boolq_acc":
                    continue
                short_name = key.replace("eval/downstream/", "")
                metrics_by_step[step][short_name] = value

    return metrics_by_step


def get_runs_from_group(entity: str, project: str, group_name: str) -> list:
    """
    Get all runs from a specific group.

    Args:
        entity: W&B entity name
        project: W&B project name
        group_name: Group name to filter runs

    Returns:
        List of run objects
    """
    api = wandb.Api()
    filters = {"group": group_name}
    runs = api.runs(f"{entity}/{project}", filters=filters)
    return list(runs)


def display_results(metrics_by_step: dict, title: str = "Eval/Downstream Average by Step"):
    """Display results using PrettyTable with step and average score."""
    if not metrics_by_step:
        print("No eval/downstream metrics found.")
        return

    table = PrettyTable()
    table.field_names = ["Step", "Average"]
    table.align["Step"] = "r"
    table.align["Average"] = "r"

    for step in sorted(metrics_by_step.keys()):
        values = list(metrics_by_step[step].values())
        if values:
            avg = sum(values) / len(values)
            table.add_row([step, f"{avg:.4f}"])

    print(f"\n=== {title} ===")
    print(table)


def merge_metrics(all_metrics: list) -> dict:
    """
    Get metrics from runs (uses first run, no averaging across runs).

    Args:
        all_metrics: List of metrics_by_step dictionaries

    Returns:
        Dictionary with step -> average score (average of tasks at each step)
    """
    if not all_metrics:
        return {}

    result = {}
    for metrics_by_step in all_metrics:
        for step, values in metrics_by_step.items():
            if values:
                avg = sum(values.values()) / len(values)
                result[step] = avg

    return result


def display_merged_results(merged: dict, all_metrics: list = None):
    """Display merged results with best step and per-task breakdown."""
    if not merged:
        print("No merged metrics found.")
        return

    table = PrettyTable()
    table.field_names = ["Step", "Average"]
    table.align["Step"] = "r"
    table.align["Average"] = "r"

    best_step = None
    best_avg = -float("inf")

    for step in sorted(merged.keys()):
        table.add_row([step, f"{merged[step]:.4f}"])
        if merged[step] > best_avg:
            best_avg = merged[step]
            best_step = step

    print("\n=== Merged Downstream Average by Step ===")
    print(table)

    # Display best step with per-task breakdown
    if best_step is not None and all_metrics:
        print(f"\n=== Best Step: {best_step} (Average: {best_avg:.4f}) ===")

        # Use the first run that has data for this step to get all tasks
        for metrics_by_step in all_metrics:
            if best_step in metrics_by_step:
                # Get all task scores at the best step from all runs
                task_table = PrettyTable()
                task_table.field_names = ["Task", "Score"]
                task_table.align["Task"] = "l"
                task_table.align["Score"] = "r"
                for task in sorted(metrics_by_step[best_step].keys()):
                    score = metrics_by_step[best_step][task]
                    task_table.add_row([task, f"{score:.4f}"])

        print(task_table)


def main():
    parser = argparse.ArgumentParser(description="Calculate average eval/downstream metrics from a W&B run")
    parser.add_argument("--run-path", help="W&B run path in format 'entity/project/run_id'")
    parser.add_argument(
        "--eval-interval",
        type=int,
        required=True,
        help="Eval interval (e.g., 539). Fetches metrics at steps that are multiples of this value.",
    )
    parser.add_argument("--entity", default="cxcscmu", help="W&B entity name")
    parser.add_argument("--project", default="healthcare", help="W&B project name")
    parser.add_argument("--group", help="W&B group name to fetch all runs from")
    parser.add_argument("--name", help="Run name to filter when fetching from group (optional)")
    args = parser.parse_args()

    if args.group:
        if not args.entity or not args.project:
            parser.error("--entity and --project are required when using --group")

        print(f"Fetching runs from group: {args.group}")
        print(f"Project: {args.entity}/{args.project}")
        print(f"Eval interval: {args.eval_interval}")

        runs = get_runs_from_group(args.entity, args.project, args.group)
        print(f"Found {len(runs)} runs in group")

        all_metrics = []
        for run in tqdm(runs, desc="Processing runs"):
            if args.name and run.name != args.name:
                continue
            print(f"\nProcessing run: {run.name} ({run.id})")
            run_path = f"{args.entity}/{args.project}/{run.id}"
            metrics = get_eval_downstream_metrics(run_path, args.eval_interval)
            all_metrics.append(metrics)
            display_results(metrics, title=f"Run: {run.name}")

        merged = merge_metrics(all_metrics)
        display_merged_results(merged, all_metrics)

    elif args.run_path:
        print(f"Fetching metrics from: {args.run_path}")
        print(f"Eval interval: {args.eval_interval}")

        metrics_by_step = get_eval_downstream_metrics(args.run_path, args.eval_interval)
        display_results(metrics_by_step)

        merged = merge_metrics([metrics_by_step])
        display_merged_results(merged, [metrics_by_step])

    else:
        parser.error("Either --run-path or --group (with --entity and --project) is required")


if __name__ == "__main__":
    main()
