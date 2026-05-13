#!/usr/bin/env python3
"""Generate figures from polyMorph search outputs."""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

from polymorph_results import BenchmarkSummary, load_summaries, slug


def require_matplotlib():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/scout-matplotlib")
    os.environ.setdefault("XDG_CACHE_HOME", "/tmp/scout-cache")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for figure generation. Install it with "
            "`pip install matplotlib` or run the terminal summary script instead."
        ) from exc
    return plt


def save_speedup_bar(plt, summaries: list[BenchmarkSummary], out_dir: Path) -> None:
    data = [item for item in summaries if item.speedup is not None]
    if not data:
        return
    data = sorted(data, key=lambda item: item.speedup or 0.0, reverse=True)
    labels = [item.label for item in data]
    values = [item.speedup or 0.0 for item in data]
    height = max(4.0, 0.32 * len(data))
    fig, ax = plt.subplots(figsize=(12, height))
    ax.barh(labels, values, color="#4C78A8")
    ax.axvline(1.0, color="#333333", linewidth=1.0, linestyle="--")
    ax.set_xlabel("Best speedup over baseline")
    ax.set_title("polyMorph Best Speedup By Benchmark")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(out_dir / "best_speedup_by_benchmark.png", dpi=180)
    plt.close(fig)


def save_opt_level_boxplot(plt, summaries: list[BenchmarkSummary], out_dir: Path) -> None:
    groups: dict[str, list[float]] = {}
    for item in summaries:
        if item.speedup is not None:
            groups.setdefault(item.opt_level, []).append(item.speedup)
    groups = {key: values for key, values in sorted(groups.items()) if values}
    if not groups:
        return
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(groups.values(), tick_labels=groups.keys(), showmeans=True)
    ax.axhline(1.0, color="#333333", linewidth=1.0, linestyle="--")
    ax.set_ylabel("Best speedup over baseline")
    ax.set_title("Speedup Distribution By Optimization Level")
    fig.tight_layout()
    fig.savefig(out_dir / "speedup_by_optimization_level.png", dpi=180)
    plt.close(fig)


def save_status_counts(plt, summaries: list[BenchmarkSummary], out_dir: Path) -> None:
    complete = sum(item.completed_trials for item in summaries)
    pruned = sum(item.pruned_trials for item in summaries)
    failed = sum(item.failed_trials for item in summaries)
    values = [complete, pruned, failed]
    if not any(values):
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(["complete", "pruned", "failed"], values, color=["#59A14F", "#F28E2B", "#E15759"])
    ax.set_ylabel("Trial count")
    ax.set_title("Trial Outcomes")
    fig.tight_layout()
    fig.savefig(out_dir / "trial_outcomes.png", dpi=180)
    plt.close(fig)


def save_transform_counts(plt, summaries: list[BenchmarkSummary], out_dir: Path) -> None:
    counts: Counter[str] = Counter()
    for item in summaries:
        counts.update(item.transform_counts)
    if not counts:
        return
    labels, values = zip(*counts.most_common(12))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(labels, values, color="#B07AA1")
    ax.set_ylabel("Occurrences in trial CSVs")
    ax.set_title("Most Frequent Transform Types")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(out_dir / "transform_type_counts.png", dpi=180)
    plt.close(fig)


def save_trial_curves(plt, summaries: list[BenchmarkSummary], out_dir: Path, limit: int) -> None:
    curve_dir = out_dir / "trial_curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    plotted = 0
    for item in summaries:
        if not item.trial_speedups:
            continue
        points = sorted(item.trial_speedups)
        xs = [trial for trial, _ in points]
        ys = [speedup for _, speedup in points]
        best_so_far = []
        best = 0.0
        for value in ys:
            best = max(best, value)
            best_so_far.append(best)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(xs, ys, marker="o", linewidth=1.0, label="trial speedup")
        ax.plot(xs, best_so_far, linewidth=2.0, label="best so far")
        ax.axhline(1.0, color="#333333", linewidth=1.0, linestyle="--")
        ax.set_xlabel("Trial")
        ax.set_ylabel("Speedup")
        ax.set_title(item.label)
        ax.legend()
        fig.tight_layout()
        fig.savefig(curve_dir / f"{slug(item.label)}.png", dpi=180)
        plt.close(fig)
        plotted += 1
        if limit > 0 and plotted >= limit:
            break


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot polyMorph result files.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root.")
    parser.add_argument("--out-dir", type=Path, default=Path("post_processing/figures"))
    parser.add_argument(
        "--trial-curve-limit",
        type=int,
        default=40,
        help="Maximum number of per-benchmark trial-curve figures. Use 0 for all.",
    )
    args = parser.parse_args()

    summaries = load_summaries(args.root.resolve())
    if not summaries:
        print("No polyMorph result/trial/history outputs found.")
        return 0

    plt = require_matplotlib()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    save_speedup_bar(plt, summaries, out_dir)
    save_opt_level_boxplot(plt, summaries, out_dir)
    save_status_counts(plt, summaries, out_dir)
    save_transform_counts(plt, summaries, out_dir)
    save_trial_curves(plt, summaries, out_dir, args.trial_curve_limit)

    print(f"Wrote figures to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
