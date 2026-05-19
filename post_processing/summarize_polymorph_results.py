#!/usr/bin/env python3
"""Print polyMorph search outputs as terminal tables."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from polymorph_results import BenchmarkSummary, format_float, load_summaries


def table(headers: list[str], rows: list[list[str]]) -> str:
    widths = [len(header) for header in headers]
    for row in rows:
        for idx, cell in enumerate(row):
            widths[idx] = max(widths[idx], len(cell))
    sep = "  "
    lines = [
        sep.join(header.ljust(widths[idx]) for idx, header in enumerate(headers)),
        sep.join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append(sep.join(cell.ljust(widths[idx]) for idx, cell in enumerate(row)))
    return "\n".join(lines)


def top_items(counter: Counter[str], limit: int = 3) -> str:
    if not counter:
        return "-"
    return ", ".join(f"{key}:{value}" for key, value in counter.most_common(limit))


def status_text(summary: BenchmarkSummary) -> str:
    if summary.total_trials:
        return (
            f"C{summary.completed_trials}/P{summary.pruned_trials}/"
            f"F{summary.failed_trials}/T{summary.total_trials}"
        )
    return "-"


def backend_text(summary: BenchmarkSummary) -> str:
    if summary.backend_sensitive is None:
        return summary.target_backend or "-"
    marker = "sensitive" if summary.backend_sensitive else "stable"
    if summary.target_backend:
        return f"{summary.target_backend},{marker}"
    return marker


def significance_text(summary: BenchmarkSummary) -> str:
    if summary.validated_significant is None:
        return "-"
    return "yes" if summary.validated_significant else "no"


def summary_rows(summaries: list[BenchmarkSummary]) -> list[list[str]]:
    rows = []
    for item in summaries:
        rows.append(
            [
                item.opt_level,
                item.benchmark,
                format_float(item.baseline, 6),
                format_float(item.best, 6),
                format_float(item.speedup, 3),
                format_float(item.validated_speedup, 3),
                significance_text(item),
                str(item.best_trial) if item.best_trial is not None else "-",
                status_text(item),
                backend_text(item),
                "; ".join(item.best_transforms) if item.best_transforms else "-",
            ]
        )
    return rows


def detail_rows(summaries: list[BenchmarkSummary]) -> list[list[str]]:
    rows = []
    for item in summaries:
        rows.append(
            [
                item.label,
                top_items(item.transform_counts),
                top_items(item.regression_counts),
                top_items(item.kernel_counts),
            ]
        )
    return rows


def aggregate_rows(summaries: list[BenchmarkSummary]) -> list[list[str]]:
    by_opt: dict[str, list[BenchmarkSummary]] = {}
    for item in summaries:
        by_opt.setdefault(item.opt_level, []).append(item)
    rows = []
    for opt_level, items in sorted(by_opt.items()):
        speedups = [item.speedup for item in items if item.speedup is not None]
        validated_speedups = [
            item.validated_speedup
            for item in items
            if item.validated_speedup is not None
        ]
        completed = sum(item.completed_trials for item in items)
        pruned = sum(item.pruned_trials for item in items)
        failed = sum(item.failed_trials for item in items)
        avg_speedup = sum(speedups) / len(speedups) if speedups else None
        avg_validated = (
            sum(validated_speedups) / len(validated_speedups)
            if validated_speedups else None
        )
        best = max(speedups) if speedups else None
        significant = sum(1 for item in items if item.validated_significant is True)
        rows.append(
            [
                opt_level,
                str(len(items)),
                format_float(avg_speedup, 3),
                format_float(avg_validated, 3),
                format_float(best, 3),
                str(significant),
                str(completed),
                str(pruned),
                str(failed),
            ]
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize polyMorph result files.")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Repository root.")
    parser.add_argument("--details", action="store_true", help="Print transform/regression details.")
    parser.add_argument("--min-speedup", type=float, default=None, help="Only show rows at or above this speedup.")
    parser.add_argument("--sort", choices=["path", "speedup"], default="path")
    args = parser.parse_args()

    summaries = load_summaries(args.root.resolve())
    if args.min_speedup is not None:
        summaries = [
            item for item in summaries
            if item.speedup is not None and item.speedup >= args.min_speedup
        ]
    if args.sort == "speedup":
        summaries = sorted(summaries, key=lambda item: item.speedup or 0.0, reverse=True)

    if not summaries:
        print("No polyMorph result/trial/history outputs found.")
        return 0

    print("polyMorph Search Summary")
    print(table(
        [
            "Opt",
            "Benchmark",
            "Baseline",
            "Best",
            "SearchSpeedup",
            "ValidSpeedup",
            "Significant",
            "BestTrial",
            "Trials",
            "Backend",
            "Best transforms",
        ],
        summary_rows(summaries),
    ))

    print("\nAggregate By Optimization Level")
    print(table(
        [
            "Opt",
            "Benchmarks",
            "AvgSearch",
            "AvgValid",
            "BestSearch",
            "Significant",
            "Complete",
            "Pruned",
            "Failed",
        ],
        aggregate_rows(summaries),
    ))

    if args.details:
        print("\nTransform And Kernel Details")
        print(table(
            ["Benchmark", "Transforms", "Regression classes", "Worst kernels"],
            detail_rows(summaries),
        ))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
