from __future__ import annotations

import csv
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


JsonDict = dict[str, Any]


@dataclass
class BenchmarkSummary:
    opt_level: str
    benchmark: str
    config_dir: Path
    result_path: Path | None = None
    trials_path: Path | None = None
    history_path: Path | None = None
    baseline: float | None = None
    best: float | None = None
    speedup: float | None = None
    validated_speedup: float | None = None
    validated_significant: bool | None = None
    best_trial: int | None = None
    target_backend: str = ""
    backend_sensitive: bool | None = None
    completed_trials: int = 0
    pruned_trials: int = 0
    failed_trials: int = 0
    total_trials: int = 0
    cache_hits: int = 0
    best_transforms: list[str] = field(default_factory=list)
    transform_counts: Counter[str] = field(default_factory=Counter)
    regression_counts: Counter[str] = field(default_factory=Counter)
    kernel_counts: Counter[str] = field(default_factory=Counter)
    trial_objectives: list[tuple[int, float]] = field(default_factory=list)
    trial_speedups: list[tuple[int, float]] = field(default_factory=list)

    @property
    def label(self) -> str:
        return f"{self.opt_level}/{self.benchmark}"


def load_json(path: Path) -> JsonDict:
    return json.loads(path.read_text())


def safe_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def safe_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def json_cell(value: str) -> Any:
    if not value:
        return None
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return None


def transform_label(transform: JsonDict) -> str:
    name = str(transform.get("tr") or transform.get("transform") or "?")
    scop = transform.get("scop", "?")
    node = transform.get("node", "?")
    args = transform.get("args", [])
    return f"{name}@s{scop}:n{node}{args}"


def compact_transform_sequence(transforms: Any, max_items: int = 3) -> list[str]:
    if not isinstance(transforms, list):
        return []
    labels = [transform_label(item) for item in transforms if isinstance(item, dict)]
    if len(labels) > max_items:
        return [*labels[:max_items], f"+{len(labels) - max_items} more"]
    return labels


def config_dirs(root: Path) -> list[Path]:
    candidates: list[Path] = []
    seen: set[Path] = set()

    def add_dir(path: Path) -> None:
        cfg_dir = path.resolve()
        if cfg_dir not in seen:
            seen.add(cfg_dir)
            candidates.append(cfg_dir)

    def add_output_dirs(base: Path) -> None:
        if not base.exists():
            return
        if base.is_file():
            if (
                base.name.endswith("-result.json")
                or base.name.endswith("-trials.csv")
                or base.name.endswith("-history.jsonl")
            ):
                add_dir(base.parent)
            return
        for pattern in ["*-result.json", "*-trials.csv", "*-history.jsonl"]:
            for path in base.rglob(pattern):
                add_dir(path.parent)

    def add_config_dirs(base: Path) -> None:
        if not base.exists():
            return
        if base.is_file() and base.name == "config.json":
            add_dir(base.parent)
            return
        if not base.is_dir():
            return
        direct = base / "config.json"
        if direct.exists():
            add_dir(base)
        for path in base.rglob("config.json"):
            add_dir(path.parent)

    def add_base(base: Path) -> None:
        if not base.exists():
            return
        add_config_dirs(base)
        add_output_dirs(base)

    add_base(root)
    add_base(root / "configs" / "polyMorph")
    return sorted(candidates)


def opt_and_benchmark(config_dir: Path) -> tuple[str, str]:
    parts = config_dir.parts
    for idx, part in enumerate(parts):
        if part in {"O1", "O2", "O3"} and idx + 1 < len(parts):
            return part, parts[idx + 1]
    return "?", config_dir.name


def first_existing(config_dir: Path, suffix: str) -> Path | None:
    matches = sorted(config_dir.glob(f"*{suffix}"))
    return matches[0] if matches else None


def load_result(summary: BenchmarkSummary) -> None:
    if not summary.result_path or not summary.result_path.exists():
        return
    data = load_json(summary.result_path)
    summary.baseline = safe_float(data.get("baseline_runtime"))
    summary.best = safe_float(data.get("best_runtime"))
    summary.speedup = safe_float(data.get("best_speedup"))
    summary.validated_speedup = safe_float(data.get("validated_best_speedup"))
    if data.get("validated_best_significant") is not None:
        summary.validated_significant = bool(data.get("validated_best_significant"))
    summary.best_trial = safe_int(data.get("best_trial_number"))
    summary.target_backend = str(data.get("target_backend") or "")
    sensitivity = data.get("baseline_backend_sensitivity")
    if isinstance(sensitivity, dict) and sensitivity.get("checked"):
        summary.backend_sensitive = bool(sensitivity.get("backend_sensitive", False))
    summary.best_transforms = compact_transform_sequence(data.get("best_transforms"), max_items=5)


def load_trials(summary: BenchmarkSummary) -> None:
    if not summary.trials_path or not summary.trials_path.exists():
        return
    with summary.trials_path.open(newline="", encoding="utf-8") as fp:
        reader = csv.DictReader(fp)
        for row in reader:
            summary.total_trials += 1
            state = str(row.get("state") or "").upper()
            if state == "COMPLETE":
                summary.completed_trials += 1
            elif state == "PRUNED":
                summary.pruned_trials += 1
            elif state:
                summary.failed_trials += 1
            trial = safe_int(row.get("trial"))
            objective = safe_float(row.get("objective"))
            speedup = safe_float(row.get("speedup"))
            if trial is not None and objective is not None:
                summary.trial_objectives.append((trial, objective))
            if trial is not None and speedup is not None:
                summary.trial_speedups.append((trial, speedup))
            transforms = json_cell(row.get("transforms", ""))
            if isinstance(transforms, list):
                for item in transforms:
                    if isinstance(item, dict):
                        summary.transform_counts[str(item.get("tr") or "?")] += 1
            feedback = json_cell(row.get("performance_feedback_analysis", ""))
            if isinstance(feedback, dict):
                regression = feedback.get("regression")
                if regression:
                    summary.regression_counts[str(regression)] += 1
                kernel_timing = feedback.get("kernel_timing")
                if isinstance(kernel_timing, dict):
                    worst = kernel_timing.get("worst_kernel")
                    if worst is not None:
                        summary.kernel_counts[str(worst)] += 1


def load_history(summary: BenchmarkSummary) -> None:
    if not summary.history_path or not summary.history_path.exists():
        return
    for line in summary.history_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            continue
        if record.get("cache_hit"):
            summary.cache_hits += 1


def load_summaries(root: Path) -> list[BenchmarkSummary]:
    summaries: list[BenchmarkSummary] = []
    for cfg_dir in config_dirs(root):
        opt_level, benchmark = opt_and_benchmark(cfg_dir)
        summary = BenchmarkSummary(
            opt_level=opt_level,
            benchmark=benchmark,
            config_dir=cfg_dir,
            result_path=first_existing(cfg_dir, "-result.json"),
            trials_path=first_existing(cfg_dir, "-trials.csv"),
            history_path=first_existing(cfg_dir, "-history.jsonl"),
        )
        load_result(summary)
        load_trials(summary)
        load_history(summary)
        if summary.result_path or summary.trials_path or summary.history_path:
            summaries.append(summary)
    return sorted(summaries, key=lambda item: (item.opt_level, item.benchmark))


def format_float(value: float | None, digits: int = 3) -> str:
    if value is None:
        return "-"
    return f"{value:.{digits}f}"


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "benchmark"
