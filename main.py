#!/usr/bin/env python3
"""
============================================================
Multi‑objective design‑space exploration tool for AdaptiveCpp / SYCL workloads.

Requirements
~~~~~~~~~~~~
* Python ≥ 3.10
* `python3 -m pip install -r requirements.txt`
* `likwid` and/or `perf` in `$PATH` for measurement

Usage
~~~~~
```bash
$ python main.py --mode parameter_tuning --config config.json --trials 50
$ python main.py --mode polymorph --config configs/polyMorph/O1/matrixT-sycl/config.json --trials 50
```
See `sample_config.json` for a minimal two‑objective example.
"""
from __future__ import annotations

###############################################################################
# Standard library imports                                                    #
###############################################################################

import argparse
import sys
from pathlib import Path
from typing import Dict, Optional
import time
from datetime import timedelta
###############################################################################
# Local imports                                                               #
###############################################################################

from src.config import Config
from src.explore import (
    explore_optuna,
    explore_wavefront,
    explore_tabu,
    explore_beam_tabu,
    explore_anneal,
)
from src.stats import STATS, resolve_stats_path
from src.logger import LOG

###############################################################################
# Type helpers                                                                #
###############################################################################

Number = float
EnvMap = Dict[str, str]
MetricDict = Dict[str, Number]


###############################################################################
# Helpers
###############################################################################

def _fmt_dur(sec: float) -> str:
    return str(timedelta(seconds=sec))

def _prompt_path(prompt: str) -> Path:
    while True:
        s = input(prompt).strip()
        if not s:
            print("Please enter a path.")
            continue
        p = Path(s)
        if p.exists():
            return p
        print(f"Path does not exist: {p}")


def _prompt_int(prompt: str, default: int) -> int:
    while True:
        s = input(f"{prompt} [{default}]: ").strip()
        if not s:
            return default
        try:
            v = int(s)
            if v <= 0:
                raise ValueError
            return v
        except ValueError:
            print("Please enter a positive integer.")


def _positive_int_arg(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("must be a positive integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def _run_from_config(
    cfg: Config,
    trials: int,
    *,
    resume: bool = False,
    iters: Optional[int] = None,
    budget: str = "lifetime",
    workroot: Optional[Path] = None,
) -> None:
    study = getattr(cfg, "search", None).study if getattr(cfg, "search", None) else "optuna"
    study = (study or "optuna").lower()

    if study == "wavefront":
        explore_wavefront(cfg)
    elif study == "tabu":
        explore_tabu(cfg, resume=resume, iters=iters, budget=budget, workroot=workroot)
    elif study == "beam_tabu":
        explore_beam_tabu(cfg)
    elif study == "anneal":
        explore_anneal(cfg)
    else:
        explore_optuna(cfg, trials)


###############################################################################
# Entry point                                                                 #
###############################################################################


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SCOuT entrypoint (select a mode, then provide mode-specific args)"
    )

    parser.add_argument(
        "--mode",
        choices=["parameter_tuning", "polymorph"],
        default=None,
        help="Select what SCOuT should do (required unless using interactive prompts).",
    )

    parser.add_argument("--config", nargs="?", type=Path, help="Path to JSON config file")
    parser.add_argument(
        "--trials",
        type=_positive_int_arg,
        default=None,
        help=(
            "Number of trials. For parameter_tuning this controls Optuna trials; "
            "for polymorph it overrides polyMorph.search.n_trials from the config."
        ),
    )
    parser.add_argument("--pareto-log", action="store_true", help="Write pareto.csv when a multi-objective config has no pareto_log")
    restart = parser.add_mutually_exclusive_group()
    restart.add_argument(
        "--resume",
        action="store_true",
        help=(
            "Continue a previous tabu run from its results CSV and state file. "
            "Configurations already measured are reused instead of re-run."
        ),
    )
    restart.add_argument(
        "--fresh",
        action="store_true",
        help=(
            "Start a new tabu run. Existing results/state are archived rather "
            "than overwritten (this is also the default)."
        ),
    )
    parser.add_argument(
        "--iters",
        type=_positive_int_arg,
        default=None,
        help=(
            "Tabu iteration budget, overriding tabu.max_iters. Interpreted as a "
            "lifetime budget unless --budget per-run is given."
        ),
    )
    parser.add_argument(
        "--budget",
        choices=["lifetime", "per-run"],
        default="lifetime",
        help=(
            "How --iters/tabu.max_iters is counted for tabu: 'lifetime' caps the "
            "total number of iterations across restarts (default), 'per-run' "
            "grants that many additional iterations on top of completed ones."
        ),
    )
    parser.add_argument(
        "--workroot",
        type=Path,
        default=None,
        help=(
            "Directory for tabu builds and run scratch (default: "
            "$SCOUT_TABU_WORKROOT, then $TMPDIR/SCOuT_tabu). Existing build "
            "artifacts there are reused."
        ),
    )
    parser.add_argument(
        "--stats-file",
        type=Path,
        default=None,
        help=(
            "Where to write build/run/runtime statistics (default: config "
            "stats_log, then $SCOUT_STATS_FILE, then ./scout_stats.csv). "
            "Rows are appended every ~5s while a parameter-tuning run is active."
        ),
    )
    parser.add_argument("--interactive", action="store_true", help="Prompt for missing args")

    args = parser.parse_args()

    # If mode isn’t provided, default to interactive selection (future-proof).
    mode = args.mode
    if mode is None:
        if args.interactive or sys.stdin.isatty():
            print("Select mode:")
            print("  1) parameter_tuning")
            print("  2) polymorph")
            choice = input("Enter choice [1]: ").strip() or "1"
            mode = "polymorph" if choice == "2" else "parameter_tuning"
        else:
            parser.error("--mode is required in non-interactive contexts.")

    # For now only one mode, but the structure is ready for more.
    if mode == "parameter_tuning":
        config_path = args.config
        trials = args.trials

        if (config_path is None or trials is None) and (args.interactive or sys.stdin.isatty()):
            if config_path is None:
                config_path = _prompt_path("Config path: ")
            if trials is None:
                trials = _prompt_int("Trials", default=50)
        else:
            if config_path is None:
                parser.error("config is required (or use --interactive)")
            if trials is None:
                trials = 50

        cfg = Config.load(config_path)
        LOG.start(config_path=config_path)
        if args.pareto_log and not cfg.pareto_log:
            cfg.pareto_log = "pareto.csv"

        stats_path = resolve_stats_path(args.stats_file, cfg.stats_log)
        STATS.start(stats_path)
        print(f"[stats] tracking build/run/runtime → {stats_path}")

        t0 = time.perf_counter()
        try:
            _run_from_config(
                cfg,
                trials,
                resume=args.resume,
                iters=args.iters,
                budget=args.budget,
                workroot=args.workroot,
            )
        finally:
            dt = time.perf_counter() - t0
            STATS.finish()
            print(f"[explore] total wall time: {_fmt_dur(dt)} ({dt:.3f}s)")
            print(STATS.summary_line())
    elif mode == "polymorph":
        # PolyMorph is an optional subsystem.  Import it only after the user
        # explicitly selects this mode so its Python/toolchain dependencies do
        # not affect normal parameter-tuning runs.
        from src.polyMorph import PolyMorphUnavailableError, run_poly_morph

        config_path = args.config
        if config_path is None:
            if args.interactive or sys.stdin.isatty():
                config_path = _prompt_path("Config path: ")
            else:
                parser.error("config is required (or use --interactive)")

        cfg = Config.load(config_path)
        LOG.start(config_path=config_path)
        t0 = time.perf_counter()
        try:
            if args.trials is not None:
                print(f"[polyMorph] overriding search.n_trials with --trials={args.trials}")
            try:
                rc = run_poly_morph(cfg, args.trials)
            except PolyMorphUnavailableError as exc:
                parser.exit(2, f"error: {exc}\n")
            if rc:
                raise SystemExit(rc)
        finally:
            dt = time.perf_counter() - t0
            print(f"[polyMorph] total wall time: {_fmt_dur(dt)} ({dt:.3f}s)")
    else:
        parser.error(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
