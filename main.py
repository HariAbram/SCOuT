#!/usr/bin/env python3
"""
============================================================
Multi‑objective design‑space exploration tool for AdaptiveCpp / SYCL workloads.

Requirements
~~~~~~~~~~~~
* Python ≥ 3.8
* `pip install optuna`
* `likwid` and/or `perf` in `$PATH` for measurement

Usage
~~~~~
```bash
$ python main.py --mode parameter_tuning config.json --trials 50
$ python main.py --mode polymorph configs/polyMorph/O1/matrixT-sycl/config.json --trials 50
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
from typing import Dict
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
from src.polyMorph import run_poly_morph

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


def _run_from_config(cfg: Config, trials: int) -> None:
    study = getattr(cfg, "search", None).study if getattr(cfg, "search", None) else "optuna"
    study = (study or "optuna").lower()

    if study == "wavefront":
        explore_wavefront(cfg)
    elif study == "tabu":
        explore_tabu(cfg)
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
    parser.add_argument("--pareto-log", action="store_true", help="Store pareto front (reserved)")
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

        t0 = time.perf_counter()
        try:
            _run_from_config(cfg, trials)
        finally:
            dt = time.perf_counter() - t0
            print(f"[explore] total wall time: {_fmt_dur(dt)} ({dt:.3f}s)")
    elif mode == "polymorph":
        config_path = args.config
        if config_path is None:
            if args.interactive or sys.stdin.isatty():
                config_path = _prompt_path("Config path: ")
            else:
                parser.error("config is required (or use --interactive)")

        cfg = Config.load(config_path)
        t0 = time.perf_counter()
        try:
            if args.trials is not None:
                print(f"[polyMorph] overriding search.n_trials with --trials={args.trials}")
            rc = run_poly_morph(cfg, args.trials)
            if rc:
                raise SystemExit(rc)
        finally:
            dt = time.perf_counter() - t0
            print(f"[polyMorph] total wall time: {_fmt_dur(dt)} ({dt:.3f}s)")
    else:
        parser.error(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()
