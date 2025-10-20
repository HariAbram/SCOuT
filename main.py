#!/usr/bin/env python3
"""
============================================================
Multi‑objective, Optuna‑driven design‑space exploration tool
for AdaptiveCpp / SYCL workloads.

Highlights
----------
* **Any SYCL build target** – same as the original script (single file or
  full CMake/Make project).
* **Search space = Cartesian product** of compiler‑flag variants and
  environment sets – represented as discrete indices for Optuna.
* **Multi‑objective optimisation** – optimise an arbitrary list of
  metrics, each with its own *min/max* goal.
* **Bayesian / evolutionary samplers** – choose between Optunaʼs
  `TPESampler` (Bayesian) or `NSGAIIISampler` (evolutionary) from the
  JSON.
* **Pareto front report** – prints the non‑dominated set at the end and
  writes the full trial table to CSV/SQLite.

Requirements
~~~~~~~~~~~~
* Python ≥ 3.8
* `pip install optuna`
* `likwid` and/or `perf` in `$PATH` for measurement

Usage
~~~~~
```bash
$ python main.py config.json --trials 50
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
from statistics import mean, variance
from typing import Dict, List, Optional, Sequence, Tuple, Any, Union

###############################################################################
# Local imports                                                               #
###############################################################################

from src.config import Config
from src.explore import explore_optuna, explore_wavefront, explore_tabu, explore_beam_tabu, explore_anneal

###############################################################################
# Type helpers                                                                #
###############################################################################

Number = float
EnvMap = Dict[str, str]
MetricDict = Dict[str, Number]

###############################################################################
# Entry point                                                                 #
###############################################################################


def main() -> None:
    parser = argparse.ArgumentParser(description="AdaptiveCpp/SYCL multi‑objective explorer with Optuna")
    parser.add_argument("config", type=Path, help="Path to JSON config file")
    parser.add_argument("--trials", type=int, default=50, help="Number of Optuna trials")
    parser.add_argument("--pareto-log", action="store_true", help="Store pareto front)")

    args = parser.parse_args()

    cfg = Config.load(args.config)

    study = getattr(cfg, "search", None).study if getattr(cfg, "search", None) else "optuna"
    study = (study or "optuna").lower()

    if study == "wavefront":
        explore_wavefront(cfg)
        return
    if study == "tabu":
        explore_tabu(cfg)
        return
    elif study == "beam_tabu":
        explore_beam_tabu(cfg)
        return
    elif study == "anneal":
        explore_anneal(cfg)
        return
    else:
        explore_optuna(cfg, args.trials)

    #explore_optuna(cfg, args.trials)


if __name__ == "__main__":
    main()