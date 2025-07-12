#!/usr/bin/env python3
"""dse_multiobj_optuna.py
============================================================
Multi‑objective, Optuna‑driven design‑space exploration tool
for AdaptiveCpp / SYCL workloads.

Highlights
----------
* **Any SYCL build target** – same as the original script (single TU or
  full CMake/Make project).
* **Search space = Cartesian product** of compiler‑flag variants and
  environment sets – represented as discrete indices for Optuna.
* **Multi‑objective optimisation** – optimise an arbitrary list of
  metrics, each with its own *min/max* goal.
* **Bayesian / evolutionary samplers** – choose between Optunaʼs
  `TPESampler` (Bayesian) or `NSGAIISampler` (evolutionary) from the
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

import csv
import json
import sys
import tempfile
from pathlib import Path
from statistics import mean, variance
from typing import Dict, List, Optional, Sequence, Tuple, Any, Union


###############################################################################
# Third‑party imports                                                         #
###############################################################################

try:
    import optuna
    from optuna.samplers import TPESampler, NSGAIIISampler
    from optuna.trial import TrialState
except ImportError as exc:  # pragma: no cover
    sys.exit("[fatal] Optuna missing – install via `pip install optuna`.")

###############################################################################
# Type helpers                                                                #
###############################################################################

Number = float
EnvMap = Dict[str, str]
MetricDict = Dict[str, Number]

###############################################################################
# Local imports                                                               #
###############################################################################

from config import Config
from metrics import measure_likwid, measure_perf
from build import compile_project, compile_single_source
from misc import suggest_compiler_flags, suggest_env

###############################################################################
# Optuna‑driven exploration                                                   #
###############################################################################

def explore_optuna(cfg: Config, n_trials: int) -> None:
    workdir_root = Path(tempfile.mkdtemp(prefix="dse_optuna_"))
    print(f"[info] working directory root: {workdir_root}\n")

    # Sampler choice
    if cfg.search.sampler == "nsga3":
        sampler = NSGAIIISampler(population_size=cfg.search.population_size,seed=cfg.search.random_seed,)
    else:
        sampler = TPESampler(n_startup_trials=cfg.search.n_startup_trials)

    directions = ["minimize" if o.goal == "min" else "maximize" for o in cfg.objectives]
    study = optuna.create_study(sampler=sampler, directions=directions)

    def trial_objective(trial: optuna.Trial):
        # --------------------------------------------------------------
        # 1) Sample discrete indices
        # --------------------------------------------------------------
        flag_key, flags = suggest_compiler_flags(
                            trial,
                            cfg.compiler_flags,
                            cfg.compiler_params,
                            cfg.compiler_flag_pool,
                        )
        env = suggest_env(trial, cfg.env)
        
        # --------------------------------------------------------------
        # 2) Build
        # --------------------------------------------------------------
        workdir = workdir_root / f"trial_{trial.number:05d}"
        workdir.mkdir()
        if cfg.source:
            binary_path = compile_single_source(cfg.compiler, cfg.source, flags, workdir / "a.out")
        else:
            binary_path = compile_project(cfg.project, cfg.compiler, flags, workdir)
        if not binary_path:
            raise optuna.TrialPruned("build failed")

        # --------------------------------------------------------------
        # 3) Measure
        # --------------------------------------------------------------
        try:
            if cfg.backend == "perf":
                metrics = measure_perf(cfg.perf, binary_path, cfg.program_args, env)  # type: ignore[arg-type]
            else:
                metrics = measure_likwid(cfg.likwid, binary_path, cfg.program_args, env, cfg.runs)  # type: ignore[arg-type]
        except Exception as exc:
            raise optuna.TrialPruned(f"measurement failed: {exc}")

        # --------------------------------------------------------------
        # 4) Extract objective values (missing ⇒ prune)
        # --------------------------------------------------------------
        obj_values: List[Number] = []
        for obj in cfg.objectives:
            if obj.metric not in metrics:
                raise optuna.TrialPruned(f"metric '{obj.metric}' missing")
            obj_values.append(metrics[obj.metric])

        # --------------------------------------------------------------
        # 5) Attach extra info for analysis
        # --------------------------------------------------------------
        trial.set_user_attr("compiler_flags", flag_key)
        trial.set_user_attr("env", env)
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("binary", str(binary_path))
        return obj_values

    study.optimize(trial_objective, n_trials=n_trials, show_progress_bar=True)

    # ------------------------------------------------------------------
    # Pareto front summary
    # ------------------------------------------------------------------
    try:
        front = study.best_trials          # Optuna ≥ 3.0
    except AttributeError:
        # Fallback for Optuna 2.x
        from optuna.visualization._pareto_front import _get_pareto_front_trials
        front = _get_pareto_front_trials(study)
    print("\n================ Pareto‑optimal configurations ================")
    for t in front:
        print(f"Trial#{t.number}: objectives={t.values} flags='{t.user_attrs['compiler_flags']}' env={t.user_attrs['env']}")
    print("==============================================================\n")

    # ------------------------------------------------------------------
    # CSV / SQLite logging
    # ------------------------------------------------------------------
    if cfg.csv_log:
        print(f"[info] writing CSV log → {cfg.csv_log}")
        with open(cfg.csv_log, "w", newline="") as fp:
            writer = csv.writer(fp)
            # Header
            header = [o.metric for o in cfg.objectives] + ["compiler_flags", "env", "binary"]
            extra_metrics: set[str] = set()
            for t in study.trials:
                extra_metrics.update(t.user_attrs.get("metrics", {}).keys())
            header += sorted(extra_metrics)
            writer.writerow(header)
            # Rows
            for t in study.trials:
                metrics: MetricDict = t.user_attrs.get("metrics", {})

                if t.state != TrialState.COMPLETE or t.values is None:
                    continue

                row = list(t.values) + [t.user_attrs.get("compiler_flags", ""), json.dumps(t.user_attrs.get("env", {})), t.user_attrs.get("binary","")]
                row += [metrics.get(k, "") for k in sorted(extra_metrics)]
                writer.writerow(row)

    if cfg.sqlite_log:
        storage = getattr(study, "storage", None) or getattr(study, "_storage", None)
        if storage is None:
            print("[warn] cannot access Study storage – skip SQLite export")
        else:
            copy_fn = getattr(storage, "copy_cached_study", None)
            if callable(copy_fn):
                # Only Cached/InMemory back-ends implement this method
                copy_fn(study._study_id, f"sqlite:///{cfg.sqlite_log}")
                print(f"[info] SQLite log written → {cfg.sqlite_log}")
            else:
                # Persistent storages (SQLite, RDB) don’t need copying
                print("[info] Study already uses persistent storage – no copy needed")
