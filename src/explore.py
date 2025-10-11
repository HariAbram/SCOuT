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
    from optuna.samplers import TPESampler, NSGAIIISampler, RandomSampler, CmaEsSampler
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

from src.config import Config
from src.metrics import measure_likwid, measure_perf, measure_parser_sycl
from src.build import compile_project, compile_single_source
from src.misc import suggest_compiler_flags, suggest_env, unique_csv_path, jit_env_for_phase, summarize_appdb, jit_enabled

###############################################################################
# Optuna‑driven exploration                                                   #
###############################################################################

def explore_optuna(cfg: Config, n_trials: int) -> None:
    workdir_root = Path(tempfile.mkdtemp(prefix="SCOuT_"))
    print(f"[info] working directory root: {workdir_root}\n")

    # Sampler choice
    is_multi = len(cfg.objectives) > 1
    if cfg.search.sampler == "nsga3":
        sampler = NSGAIIISampler(population_size=cfg.search.population_size,seed=cfg.search.random_seed,)
    elif cfg.search.sampler == "rs":
        sampler = RandomSampler(seed=cfg.search.random_seed,)
    elif cfg.search.sampler == "cmaes":
        sampler = CmaEsSampler(seed=cfg.search.random_seed,)
    else:
        startup = cfg.search.n_startup_trials or 0
        if is_multi and startup < 5:
            print("[info] MOTPE bootstrap: bumping n_startup_trials → 5")
            startup = 5
        sampler = TPESampler(n_startup_trials=cfg.search.n_startup_trials,
                            multivariate=True,
                            group=True,
                            seed=cfg.search.random_seed
                            )

    directions = ["minimize" if o.goal == "min" else "maximize" for o in cfg.objectives]
    study = optuna.create_study(sampler=sampler, directions=directions)

    # --- Optional: pick a Pareto CSV path (only for multi-objective) ---------
    pareto_path: Optional[Path] = None
    if is_multi:
        dest = getattr(cfg, "pareto_csv", None)
        if dest:
            pareto_path = Path(dest)
        elif cfg.csv_log:
            base = Path(cfg.csv_log)
            pareto_path = base.with_name(f"{base.stem}_pareto.csv")
        else:
            pareto_path = workdir_root / "pareto.csv"

    def _export_pareto_front(study: optuna.study.Study, out_csv: Path) -> None:
        """Write current Pareto set to CSV (overwrites on each call)."""
        try:
            front = study.best_trials  # Optuna ≥ 3.x
        except AttributeError:
            from optuna.visualization._pareto_front import _get_pareto_front_trials
            front = _get_pareto_front_trials(study)
        if not front:
            return
        # Collect any extra metric keys stored in user_attrs["metrics"].
        extra_metrics: set[str] = set()
        for t in front:
            extra_metrics.update((t.user_attrs.get("metrics") or {}).keys())
        header = [o.metric for o in cfg.objectives] + ["compiler_flags", "env", "binary"] + sorted(extra_metrics)
        with open(out_csv, "w", newline="") as fp:
            w = csv.writer(fp)
            w.writerow(header)
            for t in front:
                if t.values is None:
                    continue
                metrics: MetricDict = t.user_attrs.get("metrics", {})
                row = list(t.values) + [
                    t.user_attrs.get("compiler_flags", ""),
                    json.dumps(t.user_attrs.get("env", {})),
                    t.user_attrs.get("binary", ""),
                ]
                row += [metrics.get(k, "") for k in sorted(extra_metrics)]
                w.writerow(row)

    # Live-update Pareto CSV after each completed trial (multi-objective only)
    def _pareto_cb(study: optuna.study.Study, trial: optuna.trial.FrozenTrial) -> None:
        if is_multi and pareto_path is not None:
            _export_pareto_front(study, pareto_path)

    def trial_objective(trial: optuna.Trial):
        # --------------------------------------------------------------
        # 1) Sample discrete indices
        # --------------------------------------------------------------
        flag_key, flags = suggest_compiler_flags(
                            trial,
                            cfg.compiler_flags_base,
                            cfg.compiler_flags,
                            cfg.compiler_params,
                            cfg.compiler_flag_pool,
                            cfg.compiler_params_select,
                            cfg.search.n_startup_trials,
                        )
        env = suggest_env(trial, cfg.env)
        
        # --------------------------------------------------------------
        # 2) Build
        # --------------------------------------------------------------
        workdir = workdir_root / f"trial_{trial.number:05d}"
        workdir.mkdir()
        if cfg.source:
            binary_path = compile_single_source(cfg.compiler, cfg.source, flags, workdir / "a.out", trial)
        else:
            binary_path = compile_project(cfg.project, cfg.compiler, flags, workdir, trial)
        if not binary_path:
            raise optuna.TrialPruned("build failed")
        
        # ---- JIT Phase selection for search ----
        if jit_enabled(cfg):
            phase = "A" if cfg.jit.mode in ("isolate_then_adapt", "isolate_only") else "B"
        else:
            phase = "A" 

        # --------------------------------------------------------------
        # 3) Measure
        # --------------------------------------------------------------
        try:
            if cfg.backend == "perf":
                metrics = measure_perf(cfg.perf, binary_path, cfg.program_args, run_env, cfg.runs)  # type: ignore[arg-type]
            elif cfg.backend == "parser": 
                metrics = measure_parser_sycl(cfg.parser, binary_path, cfg.program_args, run_env, cfg.runs, workdir, cfg.project)
            else:
                metrics = measure_likwid(cfg.likwid, binary_path, cfg.program_args, run_env, cfg.runs)  # type: ignore[arg-type]        
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
        s = summarize_appdb(appdb)
        metrics["_appdb_files"] = s["files"]
        metrics["_appdb_bytes"] = s["bytes"]

        trial.set_user_attr("phase", phase)
        trial.set_user_attr("compiler_flags", flag_key)
        trial.set_user_attr("env", env)
        trial.set_user_attr("appdb_dir", run_env["ACPP_APPDB_DIR"])
        trial.set_user_attr("metrics", metrics)
        trial.set_user_attr("binary", str(binary_path))

        return obj_values

    study.optimize(trial_objective, n_trials=n_trials, show_progress_bar=True,
                   callbacks=([_pareto_cb] if is_multi and pareto_path is not None else None))
    
    # Phase B re-evaluation (persistent cache + adaptivity) for top_k
    if jit_enabled(cfg) and cfg.jit.mode == "isolate_then_adapt" and cfg.jit.top_k > 0:
        # pick “best” trials; for multi-objective you might pick Pareto front or first objective rank
        complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
        # Sort by first objective direction
        first_dir = study.directions[0].name if hasattr(study, "directions") else study.direction.name
        reverse = (first_dir == "maximize")
        top = sorted(complete, key=lambda t: t.values[0], reverse=reverse)[: cfg.jit.top_k]

        print(f"[jit] Phase B re-evaluation of top {len(top)} configs…")
        for t in top:
            flags_key = t.user_attrs["compiler_flags"]
            # if you stored the rendered string, keep it; otherwise rebuild flags from key
            flags_str = flags_key if isinstance(flags_key, str) else flags_key
            flags = flags_str.split() if isinstance(flags_str, str) else list(flags_str)
            base_env = t.user_attrs["env"]
            # rebuild (optional: reuse existing binary)
            workdir = workdir_root / f"phaseB_trial_{t.number:05d}"
            workdir.mkdir()
            if cfg.source:
                binary_path = compile_single_source(cfg.compiler, cfg.source, " ".join(flags), workdir / "a.out")
            else:
                binary_path = compile_project(cfg.project, cfg.compiler, " ".join(flags), workdir)
            if not binary_path:
                print(f"[jit] skip trial {t.number}: rebuild failed")
                continue

            run_env, appdb = jit_env_for_phase(cfg, "B", workdir, flags, base_env)
            if cfg.backend == "perf":
                metsB = measure_perf(cfg.perf, binary_path, cfg.program_args, run_env, cfg.runs)
            elif cfg.backend == "likwid":
                metsB = measure_likwid(cfg.likwid, binary_path, cfg.program_args, run_env, cfg.runs)
            else:
                metsB = measure_parser_sycl(cfg.parser, binary_path, cfg.program_args, run_env, cfg.runs, workdir, cfg.project)

            metsB["_appdb_files"] = summarize_appdb(Path(run_env["ACPP_APPDB_DIR"]))["files"]
            metsB["_appdb_bytes"] = summarize_appdb(Path(run_env["ACPP_APPDB_DIR"]))["bytes"]

            print(f"[jit][B] trial {t.number}: {cfg.objectives[0].metric}={metsB.get(cfg.objectives[0].metric)} appdb={run_env['ACPP_APPDB_DIR']}")
            # optional: write an extra CSV “phaseB_results.csv” here


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
    
    if is_multi and pareto_path is not None:
        _export_pareto_front(study, pareto_path)
        print(f"[info] Pareto CSV written → {pareto_path}")

    # ------------------------------------------------------------------
    # CSV / SQLite logging
    # ------------------------------------------------------------------
    if cfg.csv_log:
        csv_path = unique_csv_path(cfg.csv_log)
        print(f"[info] writing CSV log → {csv_path}")
        with open(csv_path, "w", newline="") as fp:
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

    failed   = study.get_trials(states=(TrialState.FAIL,))
    if failed:
        with open(cfg.fail_log, "w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow(["trial","flags","env","reason","log"])
            for t in failed:
                writer.writerow([
                    t.number,
                    t.user_attrs.get("compiler_flags"),
                    json.dumps(t.user_attrs.get("env")),
                    t.system_attrs.get("fail_reason"),
                    t.system_attrs.get("build_log"),
                ])
    

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


###############################################################################
# Synergy exploration                                                         #
###############################################################################

def explore_synergy(cfg: Config) -> None:
    # lightweight adapter – calls the new study module
    from src.studies.synergy_flags import run_synergy_study
    return run_synergy_study(cfg)


###############################################################################
# Wavefront exploration                                                       #
###############################################################################

def explore_wavefront(cfg: Config) -> None:
    from src.studies.wavefront_flags import run_wavefront_study
    return run_wavefront_study(cfg)