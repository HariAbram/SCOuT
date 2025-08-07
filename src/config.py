from __future__ import annotations

###############################################################################
# Standard library imports                                                    #
###############################################################################

import dataclasses
import json
import sys
from pathlib import Path
from statistics import mean, variance
from typing import Dict, List, Optional, Sequence, Tuple, Any, Union

###############################################################################
# Type helpers                                                                #
###############################################################################

Number = float
EnvMap = Dict[str, str]
MetricDict = Dict[str, Number]

###############################################################################
# Local imports                                                               #
###############################################################################

from src.misc import _normalize_args

###############################################################################
# Data‑classes for strongly‑typed config handling                             #
###############################################################################

@dataclasses.dataclass
class Objective:
    metric: str = "CPI"  # metric key to optimise
    goal: str = "min"    # "min" | "max"

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> "Objective":
        goal = d.get("goal", "min").lower()
        if goal not in {"min", "max"}:
            raise ValueError("objective.goal must be 'min' or 'max'")
        return cls(metric=d.get("metric", "CPI"), goal=goal)

    # Note: better() no longer used because Optuna handles dominance.


@dataclasses.dataclass
class PerfConfig:
    events: List[str]
    core_list: Optional[str]

    @classmethod
    def from_dict(cls, d: Dict) -> "PerfConfig":
        return cls(
            events=d.get("events", ["cycles", "instructions"]),
            core_list=d.get("core_list"),
        )


@dataclasses.dataclass
class MetricSpec:
    name: str                      # row name in LIKWID report
    agg: str = "avg"               # "avg" | "max" | "min"
    var: bool = False              # also compute variance?

    @classmethod
    def from_any(cls, raw):
        # Accept plain string for backward compat
        if isinstance(raw, str):
            return cls(name=raw)
        if isinstance(raw, dict):
            return cls(name=raw["name"],
                       agg=raw.get("agg", "avg").lower(),
                       var=bool(raw.get("var", False)))
        raise TypeError("metrics entries must be string or object")



@dataclasses.dataclass
class LikwidConfig:
    group: str | None
    events: List[str] 
    metrics: List[MetricSpec]
    core_list: Optional[str]

    @classmethod
    def from_dict(cls, d: Dict) -> "LikwidConfig":

        group   = d.get("group")
        events  = d.get("events", [])
        if isinstance(events, str):
            events = [e.strip() for e in events.split(",") if e.strip()]
        if not group and not events:
            raise ValueError("Need either 'group' or 'events' in likwid block")

        metrics_raw = d.get("metrics")                       # explicit list
        if not metrics_raw:                                  # or infer from events
            metrics_raw = [e.split(":")[0] for e in events]  if events else [group]
        metrics = [MetricSpec.from_any(m) for m in metrics_raw]

        return cls(
            group=group,
            events=events,
            metrics=metrics,
            core_list=d.get("core_list"),
        )


@dataclasses.dataclass
class BuildProject:
    dir: Path
    build_system: str = "cmake"  # "cmake" | "make"
    target: Optional[str] = None
    make_vars: Dict[str, str] = dataclasses.field(default_factory=dict)
    cmake_defs: List[str] = dataclasses.field(default_factory=list)

    @classmethod
    def from_dict(cls, d: Dict) -> "BuildProject":
        return cls(
            dir=Path(d["dir"]),
            build_system=d.get("build_system", "cmake"),
            target=d.get("target"),
            make_vars=d.get("make_vars", {}),
            cmake_defs=d.get("cmake_defs", []),
        )


@dataclasses.dataclass
class SearchSpec:
    method: str = "optuna"  # only optuna for now
    sampler: str = "tpe"     # "tpe" | "nsga3" | "rs" | "cmaes"
    n_startup_trials: int = 10  # for TPE
    population_size: int = 50
    random_seed: int | None = None

    @classmethod
    def from_dict(cls, d: Dict) -> "SearchSpec":
        return cls(
            method=d.get("method", "optuna"),
            sampler=d.get("sampler", "tpe"),
            n_startup_trials=int(d.get("n_startup_trials", 10)),
            population_size=int(d.get("population_size", 50)),
            random_seed=d.get("random_seed"),
        )


@dataclasses.dataclass
class Config:
    backend: str  # "perf" | "likwid"

    # Build description (one of the two)
    source: Optional[Path]
    project: Optional[BuildProject]

    # Compiler flags
    compiler: str
    compiler_flags_base: str
    compiler_flags: List[str]
    compiler_params: Dict[str, Union[List[Any], Dict[str, Any]]]
    

    # Program arguments and environment sets
    program_args: List[str]
    #env: Dict[str, List[str]] 
    #env_sets: List[EnvMap]
    env: Dict[str, Union[List[str], Dict[str, Any]]]

    # Backend‑specific blocks
    perf: Optional[PerfConfig]
    likwid: Optional[LikwidConfig]

    # Objectives (≥1)
    objectives: List[Objective]

    # Search algorithm details
    search: SearchSpec

    runs: int
    # CSV / SQLite log paths
    csv_log: Optional[str]
    fai_log_log: Optional[str]
    sqlite_log: Optional[str]
    compiler_flag_pool: List[str] = dataclasses.field(default_factory=list)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path | str) -> "Config":
        with open(path, "r", encoding="utf-8") as fp:
            raw = json.load(fp)

        backend = raw.get("backend", "likwid").lower()
        if backend not in {"perf", "likwid"}:
            raise ValueError("backend must be 'perf' or 'likwid'")

        # Build description
        source = raw.get("source")
        project = raw.get("project")
        if bool(source) == bool(project):
            raise ValueError("Provide exactly one of 'source' or 'project'.")

        # Environment space
        env_schema = raw.get("env")
        if not isinstance(env_schema, dict) or not env_schema:
            raise ValueError("Config must contain a non-empty 'env' object.")
        
        for var, spec in env_schema.items():
            if isinstance(spec, list):
                continue
            if not (isinstance(spec, dict) and "values" in spec):
                raise ValueError(
                    f"'env.{var}' must be a list or an object with a 'values' key"
                )

        # Program arguments
        program_args = _normalize_args(raw.get("program_args"))

        # Objectives
        objs_raw = raw.get("objectives")
        if not objs_raw:
            # Fallback to single objective block from backend
            if backend == "perf":
                o_raw = raw.get("perf", {}).get("objective", {})
            else:
                o_raw = raw.get("likwid", {}).get("objective", {})
            objs_raw = [o_raw]

        objectives = [Objective.from_dict(o) for o in objs_raw]
        if not objectives:
            raise ValueError("At least one objective required.")

        return cls(
            backend=backend,
            source=Path(source) if source else None,
            project=BuildProject.from_dict(project) if project else None,
            compiler=raw.get("compiler", "acpp"),
            compiler_flags_base=raw.get("compiler_flags_base", ""),
            compiler_flags=raw.get("compiler_flags", []),
            compiler_params = raw.get("compiler_params", {}),
            compiler_flag_pool  = raw.get("compiler_flag_pool", []),
            program_args=program_args,
            env=env_schema,
            perf=PerfConfig.from_dict(raw.get("perf", {})) if backend == "perf" else None,
            likwid=LikwidConfig.from_dict(raw.get("likwid", {})) if backend == "likwid" else None,
            objectives=objectives,
            search=SearchSpec.from_dict(raw.get("search", {})),
            runs=raw.get("runs"),
            csv_log=raw.get("csv_log"),
            fail_log=raw.get("failed_builds"),
            sqlite_log=raw.get("sqlite_log"),
        )
