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
    warmup_runs: int = 0
    

    @classmethod
    def from_dict(cls, d: Dict) -> "PerfConfig":
        return cls(
            events=d.get("events", ["cycles", "instructions"]),
            core_list=d.get("core_list"),
            warmup_runs=int(d.get("warmup_runs", 0))
        )


@dataclasses.dataclass
class MetricSpec:
    name: str                      # row name in LIKWID report
    agg: str = "avg"               # "avg" | "max" | "min" | "median"
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
class ParserConfig:
    # which lines to use from the SYCL output
    label: str = "avg"                 # "avg" | "sum"
    # which kernel ids to consider (None = all ids found)
    kernels: Optional[List[int]] = None
    # how to aggregate across multiple kernels
    aggregate: str = "sum"             # "sum" | "mean" | "max" | "min"
    warmup_runs: int = 0

    # launch options to mimic perf/likwid
    core_list: Optional[str] = None    # e.g. "0-15" → taskset -c
    prefix: Optional[List[str]] = None # e.g. ["numactl","-N","0","-m","0"]
    run_cwd: str = "binary_dir"        # "binary_dir" | "project_dir" | "workdir"

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ParserConfig":
        return cls(
            label=d.get("label", "avg"),
            kernels=d.get("kernels"),
            aggregate=d.get("aggregate", "sum"),
            core_list=d.get("core_list"),
            prefix=d.get("prefix"),
            run_cwd=d.get("run_cwd", "binary_dir"),
            warmup_runs=int(d.get("warmup_runs", 0)),
        )

@dataclasses.dataclass
class LikwidConfig:
    group: str | None
    events: List[str] 
    metrics: List[MetricSpec]
    core_list: Optional[str]
    warmup_runs: int = 0

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
            warmup_runs=int(d.get("warmup_runs", 0)),
        )


@dataclasses.dataclass
class BuildProject:
    dir: Path
    build_system: str = "cmake"  # "cmake" | "make"
    target: Optional[str] = None
    make_vars: Dict[str, str] = dataclasses.field(default_factory=dict)
    make_flags_var: str = "CXXFLAGS"

    cmake_defs: List[str] = dataclasses.field(default_factory=list)
    cmake_flag_vars: List[str] = dataclasses.field(       
        default_factory=lambda: ["CMAKE_CXX_FLAGS"]
    )

    @classmethod
    def from_dict(cls, d: Dict) -> "BuildProject":
        return cls(
            dir=Path(d["dir"]),
            build_system=d.get("build_system", "cmake"),
            target=d.get("target"),
            make_vars=d.get("make_vars", {}),
            make_flags_var=d.get("make_flags_var", "CXXFLAGS"),
            cmake_flag_vars=d.get("cmake_flag_vars", ["CMAKE_CXX_FLAGS"]),
            cmake_defs=d.get("cmake_defs", []),
        )


@dataclasses.dataclass
class SearchSpec:
    study: str = "optuna"  # only optuna for now
    sampler: str = "tpe"     # "tpe" | "nsga3" | "rs" | "cmaes"
    n_startup_trials: int = 10  # for TPE
    population_size: int = 50
    random_seed: int | None = None

    @classmethod
    def from_dict(cls, d: Dict) -> "SearchSpec":
        return cls(
            study=d.get("study", "optuna"),
            sampler=d.get("sampler", "tpe"),
            n_startup_trials=int(d.get("n_startup_trials", 10)),
            population_size=int(d.get("population_size", 50)),
            random_seed=d.get("random_seed"),
        )

@dataclasses.dataclass
class WavefrontSpec:
    base_flags: List[str] = dataclasses.field(default_factory=list)
    flag_atoms: Optional[List[str]] = None
    max_k: int = 3
    mode: str = "beam"         # "beam" | "full"
    beam_width: int = 16
    per_wave_cap: Optional[int] = None
    stop_if_no_improve: bool = True
    improvement_eps: float = 0.0
    env: Dict[str, str] = dataclasses.field(default_factory=dict)
    results_csv: str = "wavefront_results.csv"

    @classmethod
    def from_dict(cls, d: Dict) -> "WavefrontSpec":
        return cls(
            base_flags=d.get("base_flags", []),
            flag_atoms=d.get("flag_atoms"),
            max_k=int(d.get("max_k", 3)),
            mode=d.get("mode", "beam"),
            beam_width=int(d.get("beam_width", 16)),
            per_wave_cap=d.get("per_wave_cap"),
            stop_if_no_improve=bool(d.get("stop_if_no_improve", True)),
            improvement_eps=float(d.get("improvement_eps", 0.0)),
            env=d.get("env", {}) or {},
            results_csv=d.get("results_csv", "wavefront_results.csv")
        )


@dataclasses.dataclass
class PolyMorphSearchSpec:
    n_trials: int = 20
    repeat: int = 3
    generated_infix: str = "optuna"
    seed: int = 0
    timeout: int | None = None
    enumerate_only: bool = False
    baseline_exec_name: str | None = None
    max_transforms_per_trial: int = 1
    tile_sizes: List[int] = dataclasses.field(default_factory=lambda: [8, 16, 32, 64])
    scale_factors: List[int] = dataclasses.field(default_factory=lambda: [2])
    shift_values: List[int] = dataclasses.field(default_factory=lambda: [-2, -1, 1, 2])
    allow_transforms: List[str] | None = None
    block_transforms: List[str] = dataclasses.field(default_factory=list)
    explicit_args: Dict[str, List[List[Any]]] = dataclasses.field(default_factory=dict)
    result_json: str | None = None
    trial_csv: str | None = None
    static_pruning: bool = False
    analytical_model: bool = False
    constraint_aware: bool = False
    compiler_feedback: bool = False
    runtime_feedback: bool = False
    case_retrieval: bool = False
    top_k: int | None = None
    retrieval_top_k: int = 3
    history_jsonl: str | None = None
    pareto_csv: str | None = None
    constraints: Dict[str, Any] = dataclasses.field(default_factory=dict)
    compiler_feedback_flags: List[str] = dataclasses.field(default_factory=list)
    runtime_feedback_masks: List[str] = dataclasses.field(default_factory=list)
    runtime_feedback_debug_level: int = 4
    runtime_feedback_repeat: int = 1
    runtime_feedback_env: Dict[str, str] = dataclasses.field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any] | None) -> "PolyMorphSearchSpec":
        data = raw or {}
        allow = data.get("allow_transforms")
        return cls(
            n_trials=int(data.get("n_trials", 20)),
            repeat=int(data.get("repeat", 3)),
            generated_infix=str(data.get("generated_infix", "optuna")),
            seed=int(data.get("seed", 0)),
            timeout=int(data["timeout"]) if data.get("timeout") is not None else None,
            enumerate_only=bool(data.get("enumerate_only", False)),
            baseline_exec_name=data.get("baseline_exec_name"),
            max_transforms_per_trial=int(data.get("max_transforms_per_trial", 1)),
            tile_sizes=[int(x) for x in data.get("tile_sizes", [8, 16, 32, 64])],
            scale_factors=[int(x) for x in data.get("scale_factors", [2])],
            shift_values=[int(x) for x in data.get("shift_values", [-2, -1, 1, 2])],
            allow_transforms=[str(x) for x in allow] if allow is not None else None,
            block_transforms=[str(x) for x in data.get("block_transforms", [])],
            explicit_args=dict(data.get("explicit_args", {})),
            result_json=data.get("result_json"),
            trial_csv=data.get("trial_csv"),
            static_pruning=bool(data.get("static_pruning", False)),
            analytical_model=bool(data.get("analytical_model", False)),
            constraint_aware=bool(data.get("constraint_aware", False)),
            compiler_feedback=bool(data.get("compiler_feedback", False)),
            runtime_feedback=bool(data.get("runtime_feedback", False)),
            case_retrieval=bool(data.get("case_retrieval", False)),
            top_k=int(data["top_k"]) if data.get("top_k") is not None else None,
            retrieval_top_k=int(data.get("retrieval_top_k", 3)),
            history_jsonl=data.get("history_jsonl"),
            pareto_csv=data.get("pareto_csv"),
            constraints=dict(data.get("constraints", {}) or {}),
            compiler_feedback_flags=[
                str(flag) for flag in data.get("compiler_feedback_flags", [])
                if str(flag).strip()
            ],
            runtime_feedback_masks=[
                str(mask) for mask in data.get("runtime_feedback_masks", [])
                if str(mask).strip()
            ],
            runtime_feedback_debug_level=int(data.get("runtime_feedback_debug_level", 4)),
            runtime_feedback_repeat=int(data.get("runtime_feedback_repeat", 1)),
            runtime_feedback_env={
                str(k): str(v)
                for k, v in dict(data.get("runtime_feedback_env", {}) or {}).items()
            },
        )


@dataclasses.dataclass
class PolyMorphSpec:
    project_root: Path
    source: Optional[Path] = None
    compiler: str = "acpp"
    flags: List[str] = dataclasses.field(default_factory=list)
    transforms: List[Dict[str, Any]] = dataclasses.field(default_factory=list)
    allow_illegal: bool = False
    print_available_transformations: bool = False
    list_only: bool = False
    save_jscops: Optional[Path] = None
    discover: bool = False
    optuna_search: bool = False
    build_system: str = "make"
    build_target: Optional[str] = None
    build_dir: Optional[Path] = None
    exec_name: Optional[str] = None
    runtime_args: List[str] = dataclasses.field(default_factory=list)
    measure: bool = False
    generated_infix: str = "pass1"
    search: PolyMorphSearchSpec = dataclasses.field(default_factory=PolyMorphSearchSpec)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "PolyMorphSpec":
        source_raw = raw.get("source")
        save_jscops_raw = raw.get("save_jscops")
        build_dir_raw = raw.get("build_dir")
        transforms = raw.get("transforms", []) or []
        flags = raw.get("flags", []) or []
        build_system = str(raw.get("build_system", "make"))
        if build_system not in {"make", "cmake"}:
            raise ValueError("polyMorph.build_system must be 'make' or 'cmake'")
        if not raw.get("project_root"):
            raise ValueError("polyMorph.project_root is required")
        if not isinstance(transforms, list):
            raise ValueError("polyMorph.transforms must be a list")
        if not isinstance(flags, list):
            raise ValueError("polyMorph.flags must be a list")

        return cls(
            project_root=Path(raw["project_root"]).expanduser().resolve(),
            source=Path(source_raw).expanduser().resolve() if source_raw else None,
            compiler=str(raw.get("compiler", "acpp")),
            flags=[str(flag) for flag in flags if str(flag).strip()],
            transforms=[dict(item) for item in transforms],
            allow_illegal=bool(raw.get("allow_illegal", False)),
            print_available_transformations=bool(raw.get("print_available_transformations", False)),
            list_only=bool(raw.get("list_only", False)),
            save_jscops=Path(save_jscops_raw).expanduser().resolve() if save_jscops_raw else None,
            discover=bool(raw.get("discover", False)),
            optuna_search=bool(raw.get("optuna_search", False)),
            build_system=build_system,
            build_target=raw.get("build_target"),
            build_dir=Path(build_dir_raw).expanduser().resolve() if build_dir_raw else None,
            exec_name=raw.get("exec_name"),
            runtime_args=_normalize_args(raw.get("runtime_args")),
            measure=bool(raw.get("measure", False)),
            generated_infix=str(raw.get("generated_infix", "pass1")),
            search=PolyMorphSearchSpec.from_dict(raw.get("search")),
        )


@dataclasses.dataclass
class Config:
    backend: str  # "perf" | "likwid"

    # Build description (one of the two)
    source: Optional[Path]
    project: Optional[BuildProject]

    # Program arguments and environment sets
    program_args: List[str] 

    # Compiler flags
    compiler: str 
    compiler_flags_base: str 
    compiler_flags: List[str] 
    compiler_params: Dict[str, Union[List[Any], Dict[str, Any]]] 
    compiler_params_select: Dict[str, Any] 
    compiler_flag_pool: List[str]
    
    #Environment variables
    env: Dict[str, Union[List[str], Dict[str, Any]]] 

    # Backend‑specific blocks
    perf: Optional[PerfConfig]
    likwid: Optional[LikwidConfig]
    parser: Optional[ParserConfig] 

    # Objectives (≥1)
    objectives: List[Objective] 

    # Search algorithm details
    search: SearchSpec

    runs: int
    # CSV / SQLite log paths
    csv_log: Optional[str]
    pareto_log: Optional[str]
    fail_log: Optional[str]
    sqlite_log: Optional[str]

    # Wavefront
    wavefront: Optional[WavefrontSpec] = None
    #tabu
    tabu: Dict[str, Any] = dataclasses.field(default_factory=dict)
    #anneal 
    anneal: Dict[str, Any] = dataclasses.field(default_factory=dict)
    # polyMorph
    poly_morph: Optional[PolyMorphSpec] = None

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def load(cls, path: Path | str) -> "Config":
        with open(path, "r", encoding="utf-8") as fp:
            raw = json.load(fp)

        poly_morph_raw = raw.get("polyMorph")
        poly_morph = PolyMorphSpec.from_dict(poly_morph_raw) if isinstance(poly_morph_raw, dict) else None

        backend = raw.get("backend", "likwid").lower()
        if backend not in {"perf", "likwid", "parser"}:
            raise ValueError("backend must be 'perf' or 'likwid' or 'parser'")

        # Build description
        source = raw.get("source")
        project = raw.get("project")
        if bool(source) == bool(project) and poly_morph is None:
            raise ValueError("Provide exactly one of 'source' or 'project', or define 'polyMorph'.")
        
        # Compiler flags
        def _validate_params_select(sel: Dict[str, Any], available: List[str]) -> Dict[str, Any]:
            if not sel:
                return {}
            if "k" in sel and ("min" in sel or "max" in sel):
                raise ValueError("compiler_params_select: use either 'k' or 'min/max', not both")
            if "always" in sel:
                unknown = [k for k in sel["always"] if k not in available]
                if unknown:
                    print(f"[warn] compiler_params_select.always has unknown keys: {unknown}")
            return sel
        
        compiler_params = raw.get("compiler_params", {})
        compiler_params_select = _validate_params_select(
            raw.get("compiler_params_select", {}),
            list(compiler_params.keys())
        )

        # Environment space
        env_schema = raw.get("env")
        if not isinstance(env_schema, dict):
            env_schema = {}
        
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
        
        # Parser
        parser_cfg = ParserConfig.from_dict(raw.get("parser", {})) if backend == "parser" else None

        tabu = raw.get("tabu", {})

        anneal = raw.get("anneal", {})

        return cls(
            backend=backend,
            source=Path(source) if source else None,
            project=BuildProject.from_dict(project) if project else None,
            compiler=raw.get("compiler", "acpp"),
            compiler_flags_base=raw.get("compiler_flags_base", ""),
            compiler_flags=raw.get("compiler_flags", []),
            compiler_params = compiler_params,
            compiler_params_select = compiler_params_select,
            compiler_flag_pool  = raw.get("compiler_flag_pool", []),
            program_args=program_args,
            env=env_schema,
            perf=PerfConfig.from_dict(raw.get("perf", {})) if backend == "perf" else None,
            likwid=LikwidConfig.from_dict(raw.get("likwid", {})) if backend == "likwid" else None,
            parser=parser_cfg,
            objectives=objectives,
            search=SearchSpec.from_dict(raw.get("search", {})),
            wavefront=WavefrontSpec.from_dict(raw.get("wavefront", {})) if "wavefront" in raw else None,
            tabu=tabu,
            anneal=anneal,
            poly_morph=poly_morph,
            runs=int(raw.get("runs", 1)),
            csv_log=raw.get("csv_log"),
            pareto_log=raw.get("pareto_log"),
            fail_log=raw.get("failed_builds"),
            sqlite_log=raw.get("sqlite_log"),
        )
