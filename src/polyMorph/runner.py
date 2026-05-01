from __future__ import annotations

import csv
import json
import re
import shutil
import shlex
import subprocess
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import optuna

from src.config import Config, PolyMorphSpec
from src.metrics import measure_parser_sycl

try:
    from tadashi import TrEnum
    from tadashi.apps import App
    from tadashi.translators import Polly
    TADASHI_IMPORT_ERROR: Exception | None = None
except Exception as exc:
    TrEnum = Any  # type: ignore[assignment]
    App = object  # type: ignore[assignment]
    Polly = Any  # type: ignore[assignment]
    TADASHI_IMPORT_ERROR = exc


JsonDict = Dict[str, Any]


def _run(
    cmd: Sequence[str] | str,
    *,
    cwd: Path | None = None,
    env: Dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    pretty = " ".join(shlex.quote(str(part)) for part in cmd) if isinstance(cmd, Sequence) else cmd
    print(f"[exec] {pretty}" + (f"  (cwd={cwd})" if cwd else ""))
    return subprocess.run(
        cmd,
        shell=isinstance(cmd, str),
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )


def ensure_tadashi_available() -> None:
    if TADASHI_IMPORT_ERROR is None:
        return
    raise RuntimeError(
        "Tadashi could not be imported in this environment. "
        f"Original error: {TADASHI_IMPORT_ERROR}"
    )


def require_executable(name: str) -> None:
    if shutil.which(name) is not None:
        return
    raise RuntimeError(
        f"Required executable '{name}' was not found on PATH. "
        "Update polyMorph.compiler or load the toolchain environment first."
    )


def discover_sycl_sources(project_root: Path) -> List[Path]:
    exts = {".cpp", ".cc", ".cxx", ".c++", ".C"}
    skip_dirs = {".git", "build", "CMakeFiles", ".cache", "__pycache__"}
    patterns = [
        r"#\s*include\s*<sycl",
        r"#\s*include\s*<CL/sycl",
        r"\bsycl::queue\b",
        r"\bparallel_for\b",
        r"\bsingle_task\b",
        r"\bnd_range\b",
        r"\bhandler\b",
    ]

    found: List[Path] = []
    for path in project_root.rglob("*"):
        if any(part in skip_dirs for part in path.parts):
            continue
        if not path.is_file() or path.suffix not in exts:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        if any(re.search(pattern, text) for pattern in patterns):
            found.append(path.resolve())
    return sorted(found)


class SyclProjectApp(App):
    def __init__(
        self,
        *,
        project_root: str | Path,
        source: str | Path,
        exec_name: str,
        build_system: str = "make",
        build_target: str | None = None,
        build_dir: str | Path | None = None,
        runtime_args: Sequence[str] | None = None,
        compiler_executable: str | None = None,
        translator: Any | None = None,
        compiler_options: Sequence[str] | None = None,
        ephemeral: bool = False,
        populate_scops: bool = True,
    ) -> None:
        self.project_root = Path(project_root).resolve()
        self.exec_name = exec_name
        self.build_system = build_system
        self.build_target = build_target
        self.build_dir = Path(build_dir).resolve() if build_dir else self.project_root
        self.runtime_args = list(runtime_args or [])
        self.compiler_options = list(compiler_options or [])
        self.compiler_executable = compiler_executable

        if self.compiler_executable is None and translator is not None:
            compiler_candidate = getattr(translator, "compiler", None)
            if isinstance(compiler_candidate, str) and compiler_candidate.strip():
                self.compiler_executable = compiler_candidate

        super().__init__(
            source=Path(source).resolve(),
            translator=translator,
            compiler_options=self.compiler_options,
            ephemeral=ephemeral,
            populate_scops=populate_scops,
        )

    def codegen_init_args(self) -> dict[str, Any]:
        return {
            "project_root": self.project_root,
            "exec_name": self.exec_name,
            "build_system": self.build_system,
            "build_target": self.build_target,
            "build_dir": self.build_dir,
            "runtime_args": self.runtime_args,
        }

    @property
    def output_binary(self) -> Path:
        return self.build_dir / self.exec_name

    def compile_cmd(self, suffix: str) -> List[str]:
        del suffix
        src = str(self.source)
        exe = str(self.output_binary)

        if self.build_system == "make":
            extra_cflags = " ".join(self.compiler_options)
            cmd = [
                "make",
                "-C",
                str(self.project_root),
                f"SOURCE={src}",
                f"EXEC={exe}",
            ]
            if self.compiler_executable:
                cmd.append(f"CC={self.compiler_executable}")
                cmd.append(f"CXX={self.compiler_executable}")
            if extra_cflags:
                cmd.append(f"EXTRA_CFLAGS={extra_cflags}")
            if self.build_target:
                cmd.append(self.build_target)
            return cmd

        if self.build_system == "cmake":
            cmd = ["cmake", "--build", str(self.build_dir)]
            if self.build_target:
                cmd += ["--target", self.build_target]
            cmd += ["--", f"SOURCE={src}", f"EXEC={exe}"]
            return cmd

        raise ValueError(f"Unsupported build_system: {self.build_system}")

    def run_cmd(self) -> List[str]:
        return [str(self.output_binary), *self.runtime_args]

    def extract_runtime(self, proc: subprocess.CompletedProcess[str]) -> float:
        match = re.search(r"WALLTIME:\s*([0-9]*\.?[0-9]+)", proc.stdout)
        return float(match.group(1)) if match else 0.0


def enum_from_name(name: str) -> Any:
    ensure_tadashi_available()
    try:
        return getattr(TrEnum, name)
    except AttributeError as exc:
        valid = ", ".join(item.name for item in TrEnum)
        raise SystemExit(f"Unknown transformation '{name}'. Valid names: {valid}") from exc


def describe_scops(translator: Any) -> None:
    if not getattr(translator, "scops", None):
        print("No SCoPs discovered.")
        return

    print(f"Discovered {len(translator.scops)} SCoP(s)")
    for scop_idx, scop in enumerate(translator.scops):
        print(f"\n=== SCoP {scop_idx} ===")
        print(f"Nodes in schedule tree: {len(scop.schedule_tree)}")
        for node_idx, node in enumerate(scop.schedule_tree):
            node_type = getattr(node, "node_type", "<unknown>")
            available = getattr(node, "available_transformations", None)
            yaml_str = getattr(node, "yaml_str", None)

            print(f"\n  Node {node_idx}")
            print(f"    type: {node_type}")
            if available is not None:
                names = [getattr(item, "name", str(item)) for item in available]
                print(f"    available_transformations: {names}")
            if yaml_str:
                indented = "\n".join("      " + line for line in str(yaml_str).splitlines())
                print("    yaml:")
                print(indented)


def print_available_transformations_only(translator: Any) -> None:
    if not getattr(translator, "scops", None):
        print("No SCoPs discovered.")
        return

    print(f"Discovered {len(translator.scops)} SCoP(s)")
    for scop_idx, scop in enumerate(translator.scops):
        print(f"\n=== SCoP {scop_idx} ===")
        for node_idx, node in enumerate(scop.schedule_tree):
            node_type = getattr(node, "node_type", "<unknown>")
            available = getattr(node, "available_transformations", [])
            names = [getattr(item, "name", str(item)) for item in available]
            print(f"  Node {node_idx} ({node_type})")
            if names:
                for name in names:
                    print(f"    - {name}")
            else:
                print("    - <none>")


def apply_transforms_to_scops(
    scops: List[Any],
    specs: List[JsonDict],
    legality_cb: Callable[[], Any] | None = None,
) -> None:
    for idx, spec in enumerate(specs):
        try:
            scop_idx = int(spec["scop"])
            node_idx = int(spec["node"])
            tr_name = str(spec["tr"])
            tr_args = list(spec.get("args", []))
        except Exception as exc:
            raise SystemExit(f"Invalid transform spec at index {idx}: {spec}") from exc

        node = scops[scop_idx].schedule_tree[node_idx]
        tr = enum_from_name(tr_name)

        print(
            f"Applying transform #{idx}: "
            f"scop={scop_idx}, node={node_idx}, tr={tr_name}, args={tr_args}"
        )
        ok = node.transform(tr, *tr_args)
        print(f"  result={ok}")

        available = getattr(node, "available_transformations", [])
        names = [getattr(item, "name", str(item)) for item in available]
        print(f"  current node type: {getattr(node, 'node_type', '<unknown>')}")
        print(f"  currently available: {names}")

        if legality_cb is not None:
            try:
                legal = legality_cb()
                print(f"  legal()={legal}")
            except Exception as exc:
                print(f"  legality check failed: {exc}")


def copy_jscops_from_translator(translator: Any, dst_dir: Path) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for rel_path in translator.json_paths:
        src = translator.tmpdir / rel_path
        if src.exists():
            shutil.copy2(src, dst_dir / src.name)
        bak = src.with_suffix(src.suffix + ".bak")
        if bak.exists():
            shutil.copy2(bak, dst_dir / bak.name)


def remove_stale_build_output(app: SyclProjectApp) -> None:
    output = app.output_binary
    if output.is_file() or output.is_symlink():
        output.unlink()


def build_app_or_raise(app: SyclProjectApp) -> None:
    print(f"[build] source={app.source}")
    print(f"[build] output={app.output_binary}")
    remove_stale_build_output(app)
    proc = _run(app.compile_cmd(""))
    if proc.returncode == 0:
        return
    raise RuntimeError(
        "Build failed\n"
        f"stdout:\n{proc.stdout}\n"
        f"stderr:\n{proc.stderr}\n"
    )


def _parser_metric_name(cfg: Config) -> str:
    if cfg.objectives:
        return cfg.objectives[0].metric
    if cfg.parser is None:
        raise RuntimeError("Parser backend selected, but parser config is missing.")
    return f"sycl_{cfg.parser.label}_{cfg.parser.aggregate}_s"


def measure_app_or_raise(app: SyclProjectApp, repeat: int, cfg: Config | None = None) -> float:
    if not app.output_binary.exists():
        build_app_or_raise(app)

    if cfg is not None and cfg.backend == "parser":
        if cfg.parser is None:
            raise RuntimeError("Parser backend selected, but parser config is missing.")
        metrics = measure_parser_sycl(
            cfg.parser,
            app.output_binary,
            app.runtime_args,
            {},
            repeat,
            workdir=app.project_root,
            project=None,
        )
        metric_name = _parser_metric_name(cfg)
        if metric_name not in metrics:
            raise RuntimeError(
                f"Parser metric '{metric_name}' missing; got metrics: {sorted(metrics.keys())}"
            )
        return float(metrics[metric_name])

    values: List[float] = []
    cmd = app.run_cmd()
    for _ in range(repeat):
        proc = _run(cmd)
        if proc.returncode != 0:
            raise RuntimeError(
                "Run failed\n"
                f"stdout:\n{proc.stdout}\n"
                f"stderr:\n{proc.stderr}\n"
            )
        value = app.extract_runtime(proc)
        if value <= 0.0:
            raise RuntimeError(
                "Could not extract runtime. Expected output like 'WALLTIME: <number>'.\n"
                f"stdout:\n{proc.stdout}"
            )
        values.append(value)
    return min(values)


def inherit_build_settings(dst: SyclProjectApp, src: SyclProjectApp) -> SyclProjectApp:
    dst.compiler_executable = src.compiler_executable
    dst.compiler_options = list(src.compiler_options)
    dst.project_root = src.project_root
    dst.build_system = src.build_system
    dst.build_target = src.build_target
    dst.build_dir = src.build_dir
    dst.runtime_args = list(src.runtime_args)
    return dst


def _trial_csv_path(poly: PolyMorphSpec) -> Path:
    if poly.search.trial_csv:
        return Path(poly.search.trial_csv)
    if poly.search.result_json:
        result_path = Path(poly.search.result_json)
        return result_path.with_name(f"{result_path.stem}_trials.csv")
    return Path("polymorph_trials.csv")


def write_trial_csv(study: optuna.study.Study, poly: PolyMorphSpec) -> Path:
    out_csv = _trial_csv_path(poly)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    header = [
        "trial",
        "state",
        "objective",
        "speedup",
        "transforms",
        "failure",
        "params",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        for t in study.trials:
            objective = t.value if t.value is not None else ""
            speedup = t.user_attrs.get("speedup", "")
            transforms = json.dumps(t.user_attrs.get("transforms", []))
            failure = t.user_attrs.get("failure", "")
            params = json.dumps(t.params, sort_keys=True)
            writer.writerow(
                [
                    t.number,
                    str(t.state.name),
                    objective,
                    speedup,
                    transforms,
                    failure,
                    params,
                ]
            )
    return out_csv


def transform_allowed(name: str, poly: PolyMorphSpec) -> bool:
    allow = poly.search.allow_transforms
    if allow is not None and name not in allow:
        return False
    if name in poly.search.block_transforms:
        return False
    return True


def candidate_args_for_transform(name: str, poly: PolyMorphSpec) -> List[List[Any]]:
    if name == "TILE_1D":
        return [[int(x)] for x in poly.search.tile_sizes]
    if name == "TILE_2D":
        return list(poly.search.explicit_args.get(name, []))
    if name == "TILE_3D":
        return list(poly.search.explicit_args.get(name, []))
    if name == "INTERCHANGE":
        return [[]]
    if name == "FULL_FUSE":
        return [[]]
    if name == "FUSE":
        return list(poly.search.explicit_args.get(name, []))
    if name == "FULL_SPLIT":
        return [[]]
    if name == "SPLIT":
        return list(poly.search.explicit_args.get(name, []))
    if name == "SCALE":
        return [[int(x)] for x in poly.search.scale_factors]
    if name == "FULL_SHIFT_VAL":
        return [[int(x)] for x in poly.search.shift_values]
    if name == "PARTIAL_SHIFT_VAL":
        return list(poly.search.explicit_args.get(name, []))
    if name in {
        "FULL_SHIFT_VAR",
        "PARTIAL_SHIFT_VAR",
        "FULL_SHIFT_PARAM",
        "PARTIAL_SHIFT_PARAM",
        "SET_PARALLEL",
        "SET_LOOP_OPT",
    }:
        return list(poly.search.explicit_args.get(name, []))
    return [[]]


def enumerate_transform_candidates(app: SyclProjectApp, poly: PolyMorphSpec) -> List[JsonDict]:
    candidates: List[JsonDict] = []
    for scop_idx, scop in enumerate(app.scops):
        for node_idx, node in enumerate(scop.schedule_tree):
            available = getattr(node, "available_transformations", [])
            for tr in available:
                name = getattr(tr, "name", str(tr))
                if not transform_allowed(name, poly):
                    continue
                for args in candidate_args_for_transform(name, poly):
                    candidates.append(
                        {
                            "scop": scop_idx,
                            "node": node_idx,
                            "tr": name,
                            "args": list(args),
                        }
                    )
    return candidates


def print_candidates(candidates: List[JsonDict]) -> None:
    print(f"Enumerated {len(candidates)} candidate transform(s)")
    for idx, candidate in enumerate(candidates):
        print(
            f"{idx}: scop={candidate['scop']} node={candidate['node']} "
            f"tr={candidate['tr']} args={candidate.get('args', [])}"
        )


def sample_transform_combination(
    trial: optuna.Trial,
    candidates: List[JsonDict],
    poly: PolyMorphSpec,
) -> List[JsonDict]:
    max_transforms = max(1, min(poly.search.max_transforms_per_trial, len(candidates)))
    n_transforms = trial.suggest_int("n_transforms", 1, max_transforms)

    chosen: List[JsonDict] = []
    used_indices: set[int] = set()
    for pos in range(n_transforms):
        idx = trial.suggest_int(f"candidate_{pos}", 0, len(candidates) - 1)
        if idx in used_indices:
            continue
        used_indices.add(idx)
        chosen.append(candidates[idx])
    return chosen


def sample_transform_sequence(
    trial: optuna.Trial,
    app: SyclProjectApp,
    poly: PolyMorphSpec,
) -> List[JsonDict]:
    initial_candidates = enumerate_transform_candidates(app, poly)
    if not initial_candidates:
        return []

    max_transforms = max(1, min(poly.search.max_transforms_per_trial, len(initial_candidates)))
    n_transforms = trial.suggest_int("n_transforms", 1, max_transforms)

    chosen: List[JsonDict] = []
    for pos in range(n_transforms):
        current_candidates = enumerate_transform_candidates(app, poly)
        if not current_candidates:
            break

        idx = trial.suggest_int(f"candidate_{pos}", 0, len(current_candidates) - 1)
        spec = current_candidates[idx]
        chosen.append(spec)

        apply_transforms_to_scops(
            app.scops,
            [spec],
            legality_cb=lambda: app.legal,
        )

        if not poly.allow_illegal and not app.legal:
            raise optuna.TrialPruned("Illegal transformed schedule")

    return chosen


def make_project_app(
    *,
    poly: PolyMorphSpec,
    source: Path,
    exec_name: str,
    populate_scops: bool,
) -> SyclProjectApp:
    ensure_tadashi_available()
    return SyclProjectApp(
        project_root=poly.project_root,
        source=source,
        exec_name=exec_name,
        build_system=poly.build_system,
        build_target=poly.build_target,
        build_dir=poly.build_dir,
        runtime_args=poly.runtime_args,
        compiler_executable=poly.compiler,
        translator=Polly(poly.compiler) if populate_scops else None,
        compiler_options=poly.flags,
        ephemeral=False,
        populate_scops=populate_scops,
    )


def infer_source(poly: PolyMorphSpec) -> Path:
    if poly.source is not None:
        if not poly.source.exists():
            raise FileNotFoundError(f"Source file not found: {poly.source}")
        return poly.source

    candidates = discover_sycl_sources(poly.project_root)
    if len(candidates) == 1:
        print(f"Auto-selected only discovered SYCL source: {candidates[0]}")
        return candidates[0]
    if not candidates:
        raise ValueError("No SYCL-looking source files found.")

    joined = "\n".join(str(path) for path in candidates)
    raise ValueError(
        "Multiple SYCL-looking source files found. Please specify polyMorph.source in config.\n"
        f"{joined}"
    )


def run_project_mode(cfg: Config, poly: PolyMorphSpec, source: Path) -> int:
    ensure_tadashi_available()
    require_executable(poly.compiler)
    if not poly.exec_name:
        raise ValueError("polyMorph.exec_name is required for project mode.")

    app = make_project_app(
        poly=poly,
        source=source,
        exec_name=poly.exec_name,
        populate_scops=True,
    )

    if poly.print_available_transformations:
        print_available_transformations_only(app.translator)
        return 0

    describe_scops(app.translator)
    if poly.list_only:
        return 0

    if poly.transforms:
        apply_transforms_to_scops(app.scops, poly.transforms, legality_cb=lambda: app.legal)
    else:
        print("\nNo transforms were provided; project will be rebuilt around unchanged source.")

    try:
        final_legal = app.legal
    except Exception as exc:
        print(f"Final legality check failed: {exc}")
        final_legal = False

    print(f"\nFinal legality: {final_legal}")
    if not final_legal and not poly.allow_illegal:
        print(
            "Refusing to generate code because the transformed schedule is illegal. "
            "Set polyMorph.allow_illegal to true to override."
        )
        return 1

    try:
        transformed_app = app.generate_code(
            alt_infix=poly.generated_infix,
            ephemeral=False,
            populate_scops=False,
            ensure_legality=not poly.allow_illegal,
        )
        transformed_app = inherit_build_settings(transformed_app, app)
    except Exception as exc:
        print("Failed while generating transformed source for project build.")
        print(str(exc))
        return 1

    print(f"\nGenerated transformed source: {transformed_app.source}")

    try:
        build_app_or_raise(transformed_app)
    except Exception as exc:
        print("Project build failed.")
        print(str(exc))
        return 1

    print(f"Built binary: {transformed_app.output_binary}")
    if poly.measure:
        try:
            runtime = measure_app_or_raise(transformed_app, poly.search.repeat, cfg)
            print(f"Measured runtime: {runtime}")
        except Exception as exc:
            print(f"Measurement failed: {exc}")
            return 1

    if poly.save_jscops:
        copy_jscops_from_translator(app.translator, poly.save_jscops)
    return 0


def explore_optuna(cfg: Config, poly: PolyMorphSpec, source: Path) -> int:
    require_executable(poly.compiler)
    if not poly.exec_name:
        raise ValueError("polyMorph.exec_name is required for search mode.")

    baseline_exec = poly.search.baseline_exec_name or f"{poly.exec_name}-baseline"

    print("\n=== Building/measuring baseline ===")
    baseline_app = make_project_app(
        poly=poly,
        source=source,
        exec_name=baseline_exec,
        populate_scops=False,
    )
    build_app_or_raise(baseline_app)
    baseline_value = measure_app_or_raise(baseline_app, poly.search.repeat, cfg)
    print(f"Baseline objective: {baseline_value}")

    enum_app = make_project_app(
        poly=poly,
        source=source,
        exec_name=poly.exec_name,
        populate_scops=True,
    )
    candidates = enumerate_transform_candidates(enum_app, poly)
    print_candidates(candidates)

    if not candidates:
        print("No candidate transformations found.")
        return 1

    if poly.search.enumerate_only:
        return 0

    def objective(trial: optuna.Trial) -> float:
        trial_exec = f"{poly.exec_name}-trial-{trial.number}"
        trial_app = make_project_app(
            poly=poly,
            source=source,
            exec_name=trial_exec,
            populate_scops=True,
        )

        try:
            specs = sample_transform_sequence(trial, trial_app, poly)
            if not specs:
                raise optuna.TrialPruned("No candidate transformations available for this trial.")
            trial.set_user_attr("transforms", specs)

            transformed_app = trial_app.generate_code(
                alt_infix=f"{poly.search.generated_infix}-{trial.number}",
                ephemeral=False,
                populate_scops=False,
                ensure_legality=not poly.allow_illegal,
            )
            transformed_app = inherit_build_settings(transformed_app, trial_app)
            transformed_app.exec_name = trial_exec
            print(
                f"Trial {trial.number}: generated transformed source: "
                f"{transformed_app.source}"
            )

            build_app_or_raise(transformed_app)
            value = measure_app_or_raise(transformed_app, poly.search.repeat, cfg)
            speedup = baseline_value / value if value > 0 else 0.0

            trial.set_user_attr("runtime", value)
            trial.set_user_attr("speedup", speedup)
            print(f"Trial {trial.number}: objective={value}, speedup={speedup}, transforms={specs}")
            return value
        except optuna.TrialPruned:
            raise
        except Exception as exc:
            trial.set_user_attr("failure", str(exc))
            raise optuna.TrialPruned(str(exc))

    sampler = optuna.samplers.TPESampler(seed=poly.search.seed)
    direction = "minimize"
    if cfg.objectives:
        direction = "minimize" if cfg.objectives[0].goal == "min" else "maximize"
    study = optuna.create_study(direction=direction, sampler=sampler)
    study.set_user_attr("baseline_runtime", baseline_value)
    study.optimize(
        objective,
        n_trials=poly.search.n_trials,
        timeout=poly.search.timeout,
    )

    completed_trials = [
        trial
        for trial in study.trials
        if trial.state == optuna.trial.TrialState.COMPLETE
    ]
    trial_csv = write_trial_csv(study, poly)
    print(f"Wrote trial CSV: {trial_csv}")

    if not completed_trials:
        print("No successful trials.")
        return 1

    best_runtime = study.best_value
    best_specs = study.best_trial.user_attrs.get("transforms", [])
    best_speedup = baseline_value / best_runtime if best_runtime > 0 else 0.0

    print("\n=== polyMorph search result ===")
    print(f"Baseline objective: {baseline_value}")
    print(f"Best objective: {best_runtime}")
    print(f"Best speedup: {best_speedup}")
    print("Best transforms:")
    print(json.dumps(best_specs, indent=2))

    if poly.search.result_json:
        result = {
            "baseline_runtime": baseline_value,
            "best_runtime": best_runtime,
            "best_speedup": best_speedup,
            "best_transforms": best_specs,
            "best_trial_number": study.best_trial.number,
        }
        Path(poly.search.result_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote result JSON: {poly.search.result_json}")

    print(
        "Found a transformation combination better than baseline."
        if best_runtime < baseline_value
        else "No transformation combination beat the baseline."
    )
    return 0


def run_poly_morph(cfg: Config, trials_override: int | None = None) -> int:
    poly = cfg.poly_morph
    if poly is None:
        raise ValueError("Config does not define a 'polyMorph' block.")

    if trials_override is not None:
        poly.search.n_trials = int(trials_override)

    if poly.discover:
        candidates = discover_sycl_sources(poly.project_root)
        if not candidates:
            print("No SYCL-looking source files found.")
            return 0
        print("Discovered SYCL-looking source files:")
        for candidate in candidates:
            print(candidate)
        return 0

    source = infer_source(poly)
    if poly.optuna_search:
        return explore_optuna(cfg, poly, source)
    return run_project_mode(cfg, poly, source)
