from __future__ import annotations

import csv
import json
import os
import re
import shutil
import shlex
import subprocess
import stat
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

import optuna

from src.config import Config, PolyMorphSpec
from src.metrics import measure_likwid, measure_parser_sycl, measure_perf
from src.polyMorph.features import candidate_key, enrich_candidate, file_hash, sequence_signature
from src.polyMorph.feedback import (
    analyze_runtime_feedback,
    merge_feedback,
    parse_adaptivecpp_runtime_feedback,
    parse_compiler_feedback,
)
from src.polyMorph.history import append_history, load_history, retrieve_sequences
from src.polyMorph.pruning import (
    analytical_score,
    static_prune_candidate,
    static_prune_sequence,
    violates_constraints,
)

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


@contextmanager
def suppress_external_viewers() -> Iterator[None]:
    """Prevent native analysis libraries from opening graph viewer windows."""
    shim_dir = Path("/tmp/scout-noop-viewers")
    shim_dir.mkdir(parents=True, exist_ok=True)
    dotty = shim_dir / "dotty"
    if not dotty.exists():
        dotty.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        dotty.chmod(dotty.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    old_path = os.environ.get("PATH", "")
    os.environ["PATH"] = f"{shim_dir}{os.pathsep}{old_path}" if old_path else str(shim_dir)
    try:
        yield
    finally:
        os.environ["PATH"] = old_path


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
        original_source: str | Path | None = None,
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
        self.original_source = (
            Path(original_source).resolve() if original_source is not None else Path(source).resolve()
        )

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
            "original_source": self.original_source,
        }

    @property
    def output_binary(self) -> Path:
        return self.build_dir / self.exec_name

    def compile_cmd(self, suffix: str) -> List[str]:
        del suffix
        src = str(self.source)
        exe = str(self.output_binary)
        try:
            replace_src = str(self.original_source.relative_to(self.project_root))
        except ValueError:
            replace_src = str(self.original_source)

        if self.build_system == "make":
            extra_cflags = " ".join(self.compiler_options)
            cmd = [
                "make",
                "-C",
                str(self.project_root),
                f"SOURCE={src}",
                f"REPLACE_SOURCE={replace_src}",
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
            cmd += ["--", f"SOURCE={src}", f"REPLACE_SOURCE={replace_src}", f"EXEC={exe}"]
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


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def cleanup_trial_artifacts(
    app: SyclProjectApp,
    *,
    generated_infix: str | None = None,
) -> List[Path]:
    """Remove per-trial generated sources, objects, and binaries."""
    removed: List[Path] = []
    candidates: set[Path] = set()

    output = app.output_binary
    candidates.add(output)

    source = Path(app.source)
    original_source = Path(getattr(app, "original_source", source))
    if source.resolve() != original_source.resolve() and _is_relative_to(source, app.project_root):
        candidates.add(source)

    if generated_infix:
        for artifact in app.project_root.glob(f"*{generated_infix}*"):
            if artifact.is_file() or artifact.is_symlink():
                candidates.add(artifact)

    output_stem = output.name
    search_dirs = {app.project_root, app.build_dir, output.parent}
    for directory in search_dirs:
        if not directory.exists() or not _is_relative_to(directory, app.project_root):
            continue
        candidates.update(
            artifact
            for artifact in directory.glob(f"{output_stem}*.o")
            if artifact.is_file() or artifact.is_symlink()
        )

    for artifact in sorted(candidates):
        try:
            if artifact.is_file() or artifact.is_symlink():
                artifact.unlink()
                removed.append(artifact)
        except FileNotFoundError:
            continue

    if removed:
        print(f"[cleanup] removed {len(removed)} trial artifact(s)")
    return removed


def build_app_or_raise(app: SyclProjectApp) -> None:
    print(f"[build] source={app.source}")
    print(f"[build] output={app.output_binary}")
    remove_stale_build_output(app)
    proc = _run(app.compile_cmd(""))
    setattr(app, "last_build_stdout", proc.stdout)
    setattr(app, "last_build_stderr", proc.stderr)
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
    metrics = measure_metrics_or_raise(app, repeat, cfg)
    metric_name = _primary_metric_name(cfg)
    if metric_name not in metrics:
        raise RuntimeError(f"Primary metric '{metric_name}' missing; got metrics: {sorted(metrics)}")
    return float(metrics[metric_name])


def _primary_metric_name(cfg: Config | None) -> str:
    if cfg is not None and cfg.objectives:
        return cfg.objectives[0].metric
    if cfg is not None and cfg.backend == "parser":
        return _parser_metric_name(cfg)
    return "runtime"


def measure_metrics_or_raise(app: SyclProjectApp, repeat: int, cfg: Config | None = None) -> Dict[str, float]:
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
        return {str(k): float(v) for k, v in metrics.items()}

    if cfg is not None and cfg.backend == "perf":
        if cfg.perf is None:
            raise RuntimeError("Perf backend selected, but perf config is missing.")
        metrics = measure_perf(cfg.perf, app.output_binary, app.runtime_args, {}, repeat)
        return {str(k): float(v) for k, v in metrics.items()}

    if cfg is not None and cfg.backend == "likwid":
        if cfg.likwid is None:
            raise RuntimeError("LIKWID backend selected, but likwid config is missing.")
        metrics = measure_likwid(cfg.likwid, app.output_binary, app.runtime_args, {}, repeat)
        return {str(k): float(v) for k, v in metrics.items()}

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
    return {"runtime": min(values)}


def collect_runtime_feedback(app: SyclProjectApp, poly: PolyMorphSpec) -> Dict[str, JsonDict]:
    if not poly.search.runtime_feedback:
        return {}

    masks = poly.search.runtime_feedback_masks or [""]
    repeat = max(1, poly.search.runtime_feedback_repeat)
    results: Dict[str, JsonDict] = {}
    for raw_mask in masks:
        mask = str(raw_mask)
        stdout_parts: List[str] = []
        stderr_parts: List[str] = []
        last_returncode = 0
        for _ in range(repeat):
            env = {
                **os.environ,
                **poly.search.runtime_feedback_env,
                "ACPP_DEBUG_LEVEL": str(poly.search.runtime_feedback_debug_level),
            }
            if mask:
                env["ACPP_VISIBILITY_MASK"] = mask
            proc = _run(app.run_cmd(), cwd=app.output_binary.parent, env=env)
            stdout_parts.append(proc.stdout or "")
            stderr_parts.append(proc.stderr or "")
            last_returncode = proc.returncode

        parsed = parse_adaptivecpp_runtime_feedback(
            "\n".join(stdout_parts),
            "\n".join(stderr_parts),
            mask=mask or "<default>",
        )
        parsed["returncode"] = last_returncode
        results[mask or "default"] = parsed
    return results


def inherit_build_settings(dst: SyclProjectApp, src: SyclProjectApp) -> SyclProjectApp:
    dst.compiler_executable = src.compiler_executable
    dst.compiler_options = list(src.compiler_options)
    dst.project_root = src.project_root
    dst.build_system = src.build_system
    dst.build_target = src.build_target
    dst.build_dir = src.build_dir
    dst.runtime_args = list(src.runtime_args)
    dst.original_source = src.original_source
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
        "objectives",
        "speedup",
        "transforms",
        "failure",
        "metrics",
        "compiler_feedback",
        "runtime_feedback_analysis",
        "params",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        for t in study.trials:
            objective = t.values[0] if t.values is not None and len(t.values) == 1 else ""
            objectives = json.dumps(t.values) if t.values is not None else ""
            speedup = t.user_attrs.get("speedup", "")
            transforms = json.dumps(t.user_attrs.get("transforms", []))
            failure = t.user_attrs.get("failure", "")
            metrics = json.dumps(t.user_attrs.get("metrics", {}), sort_keys=True)
            feedback = json.dumps(t.user_attrs.get("compiler_feedback", {}), sort_keys=True)
            runtime_feedback_analysis = json.dumps(
                t.user_attrs.get("runtime_feedback_analysis", {}),
                sort_keys=True,
            )
            params = json.dumps(t.params, sort_keys=True)
            writer.writerow(
                [
                    t.number,
                    str(t.state.name),
                    objective,
                    objectives,
                    speedup,
                    transforms,
                    failure,
                    metrics,
                    feedback,
                    runtime_feedback_analysis,
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


def enumerate_transform_candidates(
    app: SyclProjectApp,
    poly: PolyMorphSpec,
    runtime_bias: Dict[str, JsonDict] | None = None,
) -> List[JsonDict]:
    candidates: List[JsonDict] = []
    candidate_pipeline = _candidate_pipeline_enabled(poly)
    runtime_bias = runtime_bias or {}
    for scop_idx, scop in enumerate(app.scops):
        for node_idx, node in enumerate(scop.schedule_tree):
            available = getattr(node, "available_transformations", [])
            for tr in available:
                name = getattr(tr, "name", str(tr))
                if not transform_allowed(name, poly):
                    continue
                for args in candidate_args_for_transform(name, poly):
                    candidate = {
                        "scop": scop_idx,
                        "node": node_idx,
                        "tr": name,
                        "args": list(args),
                    }
                    if candidate_pipeline:
                        candidate = enrich_candidate(candidate, node)
                        if poly.search.analytical_model:
                            prediction = analytical_score(candidate, poly.search.constraints)
                            candidate["predictions"].update(prediction)
                        key = candidate_key(candidate)
                        bias = runtime_bias.get(key)
                        if bias:
                            _apply_runtime_bias(candidate, bias, poly.search.constraints)
                        if poly.search.static_pruning:
                            reasons = static_prune_candidate(candidate, poly.search.constraints)
                            candidate["prune_reasons"].extend(reasons)
                        if poly.search.constraint_aware:
                            reasons = violates_constraints(candidate, poly.search.constraints)
                            candidate["prune_reasons"].extend(reasons)
                            reasons = _runtime_bias_prune_reasons(candidate, poly.search.constraints)
                            candidate["prune_reasons"].extend(reasons)
                        if candidate["prune_reasons"]:
                            continue
                    candidates.append(candidate)
    if candidate_pipeline and poly.search.analytical_model and poly.search.top_k:
        candidates = select_diverse_top_k(candidates, poly.search.top_k)
    return candidates


def print_candidates(candidates: List[JsonDict], *, verbose: bool = False) -> None:
    print(f"Enumerated {len(candidates)} candidate transform(s)")
    if not verbose:
        print("Candidate details suppressed during search. Use enumerate_only/list mode to print them.")
        return
    for idx, candidate in enumerate(candidates):
        pred = candidate.get("predictions") or {}
        suffix = ""
        if pred:
            suffix = f" score={pred.get('score', '')} risk={pred.get('risk', '')}"
        print(
            f"{idx}: scop={candidate['scop']} node={candidate['node']} "
            f"tr={candidate['tr']} args={candidate.get('args', [])}{suffix}"
        )


def candidate_rank_key(candidate: JsonDict) -> tuple[float, float]:
    pred = candidate.get("predictions") or {}
    return (
        float(pred.get("score", 0.0)),
        -float(pred.get("risk", 1.0)),
    )


def select_diverse_top_k(candidates: List[JsonDict], top_k: int) -> List[JsonDict]:
    if top_k <= 0 or len(candidates) <= top_k:
        return candidates

    ranked = sorted(candidates, key=candidate_rank_key, reverse=True)
    groups: Dict[str, List[JsonDict]] = {}
    for candidate in ranked:
        name = str(candidate.get("tr", ""))
        groups.setdefault(name, []).append(candidate)

    group_names = sorted(
        groups,
        key=lambda name: candidate_rank_key(groups[name][0]),
        reverse=True,
    )
    selected: List[JsonDict] = []
    offset = 0
    while len(selected) < top_k:
        added = False
        for name in group_names:
            group = groups[name]
            if offset < len(group):
                selected.append(group[offset])
                added = True
                if len(selected) >= top_k:
                    break
        if not added:
            break
        offset += 1
    return selected


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
    seeded_sequences: List[List[JsonDict]] | None = None,
    runtime_bias: Dict[str, JsonDict] | None = None,
) -> List[JsonDict]:
    initial_candidates = enumerate_transform_candidates(app, poly, runtime_bias)
    if not initial_candidates:
        return []

    if seeded_sequences and trial.number < len(seeded_sequences):
        specs = seeded_sequences[trial.number]
        reasons = static_prune_sequence(specs, poly.search.constraints)
        if poly.search.constraint_aware and reasons:
            raise optuna.TrialPruned("; ".join(reasons))
        chosen: List[JsonDict] = []
        for spec in specs:
            apply_transforms_to_scops(
                app.scops,
                [spec],
                legality_cb=lambda: app.legal,
            )
            if not poly.allow_illegal and not app.legal:
                raise optuna.TrialPruned("Illegal transformed schedule")
            chosen.append(spec)
        return chosen

    max_transforms = max(1, min(poly.search.max_transforms_per_trial, len(initial_candidates)))
    n_transforms = trial.suggest_int("n_transforms", 1, max_transforms)

    chosen: List[JsonDict] = []
    for pos in range(n_transforms):
        current_candidates = enumerate_transform_candidates(app, poly, runtime_bias)
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

        if poly.search.constraint_aware:
            reasons = static_prune_sequence(chosen, poly.search.constraints)
            if reasons:
                raise optuna.TrialPruned("; ".join(reasons))

    return chosen


def make_project_app(
    *,
    poly: PolyMorphSpec,
    source: Path,
    exec_name: str,
    populate_scops: bool,
) -> SyclProjectApp:
    ensure_tadashi_available()
    compiler_options = list(poly.flags)
    if poly.search.compiler_feedback:
        compiler_options.extend(poly.search.compiler_feedback_flags)
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
        compiler_options=compiler_options,
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


def _objective_values_from_metrics(cfg: Config, metrics: Dict[str, float]) -> List[float]:
    if cfg.objectives:
        values = []
        for obj in cfg.objectives:
            if obj.metric not in metrics:
                raise RuntimeError(f"Metric '{obj.metric}' missing; got metrics: {sorted(metrics.keys())}")
            values.append(float(metrics[obj.metric]))
        return values
    metric_name = _primary_metric_name(cfg)
    if metric_name not in metrics:
        raise RuntimeError(f"Metric '{metric_name}' missing; got metrics: {sorted(metrics.keys())}")
    return [float(metrics[metric_name])]


def _speedup_for_primary(cfg: Config, baseline_value: float, value: float) -> float:
    if value <= 0.0 or baseline_value <= 0.0:
        return 0.0
    goal = cfg.objectives[0].goal if cfg.objectives else "min"
    if goal == "max":
        return value / baseline_value
    return baseline_value / value


def _pick_representative_trial(trials: Sequence[Any], cfg: Config) -> Any:
    if not trials:
        raise RuntimeError("No Pareto trials available.")
    first_goal = cfg.objectives[0].goal if cfg.objectives else "min"
    if first_goal == "max":
        return max(trials, key=lambda trial: trial.values[0] if trial.values else float("-inf"))
    return min(trials, key=lambda trial: trial.values[0] if trial.values else float("inf"))


def _write_pareto_csv(study: optuna.study.Study, poly: PolyMorphSpec, cfg: Config) -> None:
    if not poly.search.pareto_csv:
        return
    out_csv = Path(poly.search.pareto_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    metric_names = [obj.metric for obj in cfg.objectives]
    extra_metrics: set[str] = set()
    for trial in study.best_trials:
        extra_metrics.update((trial.user_attrs.get("metrics") or {}).keys())
    extra_metrics.difference_update(metric_names)
    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["trial", *metric_names, "speedup", "transforms", *sorted(extra_metrics)])
        for trial in study.best_trials:
            if trial.values is None:
                continue
            metrics = trial.user_attrs.get("metrics", {}) or {}
            writer.writerow(
                [
                    trial.number,
                    *trial.values,
                    trial.user_attrs.get("speedup", ""),
                    json.dumps(trial.user_attrs.get("transforms", [])),
                    *[metrics.get(name, "") for name in sorted(extra_metrics)],
                ]
            )
    print(f"Wrote Pareto CSV: {out_csv}")


def _candidate_pipeline_enabled(poly: PolyMorphSpec) -> bool:
    return any(
        [
            poly.search.static_pruning,
            poly.search.analytical_model,
            poly.search.constraint_aware,
            poly.search.top_k is not None,
        ]
    )


def _apply_runtime_bias(candidate: JsonDict, bias: JsonDict, constraints: JsonDict) -> None:
    predictions = candidate.setdefault("predictions", {})
    base_score = float(predictions.get("score", 0.5) or 0.5)
    base_risk = float(predictions.get("risk", 0.1) or 0.1)
    penalty = float(bias.get("penalty", 0.0) or 0.0)
    bonus = float(bias.get("bonus", 0.0) or 0.0)
    weight = float(constraints.get("runtime_feedback_bias_weight", 0.35))

    predictions["runtime_penalty"] = penalty
    predictions["runtime_bonus"] = bonus
    predictions["runtime_observations"] = int(bias.get("observations", 0) or 0)
    predictions["runtime_backend_sensitive"] = bool(bias.get("backend_sensitive"))
    predictions["score"] = max(0.0, min(1.0, base_score - weight * penalty + 0.5 * weight * bonus))
    predictions["risk"] = max(0.0, min(1.0, base_risk + weight * penalty))
    reasons = predictions.setdefault("reasons", [])
    if isinstance(reasons, list):
        reasons.extend(str(reason) for reason in bias.get("reasons", [])[:5])


def _runtime_bias_prune_reasons(candidate: JsonDict, constraints: JsonDict) -> List[str]:
    predictions = candidate.get("predictions", {}) or {}
    observations = int(predictions.get("runtime_observations", 0) or 0)
    if observations < int(constraints.get("runtime_feedback_min_observations_for_prune", 1)):
        return []

    reasons: List[str] = []
    max_penalty = constraints.get("max_runtime_feedback_penalty")
    penalty = float(predictions.get("runtime_penalty", 0.0) or 0.0)
    if max_penalty is not None and penalty > float(max_penalty):
        reasons.append(f"runtime feedback penalty {penalty:.3f} exceeds max_runtime_feedback_penalty={max_penalty}")

    if bool(predictions.get("runtime_backend_sensitive")) and constraints.get("prune_backend_sensitive"):
        reasons.append("runtime feedback marks transform as backend-sensitive")

    return reasons


def _update_runtime_bias(
    runtime_bias: Dict[str, JsonDict],
    specs: List[JsonDict],
    analysis: JsonDict,
) -> None:
    if not specs or not analysis:
        return
    penalty = float(analysis.get("penalty", 0.0) or 0.0)
    bonus = float(analysis.get("bonus", 0.0) or 0.0)
    backend_sensitive = bool(analysis.get("backend_sensitive"))
    reasons = list(analysis.get("reasons", []) or [])
    for spec in specs:
        key = candidate_key(spec)
        current = runtime_bias.setdefault(
            key,
            {
                "observations": 0,
                "penalty": 0.0,
                "bonus": 0.0,
                "backend_sensitive": False,
                "reasons": [],
            },
        )
        n = int(current.get("observations", 0) or 0)
        current["observations"] = n + 1
        current["penalty"] = ((float(current.get("penalty", 0.0) or 0.0) * n) + penalty) / (n + 1)
        current["bonus"] = ((float(current.get("bonus", 0.0) or 0.0) * n) + bonus) / (n + 1)
        current["backend_sensitive"] = bool(current.get("backend_sensitive")) or backend_sensitive
        merged_reasons = list(current.get("reasons", []) or [])
        merged_reasons.extend(reasons[:5])
        current["reasons"] = merged_reasons[-10:]


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
    baseline_metrics = measure_metrics_or_raise(baseline_app, poly.search.repeat, cfg)
    baseline_value = _objective_values_from_metrics(cfg, baseline_metrics)[0]
    print(f"Baseline objective: {baseline_value}")

    enum_app = make_project_app(
        poly=poly,
        source=source,
        exec_name=poly.exec_name,
        populate_scops=True,
    )
    candidates = enumerate_transform_candidates(enum_app, poly)
    print_candidates(candidates, verbose=poly.search.enumerate_only)

    if not candidates:
        print("No candidate transformations found.")
        return 1

    if poly.search.enumerate_only:
        return 0

    baseline_runtime_feedback: Dict[str, JsonDict] = {}
    baseline_runtime_analysis: JsonDict = {}
    if poly.search.runtime_feedback:
        baseline_runtime_feedback = collect_runtime_feedback(baseline_app, poly)
        baseline_runtime_analysis = analyze_runtime_feedback(
            baseline_runtime_feedback,
            constraints=poly.search.constraints,
        )
        print(
            "Baseline runtime feedback: "
            f"score={baseline_runtime_analysis.get('score', '')}, "
            f"penalty={baseline_runtime_analysis.get('penalty', '')}, "
            f"backend_sensitive={baseline_runtime_analysis.get('backend_sensitive', False)}"
        )

    source_hash = file_hash(source)
    runtime_bias: Dict[str, JsonDict] = {}
    seeded_sequences: List[List[JsonDict]] = []
    if poly.search.case_retrieval:
        records = load_history(poly.search.history_jsonl)
        seeded_sequences = retrieve_sequences(
            history=records,
            source_hash=source_hash,
            candidates=candidates,
            limit=poly.search.retrieval_top_k,
        )
        if seeded_sequences:
            print(f"Retrieved {len(seeded_sequences)} previous transform sequence(s).")

    is_multi = len(cfg.objectives) > 1

    def objective(trial: optuna.Trial) -> float | List[float]:
        trial_exec = f"{poly.exec_name}-trial-{trial.number}"
        generated_infix = f"{poly.search.generated_infix}-{trial.number}"
        transformed_app: SyclProjectApp | None = None
        trial_app = make_project_app(
            poly=poly,
            source=source,
            exec_name=trial_exec,
            populate_scops=True,
        )

        try:
            specs = sample_transform_sequence(trial, trial_app, poly, seeded_sequences, runtime_bias)
            if not specs:
                raise optuna.TrialPruned("No candidate transformations available for this trial.")
            trial.set_user_attr("transforms", specs)
            trial.set_user_attr("transform_signature", sequence_signature(specs))

            transformed_app = trial_app.generate_code(
                alt_infix=generated_infix,
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
            compile_feedback = {}
            if poly.search.compiler_feedback:
                compile_feedback = parse_compiler_feedback(
                    getattr(transformed_app, "last_build_stdout", ""),
                    getattr(transformed_app, "last_build_stderr", ""),
                )

            metrics = measure_metrics_or_raise(transformed_app, poly.search.repeat, cfg)
            obj_values = _objective_values_from_metrics(cfg, metrics)
            value = obj_values[0]

            runtime_feedback = collect_runtime_feedback(transformed_app, poly)
            runtime_feedback_analysis = analyze_runtime_feedback(
                runtime_feedback,
                baseline=baseline_runtime_feedback,
                constraints=poly.search.constraints,
            ) if runtime_feedback else {}
            if runtime_feedback_analysis:
                trial.set_user_attr("runtime_feedback_analysis", runtime_feedback_analysis)
                _update_runtime_bias(runtime_bias, specs, runtime_feedback_analysis)
            feedback = merge_feedback(compile_feedback, runtime_feedback)
            if feedback:
                trial.set_user_attr("compiler_feedback", feedback)
            if runtime_feedback:
                trial.set_user_attr("runtime_feedback", runtime_feedback)

            speedup = _speedup_for_primary(cfg, baseline_value, value)

            trial.set_user_attr("runtime", value)
            trial.set_user_attr("metrics", metrics)
            trial.set_user_attr("speedup", speedup)
            print(f"Trial {trial.number}: objective={value}, speedup={speedup}, transforms={specs}")
            append_history(
                poly.search.history_jsonl,
                {
                    "status": "complete",
                    "trial": trial.number,
                    "source": str(source),
                    "source_hash": source_hash,
                    "transforms": specs,
                    "metrics": metrics,
                    "objectives": obj_values,
                    "speedup": speedup,
                    "compiler_feedback": feedback,
                    "runtime_feedback_analysis": runtime_feedback_analysis,
                },
            )
            return obj_values if is_multi else value
        except optuna.TrialPruned as exc:
            append_history(
                poly.search.history_jsonl,
                {
                    "status": "pruned",
                    "trial": trial.number,
                    "source": str(source),
                    "source_hash": source_hash,
                    "transforms": trial.user_attrs.get("transforms", []),
                    "failure": str(exc),
                },
            )
            raise
        except Exception as exc:
            trial.set_user_attr("failure", str(exc))
            append_history(
                poly.search.history_jsonl,
                {
                    "status": "failed",
                    "trial": trial.number,
                    "source": str(source),
                    "source_hash": source_hash,
                    "transforms": trial.user_attrs.get("transforms", []),
                    "failure": str(exc),
                },
            )
            raise optuna.TrialPruned(str(exc))
        finally:
            if transformed_app is not None:
                cleanup_trial_artifacts(transformed_app, generated_infix=generated_infix)

    sampler = optuna.samplers.TPESampler(seed=poly.search.seed)
    directions = ["minimize" if obj.goal == "min" else "maximize" for obj in cfg.objectives]
    if not directions:
        directions = ["minimize"]
    if is_multi:
        study = optuna.create_study(directions=directions, sampler=sampler)
    else:
        study = optuna.create_study(direction=directions[0], sampler=sampler)
    study.set_user_attr("baseline_runtime", baseline_value)
    study.set_user_attr("baseline_metrics", baseline_metrics)
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

    if is_multi:
        _write_pareto_csv(study, poly, cfg)
        best_trial = _pick_representative_trial(study.best_trials, cfg)
        best_runtime = best_trial.values[0] if best_trial.values else float("inf")
        best_specs = best_trial.user_attrs.get("transforms", [])
        best_speedup = best_trial.user_attrs.get("speedup", 0.0)
    else:
        best_trial = study.best_trial
        best_runtime = study.best_value
        best_specs = best_trial.user_attrs.get("transforms", [])
        best_speedup = _speedup_for_primary(cfg, baseline_value, best_runtime)

    print("\n=== polyMorph search result ===")
    print(f"Baseline objective: {baseline_value}")
    print(f"Best objective: {best_runtime}")
    print(f"Best speedup: {best_speedup}")
    print("Best transforms:")
    print(json.dumps(best_specs, indent=2))

    if poly.search.result_json:
        result = {
            "baseline_runtime": baseline_value,
            "baseline_metrics": baseline_metrics,
            "baseline_runtime_feedback_analysis": baseline_runtime_analysis,
            "best_runtime": best_runtime,
            "best_speedup": best_speedup,
            "best_transforms": best_specs,
            "best_trial_number": best_trial.number,
        }
        if is_multi:
            result["pareto_trials"] = [
                {
                    "trial": trial.number,
                    "objectives": trial.values,
                    "transforms": trial.user_attrs.get("transforms", []),
                    "metrics": trial.user_attrs.get("metrics", {}),
                }
                for trial in study.best_trials
            ]
        Path(poly.search.result_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"Wrote result JSON: {poly.search.result_json}")

    print(
        "Found a transformation combination better than baseline."
        if best_speedup > 1.0
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
    with suppress_external_viewers():
        if poly.optuna_search:
            return explore_optuna(cfg, poly, source)
        return run_project_mode(cfg, poly, source)
