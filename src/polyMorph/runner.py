from __future__ import annotations

import csv
import hashlib
import math
import json
import os
import random
import re
import shutil
import shlex
import subprocess
import stat
import sys
import time
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence

from src.config import BuildProject, Config, PolyMorphSpec
from src.metrics import measure_likwid, measure_parser_sycl, measure_perf
from src.polyMorph.features import (
    candidate_key,
    enrich_candidate,
    file_hash,
    sequence_feature_summary,
    sequence_signature,
    stable_json_hash,
    structural_signature,
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


def _ansi(code: str, text: str) -> str:
    if os.environ.get("NO_COLOR") or not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


def _trial_banner(trial_number: int) -> str:
    return _ansi("1;36", f"\n========== MCTS Trial {trial_number} ==========")


class InvalidTransformArgs(RuntimeError):
    pass


class TrialPruned(RuntimeError):
    pass


class CachedSequenceSelected(TrialPruned):
    pass


@dataclass
class HotKernelFilter:
    enabled: bool = False
    hot_kernels: List[int] = field(default_factory=list)
    allowed_scops: set[int] = field(default_factory=set)
    kernel_by_scop: Dict[int, int] = field(default_factory=dict)
    total_kernel_time: float = 0.0
    selected_kernel_time: float = 0.0

    @property
    def coverage(self) -> float:
        if self.total_kernel_time <= 0.0:
            return 0.0
        return self.selected_kernel_time / self.total_kernel_time

    def allows(self, candidate: JsonDict) -> bool:
        if not self.enabled:
            return True
        scop = int(candidate.get("scop", -1))
        return scop in self.allowed_scops


@dataclass
class NativeTrial:
    number: int
    state: str = "RUNNING"
    values: List[float] | None = None
    user_attrs: JsonDict = field(default_factory=dict)
    params: JsonDict = field(default_factory=dict)

    def set_user_attr(self, key: str, value: Any) -> None:
        self.user_attrs[key] = value


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
        build_compiler_options: Sequence[str] | None = None,
        run_env: Dict[str, str] | None = None,
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
        self.run_env = dict(run_env or {})
        self.compiler_options = list(compiler_options or [])
        self.build_compiler_options = (
            list(build_compiler_options)
            if build_compiler_options is not None
            else list(self.compiler_options)
        )
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
            "run_env": self.run_env,
            "build_compiler_options": self.build_compiler_options,
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
            extra_cflags = " ".join(self.build_compiler_options)
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


def _short_jscop_backup_path(path: Path) -> Path:
    digest = hashlib.sha256(path.name.encode("utf-8")).hexdigest()[:16]
    return path.with_name(f"jscop-backup-{digest}.bak")


class SafePolly(Polly):
    def generate_code(self, input_path: str, output_path: Path, options: list[str]) -> Path:
        original_copy2 = shutil.copy2

        def copy2_with_short_jscop_backup(src: Any, dst: Any, *args: Any, **kwargs: Any) -> Any:
            dst_path = Path(dst)
            if dst_path.name.endswith(".jscop.bak") and len(dst_path.name.encode("utf-8")) > 240:
                dst_path = _short_jscop_backup_path(dst_path)
            return original_copy2(src, dst_path, *args, **kwargs)

        shutil.copy2 = copy2_with_short_jscop_backup
        try:
            return super().generate_code(input_path, output_path, options)
        finally:
            shutil.copy2 = original_copy2


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
        try:
            ok = node.transform(tr, *tr_args)
        except Exception as exc:
            message = str(exc)
            if "Not valid args" in message:
                print(f"  invalid: args={tuple(tr_args)} are not valid for {tr_name}")
                raise InvalidTransformArgs(
                    f"invalid args={tuple(tr_args)} for tr={tr_name} on scop={scop_idx}, node={node_idx}"
                ) from exc
            first_line = message.splitlines()[0] if message else type(exc).__name__
            print(f"  invalid: {first_line}")
            raise InvalidTransformArgs(
                f"{tr_name} is not valid on scop={scop_idx}, node={node_idx}: {first_line}"
            ) from exc
        print(f"  result={ok}")
        if not ok:
            raise InvalidTransformArgs(
                f"{tr_name} returned False on scop={scop_idx}, node={node_idx}"
            )

        try:
            available = getattr(node, "available_transformations", [])
            names = [getattr(item, "name", str(item)) for item in available]
        except Exception as exc:
            names = [f"<unavailable: {exc}>"]
        try:
            node_type = getattr(node, "node_type", "<unknown>")
        except Exception as exc:
            node_type = f"<invalid: {exc}>"
        print(f"  current node type: {node_type}")
        print(f"  currently available: {names}")

        if legality_cb is not None:
            try:
                legal = legality_cb()
                print(f"  legal()={legal}")
            except Exception as exc:
                print(f"  legality check failed: {exc}")


def app_is_legal(app: SyclProjectApp) -> bool:
    try:
        return bool(app.legal)
    except Exception:
        return False


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


def _metric_warmup_config(cfg: Config | None) -> Any | None:
    if cfg is None:
        return None
    if cfg.backend == "parser":
        return cfg.parser
    if cfg.backend == "perf":
        return cfg.perf
    if cfg.backend == "likwid":
        return cfg.likwid
    return None


@contextmanager
def temporary_metric_warmup_runs(cfg: Config | None, extra_warmups: int) -> Iterator[None]:
    target = _metric_warmup_config(cfg)
    if target is None or extra_warmups <= 0:
        yield
        return

    old_value = int(getattr(target, "warmup_runs", 0) or 0)
    setattr(target, "warmup_runs", old_value + int(extra_warmups))
    try:
        yield
    finally:
        setattr(target, "warmup_runs", old_value)


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
            app.run_env,
            repeat,
            workdir=app.project_root,
            project=BuildProject(dir=app.project_root),
        )
        return {str(k): float(v) for k, v in metrics.items()}

    if cfg is not None and cfg.backend == "perf":
        if cfg.perf is None:
            raise RuntimeError("Perf backend selected, but perf config is missing.")
        metrics = measure_perf(cfg.perf, app.output_binary, app.runtime_args, app.run_env, repeat)
        return {str(k): float(v) for k, v in metrics.items()}

    if cfg is not None and cfg.backend == "likwid":
        if cfg.likwid is None:
            raise RuntimeError("LIKWID backend selected, but likwid config is missing.")
        metrics = measure_likwid(cfg.likwid, app.output_binary, app.runtime_args, app.run_env, repeat)
        return {str(k): float(v) for k, v in metrics.items()}

    values: List[float] = []
    cmd = app.run_cmd()
    for _ in range(repeat):
        proc = _run(cmd, env={**os.environ, **app.run_env})
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
    return {"runtime": sum(values) / len(values)}


def _app_run_cwd(app: SyclProjectApp, cfg: Config | None) -> Path:
    if cfg is not None and cfg.backend == "parser" and cfg.parser is not None:
        run_cwd = cfg.parser.run_cwd
        if run_cwd == "workdir":
            return app.project_root
        if run_cwd == "project_dir":
            return app.project_root
    return app.output_binary.parent


def _numbers_from_text(text: str) -> List[float]:
    return [
        float(match.group(0))
        for match in re.finditer(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][-+]?\d+)?", text)
    ]


def _outputs_match(expected: bytes, actual: bytes, tolerance: float) -> bool:
    if expected == actual:
        return True
    try:
        expected_text = expected.decode("utf-8")
        actual_text = actual.decode("utf-8")
    except UnicodeDecodeError:
        return False
    expected_nums = _numbers_from_text(expected_text)
    actual_nums = _numbers_from_text(actual_text)
    if expected_nums and len(expected_nums) == len(actual_nums):
        return all(
            math.isclose(left, right, rel_tol=tolerance, abs_tol=tolerance)
            for left, right in zip(expected_nums, actual_nums)
        )
    return False


def capture_correctness_outputs(app: SyclProjectApp, cfg: Config | None, poly: PolyMorphSpec) -> Dict[str, bytes]:
    outputs: Dict[str, bytes] = {}
    if not poly.search.correctness_outputs:
        return outputs
    cwd = _app_run_cwd(app, cfg)
    for raw_path in poly.search.correctness_outputs:
        path = Path(raw_path)
        if not path.is_absolute():
            path = cwd / path
        if not path.exists():
            if poly.search.correctness_required:
                raise RuntimeError(f"Correctness output not found: {path}")
            continue
        outputs[str(raw_path)] = path.read_bytes()
    return outputs


def verify_correctness_outputs(
    baseline_outputs: Dict[str, bytes],
    app: SyclProjectApp,
    cfg: Config | None,
    poly: PolyMorphSpec,
) -> JsonDict:
    if not baseline_outputs:
        return {"checked": False}
    actual_outputs = capture_correctness_outputs(app, cfg, poly)
    mismatches: List[str] = []
    missing: List[str] = []
    for name, expected in baseline_outputs.items():
        actual = actual_outputs.get(name)
        if actual is None:
            missing.append(name)
            continue
        if not _outputs_match(expected, actual, poly.search.correctness_tolerance):
            mismatches.append(name)
    ok = not mismatches and not missing
    return {
        "checked": True,
        "ok": ok,
        "mismatches": mismatches,
        "missing": missing,
        "outputs": sorted(baseline_outputs),
    }


def collect_backend_sensitivity(
    app: SyclProjectApp,
    poly: PolyMorphSpec,
    cfg: Config,
) -> JsonDict:
    masks = [str(mask) for mask in poly.search.backend_sensitivity_masks if str(mask).strip()]
    if len(masks) < 2:
        return {"checked": False, "backend_sensitive": False, "per_backend": {}}

    original_env = dict(app.run_env)
    per_backend: Dict[str, JsonDict] = {}
    repeat = max(1, int(poly.search.backend_sensitivity_repeat))
    try:
        for mask in masks:
            app.run_env = {**original_env, "ACPP_VISIBILITY_MASK": mask}
            try:
                metrics = measure_metrics_or_raise(app, repeat, cfg)
                values = _objective_values_from_metrics(cfg, metrics)
                per_backend[mask] = {
                    "ok": True,
                    "metrics": metrics,
                    "objective": values[0],
                }
            except Exception as exc:
                per_backend[mask] = {
                    "ok": False,
                    "error": str(exc),
                }
    finally:
        app.run_env = original_env

    objectives = [
        float(item["objective"])
        for item in per_backend.values()
        if item.get("ok") and item.get("objective") is not None
    ]
    rel_threshold = float(poly.search.constraints.get("backend_sensitivity_rel_threshold", 0.20))
    backend_sensitive = False
    reasons: List[str] = []
    if len(objectives) >= 2:
        min_obj = min(objectives)
        max_obj = max(objectives)
        rel = (max_obj - min_obj) / max(min_obj, 1.0e-12)
        backend_sensitive = rel >= rel_threshold
        if backend_sensitive:
            reasons.append(
                f"objective varies across backend masks by {rel:.3f}, threshold={rel_threshold:.3f}"
            )
    failed = [mask for mask, item in per_backend.items() if not item.get("ok")]
    if failed:
        backend_sensitive = True
        reasons.append(f"backend mask(s) failed: {failed}")

    return {
        "checked": True,
        "backend_sensitive": backend_sensitive,
        "reasons": reasons,
        "per_backend": per_backend,
    }


def inherit_build_settings(dst: SyclProjectApp, src: SyclProjectApp) -> SyclProjectApp:
    dst.compiler_executable = src.compiler_executable
    dst.compiler_options = list(src.compiler_options)
    dst.build_compiler_options = list(src.build_compiler_options)
    dst.project_root = src.project_root
    dst.build_system = src.build_system
    dst.build_target = src.build_target
    dst.build_dir = src.build_dir
    dst.runtime_args = list(src.runtime_args)
    dst.run_env = dict(src.run_env)
    dst.original_source = src.original_source
    return dst


def _trial_csv_path(poly: PolyMorphSpec) -> Path:
    if poly.search.trial_csv:
        return Path(poly.search.trial_csv)
    if poly.search.result_json:
        result_path = Path(poly.search.result_json)
        return result_path.with_name(f"{result_path.stem}_trials.csv")
    return Path("polymorph_trials.csv")


def _cache_path(poly: PolyMorphSpec) -> Path | None:
    if not poly.search.cache_evaluations:
        return None
    if poly.search.cache_jsonl:
        return Path(poly.search.cache_jsonl)
    if poly.search.history_jsonl:
        return Path(poly.search.history_jsonl).with_suffix(".cache.jsonl")
    if poly.search.result_json:
        return Path(poly.search.result_json).with_suffix(".cache.jsonl")
    return None


JSCOP_SAFE_POLLY_OPTIONS = ["-mllvm", "-polly-use-llvm-names=false"]


def tadashi_compiler_options(
    flags: Sequence[str],
    populate_scops: bool,
) -> List[str]:
    options = list(flags)
    if not populate_scops:
        return options
    if any("polly-use-llvm-names" in option for option in options):
        return options
    return [*options, *JSCOP_SAFE_POLLY_OPTIONS]


def _cache_key(
    *,
    source_hash: str,
    specs: List[JsonDict],
    poly: PolyMorphSpec,
    cfg: Config,
) -> str:
    return stable_json_hash(
        {
            "source_hash": source_hash,
            "transforms": _compact_transform_specs(specs),
            "flags": poly.flags,
            "compiler": poly.compiler,
            "runtime_args": poly.runtime_args,
            "objectives": [(obj.metric, obj.goal) for obj in cfg.objectives],
            "backend": cfg.backend,
            "target_backend": poly.search.target_backend,
            "multi_fidelity": poly.search.multi_fidelity,
            "early_stop_worse_than": poly.search.early_stop_worse_than,
            "repeat": poly.search.repeat,
        }
    )


def _candidate_model_key(candidate: JsonDict) -> tuple[str, tuple[Any, ...]]:
    return (str(candidate.get("tr", "")), tuple(candidate.get("args", [])))


def _record_backend_score(record: JsonDict, target_backend: str | None) -> float:
    if not target_backend:
        return 0.0
    analysis = record.get("backend_sensitivity") or {}
    per_backend = analysis.get("per_backend") if isinstance(analysis, dict) else None
    if not isinstance(per_backend, dict):
        return 0.0
    backend = per_backend.get(target_backend)
    if not isinstance(backend, dict):
        return 0.0
    if not backend.get("ok"):
        return -1.0
    return 1.0


def build_learned_candidate_model(records: Sequence[JsonDict], target_backend: str | None = None) -> Dict[tuple[str, tuple[Any, ...]], JsonDict]:
    model: Dict[tuple[str, tuple[Any, ...]], JsonDict] = {}
    for record in records:
        transforms = record.get("transforms") or []
        if not isinstance(transforms, list) or not transforms:
            continue
        speedup = float(record.get("speedup", 0.0) or 0.0)
        status = str(record.get("status", ""))
        backend_score = _record_backend_score(record, target_backend)
        for transform in transforms:
            key = _candidate_model_key(transform)
            stats = model.setdefault(
                key,
                {
                    "observations": 0,
                    "successes": 0,
                    "failures": 0,
                    "speedup_sum": 0.0,
                    "backend_score_sum": 0.0,
                },
            )
            stats["observations"] = int(stats["observations"]) + 1
            if status in {"complete", "success"}:
                stats["successes"] = int(stats["successes"]) + 1
                stats["speedup_sum"] = float(stats["speedup_sum"]) + speedup
                stats["backend_score_sum"] = float(stats["backend_score_sum"]) + backend_score
            elif status in {"failed", "pruned"}:
                stats["failures"] = int(stats["failures"]) + 1

    for stats in model.values():
        successes = max(1, int(stats.get("successes", 0) or 0))
        observations = max(1, int(stats.get("observations", 0) or 0))
        failures = int(stats.get("failures", 0) or 0)
        stats["avg_speedup"] = float(stats.get("speedup_sum", 0.0) or 0.0) / successes
        stats["avg_backend_score"] = float(stats.get("backend_score_sum", 0.0) or 0.0) / successes
        stats["failure_rate"] = failures / observations
    return model


def apply_learned_model_to_candidates(
    candidates: List[JsonDict],
    model: Dict[tuple[str, tuple[Any, ...]], JsonDict],
    *,
    min_observations: int,
) -> None:
    for candidate in candidates:
        stats = model.get(_candidate_model_key(candidate))
        if not stats or int(stats.get("observations", 0) or 0) < min_observations:
            continue
        predictions = candidate.setdefault("predictions", {})
        base_score = float(predictions.get("score", 0.5) or 0.5)
        base_risk = float(predictions.get("risk", 0.1) or 0.1)
        avg_speedup = float(stats.get("avg_speedup", 0.0) or 0.0)
        backend_score = float(stats.get("avg_backend_score", 0.0) or 0.0)
        failure_rate = float(stats.get("failure_rate", 0.0) or 0.0)
        speedup_bonus = max(-0.25, min(0.25, (avg_speedup - 1.0) * 0.2))
        backend_bonus = max(0.0, min(0.12, backend_score * 0.12))
        predictions["learned_observations"] = int(stats.get("observations", 0) or 0)
        predictions["learned_avg_speedup"] = avg_speedup
        predictions["learned_failure_rate"] = failure_rate
        predictions["learned_backend_score"] = backend_score
        predictions["score"] = max(0.0, min(1.0, base_score + speedup_bonus + backend_bonus - 0.2 * failure_rate))
        predictions["risk"] = max(0.0, min(1.0, base_risk + 0.25 * failure_rate))
        reasons = predictions.setdefault("reasons", [])
        if isinstance(reasons, list):
            reasons.append(
                f"learned model: avg_speedup={avg_speedup:.3f}, failure_rate={failure_rate:.3f}"
            )


def _compact_transform_specs(specs: List[JsonDict]) -> List[JsonDict]:
    return [
        {
            "scop": int(spec.get("scop", -1)),
            "node": int(spec.get("node", -1)),
            "tr": str(spec.get("tr", "")),
            "args": list(spec.get("args", [])),
        }
        for spec in specs
    ]


def _stop_spec() -> JsonDict:
    return {
        "scop": -1,
        "node": -1,
        "tr": "STOP",
        "args": [],
        "source_info": {},
        "scop_classification": {"labels": ["no_transform"]},
    }


def _is_stop_sequence(specs: Sequence[JsonDict]) -> bool:
    return len(specs) == 1 and str(specs[0].get("tr", "")) == "STOP"


def _transform_family(name: str) -> str:
    upper = str(name).upper()
    if upper == "STOP":
        return "STOP"
    if upper.startswith("TILE"):
        return "TILE"
    if upper == "INTERCHANGE":
        return "INTERCHANGE"
    if "FUSE" in upper:
        return "FUSE"
    if "SPLIT" in upper or upper == "SCALE":
        return "SPLIT"
    if "SHIFT" in upper:
        return "SHIFT"
    if upper in {"SET_LOOP_OPT", "SET_PARALLEL"}:
        return "PARALLEL_HINT"
    return upper or "OTHER"


def _candidate_family(candidate: JsonDict) -> str:
    return _transform_family(str(candidate.get("tr", "")))


def _format_family_counts(counts: Counter[str]) -> str:
    if not counts:
        return "none"
    return ", ".join(f"{family}:{count}" for family, count in sorted(counts.items()))


def _hot_kernel_metrics(metrics: Dict[str, float]) -> list[tuple[int, float]]:
    return sorted(
        ((kernel_id, runtime) for kernel_id, runtime in _kernel_metrics(metrics).items() if runtime > 0.0),
        key=lambda item: item[1],
        reverse=True,
    )


def build_hot_kernel_filter(
    candidates: Sequence[JsonDict],
    baseline_metrics: Dict[str, float],
    constraints: JsonDict,
) -> HotKernelFilter:
    if not bool(constraints.get("hot_kernel_filter", True)):
        return HotKernelFilter()
    kernel_times = _hot_kernel_metrics(baseline_metrics)
    scop_ids = sorted({
        int(candidate.get("scop", -1))
        for candidate in candidates
        if int(candidate.get("scop", -1)) >= 0
    })
    if len(kernel_times) <= 1 or not scop_ids:
        return HotKernelFilter()

    top_k = int(constraints.get("hot_kernel_top_k", 3) or 3)
    top_k = max(1, min(top_k, len(kernel_times)))
    min_fraction = float(constraints.get("hot_kernel_min_fraction", 0.0) or 0.0)
    total_time = sum(runtime for _, runtime in kernel_times)
    selected: list[tuple[int, float]] = []
    selected_time = 0.0
    for kernel_id, runtime in kernel_times:
        if len(selected) < top_k or (min_fraction > 0.0 and selected_time / total_time < min_fraction):
            selected.append((kernel_id, runtime))
            selected_time += runtime
            continue
        break
    hot_ids = [kernel_id for kernel_id, _ in selected]

    kernel_ids = sorted(kernel_id for kernel_id, _ in kernel_times)
    max_kernel_id = max(kernel_ids)
    mapping_mode = str(constraints.get("hot_kernel_scop_mapping", "ordinal"))
    kernel_by_scop: Dict[int, int] = {}
    if mapping_mode == "one_based" or (mapping_mode == "ordinal" and max_kernel_id >= max(scop_ids) + 1):
        kernel_by_scop = {scop: scop + 1 for scop in scop_ids}
    elif mapping_mode == "zero_based":
        kernel_by_scop = {scop: scop for scop in scop_ids}
    else:
        for idx, scop in enumerate(scop_ids):
            bucket = min(int(idx * len(kernel_ids) / max(len(scop_ids), 1)), len(kernel_ids) - 1)
            kernel_by_scop[scop] = kernel_ids[bucket]

    hot_set = set(hot_ids)
    allowed_scops = {scop for scop, kernel_id in kernel_by_scop.items() if kernel_id in hot_set}
    min_allowed = int(constraints.get("hot_kernel_min_scops", 1) or 1)
    if len(allowed_scops) < min_allowed:
        return HotKernelFilter()
    return HotKernelFilter(
        enabled=True,
        hot_kernels=hot_ids,
        allowed_scops=allowed_scops,
        kernel_by_scop=kernel_by_scop,
        total_kernel_time=total_time,
        selected_kernel_time=selected_time,
    )


def apply_hot_kernel_filter_to_candidates(
    candidates: Sequence[JsonDict],
    hot_filter: HotKernelFilter,
) -> List[JsonDict]:
    if not hot_filter.enabled:
        return list(candidates)
    filtered: List[JsonDict] = []
    for candidate in candidates:
        if not hot_filter.allows(candidate):
            continue
        scop = int(candidate.get("scop", -1))
        suspected_kernel = hot_filter.kernel_by_scop.get(scop)
        candidate.setdefault("source_info", {})["suspected_kernel"] = suspected_kernel
        candidate.setdefault("scop_classification", {})["hot_kernel_filter"] = {
            "matched": True,
            "suspected_kernel": suspected_kernel,
        }
        filtered.append(candidate)
    return filtered


def load_evaluation_cache(path: Path | None) -> Dict[str, JsonDict]:
    if path is None or not path.exists():
        return {}
    entries: Dict[str, JsonDict] = {}
    with path.open("r", encoding="utf-8") as fp:
        for line in fp:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict) and item.get("key"):
                entries[str(item["key"])] = item
    return entries


def append_evaluation_cache(path: Path | None, record: JsonDict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def write_trial_csv(trials: Sequence[NativeTrial], poly: PolyMorphSpec) -> Path:
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
        "performance_feedback_analysis",
        "backend_sensitivity",
        "params",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(header)
        for t in trials:
            objective = t.values[0] if t.values is not None and len(t.values) == 1 else ""
            objectives = json.dumps(t.values) if t.values is not None else ""
            speedup = t.user_attrs.get("speedup", "")
            transforms = json.dumps(t.user_attrs.get("transforms", []))
            failure = t.user_attrs.get("failure", "")
            metrics = json.dumps(t.user_attrs.get("metrics", {}), sort_keys=True)
            feedback = json.dumps(
                t.user_attrs.get("performance_feedback_analysis", {}),
                sort_keys=True,
            )
            backend_sensitivity = json.dumps(
                t.user_attrs.get("backend_sensitivity", {}),
                sort_keys=True,
            )
            params = json.dumps(t.params, sort_keys=True)
            writer.writerow(
                [
                    t.number,
                    str(t.state),
                    objective,
                    objectives,
                    speedup,
                    transforms,
                    failure,
                    metrics,
                    feedback,
                    backend_sensitivity,
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


def _node_loop_rank(node: Any) -> int:
    yaml_str = str(getattr(node, "yaml_str", "") or "")
    dims = {
        int(match.group(1))
        for match in re.finditer(r"\bi(\d+)\b", yaml_str)
    }
    return max(dims) + 1 if dims else 1


def _as_positive_int(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, (list, tuple)) and value:
        return len(value)
    return None


def _node_band_member_count(node: Any) -> int | None:
    if node is None:
        return None
    attr_names = [
        "band_member_count",
        "band_members",
        "n_band_members",
        "member_count",
        "members",
        "n_members",
        "n_member",
        "dimension",
        "dim",
    ]
    for attr_name in attr_names:
        if not hasattr(node, attr_name):
            continue
        value = getattr(node, attr_name)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        count = _as_positive_int(value)
        if count is not None:
            return count
    return None


def _node_param_count(node: Any) -> int:
    yaml_str = str(getattr(node, "yaml_str", "") or "")
    params = {
        int(match.group(1))
        for match in re.finditer(r"\bp_(\d+)\b", yaml_str)
    }
    return max(params) + 1 if params else 1


def _bounded_indexes(count: int, limit: int = 3) -> List[int]:
    return list(range(max(0, min(count, limit))))


def candidate_args_for_transform(name: str, poly: PolyMorphSpec, node: Any | None = None) -> List[List[Any]]:
    loop_rank = _node_loop_rank(node) if node is not None else 1
    param_count = _node_param_count(node) if node is not None else 1
    loop_indexes = _bounded_indexes(loop_rank)
    param_indexes = _bounded_indexes(param_count)

    if name == "TILE_1D":
        return [[int(x)] for x in poly.search.tile_sizes]
    if name == "TILE_2D":
        if loop_rank < 2:
            return []
        return [[int(x), int(x)] for x in poly.search.tile_sizes]
    if name == "TILE_3D":
        if loop_rank < 3:
            return []
        return [[int(x), int(x), int(x)] for x in poly.search.tile_sizes]
    if name == "INTERCHANGE":
        return [[]]
    if name == "FULL_FUSE":
        return [[]]
    if name == "FUSE":
        return []
    if name == "FULL_SPLIT":
        return [[]]
    if name == "SPLIT":
        return [[int(x)] for x in poly.search.scale_factors]
    if name == "SCALE":
        return [[int(x)] for x in poly.search.scale_factors]
    if name == "FULL_SHIFT_VAL":
        return [[int(x)] for x in poly.search.shift_values]
    if name == "PARTIAL_SHIFT_VAL":
        return [[dim, int(value)] for dim in loop_indexes for value in poly.search.shift_values]
    if name == "FULL_SHIFT_VAR":
        return [[dim, int(value)] for dim in loop_indexes for value in poly.search.shift_values]
    if name == "PARTIAL_SHIFT_VAR":
        return [
            [target_dim, shift_dim, int(value)]
            for target_dim in loop_indexes
            for shift_dim in loop_indexes
            for value in poly.search.shift_values
        ]
    if name == "FULL_SHIFT_PARAM":
        return [[param, int(value)] for param in param_indexes for value in poly.search.shift_values]
    if name == "PARTIAL_SHIFT_PARAM":
        return [
            [target_dim, param, int(value)]
            for target_dim in loop_indexes
            for param in param_indexes
            for value in poly.search.shift_values
        ]
    if name == "SET_PARALLEL":
        return [[int(x)] for x in poly.search.scale_factors]
    if name == "SET_LOOP_OPT":
        member_count = _node_band_member_count(node)
        if member_count is None:
            return []
        loop_indexes = _bounded_indexes(member_count)
        return [[dim, opt] for dim in loop_indexes for opt in [1, 3]]
    return [[]]


def candidate_args_for_node(name: str, poly: PolyMorphSpec, node: Any) -> List[List[Any]]:
    inferred = candidate_args_for_transform(name, poly, node)
    if not poly.search.legality_aware_args:
        return inferred

    child_count = _sequence_child_count(node)
    if name == "FUSE" and _node_is_sequence_like(node) and child_count >= 2:
        return [[idx, idx + 1] for idx in range(child_count - 1)]

    return inferred


def _node_is_sequence_like(node: Any) -> bool:
    try:
        node_type = str(getattr(node, "node_type", ""))
    except Exception:
        node_type = ""
    if "SEQUENCE" in node_type.upper():
        return True
    return False


def _sequence_child_count(node: Any) -> int:
    for attr_name in ["sequence_child_count", "child_count", "n_children", "children"]:
        if not hasattr(node, attr_name):
            continue
        value = getattr(node, attr_name)
        if callable(value):
            try:
                value = value()
            except Exception:
                continue
        count = _as_positive_int(value)
        if count is not None:
            return count
    yaml_str = str(getattr(node, "yaml_str", "") or "")
    return len(re.findall(r"^\s*-\s+filter:", yaml_str, flags=re.MULTILINE))


def candidate_args_invalid_for_node(name: str, args: Sequence[Any], node: Any) -> str | None:
    if name != "FUSE":
        return None
    if not _node_is_sequence_like(node):
        return "FUSE requires a sequence schedule node"
    if len(args) != 2:
        return "FUSE requires two child indexes"
    try:
        left, right = (int(args[0]), int(args[1]))
    except (TypeError, ValueError):
        return "FUSE child indexes must be integers"
    if left < 0 or right < 0:
        return "FUSE child indexes must be non-negative"
    if left == right:
        return "FUSE child indexes must be distinct"
    child_count = _sequence_child_count(node)
    if child_count and max(left, right) >= child_count:
        return f"FUSE args {list(args)} exceed sequence child count {child_count}"
    return None


def _best_effort_attr(obj: Any, names: Sequence[str]) -> str | None:
    for name in names:
        value = getattr(obj, name, None)
        if value is None:
            continue
        text = str(value)
        if text and text != "<unknown>":
            return text
    return None


def _classify_node(node: Any, available_count: int) -> JsonDict:
    yaml_str = str(getattr(node, "yaml_str", "") or "")
    loop_rank = _node_loop_rank(node)
    child_count = _sequence_child_count(node)
    access_count = len(re.findall(r"\[[^\]]+\]", yaml_str))
    statement_count = len(set(re.findall(r"\bStmt\w*", yaml_str)))
    labels: List[str] = []
    if child_count:
        labels.append("sequence")
    else:
        labels.append("band")
    if loop_rank <= 1:
        labels.append("1d_loop")
    elif loop_rank == 2:
        labels.append("2d_loop")
    else:
        labels.append("deep_loop")
    if len(yaml_str) < 700:
        labels.append("small_scop")
    if access_count >= 4:
        labels.append("memory_heavy_hint")
    if available_count <= 3:
        labels.append("limited_transform_space")
    return {
        "labels": labels,
        "loop_rank": loop_rank,
        "sequence_child_count": child_count,
        "statement_count": statement_count,
        "access_hint_count": access_count,
        "yaml_bytes": len(yaml_str.encode("utf-8")),
    }


def _candidate_source_info(source: Path, scop: Any, node: Any) -> JsonDict:
    return {
        "source": str(source),
        "function": _best_effort_attr(
            scop,
            ["function_name", "function", "name", "func_name"],
        ),
        "node_type": str(getattr(node, "node_type", "")),
    }


def enumerate_transform_candidates(
    app: SyclProjectApp,
    poly: PolyMorphSpec,
    performance_bias: Dict[str, JsonDict] | None = None,
    hot_filter: HotKernelFilter | None = None,
    *,
    print_family_summary: bool = False,
    apply_top_k: bool = True,
) -> List[JsonDict]:
    candidates: List[JsonDict] = []
    available_family_counts: Counter[str] = Counter()
    kept_family_counts: Counter[str] = Counter()
    candidate_pipeline = _candidate_pipeline_enabled(poly)
    performance_bias = performance_bias or {}
    for scop_idx, scop in enumerate(app.scops):
        for node_idx, node in enumerate(scop.schedule_tree):
            available = getattr(node, "available_transformations", [])
            for tr in available:
                name = getattr(tr, "name", str(tr))
                if not transform_allowed(name, poly):
                    continue
                for args in candidate_args_for_node(name, poly, node):
                    family = _transform_family(name)
                    available_family_counts[family] += 1
                    invalid_reason = candidate_args_invalid_for_node(name, args, node)
                    if invalid_reason:
                        continue
                    candidate = {
                        "scop": scop_idx,
                        "node": node_idx,
                        "tr": name,
                        "args": list(args),
                        "source_info": _candidate_source_info(app.original_source, scop, node),
                        "scop_classification": _classify_node(node, len(available)),
                    }
                    if candidate_pipeline:
                        candidate = enrich_candidate(candidate, node)
                        if hot_filter and hot_filter.enabled:
                            suspected_kernel = hot_filter.kernel_by_scop.get(scop_idx)
                            candidate.setdefault("source_info", {})["suspected_kernel"] = suspected_kernel
                            candidate.setdefault("scop_classification", {})["hot_kernel_filter"] = {
                                "matched": scop_idx in hot_filter.allowed_scops,
                                "suspected_kernel": suspected_kernel,
                            }
                        if poly.search.analytical_model:
                            prediction = analytical_score(candidate, poly.search.constraints)
                            candidate["predictions"].update(prediction)
                        key = candidate_key(candidate)
                        bias = performance_bias.get(key)
                        if bias:
                            _apply_performance_bias(candidate, bias, poly.search.constraints)
                        if poly.search.static_pruning:
                            reasons = static_prune_candidate(candidate, poly.search.constraints)
                            candidate["prune_reasons"].extend(reasons)
                        if poly.search.constraint_aware:
                            reasons = violates_constraints(candidate, poly.search.constraints)
                            candidate["prune_reasons"].extend(reasons)
                            reasons = _performance_bias_prune_reasons(candidate, poly.search.constraints)
                            candidate["prune_reasons"].extend(reasons)
                        if candidate["prune_reasons"]:
                            continue
                    if hot_filter and not hot_filter.allows(candidate):
                        continue
                    candidates.append(candidate)
                    kept_family_counts[_candidate_family(candidate)] += 1
    if apply_top_k and candidate_pipeline and poly.search.analytical_model and poly.search.top_k:
        candidates = select_diverse_top_k(candidates, poly.search.top_k)
        kept_family_counts = Counter(_candidate_family(candidate) for candidate in candidates)
    if print_family_summary and bool(poly.search.constraints.get("print_candidate_family_counts", True)):
        print(
            "Candidate family counts: "
            f"available={_format_family_counts(available_family_counts)}; "
            f"kept={_format_family_counts(kept_family_counts)}"
        )
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



def _candidate_exact_key(candidate: JsonDict) -> tuple[int, int, str, tuple[Any, ...]]:
    return (
        int(candidate.get("scop", -1)),
        int(candidate.get("node", -1)),
        str(candidate.get("tr", "")),
        tuple(candidate.get("args", [])),
    )


@dataclass
class PrefixStats:
    visits: int = 0
    total_reward: float = 0.0
    best_reward: float = 0.0
    pruned_count: int = 0
    early_stop_count: int = 0
    failed_count: int = 0

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.visits if self.visits else 0.0


@dataclass
class AdaptiveTreeState:
    constraints: JsonDict
    rng: random.Random
    stats: Dict[str, PrefixStats] = field(default_factory=dict)
    pair_stats: Dict[str, PrefixStats] = field(default_factory=dict)
    tried_sequences: set[str] = field(default_factory=set)
    evaluated_terminal_sequences: set[str] = field(default_factory=set)
    saturated_prefixes: set[str] = field(default_factory=set)
    terminal_repeat_counts: Counter[str] = field(default_factory=Counter)
    tried_family_counts: Counter[str] = field(default_factory=Counter)
    failed_family_candidates: Dict[str, set[str]] = field(default_factory=dict)
    successful_family_candidates: Dict[str, set[str]] = field(default_factory=dict)
    disabled_families: set[str] = field(default_factory=set)
    disabled_scops: set[int] = field(default_factory=set)

    def prefix_key(self, specs: List[JsonDict]) -> str:
        return sequence_signature(specs) if specs else "root"

    def get(self, specs: List[JsonDict]) -> PrefixStats:
        return self.stats.setdefault(self.prefix_key(specs), PrefixStats())

    def mark_tried(self, specs: List[JsonDict]) -> None:
        if specs:
            signature = sequence_signature(specs)
            if signature not in self.tried_sequences:
                self.tried_sequences.add(signature)
                self.tried_family_counts[_candidate_family(specs[0])] += 1

    def mark_existing(self, specs: List[JsonDict], reward: float, status: str = "complete") -> None:
        self.mark_tried(specs)
        self.mark_terminal(specs)
        self.update(specs, reward, status)

    def mark_terminal(self, specs: List[JsonDict]) -> None:
        if specs:
            self.evaluated_terminal_sequences.add(sequence_signature(specs))

    def is_terminal_evaluated(self, specs: List[JsonDict]) -> bool:
        return bool(specs) and sequence_signature(specs) in self.evaluated_terminal_sequences

    def mark_saturated(self, specs: List[JsonDict], reason: str | None = None) -> None:
        if not specs:
            return
        signature = sequence_signature(specs)
        if signature in self.saturated_prefixes:
            return
        self.saturated_prefixes.add(signature)
        if reason:
            print(f"[MCTS] saturated prefix after {reason}: {len(specs)} transform(s)")

    def mark_terminal_repeat(self, specs: List[JsonDict]) -> None:
        if not specs:
            return
        signature = sequence_signature(specs)
        self.terminal_repeat_counts[signature] += 1
        self.mark_saturated(specs, "terminal repeat")

    def is_saturated(self, specs: List[JsonDict]) -> bool:
        return bool(specs) and sequence_signature(specs) in self.saturated_prefixes

    def update(self, specs: List[JsonDict], reward: float, status: str) -> None:
        if specs:
            first = specs[0]
            family = _candidate_family(first)
            signature = sequence_signature([first])
            if status == "complete":
                self.successful_family_candidates.setdefault(family, set()).add(signature)
            elif status == "failed":
                self.failed_family_candidates.setdefault(family, set()).add(signature)

        root = self.get([])
        root.visits += 1
        root.total_reward += reward
        root.best_reward = max(root.best_reward, reward)
        if status == "early_stop":
            root.early_stop_count += 1
        elif status == "pruned":
            root.pruned_count += 1
        elif status == "failed":
            root.failed_count += 1

        for i in range(1, len(specs) + 1):
            stat = self.get(specs[:i])
            stat.visits += 1
            stat.total_reward += reward
            stat.best_reward = max(stat.best_reward, reward)
            if status == "early_stop":
                stat.early_stop_count += 1
            elif status == "pruned":
                stat.pruned_count += 1
            elif status == "failed":
                stat.failed_count += 1

        for left, right in zip(specs, specs[1:]):
            pair_key = stable_json_hash([_compact_transform_specs([left])[0], _compact_transform_specs([right])[0]])
            stat = self.pair_stats.setdefault(pair_key, PrefixStats())
            stat.visits += 1
            stat.total_reward += reward
            stat.best_reward = max(stat.best_reward, reward)
            if status == "early_stop":
                stat.early_stop_count += 1
            elif status == "pruned":
                stat.pruned_count += 1
            elif status == "failed":
                stat.failed_count += 1

    def is_bad_prefix(self, specs: List[JsonDict]) -> bool:
        if not specs:
            return False
        stat = self.get(specs)
        min_visits = int(self.constraints.get("tree_prune_min_visits", 2))
        min_best = float(self.constraints.get("tree_prune_best_speedup_below", 0.95))
        if stat.visits >= min_visits and stat.best_reward < min_best:
            return True
        max_bad = int(self.constraints.get("tree_prune_after_early_stops", 3))
        if max_bad > 0 and stat.early_stop_count >= max_bad and stat.best_reward < 1.0:
            return True
        return False

    def is_blacklisted_candidate(self, candidate: JsonDict) -> bool:
        if _candidate_family(candidate) in self.disabled_families:
            return True
        if int(candidate.get("scop", -1)) in self.disabled_scops:
            return True
        stat = self.get([candidate])
        failure_threshold = int(self.constraints.get("candidate_blacklist_failures", 2))
        if failure_threshold <= 0:
            return False
        failures = stat.failed_count + stat.pruned_count + stat.early_stop_count
        if failures >= failure_threshold and stat.best_reward < float(
            self.constraints.get("candidate_blacklist_min_best_speedup", 0.90)
        ):
            return True
        return False

    def disable_scops_from_specs(self, specs: Sequence[JsonDict], reason: str) -> None:
        new_scops = {
            int(spec.get("scop", -1))
            for spec in specs
            if int(spec.get("scop", -1)) >= 0
        }
        new_scops.difference_update(self.disabled_scops)
        if not new_scops:
            return
        self.disabled_scops.update(new_scops)
        formatted = ", ".join(str(scop) for scop in sorted(new_scops))
        print(f"[MCTS] disabled SCoP(s) {formatted}: {reason}")

    def disable_family_if_screening_unpromising(self, family: str, candidates: Sequence[JsonDict]) -> bool:
        if family in self.disabled_families:
            return False
        enabled = {
            sequence_signature([candidate])
            for candidate in candidates
            if _candidate_family(candidate) == family
        }
        if not enabled:
            return False
        failed = self.failed_family_candidates.get(family, set())
        successful = self.successful_family_candidates.get(family, set())
        relevant_failures = enabled.intersection(failed)
        relevant_successes = enabled.intersection(successful)
        min_failures = int(self.constraints.get("disable_family_after_failures", 4))
        fail_fraction = float(self.constraints.get("disable_family_failure_fraction", 0.50))
        enough_failures = len(relevant_failures) >= max(1, min_failures)
        enough_fraction = len(relevant_failures) / max(len(enabled), 1) >= fail_fraction
        if enough_failures and enough_fraction and not relevant_successes:
            self.disabled_families.add(family)
            print(
                f"[MCTS] disabled {family} after "
                f"{len(relevant_failures)}/{len(enabled)} screening candidates failed"
            )
            return True
        return False

    def should_expand(self, specs: List[JsonDict]) -> bool:
        if not specs:
            return True
        stat = self.get(specs)
        if stat.visits < int(self.constraints.get("tree_min_visits_before_expand", 1)):
            return False
        if self.is_bad_prefix(specs):
            return False
        return stat.best_reward >= float(self.constraints.get("tree_expand_min_best_speedup", 0.90))

    def score_child(
        self,
        parent: List[JsonDict],
        child: List[JsonDict],
        candidate: JsonDict,
    ) -> float:
        parent_visits = max(1, self.get(parent).visits)
        stat = self.get(child)
        if self.is_saturated(child):
            return -1.0e9
        prior_score, neg_risk = candidate_rank_key(candidate)
        prior = 0.35 * prior_score + 0.15 * max(0.0, 1.0 + neg_risk)
        if self.tried_family_counts[_candidate_family(candidate)] == 0:
            prior += float(self.constraints.get("tree_family_novelty_bonus", 0.20))
        if parent:
            pair_key = stable_json_hash([
                _compact_transform_specs([parent[-1]])[0],
                _compact_transform_specs([candidate])[0],
            ])
            pair = self.pair_stats.get(pair_key)
            if pair and pair.visits:
                prior += max(-0.25, min(0.25, (pair.mean_reward - 1.0) * 0.4))
        if stat.visits == 0:
            novelty = float(self.constraints.get("tree_unvisited_bonus", 1.0))
            return novelty + prior + self.rng.random() * 1.0e-6
        exploration = float(self.constraints.get("tree_exploration", 0.8))
        failure_rate = (stat.pruned_count + stat.failed_count + stat.early_stop_count) / max(stat.visits, 1)
        return (
            stat.mean_reward
            + exploration * math.sqrt(math.log(parent_visits + 1.0) / stat.visits)
            + prior / (stat.visits + 1.0)
            - float(self.constraints.get("tree_failure_penalty", 0.25)) * failure_rate
            + self.rng.random() * 1.0e-6
        )


def seed_tree_from_records(tree: AdaptiveTreeState, records: Sequence[JsonDict]) -> None:
    for record in records:
        specs = list(record.get("transforms", []) or [])
        if not specs:
            continue
        status = str(record.get("status", "complete"))
        failure = str(record.get("failure", ""))
        if "File name too long" in failure or "Errno 36" in failure:
            tree.disable_scops_from_specs(specs, "historic JScop filename too long")
            tree.mark_existing(specs, 0.0, "failed")
            continue
        if status == "complete":
            reward = float(record.get("speedup", 0.0) or 0.0)
            tree.mark_existing(specs, reward, "complete")
        else:
            lower_failure = failure.lower()
            tree.mark_existing(specs, 0.0, "early_stop" if "early stop" in lower_failure else "failed")


def _candidate_diversity_allowed(specs: List[JsonDict], candidate: JsonDict, constraints: JsonDict) -> bool:
    max_per_scop = int(constraints.get("max_transforms_per_scop", 0) or 0)
    if max_per_scop <= 0:
        max_per_scop = int(constraints.get("tree_max_transforms_per_scop", 1))
    if max_per_scop <= 0:
        return True
    scop = int(candidate.get("scop", -1))
    count = sum(1 for spec in specs if int(spec.get("scop", -1)) == scop)
    return count < max_per_scop


def _sort_tree_candidates(
    candidates: List[JsonDict],
    tree: AdaptiveTreeState,
    chosen: List[JsonDict],
) -> List[JsonDict]:
    candidates = [
        candidate
        for candidate in candidates
        if _candidate_family(candidate) not in tree.disabled_families
        and int(candidate.get("scop", -1)) not in tree.disabled_scops
        and not tree.is_saturated([*chosen, candidate])
    ]
    scored = [
        (tree.score_child(chosen, [*chosen, candidate], candidate), candidate)
        for candidate in candidates
    ]
    scored.sort(key=lambda item: item[0], reverse=True)
    return [candidate for _, candidate in scored]


def _single_transform_screening_order(
    candidates: List[JsonDict],
    tree: AdaptiveTreeState,
) -> List[JsonDict]:
    ranked = _sort_tree_candidates(candidates, tree, [])
    per_family = int(tree.constraints.get("screening_per_family", 2))
    if per_family <= 0:
        return ranked

    grouped: Dict[str, List[JsonDict]] = {}
    for candidate in ranked:
        grouped.setdefault(_candidate_family(candidate), []).append(candidate)
    family_order = sorted(
        grouped,
        key=lambda family: candidate_rank_key(grouped[family][0]),
        reverse=True,
    )

    ordered: List[JsonDict] = []
    seen: set[tuple[int, int, str, tuple[Any, ...]]] = set()
    for round_idx in range(per_family):
        for family in family_order:
            if tree.tried_family_counts[family] > round_idx:
                continue
            group = grouped[family]
            if round_idx >= len(group):
                continue
            candidate = group[round_idx]
            seen.add(_candidate_exact_key(candidate))
            ordered.append(candidate)

    for candidate in ranked:
        exact = _candidate_exact_key(candidate)
        if exact in seen:
            continue
        ordered.append(candidate)
    return ordered


def _tree_reward_for_value(cfg: Config, baseline_value: float, value: float) -> float:
    return max(0.0, _speedup_for_primary(cfg, baseline_value, value))


def _is_filename_too_long_error(exc: BaseException) -> bool:
    if isinstance(exc, OSError) and exc.errno == 36:
        return True
    text = str(exc)
    return "File name too long" in text or "Errno 36" in text


def sample_transform_sequence(
    app: SyclProjectApp,
    poly: PolyMorphSpec,
    tree: AdaptiveTreeState,
    performance_bias: Dict[str, JsonDict] | None = None,
    hot_filter: HotKernelFilter | None = None,
) -> List[JsonDict]:
    def mark_failed_and_prune(prefix: List[JsonDict], message: str) -> None:
        tree.mark_existing(prefix, 0.0, "failed")
        raise TrialPruned(message)

    initial_candidates = enumerate_transform_candidates(app, poly, performance_bias, hot_filter)
    if not initial_candidates:
        return []

    if bool(poly.search.constraints.get("mcts_include_stop_action", True)):
        stop = [_stop_spec()]
        if sequence_signature(stop) not in tree.tried_sequences:
            tree.mark_tried(stop)
            return stop

    if bool(poly.search.constraints.get("single_transform_screening", True)):
        limit = int(poly.search.constraints.get("single_transform_screening_limit", 0) or 0)
        if limit <= 0:
            limit = min(16, len(initial_candidates))
        if tree.get([]).visits < limit:
            for candidate in _single_transform_screening_order(initial_candidates, tree):
                prefix = [candidate]
                if _candidate_family(candidate) in tree.disabled_families:
                    continue
                if int(candidate.get("scop", -1)) in tree.disabled_scops:
                    continue
                if sequence_signature(prefix) in tree.tried_sequences:
                    continue
                if tree.is_saturated(prefix):
                    continue
                if tree.is_blacklisted_candidate(candidate):
                    continue
                if tree.is_bad_prefix(prefix):
                    continue
                try:
                    apply_transforms_to_scops(
                        app.scops,
                        prefix,
                        legality_cb=lambda: app_is_legal(app),
                    )
                except InvalidTransformArgs as exc:
                    tree.mark_existing(prefix, 0.0, "failed")
                    if _candidate_family(candidate) == "FUSE":
                        tree.disable_family_if_screening_unpromising("FUSE", initial_candidates)
                    continue
                except Exception as exc:
                    mark_failed_and_prune(prefix, str(exc))
                if not poly.allow_illegal and not app_is_legal(app):
                    mark_failed_and_prune(prefix, "Illegal transformed schedule")
                tree.mark_tried(prefix)
                return prefix

    max_transforms = max(1, min(poly.search.max_transforms_per_trial, len(initial_candidates)))
    chosen: List[JsonDict] = []
    used_exact: set[tuple[int, int, str, tuple[Any, ...]]] = set()
    for _pos in range(max_transforms):
        current_candidates = enumerate_transform_candidates(app, poly, performance_bias, hot_filter)
        if not current_candidates:
            break

        viable: List[JsonDict] = []
        for candidate in current_candidates:
            if _candidate_family(candidate) in tree.disabled_families:
                continue
            if int(candidate.get("scop", -1)) in tree.disabled_scops:
                continue
            exact = _candidate_exact_key(candidate)
            if exact in used_exact:
                continue
            if tree.is_blacklisted_candidate(candidate):
                continue
            if not _candidate_diversity_allowed(chosen, candidate, poly.search.constraints):
                continue
            next_prefix = [*chosen, candidate]
            if tree.is_saturated(next_prefix):
                continue
            if tree.is_bad_prefix(next_prefix):
                continue
            next_is_terminal_repeat = tree.is_terminal_evaluated(next_prefix)
            next_can_expand = len(next_prefix) < max_transforms and tree.should_expand(next_prefix)
            if next_is_terminal_repeat and not next_can_expand:
                continue
            if sequence_signature(next_prefix) in tree.tried_sequences and not next_can_expand:
                continue
            if poly.search.constraint_aware:
                reasons = static_prune_sequence(next_prefix, poly.search.constraints)
                if reasons:
                    continue
            viable.append(candidate)
        if not viable:
            if chosen and tree.is_terminal_evaluated(chosen):
                tree.mark_terminal_repeat(chosen)
                raise TrialPruned("terminal sequence already evaluated")
            break

        spec = None
        last_invalid: InvalidTransformArgs | None = None
        for candidate in _sort_tree_candidates(viable, tree, chosen):
            try:
                apply_transforms_to_scops(
                    app.scops,
                    [candidate],
                    legality_cb=lambda: app_is_legal(app),
                )
                if not poly.allow_illegal and not app_is_legal(app):
                    mark_failed_and_prune([*chosen, candidate], "Illegal transformed schedule")
            except InvalidTransformArgs as exc:
                tree.mark_existing([*chosen, candidate], 0.0, "failed")
                last_invalid = exc
                continue
            except Exception as exc:
                mark_failed_and_prune([*chosen, candidate], str(exc))
            spec = candidate
            break
        if spec is None:
            raise TrialPruned(str(last_invalid) if last_invalid else "No valid transform candidates")
        chosen.append(spec)
        used_exact.add(_candidate_exact_key(spec))

        if not poly.allow_illegal and not app_is_legal(app):
            mark_failed_and_prune(chosen, "Illegal transformed schedule")

        if poly.search.constraint_aware:
            reasons = static_prune_sequence(chosen, poly.search.constraints)
            if reasons:
                raise TrialPruned("; ".join(reasons))

        if not tree.should_expand(chosen):
            break

    if chosen and tree.is_terminal_evaluated(chosen):
        tree.mark_terminal_repeat(chosen)
        raise TrialPruned("terminal sequence already evaluated")
    return chosen


def make_project_app(
    *,
    poly: PolyMorphSpec,
    source: Path,
    exec_name: str,
    populate_scops: bool,
) -> SyclProjectApp:
    ensure_tadashi_available()
    compiler_options = tadashi_compiler_options(
        poly.flags,
        populate_scops,
    )
    build_compiler_options = list(poly.flags)
    run_env: Dict[str, str] = {}
    if poly.search.target_backend:
        run_env["ACPP_VISIBILITY_MASK"] = str(poly.search.target_backend)

    return SyclProjectApp(
        project_root=poly.project_root,
        source=source,
        exec_name=exec_name,
        build_system=poly.build_system,
        build_target=poly.build_target,
        build_dir=poly.build_dir,
        runtime_args=poly.runtime_args,
        compiler_executable=poly.compiler,
        translator=SafePolly(poly.compiler) if populate_scops else None,
        compiler_options=compiler_options,
        build_compiler_options=build_compiler_options,
        run_env=run_env,
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


_SYCL_KERNEL_METRIC_RE = re.compile(r"^sycl_kernel_(\d+)_(avg|sum)_s$")


def _kernel_metrics(metrics: Dict[str, float]) -> Dict[int, float]:
    kernels: Dict[int, float] = {}
    for name, value in metrics.items():
        match = _SYCL_KERNEL_METRIC_RE.match(str(name))
        if not match:
            continue
        label = match.group(2)
        if label != "avg":
            continue
        kernels[int(match.group(1))] = float(value)
    if kernels:
        return kernels
    for name, value in metrics.items():
        match = _SYCL_KERNEL_METRIC_RE.match(str(name))
        if match:
            kernels[int(match.group(1))] = float(value)
    return kernels


def analyze_kernel_timing_deltas(
    baseline_metrics: Dict[str, float],
    metrics: Dict[str, float],
    specs: List[JsonDict],
) -> JsonDict:
    baseline_kernels = _kernel_metrics(baseline_metrics)
    candidate_kernels = _kernel_metrics(metrics)
    common = sorted(set(baseline_kernels) & set(candidate_kernels))
    per_kernel: Dict[str, JsonDict] = {}
    regressions: List[tuple[int, float]] = []
    improvements: List[tuple[int, float]] = []
    for kernel_id in common:
        baseline = baseline_kernels[kernel_id]
        candidate = candidate_kernels[kernel_id]
        if baseline <= 0.0:
            continue
        ratio = candidate / baseline
        entry = {
            "baseline": baseline,
            "candidate": candidate,
            "ratio": ratio,
            "speedup": baseline / candidate if candidate > 0.0 else 0.0,
        }
        per_kernel[str(kernel_id)] = entry
        if ratio > 1.05:
            regressions.append((kernel_id, ratio))
        elif ratio < 0.95:
            improvements.append((kernel_id, ratio))

    if not common:
        classification = "no_kernel_detail"
    elif len(regressions) == 1:
        classification = "single_kernel_regression"
    elif len(regressions) > 1 and len(regressions) >= max(2, len(common) // 2):
        classification = "broad_kernel_regression"
    elif improvements and not regressions:
        classification = "kernel_improvement"
    elif regressions:
        classification = "mixed_kernel_regression"
    else:
        classification = "kernel_neutral"

    worst_kernel = None
    worst_ratio = 1.0
    if regressions:
        worst_kernel, worst_ratio = max(regressions, key=lambda item: item[1])

    attribution = []
    for spec in specs:
        attribution.append(
            {
                "scop": int(spec.get("scop", -1)),
                "node": int(spec.get("node", -1)),
                "transform": str(spec.get("tr", "")),
                "args": list(spec.get("args", [])),
                "suspected_kernel": worst_kernel,
                "confidence": "heuristic" if worst_kernel is not None else "none",
            }
        )

    return {
        "classification": classification,
        "per_kernel": per_kernel,
        "worst_kernel": worst_kernel,
        "worst_ratio": worst_ratio,
        "improved_kernel_count": len(improvements),
        "regressed_kernel_count": len(regressions),
        "attribution": attribution,
    }


def classify_performance_regression(
    cfg: Config,
    baseline_value: float,
    value: float,
    kernel_analysis: JsonDict,
) -> str:
    speedup = _speedup_for_primary(cfg, baseline_value, value)
    if speedup >= 1.0:
        return "objective_improvement"
    classification = str(kernel_analysis.get("classification", "no_kernel_detail"))
    if classification != "no_kernel_detail":
        return classification
    return "objective_regression"


def _is_bad_first_fidelity(cfg: Config, baseline_value: float, value: float, worse_than: float) -> bool:
    if value <= 0.0 or baseline_value <= 0.0:
        return False
    goal = cfg.objectives[0].goal if cfg.objectives else "min"
    if goal == "max":
        return value < baseline_value / max(worse_than, 1.0)
    return value > baseline_value * max(worse_than, 1.0)


def _pick_representative_trial(trials: Sequence[Any], cfg: Config) -> Any:
    if not trials:
        raise RuntimeError("No Pareto trials available.")
    first_goal = cfg.objectives[0].goal if cfg.objectives else "min"
    if first_goal == "max":
        return max(trials, key=lambda trial: trial.values[0] if trial.values else float("-inf"))
    return min(trials, key=lambda trial: trial.values[0] if trial.values else float("inf"))


def _backend_specific_best(trials: Sequence[NativeTrial], cfg: Config) -> JsonDict:
    best: Dict[str, JsonDict] = {}
    for trial in trials:
        if trial.state != "COMPLETE":
            continue
        analysis = trial.user_attrs.get("backend_sensitivity") or {}
        per_backend = analysis.get("per_backend") or {}
        if not isinstance(per_backend, dict):
            continue
        speedup = float(trial.user_attrs.get("speedup", 0.0) or 0.0)
        value = trial.values[0] if trial.values else trial.user_attrs.get("runtime")
        for backend, backend_analysis in per_backend.items():
            if not isinstance(backend_analysis, dict):
                continue
            objective = backend_analysis.get("objective", value)
            quality = speedup if backend_analysis.get("ok") else -1.0
            current = best.get(str(backend))
            if current is None or quality > float(current.get("quality", float("-inf"))):
                best[str(backend)] = {
                    "trial": trial.number,
                    "quality": quality,
                    "objective": objective,
                    "speedup": speedup,
                    "transforms": trial.user_attrs.get("transforms", []),
                    "backend_sensitivity": backend_analysis,
                }
    return best


def _dominates(left: NativeTrial, right: NativeTrial, cfg: Config) -> bool:
    if left.values is None or right.values is None:
        return False
    better_or_equal = True
    strictly_better = False
    for idx, objective in enumerate(cfg.objectives):
        if idx >= len(left.values) or idx >= len(right.values):
            return False
        lval = left.values[idx]
        rval = right.values[idx]
        if objective.goal == "max":
            better_or_equal = better_or_equal and lval >= rval
            strictly_better = strictly_better or lval > rval
        else:
            better_or_equal = better_or_equal and lval <= rval
            strictly_better = strictly_better or lval < rval
    return better_or_equal and strictly_better


def pareto_trials(trials: Sequence[NativeTrial], cfg: Config) -> List[NativeTrial]:
    completed = [trial for trial in trials if trial.state == "COMPLETE" and trial.values is not None]
    return [
        trial for trial in completed
        if not any(_dominates(other, trial, cfg) for other in completed if other is not trial)
    ]


def _write_pareto_csv(trials: Sequence[NativeTrial], poly: PolyMorphSpec, cfg: Config) -> None:
    if not poly.search.pareto_csv:
        return
    best_trials = pareto_trials(trials, cfg)
    out_csv = Path(poly.search.pareto_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    metric_names = [obj.metric for obj in cfg.objectives]
    extra_metrics: set[str] = set()
    for trial in best_trials:
        extra_metrics.update((trial.user_attrs.get("metrics") or {}).keys())
    extra_metrics.difference_update(metric_names)
    with open(out_csv, "w", newline="", encoding="utf-8") as fp:
        writer = csv.writer(fp)
        writer.writerow(["trial", *metric_names, "speedup", "transforms", *sorted(extra_metrics)])
        for trial in best_trials:
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


def _apply_performance_bias(candidate: JsonDict, bias: JsonDict, constraints: JsonDict) -> None:
    predictions = candidate.setdefault("predictions", {})
    base_score = float(predictions.get("score", 0.5) or 0.5)
    base_risk = float(predictions.get("risk", 0.1) or 0.1)
    penalty = float(bias.get("penalty", 0.0) or 0.0)
    bonus = float(bias.get("bonus", 0.0) or 0.0)
    weight = float(constraints.get("performance_feedback_bias_weight", 0.35))

    predictions["performance_penalty"] = penalty
    predictions["performance_bonus"] = bonus
    predictions["performance_observations"] = int(bias.get("observations", 0) or 0)
    predictions["score"] = max(0.0, min(1.0, base_score - weight * penalty + 0.5 * weight * bonus))
    predictions["risk"] = max(0.0, min(1.0, base_risk + weight * penalty))
    reasons = predictions.setdefault("reasons", [])
    if isinstance(reasons, list):
        reasons.extend(str(reason) for reason in bias.get("reasons", [])[:5])


def _performance_bias_prune_reasons(candidate: JsonDict, constraints: JsonDict) -> List[str]:
    predictions = candidate.get("predictions", {}) or {}
    observations = int(predictions.get("performance_observations", 0) or 0)
    if observations < int(constraints.get("performance_feedback_min_observations_for_prune", 1)):
        return []

    reasons: List[str] = []
    max_penalty = constraints.get("max_performance_feedback_penalty")
    penalty = float(predictions.get("performance_penalty", 0.0) or 0.0)
    if max_penalty is not None and penalty > float(max_penalty):
        reasons.append(f"performance feedback penalty {penalty:.3f} exceeds max_performance_feedback_penalty={max_penalty}")

    return reasons


def _update_performance_bias(
    performance_bias: Dict[str, JsonDict],
    specs: List[JsonDict],
    speedup: float,
    analysis: JsonDict,
) -> None:
    if not specs:
        return
    regression = str(analysis.get("regression", ""))
    kernel_timing = analysis.get("kernel_timing", {}) or {}
    worst_ratio = float(kernel_timing.get("worst_ratio", 1.0) or 1.0)
    penalty = max(0.0, min(1.0, 1.0 - speedup))
    if regression in {"single_kernel_regression", "mixed_kernel_regression", "broad_kernel_regression"}:
        penalty = max(penalty, min(1.0, 0.2 * max(0.0, worst_ratio - 1.0)))
    bonus = max(0.0, min(0.5, speedup - 1.0))
    reasons = [regression] if regression else []
    if kernel_timing.get("worst_kernel") is not None:
        reasons.append(
            f"worst kernel {kernel_timing.get('worst_kernel')} ratio={worst_ratio:.3f}"
        )

    for spec in specs:
        key = candidate_key(spec)
        current = performance_bias.setdefault(
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
        apply_transforms_to_scops(app.scops, poly.transforms, legality_cb=lambda: app_is_legal(app))
    else:
        print("\nNo transforms were provided; project will be rebuilt around unchanged source.")

    final_legal = app_is_legal(app)

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


def explore_mcts(cfg: Config, poly: PolyMorphSpec, source: Path) -> int:
    require_executable(poly.compiler)
    if not poly.exec_name:
        raise ValueError("polyMorph.exec_name is required for search mode.")

    baseline_exec = poly.search.baseline_exec_name or f"{poly.exec_name}-baseline"
    if poly.search.target_backend:
        print(f"[polyMorph] measuring baseline and trials with ACPP_VISIBILITY_MASK={poly.search.target_backend}")

    print("\n=== Building/measuring baseline ===")
    baseline_app = make_project_app(
        poly=poly,
        source=source,
        exec_name=baseline_exec,
        populate_scops=False,
    )
    build_app_or_raise(baseline_app)
    baseline_metrics = measure_metrics_or_raise(baseline_app, poly.search.repeat, cfg)
    baseline_correctness_outputs = capture_correctness_outputs(baseline_app, cfg, poly)
    if baseline_correctness_outputs:
        print(f"Captured {len(baseline_correctness_outputs)} baseline correctness output(s).")
    baseline_value = _objective_values_from_metrics(cfg, baseline_metrics)[0]
    print(f"Baseline objective: {baseline_value}")

    enum_app = make_project_app(
        poly=poly,
        source=source,
        exec_name=poly.exec_name,
        populate_scops=True,
    )
    all_candidates = enumerate_transform_candidates(enum_app, poly, apply_top_k=False)
    hot_kernel_filter = build_hot_kernel_filter(all_candidates, baseline_metrics, poly.search.constraints)
    if hot_kernel_filter.enabled:
        candidates = enumerate_transform_candidates(
            enum_app,
            poly,
            hot_filter=hot_kernel_filter,
            print_family_summary=True,
        )
        print(
            "Hot-kernel candidate filter: "
            f"kernels={hot_kernel_filter.hot_kernels}, "
            f"allowed_scops={sorted(hot_kernel_filter.allowed_scops)}, "
            f"coverage={hot_kernel_filter.coverage:.3f}, "
            f"candidates={len(all_candidates)}->{len(candidates)}"
        )
    else:
        candidates = enumerate_transform_candidates(enum_app, poly, print_family_summary=True)
        if bool(poly.search.constraints.get("hot_kernel_filter", True)):
            print("Hot-kernel candidate filter: disabled automatically; no reliable multi-kernel timing/SCoP map.")
    print_candidates(candidates, verbose=poly.search.enumerate_only)

    if not candidates:
        print("No candidate transformations found.")
        return 1

    if poly.search.enumerate_only:
        return 0

    baseline_backend_sensitivity = collect_backend_sensitivity(baseline_app, poly, cfg)
    if baseline_backend_sensitivity.get("checked"):
        print(
            "Baseline backend sensitivity: "
            f"backend_sensitive={baseline_backend_sensitivity.get('backend_sensitive', False)}"
        )
        for reason in baseline_backend_sensitivity.get("reasons", []):
            print(f"  - {reason}")

    source_hash = file_hash(source)
    source_structural_signature = structural_signature(candidates)
    cache_path = _cache_path(poly)
    evaluation_cache = load_evaluation_cache(cache_path)
    if evaluation_cache:
        print(f"Loaded {len(evaluation_cache)} cached evaluation record(s) from {cache_path}.")
    history_records = load_history(poly.search.history_jsonl)
    if poly.search.learned_model:
        learned_records = [*history_records, *evaluation_cache.values()]
        learned_model = build_learned_candidate_model(
            learned_records,
            target_backend=poly.search.target_backend,
        )
        apply_learned_model_to_candidates(
            candidates,
            learned_model,
            min_observations=poly.search.learned_model_min_observations,
        )
        if learned_model:
            print(f"Loaded learned candidate model with {len(learned_model)} transform key(s).")
        if poly.search.analytical_model and poly.search.top_k:
            candidates = select_diverse_top_k(candidates, poly.search.top_k)
    performance_bias: Dict[str, JsonDict] = {}
    tree_state = AdaptiveTreeState(
        constraints=poly.search.constraints,
        rng=random.Random(poly.search.seed),
    )
    if poly.search.case_retrieval:
        retrieval_records = [*history_records, *evaluation_cache.values()]
        retrieved_sequences = retrieve_sequences(
            history=retrieval_records,
            source_hash=source_hash,
            candidates=candidates,
            limit=poly.search.retrieval_top_k,
            structural_sig=source_structural_signature if poly.search.structural_retrieval else None,
        )
        for seq in retrieved_sequences:
            tree_state.mark_existing(seq, 1.0, "complete")
        if retrieved_sequences:
            print(f"Seeded adaptive tree with {len(retrieved_sequences)} retrieved sequence(s).")
    seed_tree_from_records(tree_state, [*history_records, *evaluation_cache.values()])
    print(
        "Using Monte Carlo tree search "
        f"with {len(tree_state.stats)} seeded prefix statistic(s)."
    )

    is_multi = len(cfg.objectives) > 1

    def objective(trial: NativeTrial) -> float | List[float]:
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
            specs = sample_transform_sequence(trial_app, poly, tree_state, performance_bias, hot_kernel_filter)
            if not specs:
                raise TrialPruned("No candidate transformations available for this trial.")
            trial.set_user_attr("transforms", specs)
            transform_signature = sequence_signature(specs)
            trial.set_user_attr("transform_signature", transform_signature)
            tree_state.mark_terminal(specs)

            cache_key = _cache_key(source_hash=source_hash, specs=specs, poly=poly, cfg=cfg)
            cached = evaluation_cache.get(cache_key)
            if cached:
                status = str(cached.get("status", "complete"))
                trial.set_user_attr("cache_hit", True)
                if status == "failed":
                    tree_state.update(specs, 0.0, "failed")
                    tree_state.mark_saturated(specs, "cached failed terminal")
                    trial.set_user_attr("tree_updated", True)
                    raise TrialPruned(str(cached.get("failure", "cached failed evaluation")))
                metrics = dict(cached.get("metrics", {}) or {})
                obj_values = [float(x) for x in cached.get("objectives", [])]
                if not obj_values:
                    obj_values = _objective_values_from_metrics(cfg, metrics)
                speedup = float(cached.get("speedup", _speedup_for_primary(cfg, baseline_value, obj_values[0])) or 0.0)
                cache_reward_scale = float(poly.search.constraints.get("cache_hit_reward_scale", 0.95))
                tree_state.update(specs, max(0.0, speedup * cache_reward_scale), "complete")
                tree_state.mark_saturated(specs, "cache hit")
                trial.set_user_attr("tree_updated", True)
                trial.set_user_attr("runtime", obj_values[0])
                trial.set_user_attr("metrics", metrics)
                trial.set_user_attr("speedup", speedup)
                trial.set_user_attr("correctness", cached.get("correctness", {}))
                trial.set_user_attr(
                    "performance_feedback_analysis",
                    cached.get("performance_feedback_analysis", {}),
                )
                trial.set_user_attr("backend_sensitivity", cached.get("backend_sensitivity", {}))
                print(f"Trial {trial.number}: cache hit objective={obj_values[0]}, transforms={specs}")
                raise CachedSequenceSelected(
                    f"cached terminal sequence skipped after tree update; objective={obj_values[0]}"
                )

            if _is_stop_sequence(specs):
                obj_values = _objective_values_from_metrics(cfg, baseline_metrics)
                value = obj_values[0]
                speedup = _speedup_for_primary(cfg, baseline_value, value)
                tree_state.update(specs, max(0.0, speedup), "complete")
                trial.set_user_attr("tree_updated", True)
                trial.set_user_attr("runtime", value)
                trial.set_user_attr("metrics", baseline_metrics)
                trial.set_user_attr("speedup", speedup)
                trial.set_user_attr("correctness", {"checked": False, "reason": "baseline stop action"})
                record = {
                    "key": cache_key,
                    "status": "complete",
                    "source": str(source),
                    "source_hash": source_hash,
                    "structural_signature": source_structural_signature,
                    "sequence_features": sequence_feature_summary(specs),
                    "target_backend": poly.search.target_backend,
                    "transform_signature": transform_signature,
                    "transforms": specs,
                    "metrics": baseline_metrics,
                    "objectives": obj_values,
                    "speedup": speedup,
                    "performance_feedback_analysis": {"stop_action": True},
                    "backend_sensitivity": baseline_backend_sensitivity,
                    "correctness": {"checked": False, "reason": "baseline stop action"},
                }
                evaluation_cache[cache_key] = record
                append_evaluation_cache(cache_path, record)
                append_history(poly.search.history_jsonl, record)
                print(f"Trial {trial.number}: STOP action objective={value}, speedup={speedup}")
                return obj_values if is_multi else value

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

            if poly.search.multi_fidelity and poly.search.repeat > 1:
                early_stop_warmups = int(
                    poly.search.constraints.get("early_stop_warmup_runs", 1)
                )
                early_stop_runs = max(
                    1,
                    int(poly.search.constraints.get("early_stop_measure_runs", 1)),
                )
                with temporary_metric_warmup_runs(cfg, early_stop_warmups):
                    first_metrics = measure_metrics_or_raise(transformed_app, early_stop_runs, cfg)
                first_value = _objective_values_from_metrics(cfg, first_metrics)[0]
                first_kernel_analysis = analyze_kernel_timing_deltas(
                    baseline_metrics,
                    first_metrics,
                    specs,
                )
                first_regression = classify_performance_regression(
                    cfg,
                    baseline_value,
                    first_value,
                    first_kernel_analysis,
                )
                trial.set_user_attr("first_fidelity_metrics", first_metrics)
                trial.set_user_attr("first_fidelity_warmup_runs", early_stop_warmups)
                trial.set_user_attr(
                    "first_fidelity_performance_feedback",
                    {
                        "kernel_timing": first_kernel_analysis,
                        "regression": first_regression,
                    },
                )
                if _is_bad_first_fidelity(
                    cfg,
                    baseline_value,
                    first_value,
                    poly.search.early_stop_worse_than,
                ):
                    trial.set_user_attr("early_stopped", True)
                    tree_state.update(
                        specs,
                        _tree_reward_for_value(cfg, baseline_value, first_value),
                        "early_stop",
                    )
                    _update_performance_bias(
                        performance_bias,
                        specs,
                        _speedup_for_primary(cfg, baseline_value, first_value),
                        {
                            "kernel_timing": first_kernel_analysis,
                            "regression": first_regression,
                        },
                    )
                    trial.set_user_attr("tree_updated", True)
                    raise TrialPruned(
                        f"early stop: first-fidelity objective after {early_stop_warmups} warmup run(s) "
                        f"{first_value} worse than baseline {baseline_value} "
                        f"by factor {poly.search.early_stop_worse_than}"
                    )

            metrics = measure_metrics_or_raise(transformed_app, poly.search.repeat, cfg)
            obj_values = _objective_values_from_metrics(cfg, metrics)
            value = obj_values[0]
            kernel_analysis = analyze_kernel_timing_deltas(baseline_metrics, metrics, specs)
            regression_class = classify_performance_regression(
                cfg,
                baseline_value,
                value,
                kernel_analysis,
            )
            performance_feedback_analysis = {
                "kernel_timing": kernel_analysis,
                "regression": regression_class,
            }
            trial.set_user_attr("performance_feedback_analysis", performance_feedback_analysis)
            correctness = verify_correctness_outputs(baseline_correctness_outputs, transformed_app, cfg, poly)
            if correctness.get("checked"):
                trial.set_user_attr("correctness", correctness)
                if not correctness.get("ok", False):
                    raise TrialPruned(f"correctness check failed: {correctness}")

            backend_sensitivity: JsonDict = {}
            if poly.search.backend_sensitivity_per_trial:
                backend_sensitivity = collect_backend_sensitivity(transformed_app, poly, cfg)
                if backend_sensitivity.get("checked"):
                    trial.set_user_attr("backend_sensitivity", backend_sensitivity)

            speedup = _speedup_for_primary(cfg, baseline_value, value)
            tree_state.update(specs, max(0.0, speedup), "complete")
            _update_performance_bias(performance_bias, specs, speedup, performance_feedback_analysis)
            trial.set_user_attr("tree_updated", True)

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
                    "structural_signature": source_structural_signature,
                    "sequence_features": sequence_feature_summary(specs),
                    "target_backend": poly.search.target_backend,
                    "transforms": specs,
                    "metrics": metrics,
                    "objectives": obj_values,
                    "speedup": speedup,
                    "performance_feedback_analysis": performance_feedback_analysis,
                    "backend_sensitivity": backend_sensitivity,
                    "correctness": correctness,
                },
            )
            record = {
                "key": cache_key,
                "status": "complete",
                "source": str(source),
                "source_hash": source_hash,
                "structural_signature": source_structural_signature,
                "sequence_features": sequence_feature_summary(specs),
                "target_backend": poly.search.target_backend,
                "transform_signature": transform_signature,
                "transforms": specs,
                "metrics": metrics,
                "objectives": obj_values,
                "speedup": speedup,
                "performance_feedback_analysis": performance_feedback_analysis,
                "backend_sensitivity": backend_sensitivity,
                "correctness": correctness,
            }
            evaluation_cache[cache_key] = record
            append_evaluation_cache(cache_path, record)
            return obj_values if is_multi else value
        except CachedSequenceSelected:
            raise
        except TrialPruned as exc:
            specs = trial.user_attrs.get("transforms", [])
            if not specs:
                raise
            if specs and not trial.user_attrs.get("tree_updated", False):
                tree_state.update(specs, 0.0, "pruned")
            pruned_feedback = trial.user_attrs.get("first_fidelity_performance_feedback", {})
            append_history(
                poly.search.history_jsonl,
                {
                    "status": "pruned",
                    "trial": trial.number,
                    "source": str(source),
                    "source_hash": source_hash,
                    "structural_signature": source_structural_signature,
                    "sequence_features": sequence_feature_summary(specs),
                    "target_backend": poly.search.target_backend,
                    "transforms": specs,
                    "performance_feedback_analysis": pruned_feedback,
                    "failure": str(exc),
                },
            )
            if specs:
                cache_key = _cache_key(source_hash=source_hash, specs=specs, poly=poly, cfg=cfg)
                record = {
                    "key": cache_key,
                    "status": "failed",
                    "source": str(source),
                    "source_hash": source_hash,
                    "structural_signature": source_structural_signature,
                    "sequence_features": sequence_feature_summary(specs),
                    "target_backend": poly.search.target_backend,
                    "transform_signature": sequence_signature(specs),
                    "transforms": specs,
                    "performance_feedback_analysis": pruned_feedback,
                    "failure": str(exc),
                }
                evaluation_cache[cache_key] = record
                append_evaluation_cache(cache_path, record)
            raise
        except Exception as exc:
            trial.set_user_attr("failure", str(exc))
            specs = trial.user_attrs.get("transforms", [])
            if specs and _is_filename_too_long_error(exc):
                tree_state.disable_scops_from_specs(specs, "JScop filename too long")
                if not trial.user_attrs.get("tree_updated", False):
                    tree_state.update(specs, 0.0, "failed")
                    trial.set_user_attr("tree_updated", True)
                raise CachedSequenceSelected("JScop filename too long; disabled affected SCoP(s)")
            if specs and not trial.user_attrs.get("tree_updated", False):
                tree_state.update(specs, 0.0, "failed")
            append_history(
                poly.search.history_jsonl,
                {
                    "status": "failed",
                    "trial": trial.number,
                    "source": str(source),
                    "source_hash": source_hash,
                    "structural_signature": source_structural_signature,
                    "transforms": trial.user_attrs.get("transforms", []),
                    "failure": str(exc),
                },
            )
            if specs:
                cache_key = _cache_key(source_hash=source_hash, specs=specs, poly=poly, cfg=cfg)
                record = {
                    "key": cache_key,
                    "status": "failed",
                    "source": str(source),
                    "source_hash": source_hash,
                    "structural_signature": source_structural_signature,
                    "sequence_features": sequence_feature_summary(specs),
                    "target_backend": poly.search.target_backend,
                    "transform_signature": sequence_signature(specs),
                    "transforms": specs,
                    "failure": str(exc),
                }
                evaluation_cache[cache_key] = record
                append_evaluation_cache(cache_path, record)
            raise TrialPruned(str(exc))
        finally:
            if transformed_app is not None:
                cleanup_trial_artifacts(transformed_app, generated_infix=generated_infix)

    def evaluate_sequence_for_analysis(specs: List[JsonDict], label: str) -> JsonDict:
        cache_key = _cache_key(source_hash=source_hash, specs=specs, poly=poly, cfg=cfg)
        cached = evaluation_cache.get(cache_key)
        if cached and cached.get("status") == "complete":
            return {**cached, "analysis_label": label, "cache_hit": True}
        if _is_stop_sequence(specs):
            obj_values = _objective_values_from_metrics(cfg, baseline_metrics)
            return {
                "analysis_label": label,
                "status": "complete",
                "cache_hit": False,
                "target_backend": poly.search.target_backend,
                "transforms": specs,
                "metrics": baseline_metrics,
                "objectives": obj_values,
                "speedup": _speedup_for_primary(cfg, baseline_value, obj_values[0]),
            }

        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", label).strip("-") or "analysis"
        generated_infix = f"{poly.search.generated_infix}-{safe_label}"
        analysis_app: SyclProjectApp | None = None
        try:
            app = make_project_app(
                poly=poly,
                source=source,
                exec_name=f"{poly.exec_name}-{safe_label}",
                populate_scops=True,
            )
            apply_transforms_to_scops(app.scops, specs, legality_cb=lambda: app_is_legal(app))
            if not poly.allow_illegal and not app_is_legal(app):
                raise TrialPruned("analysis sequence is not legal")
            analysis_app = app.generate_code(
                alt_infix=generated_infix,
                ephemeral=False,
                populate_scops=False,
                ensure_legality=not poly.allow_illegal,
            )
            analysis_app = inherit_build_settings(analysis_app, app)
            analysis_app.exec_name = f"{poly.exec_name}-{safe_label}"
            build_app_or_raise(analysis_app)
            metrics = measure_metrics_or_raise(analysis_app, poly.search.repeat, cfg)
            obj_values = _objective_values_from_metrics(cfg, metrics)
            speedup = _speedup_for_primary(cfg, baseline_value, obj_values[0])
            record = {
                "key": cache_key,
                "analysis_label": label,
                "status": "complete",
                "cache_hit": False,
                "source": str(source),
                "source_hash": source_hash,
                "structural_signature": source_structural_signature,
                "sequence_features": sequence_feature_summary(specs),
                "target_backend": poly.search.target_backend,
                "transform_signature": sequence_signature(specs),
                "transforms": specs,
                "metrics": metrics,
                "objectives": obj_values,
                "speedup": speedup,
            }
            evaluation_cache[cache_key] = record
            append_evaluation_cache(cache_path, record)
            return record
        except Exception as exc:
            return {
                "analysis_label": label,
                "status": "failed",
                "target_backend": poly.search.target_backend,
                "transforms": specs,
                "failure": str(exc),
            }
        finally:
            if analysis_app is not None:
                cleanup_trial_artifacts(analysis_app, generated_infix=generated_infix)

    trials: List[NativeTrial] = []
    start_time = time.monotonic()
    selection_retries = 0
    max_selection_retries = max(
        poly.search.n_trials * 10,
        int(poly.search.constraints.get("mcts_max_selection_retries", 100)),
    )
    while len(trials) < poly.search.n_trials:
        if poly.search.timeout is not None and time.monotonic() - start_time >= poly.search.timeout:
            print(f"[polyMorph] stopping MCTS search after timeout={poly.search.timeout}s")
            break
        if selection_retries >= max_selection_retries:
            print(
                "[polyMorph] stopping MCTS search after too many illegal/no-op "
                f"selection retries ({selection_retries})."
            )
            break
        trial = NativeTrial(number=len(trials))
        print(_trial_banner(trial.number))
        try:
            result = objective(trial)
            values = [float(x) for x in result] if isinstance(result, list) else [float(result)]
            trial.values = values
            trial.state = "COMPLETE"
            trials.append(trial)
            completed_so_far = [t for t in trials if t.state == "COMPLETE"]
            best_so_far = _pick_representative_trial(completed_so_far, cfg)
            print(
                f"[MCTS] Trial {trial.number} finished with value: {values[0]}. "
                f"Best so far: {(best_so_far.values or [None])[0]}"
            )
        except CachedSequenceSelected as exc:
            selection_retries += 1
            print(f"[MCTS] cache retry {selection_retries}. {exc}")
            continue
        except TrialPruned as exc:
            if not trial.user_attrs.get("transforms"):
                selection_retries += 1
                print(f"[MCTS] selection retry {selection_retries}. {exc}")
                continue
            trial.state = "PRUNED"
            trial.set_user_attr("failure", str(exc))
            trials.append(trial)
            print(f"[MCTS] Trial {trial.number} pruned. {exc}")
        except Exception as exc:
            trial.state = "FAIL"
            trial.set_user_attr("failure", str(exc))
            trials.append(trial)
            print(f"[MCTS] Trial {trial.number} failed. {exc}")

    completed_trials = [trial for trial in trials if trial.state == "COMPLETE"]
    trial_csv = write_trial_csv(trials, poly)
    print(f"Wrote trial CSV: {trial_csv}")

    if not completed_trials:
        print("No successful trials.")
        if poly.search.result_json:
            result = {
                "baseline_runtime": baseline_value,
                "baseline_metrics": baseline_metrics,
                "baseline_backend_sensitivity": baseline_backend_sensitivity,
                "baseline_correctness_outputs": sorted(baseline_correctness_outputs),
                "target_backend": poly.search.target_backend,
                "completed_trials": 0,
                "best_runtime": None,
                "best_speedup": 0.0,
                "best_transforms": [],
            }
            Path(poly.search.result_json).write_text(json.dumps(result, indent=2), encoding="utf-8")
            print(f"Wrote result JSON: {poly.search.result_json}")
        return 1

    if is_multi:
        best_trials = pareto_trials(trials, cfg)
        _write_pareto_csv(trials, poly, cfg)
        best_trial = _pick_representative_trial(best_trials, cfg)
        best_runtime = best_trial.values[0] if best_trial.values else float("inf")
        best_specs = best_trial.user_attrs.get("transforms", [])
        best_speedup = best_trial.user_attrs.get("speedup", 0.0)
    else:
        best_trial = _pick_representative_trial(completed_trials, cfg)
        best_runtime = best_trial.values[0] if best_trial.values else best_trial.user_attrs.get("runtime")
        best_specs = best_trial.user_attrs.get("transforms", [])
        best_speedup = _speedup_for_primary(cfg, baseline_value, float(best_runtime))

    print("\n=== polyMorph search result ===")
    print(f"Baseline objective: {baseline_value}")
    print(f"Best objective: {best_runtime}")
    print(f"Best speedup: {best_speedup}")
    print("Best transforms:")
    print(json.dumps(best_specs, indent=2))

    ablation_results: List[JsonDict] = []
    if poly.search.ablation_enabled and best_specs and not _is_stop_sequence(best_specs):
        seen_analysis: set[str] = set()
        analysis_sequences: List[tuple[str, List[JsonDict]]] = [("best_full", list(best_specs))]
        if len(best_specs) > 1:
            for idx in range(len(best_specs)):
                reduced = [spec for pos, spec in enumerate(best_specs) if pos != idx]
                if reduced:
                    analysis_sequences.append((f"best_minus_{idx}", reduced))
        for idx, spec in enumerate(best_specs):
            analysis_sequences.append((f"best_single_{idx}", [spec]))
        for label, specs in analysis_sequences:
            sig = sequence_signature(specs)
            if sig in seen_analysis:
                continue
            seen_analysis.add(sig)
            print(f"[analysis] replaying {label}")
            ablation_results.append(evaluate_sequence_for_analysis(specs, label))

    replay_results: List[JsonDict] = []
    if poly.search.replay_top_k > 0:
        ranked_trials = sorted(
            completed_trials,
            key=lambda trial: trial.values[0] if trial.values else float("inf"),
            reverse=(cfg.objectives[0].goal == "max" if cfg.objectives else False),
        )
        for trial in ranked_trials[: poly.search.replay_top_k]:
            specs = list(trial.user_attrs.get("transforms", []) or [])
            if not specs:
                continue
            label = f"replay_trial_{trial.number}"
            print(f"[analysis] replaying {label}")
            replay_results.append(evaluate_sequence_for_analysis(specs, label))

    if poly.search.result_json:
        result = {
            "baseline_runtime": baseline_value,
            "baseline_metrics": baseline_metrics,
            "baseline_backend_sensitivity": baseline_backend_sensitivity,
            "baseline_correctness_outputs": sorted(baseline_correctness_outputs),
            "target_backend": poly.search.target_backend,
            "best_runtime": best_runtime,
            "best_speedup": best_speedup,
            "best_transforms": best_specs,
            "best_trial_number": best_trial.number,
            "backend_best_configs": _backend_specific_best(trials, cfg),
            "ablation_results": ablation_results,
            "replay_results": replay_results,
        }
        if is_multi:
            best_trials = pareto_trials(trials, cfg)
            result["pareto_trials"] = [
                {
                    "trial": trial.number,
                    "objectives": trial.values,
                    "transforms": trial.user_attrs.get("transforms", []),
                    "metrics": trial.user_attrs.get("metrics", {}),
                }
                for trial in best_trials
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
        if poly.mcts_search:
            return explore_mcts(cfg, poly, source)
        return run_project_mode(cfg, poly, source)
