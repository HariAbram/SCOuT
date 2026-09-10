from __future__ import annotations

###############################################################################
# Standard library imports                                                    #
###############################################################################

import os
import shlex
import subprocess
import sys
import uuid
from pathlib import Path
from statistics import mean, variance
from typing import Dict, List, Optional, Sequence, Tuple, Any, Union
import optuna

###############################################################################
# Type helpers                                                                #
###############################################################################

Number = float
EnvMap = Dict[str, str]
MetricDict = Dict[str, Number]

###############################################################################
# Local imports                                                               #
###############################################################################

from src.config import BuildProject

###############################################################################
# Shell helpers                                                               #
###############################################################################

def _run(cmd: Sequence[str] | str, *, cwd: Path | None = None, env: EnvMap | None = None) -> subprocess.CompletedProcess:
    """Run a command, capturing output, and echo it to the console."""
    pretty = cmd if isinstance(cmd, str) else " ".join(shlex.quote(str(c)) for c in cmd)
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

def _trial_tag(trial: Optional["optuna.Trial"]) -> str:
    return f"trial_{trial.number:05d}" if trial is not None else f"phaseB_{uuid.uuid4().hex[:8]}"

def _save_log(workdir: Path,
              trial: Optional["optuna.Trial"],
              step: str,
              proc) -> None:
    """
    Save stdout/stderr of a subprocess to workdir/logs.
    Works even when trial is None (e.g., Phase-B rebuilds).
    """
    log_dir = Path(workdir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    tag = _trial_tag(trial)
    (log_dir / f"{tag}_{step}.out").write_text(proc.stdout or "")
    (log_dir / f"{tag}_{step}.err").write_text(proc.stderr or "")

###############################################################################
# Build logic (identical to original)                                         #
###############################################################################

def compile_single_source(compiler: str, src: Path, flags: str, out: Path, trial: Optional[optuna.Trial] = None) -> Optional[Path]:
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [*shlex.split(compiler), *shlex.split(flags), str(src), "-o", str(out)]
    proc = _run(cmd)
    if proc.returncode:
        _save_log(out.parent, trial, "compile", proc)
        return None
    return out if out.is_file() else None


def _last_executable(root: Path) -> Optional[Path]:
    latest: Optional[Path] = None
    latest_mtime = -1.0
    for p in root.rglob("*"):
        if p.is_file() and os.access(p, os.X_OK):
            mtime = p.stat().st_mtime
            if mtime > latest_mtime:
                latest_mtime, latest = mtime, p
    return latest


def compile_project(cfg: BuildProject, compiler: str, flags: str, workdir: Path, trial: Optional[optuna.Trial] = None) -> Optional[Path]:
    workdir.mkdir(parents=True, exist_ok=True)
    if cfg.build_system == "cmake":
        build_dir = workdir / f"cmake_{uuid.uuid4().hex[:8]}"
        build_dir.mkdir()
        defs = cfg.cmake_defs

        cmake_cmd = [
            "cmake", "-S", str(cfg.dir), "-B", str(build_dir),
             f"-DCMAKE_CXX_COMPILER={compiler}",
             "-DCMAKE_BUILD_TYPE=Release"
            ] 
        
        if cfg.cmake_flag_vars:
            for var in cfg.cmake_flag_vars:
                cmake_cmd += [f"-D{var}:STRING={flags}"]
        
        if cfg.cmake_defs:
            cmake_cmd +=[f"-D{d}" for d in defs]

        proc = _run(cmake_cmd)
        if proc.returncode:
            _save_log(workdir, trial, "cmake_config", proc)
            return None
        
        build_cmd = ["cmake", "--build", str(build_dir), "--parallel"]
        if cfg.target:
            build_cmd += ["--target", cfg.target]
        proc = _run(build_cmd)
        if proc.returncode:
            _save_log(workdir, trial, "cmake_build", proc)
            return None
        
        binary = _resolve_project_executable(cfg, build_dir)
        if binary is None:
            _save_message(workdir, trial, "cmake_artifact", "Build succeeded but no executable was found")
        return binary

    if cfg.build_system == "make":
        clean = _run(["make", "clean"], cwd=cfg.dir)
        if clean.returncode:
            _save_log(workdir, trial, "make_clean", clean)
            return None
        build_cmd = ["make", f"CXX={compiler}", "-j"]

        if flags:
            build_cmd.append(f"{cfg.make_flags_var}+={flags}")
        for var, val in cfg.make_vars.items():
            build_cmd.append(f"{var}={val}")
        if cfg.target:
            build_cmd.append(cfg.target)

        proc = _run(build_cmd, cwd=cfg.dir)
        if proc.returncode:
            _save_log(workdir, trial, "make", proc)
            return None
        
        binary = _resolve_project_executable(cfg, cfg.dir)
        if binary is None:
            _save_message(workdir, trial, "make_artifact", "Build succeeded but no executable was found")
        return binary

    raise ValueError(f"unknown build_system '{cfg.build_system}'")


def _resolve_project_executable(cfg: BuildProject, build_root: Path) -> Optional[Path]:
    """Resolve the runnable artifact independently from the build target name."""
    if cfg.executable:
        candidate = cfg.executable if cfg.executable.is_absolute() else build_root / cfg.executable
        return candidate if candidate.is_file() and os.access(candidate, os.X_OK) else None

    if cfg.target:
        candidate = build_root / cfg.target
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate
    return _last_executable(build_root)


def _save_message(workdir: Path, trial: Optional["optuna.Trial"], step: str, message: str) -> None:
    class _Message:
        stdout = ""
        stderr = message

    _save_log(workdir, trial, step, _Message())
