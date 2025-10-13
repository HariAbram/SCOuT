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
    pretty = " ".join(shlex.quote(str(c)) for c in cmd) if isinstance(cmd, Sequence) else cmd
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
    cmd = f"{compiler} {flags} {shlex.quote(str(src))} -o {shlex.quote(str(out))}"
    proc = _run(cmd)
    if proc.returncode:
        _save_log(out, trial, "make", proc)
        return None
    return out if proc.returncode == 0 else None


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
                cmake_cmd += [f"-D{var}+={flags}"]
        
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
        
        return (build_dir / cfg.target) if cfg.target else _last_executable(build_dir)

    if cfg.build_system == "make":
        _run(["make", "clean"], cwd=cfg.dir)
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
        
        return (cfg.dir / cfg.target) if cfg.target else _last_executable(cfg.dir)

    raise ValueError(f"unknown build_system '{cfg.build_system}'")
