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

import os
import shlex
import subprocess
import sys
import uuid
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

###############################################################################
# Build logic (identical to original)                                         #
###############################################################################

def compile_single_source(compiler: str, src: Path, flags: str, out: Path) -> Optional[Path]:
    cmd = f"{compiler} {flags} {shlex.quote(str(src))} -o {shlex.quote(str(out))}"
    return out if _run(cmd).returncode == 0 else None


def _last_executable(root: Path) -> Optional[Path]:
    latest: Optional[Path] = None
    latest_mtime = -1.0
    for p in root.rglob("*"):
        if p.is_file() and os.access(p, os.X_OK):
            mtime = p.stat().st_mtime
            if mtime > latest_mtime:
                latest_mtime, latest = mtime, p
    return latest


def compile_project(cfg: BuildProject, compiler: str, flags: str, workdir: Path) -> Optional[Path]:
    if cfg.build_system == "cmake":
        build_dir = workdir / f"cmake_{uuid.uuid4().hex[:8]}"
        build_dir.mkdir()
        defs = cfg.cmake_defs

        cmake_cmd = [
            "cmake", "-S", str(cfg.dir), "-B", str(build_dir),
             f"-DCMAKE_CXX_COMPILER={compiler}",
             f"-DCMAKE_CXX_FLAGS={flags}",
             "-DCMAKE_BUILD_TYPE=Release"
            ] + [f"-D{d}" for d in defs]

        if _run(cmake_cmd).returncode:
            return None
        
        build_cmd = ["cmake", "--build", str(build_dir), "--parallel"]
        if cfg.target:
            build_cmd += ["--target", cfg.target]
        if _run(build_cmd).returncode:
            return None
        return (build_dir / cfg.target) if cfg.target else _last_executable(build_dir)

    if cfg.build_system == "make":
        _run(["make", "clean"], cwd=cfg.dir)
        build_cmd = ["make", f"CXX={compiler}", f"CXXFLAGS+={flags}", "-j"]
        for var, val in cfg.make_vars.items():
            build_cmd.append(f"{var}={val}")
        if cfg.target:
            build_cmd.append(cfg.target)
        if _run(build_cmd, cwd=cfg.dir).returncode:
            return None
        return (cfg.dir / cfg.target) if cfg.target else _last_executable(cfg.dir)

    raise ValueError(f"unknown build_system '{cfg.build_system}'")
