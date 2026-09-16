from __future__ import annotations

###############################################################################
# Standard library imports                                                    #
###############################################################################

import os
import re
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
    with open("./scout.log", "a") as log_file:
        log_file.write(f"[exec] {pretty}" + (f"  (cwd={cwd})" if cwd else "") + "\n")
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

# Progress line patterns: Ninja prints "[N/M] …", CMake Makefiles prints "[ NN%] …".
_NINJA_RE = re.compile(r"^\[(\d+)\s*/\s*(\d+)\]")
_CMAKE_RE = re.compile(r"^\[\s*(\d{1,3})\s*%\]")


def _progress_pct(line: str) -> Optional[int]:
    """Return the integer progress percentage encoded in a build line, else None."""
    m = _NINJA_RE.match(line)
    if m:
        done, total = int(m.group(1)), int(m.group(2))
        return round(100 * done / total) if total else None
    m = _CMAKE_RE.match(line)
    if m:
        return int(m.group(1))
    return None


def _run_stream(cmd: Sequence[str] | str, *, cwd: Path | None = None, env: EnvMap | None = None) -> subprocess.CompletedProcess:
    """
    Run a command, streaming its output to the console with live progress
    updates, while capturing everything for later logging.

    stdout and stderr are merged (stderr=STDOUT). Progress is detected from
    Ninja ('[N/M]') or CMake ('[ NN%]') lines and echoed once per whole-percent
    step. Returns a CompletedProcess whose .stdout holds the merged output and
    whose .stderr is empty.
    """
    pretty = " ".join(shlex.quote(str(c)) for c in cmd) if isinstance(cmd, Sequence) else cmd
    print(f"[exec] {pretty}" + (f"  (cwd={cwd})" if cwd else ""))
    print("[build] starting …")
    with open("./scout.log", "a") as log_file:
        log_file.write(f"[exec] {pretty}" + (f"  (cwd={cwd})" if cwd else "") + "\n")
        log_file.write("[build] starting …\n")

    proc = subprocess.Popen(
        cmd,
        shell=isinstance(cmd, str),
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    lines: List[str] = []
    last_pct: Optional[int] = None
    assert proc.stdout is not None
    for line in proc.stdout:
        lines.append(line)
        pct = _progress_pct(line)
        if pct is not None and pct != last_pct:
            last_pct = pct
            print(f"[build] {pct:3d}%  {line.strip()}")
            with open("./scout.log", "a") as log_file:
                log_file.write(f"[build] {pct:3d}%  {line.strip()}\n")

    rc = proc.wait()
    print(f"[build] finished (rc={rc})")
    with open("./scout.log", "a") as log_file:
        log_file.write(f"[build] finished (rc={rc})\n")
    return subprocess.CompletedProcess(cmd, rc, stdout="".join(lines), stderr="")

def _trial_tag(trial: Optional["optuna.Trial"]) -> str:
    return f"trial_{trial.number:05d}" if trial is not None else f"phaseB_{uuid.uuid4().hex[:8]}"

def _save_log(workdir: Path,
              trial: Optional["optuna.Trial"],
              step: str,
              proc) -> Path:
    """
    Save stdout/stderr of a subprocess to workdir/logs and return the log dir.
    Works even when trial is None (e.g., Phase-B rebuilds).
    """
    log_dir = Path(workdir) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    tag = _trial_tag(trial)
    (log_dir / f"{tag}_{step}.out").write_text(proc.stdout or "")
    (log_dir / f"{tag}_{step}.err").write_text(proc.stderr or "")
    return log_dir


def _report_build_failure(step: str, proc, log_dir: Path) -> None:
    """Print a concise terminal report for a failed build step."""
    print(f"[build] ✗ {step} FAILED (rc={proc.returncode})")
    print(f"[build] logs → {log_dir}")
    text = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
    tail = text.strip().splitlines()[-20:]
    if tail:
        print(f"[build] {step} — last output:")
        for ln in tail:
            print(f"  {ln}")

###############################################################################
# Build logic (identical to original)                                         #
###############################################################################

def compile_single_source(compiler: str, src: Path, flags: str, out: Path, trial: Optional[optuna.Trial] = None) -> Optional[Path]:
    cmd = f"{compiler} {flags} {shlex.quote(str(src))} -o {shlex.quote(str(out))}"
    proc = _run(cmd)
    if proc.returncode:
        log_dir = _save_log(out, trial, "make", proc)
        _report_build_failure("make", proc, log_dir)
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


def _resolve_cmake_target_executable(build_dir: Path, target: str) -> Optional[Path]:
    # Common CMake layout: executable under build root or build/bin.
    direct = build_dir / target
    if direct.is_file() and os.access(direct, os.X_OK):
        return direct

    in_bin = build_dir / "bin" / target
    if in_bin.is_file() and os.access(in_bin, os.X_OK):
        return in_bin

    # Fallback: search exact filename under the build tree.
    matches: List[Path] = []
    for p in build_dir.rglob(target):
        if p.is_file() and os.access(p, os.X_OK):
            matches.append(p)
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


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
        
        if flags and cfg.cmake_flag_vars:
            for var in cfg.cmake_flag_vars:
                cmake_cmd += [f"-D{var}={flags}"]
        
        if cfg.cmake_defs:
            cmake_cmd +=[f"-D{d}" for d in defs]

        proc = _run(cmake_cmd)
        if proc.returncode:
            log_dir = _save_log(workdir, trial, "cmake_config", proc)
            _report_build_failure("cmake_config", proc, log_dir)
            return None

        build_cmd = ["cmake", "--build", str(build_dir)]
        if cfg.build_jobs and cfg.build_jobs > 0:
            build_cmd += ["--parallel", str(cfg.build_jobs)]
        else:
            build_cmd += ["--parallel"]
        if cfg.target:
            build_cmd += ["--target", cfg.target]
        proc = _run_stream(build_cmd)
        if proc.returncode:
            log_dir = _save_log(workdir, trial, "cmake_build", proc)
            _report_build_failure("cmake_build", proc, log_dir)
            return None
        
        if cfg.target:
            resolved = _resolve_cmake_target_executable(build_dir, cfg.target)
            return resolved
        return _last_executable(build_dir)

    if cfg.build_system == "make":
        _run(["make", "clean"], cwd=cfg.dir)
        build_cmd = ["make", f"CXX={compiler}"]
        if cfg.build_jobs and cfg.build_jobs > 0:
            build_cmd.append(f"-j{cfg.build_jobs}")
        else:
            build_cmd.append("-j")

        if flags:
            build_cmd.append(f"{cfg.make_flags_var}+={flags}")
        for var, val in cfg.make_vars.items():
            build_cmd.append(f"{var}={val}")
        if cfg.target:
            build_cmd.append(cfg.target)

        proc = _run(build_cmd, cwd=cfg.dir)
        if proc.returncode:
            log_dir = _save_log(workdir, trial, "make", proc)
            _report_build_failure("make", proc, log_dir)
            return None
        
        return (cfg.dir / cfg.target) if cfg.target else _last_executable(cfg.dir)

    raise ValueError(f"unknown build_system '{cfg.build_system}'")
