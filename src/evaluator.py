from __future__ import annotations

import hashlib
import json
import shlex
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Sequence

from src.build import compile_project, compile_single_source
from src.config import Config
from src.metrics import measure_likwid, measure_parser_sycl, measure_perf


MetricDict = Dict[str, float]


@dataclass(frozen=True)
class Evaluation:
    metrics: MetricDict
    binary: Path
    cached: bool = False


class Evaluator:
    """Shared build and measurement service with separate build/evaluation caches."""

    def __init__(self, cfg: Config, workroot: Path, *, cache_evaluations: bool = True):
        self.cfg = cfg
        self.workroot = Path(workroot)
        self.workroot.mkdir(parents=True, exist_ok=True)
        self.cache_evaluations = cache_evaluations
        self._build_cache: Dict[str, Path] = {}
        self._build_failures: Dict[str, str] = {}
        self._evaluation_cache: Dict[tuple[str, tuple[tuple[str, str], ...]], Evaluation] = {}

    @staticmethod
    def normalize_flags(flags: str | Sequence[str]) -> str:
        if isinstance(flags, str):
            return shlex.join(shlex.split(flags)) if flags.strip() else ""
        tokens: list[str] = []
        for fragment in flags:
            tokens.extend(shlex.split(str(fragment)))
        return shlex.join(tokens) if tokens else ""

    def compile_key(self, flags: str | Sequence[str]) -> str:
        normalized = self.normalize_flags(flags)
        project = self.cfg.project
        payload = {
            "compiler": self.cfg.compiler,
            "flags": normalized,
            "source": str(self.cfg.source) if self.cfg.source else None,
            "project": (
                {
                    "dir": str(project.dir),
                    "build_system": project.build_system,
                    "target": project.target,
                    "executable": str(project.executable) if project.executable else None,
                    "make_vars": project.make_vars,
                    "make_flags_var": project.make_flags_var,
                    "cmake_defs": project.cmake_defs,
                    "cmake_flag_vars": project.cmake_flag_vars,
                }
                if project
                else None
            ),
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

    def get_or_build(self, flags: str | Sequence[str]) -> tuple[str, Path]:
        key = self.compile_key(flags)
        cached = self._build_cache.get(key)
        if cached is not None and cached.is_file():
            return key, cached
        if key in self._build_failures:
            raise RuntimeError(self._build_failures[key])

        build_dir = self.workroot / "builds" / key[:16]
        build_dir.mkdir(parents=True, exist_ok=True)
        normalized = self.normalize_flags(flags)
        # normalize_flags uses shell quoting for stable keys; compiler APIs expect a
        # normal shell fragment and tokenize it safely themselves.
        flags_str = " ".join(shlex.quote(token) for token in shlex.split(normalized))
        if self.cfg.source:
            binary = compile_single_source(self.cfg.compiler, self.cfg.source, flags_str, build_dir / "program")
        elif self.cfg.project:
            binary = compile_project(self.cfg.project, self.cfg.compiler, flags_str, build_dir)
        else:  # Config validation should make this unreachable.
            raise RuntimeError("no source or project configured")
        if not binary:
            message = f"build failed; logs are under {build_dir / 'logs'}"
            self._build_failures[key] = message
            raise RuntimeError(message)

        binary = Path(binary)
        artifact = build_dir / "artifact" / binary.name
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if binary.resolve() != artifact.resolve():
            shutil.copy2(binary, artifact)
            artifact.chmod(artifact.stat().st_mode | 0o111)
        else:
            artifact = binary
        self._build_cache[key] = artifact
        return key, artifact

    def evaluate(self, flags: str | Sequence[str], env: Dict[str, str], run_workdir: Path) -> Evaluation:
        compile_key, binary = self.get_or_build(flags)
        env_key = tuple(sorted((str(k), str(v)) for k, v in env.items()))
        key = (compile_key, env_key)
        cached = self._evaluation_cache.get(key)
        if self.cache_evaluations and cached is not None:
            return Evaluation(dict(cached.metrics), cached.binary, True)

        run_workdir = Path(run_workdir)
        run_workdir.mkdir(parents=True, exist_ok=True)
        managed_keys = tuple((self.cfg.env or {}).keys())
        clear_runtime = self.cfg.runtime_cache_policy == "cold"

        if self.cfg.backend == "perf":
            metrics = measure_perf(
                self.cfg.perf, binary, self.cfg.program_args, env, self.cfg.runs,
                clear_runtime_cache=clear_runtime, managed_env_keys=managed_keys,
            )
        elif self.cfg.backend == "parser":
            metrics = measure_parser_sycl(
                self.cfg.parser, binary, self.cfg.program_args, env, self.cfg.runs,
                run_workdir, self.cfg.project,
                clear_runtime_cache=clear_runtime, managed_env_keys=managed_keys,
            )
        else:
            metrics = measure_likwid(
                self.cfg.likwid, binary, self.cfg.program_args, env, self.cfg.runs,
                clear_runtime_cache=clear_runtime, managed_env_keys=managed_keys,
            )

        result = Evaluation({str(k): float(v) for k, v in metrics.items()}, binary)
        if self.cache_evaluations:
            self._evaluation_cache[key] = result
        return result

    @property
    def build_count(self) -> int:
        return len(self._build_cache)
