from __future__ import annotations

import csv
import json
import math
import random
import tempfile
from dataclasses import dataclass, field
from itertools import combinations, permutations
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Iterable

from src.config import Config
from src.build import compile_project, compile_single_source
from src.metrics import measure_likwid, measure_perf

Number = float
MetricDict = Dict[str, Number]


@dataclass
class _WFParams:
    # Flag atoms to combine; if None, we will try to infer from config.
    flag_atoms: Optional[List[str]] = None
    # Flags that are always present (baseline)
    base_flags: List[str]
    # Max combination size (k). Beware C(n,k) growth.
    max_k: int = 3
    # Search flavor:
    #  - "full": evaluate ALL C(n,k) at each level k
    #  - "beam": expand only the top-N combos from previous level
    mode: str = "beam"
    beam_width: int = 16
    # Optional cap on the number of candidates per wave (after generation); None = no cap
    per_wave_cap: Optional[int] = None
    # Stop early if no improvement vs best-so-far by at least eps
    stop_if_no_improve: bool = True
    improvement_eps: float = 0.0
    # Environment for all runs in this study (constant)
    env: Dict[str, str] = field(default_factory=dict)
    # Output
    results_csv: str = "wavefront_results.csv"


def _params_from_cfg(cfg: Config) -> _WFParams:
    wf = getattr(cfg, "wavefront", None)
    p = _WFParams()
    if wf:
        for k in _WFParams().__dict__.keys():
            if hasattr(wf, k):
                setattr(p, k, getattr(wf, k))
    return p


def _choose_objective(cfg: Config) -> Tuple[str, str]:
    """Return (metric_name, goal) with goal in {'min','max'}; use first objective."""
    if not cfg.objectives:
        raise ValueError("No objectives in config.")
    obj = cfg.objectives[0]
    return obj.metric, ("min" if obj.goal == "min" else "max")


def _collect_atoms(cfg: Config, p: _WFParams) -> List[str]:
    if p.flag_atoms:
        atoms = list(dict.fromkeys(p.flag_atoms))
    else:
        atoms: List[str] = []
        pool = getattr(cfg, "compiler_flag_pool", None)
        if pool: atoms.extend(list(pool))
        cf = getattr(cfg, "compiler_flags", None)
        if isinstance(cf, dict): atoms.extend(list(cf.values()))
        atoms = list(dict.fromkeys(atoms))
    if not atoms:
        raise ValueError("wavefront: no flag atoms. Provide wavefront.flag_atoms or populate compiler_flag_pool/compiler_flags.")
    return atoms


def _compile_and_measure(cfg: Config, flags: Sequence[str], env: Dict[str, str], work: Path) -> Tuple[float, MetricDict, str]:
    work.mkdir(parents=True, exist_ok=True)
    if cfg.source:
        binary = compile_single_source(cfg.compiler, cfg.source, list(flags), work / "a.out", None)
    else:
        binary = compile_project(cfg.project, cfg.compiler, list(flags), work, None)
    if not binary:
        raise RuntimeError("build failed")
    if cfg.backend == "perf":
        metrics = measure_perf(cfg.perf, binary, cfg.program_args, env)  # type: ignore[arg-type]
    else:
        metrics = measure_likwid(cfg.likwid, binary, cfg.program_args, env, cfg.runs)  # type: ignore[arg-type]
    return float(metrics[_choose_objective(cfg)[0]]), metrics, str(binary)


def _score(value: float, goal: str) -> float:
    """Lower is always better for internal comparisons."""
    return value if goal == "min" else -value


def _canonical(flags: Iterable[str]) -> Tuple[str, ...]:
    """Order-insensitive canonical tuple (sorted unique) to dedup combos."""
    # Keep deterministic ordering by string sort; adapt if flag order matters in your toolchain.
    return tuple(sorted(dict.fromkeys(flags)))


def _generate_full(atoms: List[str], k: int) -> Iterable[Tuple[str, ...]]:
    for combo in combinations(atoms, k):
        yield _canonical(combo)


def _generate_beam(expand_from: List[Tuple[str, ...]], atoms: List[str]) -> Iterable[Tuple[str, ...]]:
    # Expand each previous combo by appending a new atom not already present,
    # ensuring lexicographic growth to avoid duplicates.
    index = {a: i for i, a in enumerate(atoms)}
    seen: set[Tuple[str, ...]] = set()
    for combo in expand_from:
        if not combo:
            start = 0
        else:
            start = index[combo[-1]] + 1
        for j in range(start, len(atoms)):
            a = atoms[j]
            if a in combo:  # should not happen with our start rule
                continue
            newc = _canonical(combo + (a,))
            if newc not in seen:
                seen.add(newc)
                yield newc


def run_wavefront_study(cfg: Config) -> None:
    """
    Wave-front search over flag atoms:
      k=0: baseline (base_flags)
      k=1: all singles
      k=2: all pairs (or beam expansions)
      ...
    Logs CSV and prints best per wave.
    """
    params = _params_from_cfg(cfg)
    atoms = _collect_atoms(cfg, params)
    metric_name, goal = _choose_objective(cfg)

    rng = random.Random(getattr(getattr(cfg, "search", object()), "random_seed", None))
    workroot = Path(tempfile.mkdtemp(prefix="SCOuT_wave_"))
    print(f"[wavefront] workdir: {workroot}")
    print(f"[wavefront] atoms={len(atoms)} max_k={params.max_k} mode={params.mode} beam_width={params.beam_width}")
    if params.per_wave_cap:
        print(f"[wavefront] per_wave_cap={params.per_wave_cap}")

    base_env: Dict[str, str] = dict(getattr(getattr(cfg, "wavefront", object()), "env", {}) or {})
    base_flags = list(params.base_flags or [])

    # Baseline
    print("[wavefront] evaluating baseline …")
    base_dir = workroot / "k00_baseline"
    base_val, base_metrics, base_bin = _compile_and_measure(cfg, base_flags, base_env, base_dir)
    best_global_score = _score(base_val, goal)
    best_global_combo: Tuple[str, ...] = tuple()
    print(f"[wavefront] baseline {metric_name} = {base_val:.6g}")

    # Prepare CSV
    results_path = workroot / params.results_csv
    with open(results_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["k", "value", "flags", "metrics_json", "binary"])

        # log baseline as k=0
        w.writerow([0, base_val, json.dumps(base_flags), json.dumps(base_metrics, sort_keys=True), base_bin])
        fp.flush()

        # Wave k >= 1
        prev_top: List[Tuple[str, ...]] = []
        improved_any = False
        for k in range(1, params.max_k + 1):
            print(f"[wavefront] ===== Wave k={k} =====")

            # Generate candidates
            if params.mode == "full" or (k == 1 and not prev_top):
                gen_iter = _generate_full(atoms, k)
            else:
                gen_iter = _generate_beam(prev_top, atoms)
            candidates = list(gen_iter)
            if params.per_wave_cap and len(candidates) > params.per_wave_cap:
                rng.shuffle(candidates)
                candidates = candidates[: params.per_wave_cap]
                candidates.sort()  # keep deterministic order for prints

            print(f"[wavefront] candidates={len(candidates)}")
            if not candidates:
                print("[wavefront] no candidates; stop.")
                break

            # Evaluate all candidates for this wave
            scored: List[Tuple[float, float, Tuple[str, ...], MetricDict, str]] = []
            # tuple: (score, value, combo, metrics, binary_path)

            for idx, combo in enumerate(candidates, 1):
                flags = base_flags + list(combo)
                run_dir = workroot / f"k{k:02d}" / f"c{idx:05d}"
                try:
                    value, metrics, binary = _compile_and_measure(cfg, flags, base_env, run_dir)
                    sc = _score(value, goal)
                except Exception as exc:
                    # Mark failed combos with +inf; still log for traceability.
                    value, sc, metrics, binary = (math.inf, math.inf, {"error": str(exc)}, "")
                scored.append((sc, value, combo, metrics, binary))
                # stream log row
                w.writerow([k, value, json.dumps(flags), json.dumps(metrics, sort_keys=True), binary])
                if idx % 25 == 0:
                    fp.flush()

            # Sort by score (lower is better)
            scored.sort(key=lambda t: t[0])
            best_sc, best_val, best_combo, best_metrics, best_bin = scored[0]
            print(f"[wavefront] best@k={k}: {metric_name}={best_val:.6g}  flags={list(best_combo)}")

            # Prepare next-wave seeds (beam)
            if params.mode == "beam":
                prev_top = [c for _sc, _val, c, _m, _b in scored[: params.beam_width]]
            else:
                prev_top = []  # not used in 'full' mode

            # Early stop if not improving globally
            if best_sc + params.improvement_eps < best_global_score:
                best_global_score = best_sc
                best_global_combo = best_combo
                improved_any = True
            else:
                if params.stop_if_no_improve:
                    print("[wavefront] no global improvement; stopping early.")
                    break

        print("\n[wavefront] ===== Summary =====")
        print(f"baseline: {metric_name}={base_val:.6g} flags={base_flags}")
        if improved_any:
            print(f"best:     {metric_name}={(-best_global_score if goal=='max' else best_global_score):.6g} "
                  f"flags={base_flags + list(best_global_combo)}")
        print(f"[wavefront] results → {results_path}")
