from __future__ import annotations

###############################################################################
# Standard library imports                                                    #
###############################################################################

import csv, json, math, random, tempfile, os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Iterable, Any, Union
from statistics import mean

###############################################################################
# Local imports                                                               #
###############################################################################

from src.config import Config, ParserConfig, BuildProject
from src.build import compile_project, compile_single_source, _run
from src.metrics import measure_likwid, measure_perf
from src.misc import unique_csv_path, clear_acpp_runtime_cache, is_significant_improvement

###############################################################################
# Type helpers                                                                #
###############################################################################

Number = float
MetricDict = Dict[str, Number]

# ===== Helpers (reuse wavefront ideas)

def _choose_objective(cfg: Config) -> Tuple[str, str]:
    if not cfg.objectives:
        raise ValueError("No objectives in config.")
    obj = cfg.objectives[0]
    return obj.metric, ("min" if obj.goal == "min" else "max")

def _score(value: float, goal: str) -> float:
    # Lower is better internally
    return value if goal == "min" else -value

def _canonical(flags: Iterable[str]) -> Tuple[str, ...]:
    return tuple(sorted(dict.fromkeys(flags)))

def _enumerate_env_schema(schema: Dict[str, Union[List[str], Dict[str, Any]]]) -> List[Dict[str, str]]:
    keys = list(schema.keys())
    out: List[Dict[str, str]] = []

    def rec(i: int, partial: Dict[str, str]):
        if i == len(keys):
            out.append(dict(partial))
            return
        var = keys[i]
        spec = schema[var]

        if isinstance(spec, list):
            for v in spec:
                partial[var] = str(v)
                rec(i + 1, partial)
            partial.pop(var, None)
            return

        if isinstance(spec, dict) and "values" in spec:
            pred = spec.get("when", {})
            if all(partial.get(k) == str(v) for k, v in pred.items()):
                for v in spec["values"]:
                    partial[var] = str(v)
                    rec(i + 1, partial)
                partial.pop(var, None)
            else:
                rec(i + 1, partial)
            return

        rec(i + 1, partial)

    rec(0, {})
    # dedup
    uniq = {tuple(sorted(d.items())): d for d in out}
    return list(uniq.values())

def _env_combos(cfg: Config, mode: str, cap: Optional[int], rng: random.Random, fixed: Dict[str, str] | None = None) -> List[Dict[str, str]]:
    mode = (mode or "product").lower()
    if mode == "fixed":
        return [dict(fixed or {})]

    schema = getattr(cfg, "env", {}) or {}
    combos = _enumerate_env_schema(schema)
    if not combos:
        return [{}]

    if not cap or cap <= 0 or len(combos) <= cap:
        return combos

    if mode == "product":
        rng.shuffle(combos)
        return combos[:cap]
    if mode == "sample":
        return rng.sample(combos, cap)
    return combos

def _compile_and_measure(cfg: Config, flags: Sequence[str], env: Dict[str, str], work: Path) -> Tuple[float, MetricDict, str]:
    work.mkdir(parents=True, exist_ok=True)
    flags_str = " ".join(flags)
    print(f"[anneal] using flags: {flags_str} env={json.dumps(env)}")

    if cfg.source:
        binary = compile_single_source(cfg.compiler, cfg.source, flags_str, work / "a.out")
    else:
        binary = compile_project(cfg.project, cfg.compiler, flags_str, work)
    if not binary:
        raise RuntimeError("build failed")

    # Measure via selected backend
    if cfg.backend == "perf":
        metrics = measure_perf(cfg.perf, binary, cfg.program_args, env, cfg.runs)  # type: ignore[arg-type]
    elif cfg.backend == "parser":
        metrics = _measure_parser_sycl_sa(cfg.parser, Path(binary), cfg.program_args, env, cfg.runs, work, cfg.project)
    else:
        metrics = measure_likwid(cfg.likwid, binary, cfg.program_args, env, cfg.runs)  # type: ignore[arg-type]

    # Clear ACPP runtime cache after each run (only relevant for acpp, harmless otherwise)
    try:
        clear_acpp_runtime_cache()
    except Exception:
        pass

    metric_name, goal = _choose_objective(cfg)
    if metric_name not in metrics:
        raise RuntimeError(f"objective metric '{metric_name}' missing; got: {list(metrics.keys())}")
    return float(metrics[metric_name]), metrics, str(binary)

# Minimal copy of your parser flow (matches wavefront), but local name
import re
_SYCL_RE = re.compile(
    r'^\[SYCL\]\[(?P<label>avg|sum)\]\s*kernel\s*(?P<kid>\d+)\s*:\s*'
    r'(?P<val>[0-9]*\.?[0-9]+)\s*s\s*over\s*(?P<iters>\d+)\s*iters\s*$',
    re.IGNORECASE | re.MULTILINE
)

def _wf_resolve_cwd(run_cwd: str, bin_path: Path, workdir: Optional[Path], project: Optional[BuildProject]) -> Path:
    if run_cwd == "workdir" and workdir:
        return workdir
    if run_cwd == "project_dir" and project:
        return project.dir
    return bin_path.parent

def _measure_parser_sycl_sa(
    pcfg: ParserConfig,
    bin_path: Path,
    prog_args: List[str],
    env: Dict[str, str],
    runs: int,
    workdir: Optional[Path] = None,
    project: Optional[BuildProject] = None,
) -> Dict[str, float]:
    merged_env = {**os.environ, **env}
    cmd: List[str] = []
    if pcfg.prefix: cmd.extend(pcfg.prefix)
    if pcfg.core_list: cmd.extend(["taskset", "-c", pcfg.core_list])
    cmd.append(str(bin_path))
    cmd.extend(prog_args)

    cwd = _wf_resolve_cwd(getattr(pcfg, "run_cwd", "binary_dir"), Path(bin_path), workdir, project)
    want_label = (pcfg.label or "avg").lower()
    total_runs = int(getattr(pcfg, "warmup_runs", 0)) + max(1, runs)
    warmup_cut = int(getattr(pcfg, "warmup_runs", 0))

    per_runs: List[Dict[int, float]] = []
    iters_seen: List[int] = []

    for i in range(total_runs):
        proc = _run(cmd, cwd=cwd, env=merged_env)
        if proc.returncode != 0:
            raise RuntimeError(f"program exited with rc={proc.returncode}")
        if i < warmup_cut:
            continue

        text = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")
        per_kernel: Dict[int, float] = {}
        itv: Optional[int] = None
        for m in _SYCL_RE.finditer(text):
            if m.group("label").lower() != want_label:
                continue
            per_kernel[int(m.group("kid"))] = float(m.group("val"))
            itv = int(m.group("iters"))
        if not per_kernel:
            raise RuntimeError("Parser backend (SYCL): no matching [SYCL] lines found.")
        per_runs.append(per_kernel)
        iters_seen.append(itv if itv is not None else -1)

    all_kids = sorted({k for d in per_runs for k in d})
    per_kernel_avg = {k: float(mean([d[k] for d in per_runs if k in d])) for k in all_kids}
    agg = getattr(pcfg, "aggregate", "sum").lower()
    if agg == "mean": agg_val = float(mean(per_kernel_avg.values()))
    elif agg == "min": agg_val = float(min(per_kernel_avg.values()))
    elif agg == "max": agg_val = float(max(per_kernel_avg.values()))
    else: agg_val = float(sum(per_kernel_avg.values()))

    mets: Dict[str, float] = {f"sycl_kernel_{k}_{want_label}_s": v for k, v in per_kernel_avg.items()}
    mets[f"sycl_{want_label}_{getattr(pcfg, 'aggregate', 'sum')}_s"] = agg_val
    if iters_seen and all(i == iters_seen[0] and i >= 0 for i in iters_seen):
        mets["sycl_iters"] = float(iters_seen[0])
    return mets

# ===== Anneal parameters holder
@dataclass
class _AnnealParams:
    T0: float = 1.0
    alpha: float = 0.95
    max_iters: int = 200
    max_no_improve: int = 30
    neighbor_mode: str = "mix"     # "add" | "remove" | "swap" | "mix"
    env_mode: str = "product"      # "fixed" | "product" | "sample"
    env_cap: Optional[int] = 8
    results_csv: Optional[str] = None

def _params_from_cfg(cfg: Config) -> _AnnealParams:
    p = _AnnealParams()
    an = getattr(cfg, "anneal", None)
    if not an: return p
    if isinstance(an, dict):
        for k in p.__dataclass_fields__.keys():
            if k in an and an[k] is not None:
                setattr(p, k, an[k])
    else:
        for k in p.__dataclass_fields__.keys():
            if hasattr(an, k):
                setattr(p, k, getattr(an, k))
    return p

# ===== Neighborhood moves

def _neighbors(current: Tuple[str, ...], atoms: List[str], mode: str, rng: random.Random) -> Iterable[Tuple[str, ...]]:
    """Generate neighboring flag sets."""
    s = set(current)
    available = [a for a in atoms if a not in s]
    present = list(s)

    if mode in ("add", "mix"):
        rng.shuffle(available)
        for a in available:
            yield _canonical(list(current) + [a])

    if mode in ("remove", "mix"):
        rng.shuffle(present)
        for a in present:
            nxt = list(current)
            nxt.remove(a)
            yield _canonical(nxt)

    if mode in ("swap", "mix"):
        rng.shuffle(present)
        rng.shuffle(available)
        lim = min(len(present), len(available))
        for i in range(lim):
            a_out = present[i]
            a_in  = available[i]
            nxt = list(current)
            nxt.remove(a_out)
            nxt.append(a_in)
            yield _canonical(nxt)

# ===== Main study

def run_anneal_study(cfg: Config) -> None:
    params = _params_from_cfg(cfg)
    rng = random.Random(getattr(getattr(cfg, "search", object()), "random_seed", None))

    # Flag universe
    atoms: List[str] = []
    wf = getattr(cfg, "wavefront", None)
    if wf and getattr(wf, "flag_atoms", None):
        atoms = list(dict.fromkeys(wf.flag_atoms))
    else:
        pool = getattr(cfg, "compiler_flag_pool", None) or []
        atoms = list(dict.fromkeys(list(pool)))
    rng.shuffle(atoms)

    base_flags = []
    if wf and getattr(wf, "base_flags", None):
        base_flags = list(wf.base_flags)

    metric_name, goal = _choose_objective(cfg)
    sig = getattr(cfg, "significance", {}) or {}
    MIN_REL = float(sig.get("min_rel_gain", 0.15))
    MIN_ABS = sig.get("min_abs_gain", None)
    no_improve = 0

    # Env combos to consider each evaluation
    envs = _env_combos(cfg, params.env_mode, params.env_cap, rng, fixed=(getattr(getattr(cfg, "wavefront", object()), "env", None)))

    workroot = Path(tempfile.mkdtemp(prefix="SCOuT_anneal_"))
    print(f"[anneal] workdir: {workroot}")
    print(f"[anneal] atoms={len(atoms)} T0={params.T0} alpha={params.alpha} max_iters={params.max_iters}")
    print(f"[anneal] env_mode={params.env_mode} env_combos={len(envs)}")

    # Start from baseline (no atoms, only base_flags)
    cur_flags = tuple(base_flags)
    def eval_flags(flags_tuple: Tuple[str, ...]) -> Tuple[float, Dict[str, float], str, Dict[str, str]]:
        best_val = math.inf
        best = (math.inf, {}, "", {})
        for i, env in enumerate(envs, 1):
            run_dir = workroot / f"iter_tmp" / f"env{i:03d}"
            try:
                v, m, b = _compile_and_measure(cfg, list(flags_tuple), env, run_dir)
            except Exception as exc:
                v, m, b = (math.inf, {"error": str(exc)}, "")
            sc = _score(v, goal)
            if sc < _score(best_val, goal):
                best_val = v
                best = (v, m, b, env)
        return best

    cur_val, cur_mets, cur_bin, cur_env = eval_flags(cur_flags)
    best_val, best_mets, best_bin, best_flags, best_env = cur_val, cur_mets, cur_bin, cur_flags, cur_env

    # CSV buffer (identical to Optuna header)
    rows: List[Tuple[List[float], str, Dict[str, str], str, Dict[str, float]]] = []
    extra_metric_keys: set[str] = set(cur_mets.keys())
    def _flags_key(t: Tuple[str, ...]) -> str:
        return "|".join(list(t)) if t else "default"

    rows.append(([float(cur_val)], _flags_key(cur_flags), dict(cur_env), str(cur_bin), cur_mets))

    T = float(max(params.T0, 1e-12))
    no_improve = 0

    for it in range(1, int(params.max_iters) + 1):
        # Propose a neighbor
        neigh_iter = list(_neighbors(cur_flags, atoms, params.neighbor_mode, rng))
        if not neigh_iter:
            print("[anneal] no neighbors; stopping.")
            break
        rng.shuffle(neigh_iter)
        cand_flags = neigh_iter[0]

        cand_val, cand_mets, cand_bin, cand_env = eval_flags(cand_flags)

        # Log row
        extra_metric_keys.update(cand_mets.keys())
        rows.append(([float(cand_val)], _flags_key(cand_flags), dict(cand_env), str(cand_bin), cand_mets))

        # Acceptance
        cur_score = _score(cur_val, goal)
        new_score = _score(cand_val, goal)
        accept = False
        if new_score < cur_score:
            accept = True
        else:
            # SA probability
            delta = new_score - cur_score   # >0 means worse
            p = math.exp(-delta / max(T, 1e-12))
            accept = (rng.random() < p)

        if accept:
            cur_flags, cur_val, cur_mets, cur_bin, cur_env = cand_flags, cand_val, cand_mets, cand_bin, cand_env

        if is_significant_improvement(best_val, cur_val, goal, MIN_REL, MIN_ABS):
            best_val, best_flags, best_env, best_mets, best_bin = cur_val, cur_flags, cur_env, cur_mets, cur_bin
            no_improve = 0
        else:
            no_improve += 1

        '''
        # Track best
        if _score(cur_val, goal) < _score(best_val, goal):
            best_val, best_mets, best_bin, best_flags, best_env = cur_val, cur_mets, cur_bin, cur_flags, cur_env
            no_improve = 0
        else:
            no_improve += 1
        '''
        
        # Cool
        T *= float(params.alpha)

        # Early stop
        if no_improve >= int(params.max_no_improve):
            print("[anneal] early stop: no improvement plateau.")
            break

        print(f"[anneal] iter={it} T={T:.4g} cur={cur_val:.6g} best={best_val:.6g} flags={list(best_flags)}")

    # ---- Write CSV (identical schema to Optuna) ----
    if getattr(cfg, "csv_log", None) and not params.results_csv:
        results_path = unique_csv_path(cfg.csv_log)
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
    else:
        results_path = Path(params.results_csv) if params.results_csv else (workroot / "anneal_results.csv")

    obj_headers = [o.metric for o in cfg.objectives]
    extra_cols = sorted(extra_metric_keys)
    header = obj_headers + ["compiler_flags", "env", "binary"] + extra_cols

    with open(results_path, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(header)
        for obj_vals, flags_key, env_row, binary_path, metrics in rows:
            if not obj_vals or math.isinf(obj_vals[0]):
                continue
            row = list(obj_vals) + [flags_key, json.dumps(env_row), binary_path]
            row += [metrics.get(k, "") for k in extra_cols]
            w.writerow(row)

    print("\n[anneal] ===== Summary =====")
    print(f"best: {metric_name}={best_val:.6g} flags={list(best_flags)} env={json.dumps(best_env)}")
    print(f"[anneal] results → {results_path}")
