from __future__ import annotations

###############################################################################
# Standard library imports                                                    #
###############################################################################

import os
import re
import sys, subprocess
from pathlib import Path
from statistics import mean, variance, median
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

from src.config import PerfConfig, MetricSpec, LikwidConfig, ParserConfig, BuildProject
from src.build import _run
from src.misc import clear_acpp_runtime_cache

###############################################################################
# Measurement helpers (perf & likwid)                                         #
###############################################################################

_PERF_LINE_RE = re.compile(r"^\s*([0-9,]+)\s+([^\s#]+)")


def perf_parse(stderr: str, events: Sequence[str]) -> MetricDict:
    accum: Dict[str, Number] = {}
    for line in stderr.splitlines():
        match = _PERF_LINE_RE.match(line)
        if not match:
            continue
        raw_value, raw_event = match.groups()
        try:
            count = float(raw_value.replace(",", ""))
        except ValueError:
            continue
        base_event = raw_event.strip("/").split("/")[-1]
        if base_event not in events:
            continue
        accum[base_event] = accum.get(base_event, 0.0) + count
    if {"cycles", "instructions"}.issubset(accum) and accum["instructions"]:
        accum["CPI"] = accum["cycles"] / accum["instructions"]
    return accum


def measure_perf(cfg: PerfConfig, bin_path: Path, prog_args: List[str], env: EnvMap, runs: int = 1) -> MetricDict:
    meas_runs = max(1, runs)
    total_runs = cfg.warmup_runs + meas_runs
    buckets: Dict[str, List[Number]] = {e: [] for e in cfg.events + ["CPI"]}

    for i in range(total_runs):
        cmd = ["perf", "stat", "-e", ",".join(cfg.events), "--", str(bin_path), *prog_args]
        if cfg.core_list:
            cmd = ["taskset", "-c", cfg.core_list, *cmd]
        proc = _run(cmd, env={**os.environ, **env})

        # Parse always, but only store if not warm-up
        data = perf_parse(proc.stderr, cfg.events)
        if i < cfg.warmup_runs:
            continue
        if not data:
            raise RuntimeError("Perf parse failure – received no matching events.")
        for k, v_list in buckets.items():
            if k in data:
                v_list.append(data[k])
    
    clear_acpp_runtime_cache()
    return {k: mean(v) for k, v in buckets.items() if v}



_ROW_RE   = re.compile(r"^\|\s*([^|]+?)\s*\|(.+)$")
_SEP_RE   = re.compile(r"(?<=\d)[.'\u202F](?=\d{3}\b)")  # 1.234.567 or 1'234'567
_DEC_COMMA = re.compile(r"^(\d+),(\d+)$")  
_LIKWID_ROW_RE = re.compile(r"^\|\s*([^|]+?)\s*\|(.+)$")

def _parse_num(text: str) -> float | None:
    """Parse tolerant float or return None."""
    t = text.strip()
    t = _SEP_RE.sub("", t)              # 1.234.567 -> 1234567
    m = _DEC_COMMA.match(t)             # decimal comma?
    if m:
        t = f"{m.group(1)}.{m.group(2)}"
    try:
        return float(t)
    except ValueError:
        return None


def likwid_parse(out: str, specs: Sequence[MetricSpec]) -> MetricDict:
    wanted = {s.name: s for s in specs}

    # Prepare buckets
    per_thread: Dict[str, List[Number]] = {s.name: [] for s in specs}
    stat_avg:   Dict[str, Number]       = {}

    # ── scan once ────────────────────────────────────────────────────
    for line in out.splitlines():
        m = _ROW_RE.match(line)
        if not m:                   # skip non-table lines
            continue
        name, cells_raw = m.group(1).strip(), m.group(2)

        # 1) STAT rows  → grab Avg column (index 3)
        if name.endswith("STAT"):
            base = name[:-4].rstrip()
            if base in wanted:
                cells = [c.strip() for c in cells_raw.split("|") if c.strip()]
                if len(cells) >= 4:
                    v = _parse_num(cells[3])
                    if v is not None:
                        stat_avg[base] = v
            continue

        # 2) per-thread rows
        if name in wanted:
            for cell in (c for c in cells_raw.split("|") if c.strip()):
                v = _parse_num(cell)
                if v is not None:
                    per_thread[name].append(v)

    # ── reduce according to spec ─────────────────────────────────────
    result: MetricDict = {}
    for spec in specs:
        values = per_thread[spec.name]

        # if we have STAT Avg and user asked for avg—use it (cheaper)
        if spec.agg == "avg" and spec.name in stat_avg:
            agg_val = stat_avg[spec.name]
        elif values:
            if   spec.agg == "avg": agg_val = mean(values)
            elif spec.agg == "max": agg_val = max(values)
            elif spec.agg == "min": agg_val = min(values)
            elif spec.agg == "median": agg_val = median(values)
            else:
                raise ValueError(f"Unknown agg mode '{spec.agg}'")
        else:
            continue  # metric missing

        result[spec.name] = agg_val

        if spec.var and len(values) > 1:
            result[f"{spec.name}_var"] = variance(values)

    return result


def measure_likwid(cfg: LikwidConfig, bin_path: Path, prog_args: List[str], env: EnvMap, runs: int = 1) -> MetricDict:
    specs   = cfg.metrics
    buckets: Dict[str, List[Number]] = {s.name: [] for s in specs}
    for s in specs:
        if s.var:
            buckets[f"{s.name}_var"] = []
    
    meas_runs = max(1, runs)
    total_runs = cfg.warmup_runs + meas_runs

    for i in range(total_runs):
        cmd = ["likwid-perfctr"]

        if cfg.core_list:
            cmd += ["-C", cfg.core_list]

        if cfg.group:
            cmd += ["-g", cfg.group]
        else:                       # raw events
            cmd += ["-g", ",".join(cfg.events)]
        cmd += [str(bin_path), *prog_args]

        proc = _run(cmd, env={**os.environ, **env})
        data = likwid_parse(proc.stdout, cfg.metrics)

        if i < cfg.warmup_runs:
            continue
        if not data:
            raise RuntimeError("LIKWID parse failure – no metrics captured.")
        for k, v in data.items():
            if k in buckets:
                buckets[k].append(v)

    clear_acpp_runtime_cache()
    return {k: mean(v) for k, v in buckets.items() if v}


###############################################################################
# Parser helpers (Parsers for HeCBench)                                       #
###############################################################################

# Matches: [SYCL][avg] kernel 2: 0.000664 s over 1000 iters
_SYCL_RE = re.compile(
    r'^\[SYCL\]\[(?P<label>avg|sum)\]\s*kernel\s*(?P<kid>\d+)\s*:\s*'
    r'(?P<val>[0-9]*\.?[0-9]+)\s*s\s*over\s*(?P<iters>\d+)\s*iters\s*$',
    re.IGNORECASE | re.MULTILINE
)

def _resolve_cwd(run_cwd: str, bin_path: Path, workdir: Optional[Path], project: Optional[BuildProject]) -> Path:
    if run_cwd == "workdir" and workdir: return workdir
    if run_cwd == "project_dir" and project: return project.dir
    return bin_path.parent

def _aggregate(vals: List[float], how: str) -> float:
    how = how.lower()
    if how == "sum":  return float(sum(vals))
    if how == "mean": return float(mean(vals))
    if how == "max":  return float(max(vals))
    if how == "min":  return float(min(vals))
    return float(sum(vals))

def measure_parser_sycl(
    cfg: ParserConfig,
    bin_path: Path,
    prog_args: List[str],
    env: EnvMap,
    runs: int,
    workdir: Optional[Path] = None,
    project: Optional[BuildProject] = None,
) -> MetricDict:
    """
    Runs the binary like perf/likwid (taskset/prefix, controlled cwd),
    parses lines printed as:
        [SYCL][avg] kernel K: <seconds> s over N iters
        [SYCL][sum] kernel K: <seconds> s over N iters
    Returns per-kernel metrics (seconds) and one aggregate key:
        sycl_<label>_<aggregate>_s
    """
    merged_env = {**os.environ, **env}
    cmd: List[str] = []
    if cfg.prefix:     cmd.extend(cfg.prefix)
    if cfg.core_list:  cmd.extend(["taskset", "-c", cfg.core_list])
    cmd.append(str(bin_path))
    cmd.extend(prog_args)
    cwd = _resolve_cwd(cfg.run_cwd, bin_path, workdir, project)

    # collect per-run kernel maps {kid -> value_s}
    runs_kernel_vals: List[Dict[int, float]] = []
    iterations_seen: List[int] = []

    meas_runs = max(1, runs)
    total_runs = cfg.warmup_runs + meas_runs

    for i in range(total_runs):
        proc = _run(cmd, cwd=cwd, env=merged_env)
        if proc.returncode != 0:
            raise RuntimeError(f"program exited with rc={proc.returncode}")

        # For warm-up iterations, don’t enforce parsing or errors
        is_warmup = i < cfg.warmup_runs
        text = (proc.stdout or "") + (("\n" + proc.stderr) if proc.stderr else "")

        per_kernel: Dict[int, float] = {}
        iters_val: Optional[int] = None

        for m in _SYCL_RE.finditer(text):
            label = m.group("label").lower()
            if label != cfg.label.lower():
                continue
            kid   = int(m.group("kid"))
            val   = float(m.group("val"))       # seconds
            iters = int(m.group("iters"))

            iters_val = iters
            per_kernel[kid] = val

        if is_warmup:
            continue

        if not per_kernel:
        
            logs = (workdir or cwd) / "parser_logs"
            logs.mkdir(parents=True, exist_ok=True)
            (logs / f"no_match_{i:02d}.out").write_text(proc.stdout or "")
            (logs / f"no_match_{i:02d}.err").write_text(proc.stderr or "")
            raise RuntimeError("Parser backend (SYCL): no matching [SYCL] lines found.")

        runs_kernel_vals.append(per_kernel)
        iterations_seen.append(iters_val if iters_val is not None else -1)

    # average across runs per kernel
    all_kids = sorted({k for d in runs_kernel_vals for k in d.keys()})
    # filter by cfg.kernels if provided
    if cfg.kernels:
        selected = [k for k in all_kids if k in set(cfg.kernels)]
    else:
        selected = all_kids

    per_kernel_avg: Dict[int, float] = {}
    for k in selected:
        vals = [d[k] for d in runs_kernel_vals if k in d]
        if vals:
            per_kernel_avg[k] = float(mean(vals))

    if not per_kernel_avg:
        raise RuntimeError("Parser backend (SYCL): selected kernels missing in output.")

    # aggregate across selected kernels
    aggregate_val = _aggregate(list(per_kernel_avg.values()), cfg.aggregate)

    # build metrics dict
    mets: MetricDict = {}
    # per-kernel seconds
    for k, v in per_kernel_avg.items():
        mets[f"sycl_kernel_{k}_{cfg.label}_s"] = v
    # aggregate seconds
    mets[f"sycl_{cfg.label}_{cfg.aggregate}_s"] = aggregate_val

    # if iterations consistent and positive, report it too
    if all(i == iterations_seen[0] and i >= 0 for i in iterations_seen):
        mets["sycl_iters"] = float(iterations_seen[0])

    clear_acpp_runtime_cache()
    return mets
