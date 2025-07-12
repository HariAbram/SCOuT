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
import re
import sys
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

from src.config import PerfConfig, MetricSpec, LikwidConfig
from src.build import _run

###############################################################################
# Measurement helpers (perf & likwid) – unchanged from original               #
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
    buckets: Dict[str, List[Number]] = {e: [] for e in cfg.events + ["CPI"]}
    for _ in range(runs):
        cmd = ["perf", "stat", "-e", ",".join(cfg.events), "--", str(bin_path), *prog_args]
        if cfg.core_list:
            cmd = ["taskset", "-c", cfg.core_list, *cmd]
        proc = _run(cmd, env={**os.environ, **env})
        data = perf_parse(proc.stderr, cfg.events)
        if not data:
            raise RuntimeError("Perf parse failure – received no matching events.")
        for k, v_list in buckets.items():
            if k in data:
                v_list.append(data[k])
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
    for _ in range(runs):
        cmd = ["likwid-perfctr"]
        if cfg.core_list:
            cmd += ["-C", cfg.core_list]
        
        if cfg.group:
            cmd += ["-g", cfg.group]
        else:                       # raw events
            cmd += ["-g", ",".join(cfg.events)]
        cmd += [str(bin_path), *prog_args]
        #cmd += ["-g", cfg.group, str(bin_path), *prog_args]
        proc = _run(cmd, env={**os.environ, **env})
        data = likwid_parse(proc.stdout, cfg.metrics)
        if not data:
            raise RuntimeError("LIKWID parse failure – no metrics captured.")
        for k, v in data.items():
            if k in buckets:
                buckets[k].append(v)
    return {k: mean(v) for k, v in buckets.items() if v}
