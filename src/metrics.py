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

from src.config import PerfConfig, MetricSpec, LikwidConfig
from src.build import _run

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


###############################################################################
# Parser helpers (Parsers for HeCBench)                                       #
###############################################################################


# Reuse the same patterns from output_parse.py (simplified here)
_UNIT = r"\(?\s*(ns|µs|us|ms|s|sec|secs|seconds)\s*\)?"
_DEFAULT_PATTERNS = [
    r"\bkernel(?:\s+execution)?\s*time[^0-9]*=\s*([0-9]*\.?[0-9]+)\s*" + _UNIT + r"\b",
    r"\b(?:avg|average)\s+kernel\s*time[^0-9]*=\s*([0-9]*\.?[0-9]+)\s*" + _UNIT + r"\b",
    r"\btotal\s+kernel\s+time[^0-9]*=\s*([0-9]*\.?[0-9]+)\s*" + _UNIT + r"\b",
    r"\btime\s*\(kernel\)[^0-9]*([0-9]*\.?[0-9]+)\s*" + _UNIT + r"\b",
    r"\btime\s*\(ms\)\s*[:=]\s*([0-9]*\.?[0-9]+)\b",  # unit implied ms
    r"\bkernel\s*time[^0-9]*[:=]\s*([0-9]*\.?[0-9]+)\s*" + _UNIT + r"\b",
    r"\bgpu\s+kernel\s*time[^0-9]*[:=]\s*([0-9]*\.?[0-9]+)\s*" + _UNIT + r"\b",
    r"\bdevice\s+execution\s*time[^0-9]*[:=]\s*([0-9]*\.?[0-9]+)\s*" + _UNIT + r"\b",
    r"\bexecution\s*time[^0-9]*[:=]\s*([0-9]*\.?[0-9]+)\s*" + _UNIT + r"\b",
    r"\bdevice\s+offloading\s+time[^0-9]*=\s*([0-9]*\.?[0-9]+)\s*" + _UNIT + r"\b",
    r"\btotal\s+execution\s+time\s+of\s+kernels[^0-9]*=\s*([0-9]*\.?[0-9]+)\s*" + _UNIT + r"\b",
    r"\b([0-9]*\.?[0-9]+)\s*" + _UNIT + r"\b",  # fallback
]

def _compile_patterns(patterns: Optional[List[str]]) -> List[re.Pattern]:
    pats = patterns if patterns else _DEFAULT_PATTERNS
    return [re.compile(p, re.IGNORECASE) for p in pats]

def _to_ms(val_s: str, unit: Optional[str]) -> float:
    # strip commas in numbers like "12,345.6"
    v = float(val_s.replace(",", ""))
    if not unit:
        return v  # assume ms if unit omitted in some patterns
    u = unit.lower()
    if u in ("s", "sec", "seconds"): return v * 1000.0
    if u in ("ms", "msec"):          return v
    if u in ("us", "µs"):            return v / 1000.0
    if u == "ns":                    return v / 1_000_000.0
    return v  # fallback

def _reduce(vals: List[float], how: str) -> Optional[float]:
    if not vals:
        return None
    h = how.lower()
    if h == "min":   return min(vals)
    if h == "max":   return max(vals)
    if h == "mean":  return mean(vals)
    if h == "first": return vals[0]
    if h == "last":  return vals[-1]
    return min(vals)

def _capture(proc_out: subprocess.CompletedProcess, source: str) -> str:
    if source == "stdout": return proc_out.stdout or ""
    if source == "stderr": return proc_out.stderr or ""
    return ((proc_out.stdout or "") + "\n" + (proc_out.stderr or "")).strip()

def _filter_lines(text: str, require_any: Optional[List[str]]) -> str:
    if not require_any:
        return text
    wanted = []
    low_needles = [s.lower() for s in require_any]
    for line in text.splitlines():
        low = line.lower()
        if any(n in low for n in low_needles):
            wanted.append(line)
    return "\n".join(wanted) if wanted else text  # if filter yields nothing, keep original

def _log_raw(cfg, workdir: Path, idx: int, content: str, why: str) -> None:
    if not cfg.log_raw:
        return
    outdir = workdir / cfg.log_dir
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / f"run_{idx:02d}_{why}.log").write_text(content)

def measure_parser(
    cfg,                     # ParserConfig
    bin_path: Path,
    prog_args: List[str],
    env: EnvMap,
    runs: int,
    workdir: Optional[Path] = None,     # pass the trial workdir if you have it
) -> MetricDict:
    regexes = _compile_patterns(cfg.patterns)
    values_ms: List[float] = []

    argv = [str(bin_path), *prog_args]
    print(argv)

    for i in range(max(1, runs)):
        proc = subprocess.run(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, **env},
            timeout=cfg.timeout,
            check=False,
        )
        text = _capture(proc, cfg.source)
        text = _filter_lines(text, cfg.require_any)

        found_this_run: List[float] = []
        for rx in regexes:
            for m in rx.finditer(text):
                # groups: num + optional exponent is group 1, unit is last group
                num_val = m.group(1)
                unit    = m.groups()[-1] if m.groups() else None
                try:
                    found_this_run.append(_to_ms(num_val, unit))
                except Exception:
                    continue

        if not found_this_run:
            _log_raw(cfg, workdir or Path("."), i, text, "no_match")
        picked = _reduce(found_this_run, cfg.selector)
        if picked is not None:
            values_ms.append(picked)

    if not values_ms:
        raise RuntimeError("Parser backend: no kernel times found in output.")

    avg_ms = mean(values_ms)
    # provide both ms and requested unit
    mets: MetricDict = {"kernel_time_ms": avg_ms}
    if cfg.out_unit != "ms":
        if cfg.out_unit == "s":  mets["kernel_time_s"]  = avg_ms / 1000.0
        if cfg.out_unit == "us": mets["kernel_time_us"] = avg_ms * 1000.0
        if cfg.out_unit == "ns": mets["kernel_time_ns"] = avg_ms * 1_000_000.0
    return mets
