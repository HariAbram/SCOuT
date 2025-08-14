
###############################################################################
# Standard library imports                                                    #
###############################################################################

import itertools
import shlex
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple, Any, Union

###############################################################################
# Type helpers                                                                #
###############################################################################

Number = float
EnvMap = Dict[str, str]
MetricDict = Dict[str, Number]

###############################################################################
# Misc helpers                                                                #
###############################################################################

def _render_one_param(trial, opt: str, spec: Any) -> Tuple[str, str]:
    """
    Returns (key_for_logging, flag_string) for a single compiler param.
    Supports:
      - spec = [v1, v2, ...]  ➜ default "=" glue
      - spec = {"sep": " ", "values": [...]}
      - opt contains "{}" and spec is [values] ➜ format string
    """
    if isinstance(spec, list):
        val = trial.suggest_categorical(opt, spec)
        return f"{opt}={val}", f"{opt}={val}"

    if isinstance(spec, dict) and "values" in spec:
        val = trial.suggest_categorical(opt, spec["values"])
        sep = spec.get("sep", "=")
        return f"{opt}{sep}{val}", f"{opt}{sep}{val}"

    if "{}" in opt and isinstance(spec, list):
        val = trial.suggest_categorical(opt, spec)
        rendered = opt.format(val)
        return rendered, rendered

    raise ValueError(f"Unsupported compiler_param entry for '{opt}'")

def _select_param_subset(trial, keys: List[str], sel: Dict[str, Any]) -> List[str]:
    if not sel:                 # default: include ALL params (old behaviour)
        return keys

    always = set(sel.get("always", [])) & set(keys)

    if "k" in sel:
        k = int(sel["k"])
    else:
        k = trial.suggest_int("params_k", int(sel.get("min", 0)), int(sel.get("max", len(keys))))

    # scores for a permutation; higher score = more likely to be chosen
    scored = []
    for kname in keys:
        if kname in always:
            scored.append((1.0, kname))    # force to the top
        else:
            s = trial.suggest_float(f"score::{kname}", 0.0, 1.0)
            scored.append((float(s), kname))

    scored.sort(reverse=True)              # by score
    chosen = [name for _, name in scored[:max(k, len(always))]]

    # Ensure all 'always' keys are included
    for a in always:
        if a not in chosen:
            chosen[-1] = a                 # replace the last one (k>=len(always) guaranteed)
    return chosen


def suggest_compiler_flags(trial,
                           variants: List[str],
                           params_schema: Dict[str, Union[List[Any], Dict[str, Any]]],
                           flag_pool: List[str],
                           compiler_params_select: Optional[Dict[str, Any]] = None
                           ) -> Tuple[str, str]:
    
    chosen: List[str] = []
    label_parts: List[str] = []

    # ── 1. Variant strings
    if variants:
        variant = trial.suggest_categorical("flag_variant", variants)
        chosen.append(variant)
        label_parts.append(variant)

    # ── 2. Parametric flags
    param_keys = list(params_schema.keys())
    active_keys = _select_param_subset(trial, param_keys, compiler_params_select or {})

    for opt in active_keys:
        key_for_log, flag = _render_one_param(trial, opt, params_schema[opt])
        chosen.append(flag)
        label_parts.append(key_for_log)

    # ── 2. flag pool
    for flag in flag_pool:
        use_it = trial.suggest_categorical(flag, [0, 1])
        if use_it:
            chosen.append(flag)
            label_parts.append(flag)

    flag_str  = " ".join(chosen)
    pretty_id = "|".join(chosen) or "default"
    return pretty_id, flag_str



def suggest_env(trial, schema: Dict[str, Union[List[str], Dict[str, Any]]]
               ) -> Dict[str, str]:
    
    env: Dict[str, str] = {}

    # One linear pass is enough because predicates only look backwards
    # (already chosen vars).  If you need forward refs, convert to a loop.
    for var, spec in schema.items():
        # 1. Unconditional
        if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
            env[var] = trial.suggest_categorical(var, list(spec))
            continue

        # 2. Conditional
        pred  = spec.get("when", {})
        if all(env.get(k) == v for k, v in pred.items()):
            env[var] = trial.suggest_categorical(var, spec["values"])
        # else – silently skip

    return env



def _normalize_args(raw) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return shlex.split(raw)
    if isinstance(raw, Sequence):
        merged: List[str] = []
        for elem in raw:
            merged.extend(shlex.split(elem) if isinstance(elem, str) else [str(elem)])
        return merged
    raise TypeError("program_args must be string or list of strings")


def _expand_env(env: Dict[str, Sequence[str]]) -> List[EnvMap]:
    """
    Turn {"OMP_PLACES":["cores","sockets"], "ACPP_VISIBILITY_MASK":["a","b"]}
    into [{'OMP_PLACES':'cores','ACPP_VISIBILITY_MASK':'a'}, …]  (Cartesian product).
    """
    if not env:
        return [{}]
    # normalise: each value → list
    normd = {k: (v if isinstance(v, Sequence) and not isinstance(v, str) else [v])
             for k, v in env.items()}
    keys = list(normd)
    combos = itertools.product(*(normd[k] for k in keys))
    return [{k: str(val) for k, val in zip(keys, combo)} for combo in combos]

def unique_csv_path(path: Union[str, Path]) -> Path:
    """
    Return a writable CSV filename.
    If <path> does **not** exist → return as‑is.
    Otherwise append an ISO timestamp:  results.csv  → results_20250715‑1423.csv
    """
    p = Path(path)
    if not p.exists():
        return p

    stamp = datetime.now().strftime("%Y%m%d-%H%M")
    return p.with_stem(f"{p.stem}_{stamp}")