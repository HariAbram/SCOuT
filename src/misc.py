
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

def suggest_compiler_flags(trial,
                           variants: List[str],
                           params_schema: Dict[str, Union[List[Any], Dict[str, Any]]],
                           flag_pool: List[str],
                           ) -> Tuple[str, str]:
    
    chosen: List[str] = []

    # ── 1. Variant strings (old list)
    if variants:
        variant = trial.suggest_categorical("flag_variant", variants)
        chosen.append(variant)

    # ── 2. Parametric flags
    for opt, spec in params_schema.items():
        # unconditional list
        if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
            val = trial.suggest_categorical(opt, list(spec))
        else:
            pred = spec.get("when", {})
            if all(trial.params.get(k) == v for k, v in pred.items()):
                val = trial.suggest_categorical(opt, spec["values"])
            else:
                continue  # predicate false → skip flag
        # boolean → presence
        if isinstance(val, bool):
            if val:                    # true → flag present, false → omitted
                chosen.append(opt)
        else:
            # Need a separator?
            if opt.endswith("="):      # "-march="
                chosen.append(f"{opt}{val}")
            elif opt.endswith(" "):    # "-I "  (space-terminated)
                chosen.append(f"{opt}{val}")
            else:                      # default to "="
                chosen.append(f"{opt}={val}")

    for flag in flag_pool:
        use_it = trial.suggest_categorical(flag, [0, 1])
        if use_it:
            chosen.append(flag)

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
    Turn {"OMP_PLACES":["cores","sockets"], "MASK":["a","b"]}
    into [{'OMP_PLACES':'cores','MASK':'a'}, …]  (Cartesian product).
    """
    if not env:
        return [{}]
    # normalise: each value → list
    normd = {k: (v if isinstance(v, Sequence) and not isinstance(v, str) else [v])
             for k, v in env.items()}
    keys = list(normd)
    combos = itertools.product(*(normd[k] for k in keys))
    return [{k: str(val) for k, val in zip(keys, combo)} for combo in combos]

def unique_csv_path(path: str | Path) -> Path:
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