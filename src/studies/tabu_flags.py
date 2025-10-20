from __future__ import annotations

###############################################################################
# Standard library imports                                                    #
###############################################################################
import csv
import json
import math
import random
import tempfile
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

###############################################################################
# Type helpers                                                                #
###############################################################################
Number = float
MetricDict = Dict[str, Number]

###############################################################################
# Local imports                                                               #
###############################################################################
from src.config import Config, ParserConfig, BuildProject
from src.build import compile_project, compile_single_source, _run
from src.metrics import measure_likwid, measure_perf
from src.misc import unique_csv_path, is_significant_improvement



# -----------------------------
# Config block for Tabu Search
# -----------------------------
@dataclass
class TabuSpec:
    # Search limits
    max_iters: int = 200
    max_no_improve: int = 50

    # Tabu list length (tenure)
    tabu_tenure: int = 20

    # Neighborhood sampling per iteration
    neighborhood: int = 24

    # Controls which components are allowed to change in neighbors
    allow_variant_moves: bool = True
    allow_param_moves: bool = True
    allow_pool_moves: bool = True
    allow_env_moves: bool = True

    # Environment exploration (matches wavefront style)
    env_mode: str = "product"         # "fixed" | "product" | "sample"
    env_cap: Optional[int] = None     # cap/compress env combos if large

    # CSV file; default will fall back to cfg.csv_log
    results_csv: Optional[str] = None


# -----------------------------
# Helpers: env enumeration
# -----------------------------
def _enumerate_env_schema(schema: Dict[str, Union[List[str], Dict[str, Any]]]) -> List[Dict[str, str]]:
    """
    Turn an 'env' schema with optional 'when' predicates into a list of concrete env dicts.
    """
    keys = list(schema.keys())

    def rec(i: int, partial: Dict[str, str], out: List[Dict[str, str]]):
        if i == len(keys):
            out.append(dict(partial))
            return
        var = keys[i]
        spec = schema[var]

        # Unconditional list
        if isinstance(spec, list):
            for v in spec:
                partial[var] = str(v)
                rec(i + 1, partial, out)
            partial.pop(var, None)
            return

        # Conditional object
        if isinstance(spec, dict) and "values" in spec:
            pred = spec.get("when", {})
            if all(partial.get(k) == str(v) for k, v in pred.items()):
                for v in spec["values"]:
                    partial[var] = str(v)
                    rec(i + 1, partial, out)
                partial.pop(var, None)
            else:
                # predicate not satisfied → skip var entirely
                rec(i + 1, partial, out)
            return

        # Unknown — ignore
        rec(i + 1, partial, out)

    results: List[Dict[str, str]] = []
    rec(0, {}, results)
    # dedup (just in case)
    uniq = {tuple(sorted(d.items())): d for d in results}
    return list(uniq.values())


def _env_combos(cfg: Config, tabu: TabuSpec, rng: random.Random) -> List[Dict[str, str]]:
    mode = (tabu.env_mode or "product").lower()
    if mode == "fixed":
        # If fixed, we’ll use an empty env unless cfg.wavefront/env exists; you can add a separate “tabu.env” later if needed
        return [dict(getattr(getattr(cfg, "wavefront", object()), "env", {}) or {})]

    schema = getattr(cfg, "env", {}) or {}
    combos = _enumerate_env_schema(schema)
    if not combos:
        return [{}]

    cap = tabu.env_cap or 0
    if cap <= 0 or len(combos) <= cap:
        return combos

    if mode == "sample":
        return rng.sample(combos, cap)

    # default/product: deterministic slice after shuffle for diversity
    rng.shuffle(combos)
    return combos[:cap]


# -----------------------------
# Flag rendering
# -----------------------------
def _render_flags(
    cfg: Config,
    base_flags: str,
    variant: Optional[str],
    params_choice: Dict[str, Any],
    pool_set: Sequence[str],
) -> Tuple[str, str]:
    """
    Return (flags_key_for_csv, flags_shell_string).
    params_choice: { "-march": "native", "-mllvm -force-vector-interleave": 4, ... }
    """
    parts: List[str] = []
    label_parts: List[str] = []

    if base_flags:
        parts.append(base_flags)

    if variant:
        parts.append(variant)
        label_parts.append(variant)

    # params
    for opt, spec in (cfg.compiler_params or {}).items():
        if opt not in params_choice:
            continue
        val = params_choice[opt]
        if isinstance(spec, dict) and "sep" in spec:
            sep = spec.get("sep", "=")
            frag = f"{opt}{sep}{val}"
        else:
            # default "=" glue
            frag = f"{opt}={val}"
        parts.append(frag)
        label_parts.append(frag)

    # pool
    for f in pool_set:
        parts.append(f)
        label_parts.append(f)

    flags_str = " ".join(parts).strip()
    # Optuna-like pretty ID: join the *actual* fragments with |
    # (we prefer using parts, not label_parts, so base flags are present in the key)
    flags_key = "|".join(parts) if parts else "default"
    return flags_key, flags_str


# -----------------------------
# Neighborhood moves
# -----------------------------
def _neighbors(
    cfg: Config,
    state: Tuple[Optional[str], Dict[str, Any], List[str], Dict[str, str]],
    tabu: TabuSpec,
    rng: random.Random,
    num: int,
    PARAM_MIN: int,
    PARAM_MAX: int,
) -> List[Tuple[Optional[str], Dict[str, Any], List[str], Dict[str, str]]]:
    variant, params_choice, pool_list, env_dict = state
    pool_all: List[str] = list(cfg.compiler_flag_pool or [])
    variants_all: List[str] = list(cfg.compiler_flags or [])
    params_schema = cfg.compiler_params or {}

    out: List[Tuple[Optional[str], Dict[str, Any], List[str], Dict[str, str]]] = []

    def move_variant():
        if not tabu.allow_variant_moves or not variants_all:
            return
        choices = [v for v in variants_all if v != variant] or variants_all
        nv = rng.choice(choices)
        out.append((nv, dict(params_choice), list(pool_list), dict(env_dict)))

    def _values_for_param(key: str) -> List[Any]:
        spec = params_schema[key]
        return list(spec["values"]) if isinstance(spec, dict) and "values" in spec else list(spec)

    def move_param():
        if not tabu.allow_param_moves or not params_schema:
            return

        active = set(params_choice.keys())
        all_keys = list(params_schema.keys())
        rng.shuffle(all_keys)

        # Randomly choose an action class respecting bounds
        actions = []
        if len(active) < PARAM_MAX:
            actions.append("add")
        if len(active) > PARAM_MIN:
            actions.append("remove")
        if len(active) > 0:
            actions.append("change")
        if len(active) > 0 and len(active) <= PARAM_MAX and len(active) >= PARAM_MIN and len(active) < len(all_keys):
            actions.append("swap")  # remove one, add a different one

        if not actions:
            # If bounds block add/remove, at least try change
            if len(active) > 0:
                actions = ["change"]
            else:
                return

        act = rng.choice(actions)

        if act == "add":
            candidates = [k for k in all_keys if k not in active]
            if not candidates:
                return
            k = rng.choice(candidates)
            vals = _values_for_param(k)
            if not vals:
                return
            new_params = dict(params_choice)
            new_params[k] = rng.choice(vals)
            out.append((variant, new_params, list(pool_list), dict(env_dict)))

        elif act == "remove":
            k = rng.choice(list(active))
            new_params = dict(params_choice)
            new_params.pop(k, None)
            out.append((variant, new_params, list(pool_list), dict(env_dict)))

        elif act == "change":
            k = rng.choice(list(active))
            vals = _values_for_param(k)
            if not vals:
                return
            cur = params_choice.get(k)
            choices = [v for v in vals if v != cur] or vals
            new_params = dict(params_choice)
            new_params[k] = rng.choice(choices)
            out.append((variant, new_params, list(pool_list), dict(env_dict)))

        elif act == "swap":
            # remove one active, add a different inactive
            rem_k = rng.choice(list(active))
            add_candidates = [k for k in all_keys if k not in active]
            if not add_candidates:
                return
            add_k = rng.choice(add_candidates)
            vals = _values_for_param(add_k)
            if not vals:
                return
            new_params = dict(params_choice)
            new_params.pop(rem_k, None)
            new_params[add_k] = rng.choice(vals)
            out.append((variant, new_params, list(pool_list), dict(env_dict)))

    def move_pool():
        if not tabu.allow_pool_moves or not pool_all:
            return
        new_pool = set(pool_list)
        if rng.random() < 0.5 and new_pool:
            # remove one
            rem = rng.choice(list(new_pool))
            new_pool.remove(rem)
        else:
            # add one or toggle
            cand = rng.choice(pool_all)
            if cand in new_pool:
                new_pool.remove(cand)
            else:
                new_pool.add(cand)
        out.append((variant, dict(params_choice), sorted(new_pool), dict(env_dict)))

    # Compose move bag
    moves = []
    if tabu.allow_variant_moves and (cfg.compiler_flags or []):
        moves.append(move_variant)
    if tabu.allow_param_moves and (cfg.compiler_params or {}):
        # Important: capture bounds via closure
        def bounded_move_param():
            move_param()
        moves.append(bounded_move_param)
    if tabu.allow_pool_moves and (cfg.compiler_flag_pool or []):
        moves.append(move_pool)

    if not moves:
        return out

    for _ in range(num):
        rng.choice(moves)()

    return out



# -----------------------------
# Evaluate a (flags, env) config
# -----------------------------
def _compile_and_measure(
    cfg: Config,
    flags_str: str,
    env: Dict[str, str],
    work: Path,
) -> Tuple[float, MetricDict, str]:
    work.mkdir(parents=True, exist_ok=True)

    # Build
    if cfg.source:
        binary = compile_single_source(cfg.compiler, cfg.source, flags_str, work / "a.out")
    else:
        binary = compile_project(cfg.project, cfg.compiler, flags_str, work)
    if not binary:
        raise RuntimeError("build failed")

    # Measure
    if cfg.backend == "perf":
        metrics = measure_perf(cfg.perf, binary, cfg.program_args, env, cfg.runs)  # type: ignore[arg-type]
    elif cfg.backend == "parser":
        # re-use the wavefront SYCL parser if you placed it there; if not, fallback to perf/likwid only
        from src.studies.wavefront_flags import measure_parser_sycl_wavefront  # local import to avoid cycle
        metrics = measure_parser_sycl_wavefront(cfg.parser, Path(binary), cfg.program_args, env, cfg.runs, work, cfg.project)
    else:
        metrics = measure_likwid(cfg.likwid, binary, cfg.program_args, env, cfg.runs)  # type: ignore[arg-type]

    # Objective
    obj = cfg.objectives[0]
    metric = obj.metric
    if metric not in metrics:
        raise RuntimeError(f"objective metric '{metric}' missing; got {list(metrics)}")

    value = float(metrics[metric])
    return value, metrics, str(binary)


# -----------------------------
# Main entry
# -----------------------------
def run_tabu_study(cfg: Config) -> None:
    rng = random.Random(getattr(getattr(cfg, "search", object()), "random_seed", None))

    # TabuSpec from cfg.search.tabu (if provided) or defaults
    raw = getattr(cfg, "tabu", {}) or {}
    tabu = TabuSpec(**{k: v for k, v in raw.items() if k in TabuSpec.__annotations__})

    env_list = _env_combos(cfg, tabu, rng)
    print(f"[tabu] env_mode={tabu.env_mode} env_combos={len(env_list)}")

    workroot = Path(tempfile.mkdtemp(prefix="SCOuT_tabu_"))
    print(f"[tabu] workdir: {workroot}")

    # Initial state: start small (base + first variant; params unset; pool empty)
    base = cfg.compiler_flags_base or ""
    variants = list(cfg.compiler_flags or [])
    variant0: Optional[str] = variants[0] if variants else None

    params_choice: Dict[str, Any] = {}
    pool_list: List[str] = []
    env0 = env_list[0] if env_list else {}

    sel = getattr(cfg, "compiler_params_select", {}) or {}
    PARAM_MIN = int(sel.get("min", 0))
    # if max omitted, allow all params
    PARAM_MAX = int(sel.get("max", len((cfg.compiler_params or {}))))
    if PARAM_MAX < PARAM_MIN:
        PARAM_MAX = PARAM_MIN

    params_schema = cfg.compiler_params or {}
    if PARAM_MIN > 0 and params_schema:
        keys = list(params_schema.keys())
        rng.shuffle(keys)
        need = min(PARAM_MIN, len(keys))
        for k in keys[:need]:
            spec = params_schema[k]
            if isinstance(spec, dict) and "values" in spec:
                vals = list(spec["values"])
            else:
                vals = list(spec)
            if not vals:
                continue
            params_choice[k] = rng.choice(vals)


    # Evaluate baseline across all envs and pick the best as the starting point
    best_start = None
    best_start_sc = math.inf
    goal_min = (cfg.objectives[0].goal == "min")

    sig = getattr(cfg, "significance", {}) or {}
    MIN_REL = float(sig.get("min_rel_gain", 0.15))
    MIN_ABS = sig.get("min_abs_gain", None)

    no_improve = 0

    def score(v: float) -> float:
        return v if goal_min else -v

    print("[tabu] evaluating baseline across environments…")
    for i, e in enumerate(env_list, 1):
        key, flags_str = _render_flags(cfg, base, variant0, params_choice, pool_list)
        try:
            v, mets, binp = _compile_and_measure(cfg, flags_str, e, workroot / "baseline" / f"env{i:03d}")
            sc = score(v)
        except Exception as ex:
            v, mets, binp, sc = (math.inf, {"error": str(ex)}, "", math.inf)
        if sc < best_start_sc:
            best_start_sc = sc
            best_start = (variant0, dict(params_choice), list(pool_list), dict(e), v, mets, binp, key, flags_str)

    if best_start is None:
        raise RuntimeError("tabu: all baseline evaluations failed.")

    variant, params_choice, pool_list, env, cur_val, cur_mets, cur_bin, cur_key, cur_flags = best_start
    best_val = cur_val
    best_key = cur_key
    best_flags = cur_flags
    best_env = dict(env)
    best_metrics = dict(cur_mets)
    best_binary = cur_bin

    print(f"[tabu] start: {cfg.objectives[0].metric}={best_val:.6g} env={json.dumps(best_env)}")

    # Tabu memory: store config keys; aspiration allows override if improves best
    tabu_q: deque[str] = deque(maxlen=tabu.tabu_tenure)

    # Cache: avoid rebuilding identical (flags, env)
    cache: Dict[Tuple[str, Tuple[Tuple[str, str], ...]], Tuple[float, Dict[str, float], str]] = {}
    def cfg_key(k: str, e: Dict[str, str]):
        return (k, tuple(sorted(e.items())))

    cache[cfg_key(cur_key, env)] = (cur_val, cur_mets, cur_bin)
    tabu_q.append(cur_key + "|" + json.dumps(best_env, sort_keys=True))

    no_improve = 0
    iters = 0

    # Prepare CSV like Optuna
    if getattr(cfg, "csv_log", None):
        out_csv = unique_csv_path(cfg.csv_log)
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    else:
        out_csv = workroot / (tabu.results_csv or "tabu_results.csv")
    print(f"[tabu] writing CSV → {out_csv}")

    extra_keys: set[str] = set(best_metrics.keys())
    hdr = [o.metric for o in cfg.objectives] + ["compiler_flags", "env", "binary"]
    # rows are streamed; we’ll write header now and append
    with open(out_csv, "w", newline="") as fp:
        w = csv.writer(fp)
        # header will include extra metrics later (after we know them)
        # To keep exactly the same schema as Optuna output, expand now:
        extra_cols = sorted(extra_keys)
        w.writerow(hdr + extra_cols)
        # baseline row
        w.writerow([best_val, best_key, json.dumps(best_env), best_binary] + [best_metrics.get(k, "") for k in extra_cols])
        fp.flush()

        # Main loop
        while iters < tabu.max_iters and no_improve < tabu.max_no_improve:
            iters += 1

            # Generate neighbors for the current state (flag-side)
            neigh = _neighbors(cfg, (variant, params_choice, pool_list, env), tabu, rng, tabu.neighborhood, PARAM_MIN, PARAM_MAX,)


            # Add environment neighbors if allowed
            if tabu.allow_env_moves and len(env_list) > 1:
                # a few random env flips
                for _ in range(min(8, len(env_list))):
                    e = rng.choice(env_list)
                    neigh.append((variant, dict(params_choice), list(pool_list), dict(e)))

            # Evaluate neighbors, pick the best admissible (or aspirated)
            cand_best = None
            cand_best_sc = math.inf
            cand_best_pkg = None  # (v, mets, binp, key, flags_str, env)

            for nv, nparams, npool, nenv in neigh:
                nkey, nflags = _render_flags(cfg, base, nv, nparams, npool)
                tabu_key = nkey + "|" + json.dumps(nenv, sort_keys=True)
                # admissible if not tabu OR aspirates (improves global best)
                is_tabu = (tabu_key in tabu_q)

                # cached?
                ck = cfg_key(nkey, nenv)
                if ck in cache:
                    v, mets, binp = cache[ck]
                else:
                    try:
                        v, mets, binp = _compile_and_measure(cfg, nflags, nenv, workroot / f"iter{iters:04d}")
                        cache[ck] = (v, mets, binp)
                    except Exception as ex:
                        v, mets, binp = (math.inf, {"error": str(ex)}, "")

                sc = score(v)
                if math.isinf(sc):
                    continue

                aspirates = sc < score(best_val)
                if (not is_tabu) or aspirates:
                    if sc < cand_best_sc:
                        cand_best_sc = sc
                        cand_best = (nv, nparams, npool, nenv)
                        cand_best_pkg = (v, mets, binp, nkey, nflags, nenv)

                # stream a row (like Optuna): write even if tabu (we evaluated it)
                # update header if new extra metrics arrive
                new_keys = set(mets.keys())
                if new_keys - set(extra_cols):
                    # expand header once: rewrite file from scratch with new header
                    extra_cols = sorted(set(extra_cols) | new_keys)
                    fp.seek(0)
                    rows = list(csv.reader(open(out_csv)))
                    # rows[0] was old header — rewrite
                    with open(out_csv, "w", newline="") as fp2:
                        w2 = csv.writer(fp2)
                        w2.writerow(hdr + extra_cols)
                        if len(rows) > 1:
                            # rewrite previous rows with new extra columns
                            for r in rows[1:]:
                                # r = [val, flags, env, bin, ... old extras]
                                base_len = 4
                                # rebuild into dict for old extras
                                old_extra_vals = r[base_len:]
                                old_keys = sorted(set(extra_keys))
                                old_map = dict(zip(old_keys, old_extra_vals))
                                w2.writerow(r[:base_len] + [old_map.get(k, "") for k in extra_cols])
                    # refresh in-memory extra keys
                    extra_keys = set(extra_cols)
                    # reopen append handle
                    fp = open(out_csv, "a", newline="")
                    w = csv.writer(fp)

                w.writerow([v, nkey, json.dumps(nenv), binp] + [mets.get(k, "") for k in extra_cols])
                fp.flush()

            if cand_best is None:
                print(f"[tabu] iter {iters}: no admissible neighbor; stopping.")
                break

            # Move to the best candidate
            variant, params_choice, pool_list, env = cand_best
            v, mets, binp, k, fstr, e = cand_best_pkg  # type: ignore[misc]
            tabu_q.append(k + "|" + json.dumps(e, sort_keys=True))
            '''
            # Improvement?
            if score(v) < score(best_val):
                best_val = v
                best_key = k
                best_flags = fstr
                best_env = dict(e)
                best_metrics = dict(mets)
                best_binary = binp
                no_improve = 0
                print(f"[tabu] iter {iters}: IMPROVED → {cfg.objectives[0].metric}={best_val:.6g}")
            else:
                no_improve += 1
                print(f"[tabu] iter {iters}: best={best_val:.6g} (no_improve={no_improve})")
            '''
            sig = getattr(cfg, "significance", {}) or {}
            MIN_REL = float(sig.get("min_rel_gain", 0.15))
            MIN_ABS = sig.get("min_abs_gain", None)

            # Did we significantly improve the global best?
            if is_significant_improvement(old=best_val, new=v,
                                        goal=("min" if cfg.objectives[0].goal == "min" else "max"),
                                        min_rel_gain=MIN_REL, min_abs_gain=MIN_ABS):
                best_val = v
                best_key = k
                best_flags = fstr
                best_env = dict(e)
                best_metrics = dict(mets)
                best_binary = binp
                no_improve = 0
                print(f"[tabu] iter {iters}: IMPROVED → {cfg.objectives[0].metric}={best_val:.6g}")
            else:
                no_improve += 1
                print(f"[tabu] iter {iters}: best={best_val:.6g} (no_improve={no_improve})")

    print("\n[tabu] ===== Summary =====")
    print(f"best: {cfg.objectives[0].metric}={best_val:.6g}")
    print(f"flags: {best_key}")
    print(f"env: {json.dumps(best_env)}")
    print(f"[tabu] results → {out_csv}")
