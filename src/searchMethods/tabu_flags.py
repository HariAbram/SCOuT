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
from dataclasses import dataclass, field
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
from src.config import Config
from src.evaluator import Evaluator
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
    env: Dict[str, str] = field(default_factory=dict)

    # CSV file; default will fall back to cfg.csv_log
    results_csv: Optional[str] = None


# -----------------------------
# Helpers: env enumeration
# -----------------------------
def _enumerate_env_schema(schema: Dict[str, Union[List[str], Dict[str, Any]]]) -> List[Dict[str, str]]:
    """
    Turn an 'env' schema with optional 'when' predicates into a list of concrete env dicts.
    """
    if not isinstance(schema, dict) or not schema:
        return [{}]
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
    """Return a non-empty list of env dicts; fall back to [{}] if cfg.env is empty/missing."""
    mode = (tabu.env_mode or "product").lower()
    if mode == "fixed":
        return [dict(tabu.env or {})]

    # Normal (product/sample) modes
    schema = getattr(cfg, "env", None)
    combos = _enumerate_env_schema(schema if isinstance(schema, dict) else {})
    if not combos:
        combos = [{}]

    cap = tabu.env_cap or 0
    if cap <= 0 or len(combos) <= cap:
        return combos

    if mode == "sample" and len(combos) > 0:
        return rng.sample(combos, cap)

    # default/product: deterministic slice after shuffle for diversity
    rng.shuffle(combos)
    return combos[:cap] if cap > 0 else combos


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

    if variant and variant != base_flags:
        parts.append(variant)
        label_parts.append(variant)

    # params
    for opt, spec in (cfg.compiler_params or {}).items():
        if opt not in params_choice:
            continue
        val = params_choice[opt]
        if "{}" in opt:
            frag = opt.format(val)
        elif isinstance(spec, dict) and "sep" in spec:
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
    always = set((cfg.compiler_params_select or {}).get("always", []))

    # Helper to get domain of a param
    def _values_for_param(key: str) -> List[Any]:
        spec = params_schema[key]
        return list(spec["values"]) if isinstance(spec, dict) and "values" in spec else list(spec)

    neigh: List[Tuple[Optional[str], Dict[str, Any], List[str], Dict[str, str]]] = []

    # ---- VARIANT NEIGHBORS (change variant) ----
    if tabu.allow_variant_moves and variants_all:
        cand_variants = [v for v in variants_all if v != variant]
        rng.shuffle(cand_variants)
        for nv in cand_variants:
            neigh.append((nv, dict(params_choice), list(pool_list), dict(env_dict)))

    # ---- PARAM NEIGHBORS (add/remove/change/swap) ----
    if tabu.allow_param_moves and params_schema:
        active = set(params_choice.keys())
        all_keys = list(params_schema.keys())
        inactive = [k for k in all_keys if k not in active]

        # ADD (respect max)
        if len(active) < PARAM_MAX and inactive:
            rng.shuffle(inactive)
            for k in inactive:
                vals = _values_for_param(k)
                if not vals: continue
                new_params = dict(params_choice)
                new_params[k] = rng.choice(vals)
                neigh.append((variant, new_params, list(pool_list), dict(env_dict)))

        # REMOVE (respect min)
        if len(active) > PARAM_MIN and active:
            rem_keys = list(active - always)
            rng.shuffle(rem_keys)
            for k in rem_keys:
                new_params = dict(params_choice)
                new_params.pop(k, None)
                neigh.append((variant, new_params, list(pool_list), dict(env_dict)))

        # CHANGE (pick different value)
        if active:
            chg_keys = list(active)
            rng.shuffle(chg_keys)
            for k in chg_keys:
                vals = _values_for_param(k)
                if not vals: continue
                cur = params_choice.get(k)
                choices = [v for v in vals if v != cur] or vals
                if len(choices) == 1 and choices[0] == cur:
                    continue  # no real change available
                new_params = dict(params_choice)
                new_params[k] = rng.choice(choices)
                if new_params[k] == cur:
                    continue
                neigh.append((variant, new_params, list(pool_list), dict(env_dict)))

        # SWAP (remove one, add a different one)
        if active and inactive and (len(active) >= PARAM_MIN) and (len(active) <= PARAM_MAX):
            rem_keys = list(active - always); rng.shuffle(rem_keys)
            add_keys = list(inactive); rng.shuffle(add_keys)
            for rk in rem_keys:
                for ak in add_keys:
                    vals = _values_for_param(ak)
                    if not vals: continue
                    new_params = dict(params_choice)
                    new_params.pop(rk, None)
                    new_params[ak] = rng.choice(vals)
                    neigh.append((variant, new_params, list(pool_list), dict(env_dict)))

    # ---- POOL NEIGHBORS (add/remove single flag) ----
    if tabu.allow_pool_moves and pool_all:
        s = set(pool_list)
        # ADD flags not present
        addables = [f for f in pool_all if f not in s]
        rng.shuffle(addables)
        for f in addables:
            new_pool = sorted(s | {f})
            neigh.append((variant, dict(params_choice), new_pool, dict(env_dict)))
        # REMOVE flags present
        removables = list(s)
        rng.shuffle(removables)
        for f in removables:
            new_pool = sorted(s - {f})
            neigh.append((variant, dict(params_choice), new_pool, dict(env_dict)))

    # ---- DEDUP + NO-OP FILTER + CAP ----
    uniq: List[Tuple[Optional[str], Dict[str, Any], List[str], Dict[str, str]]] = []
    seen = set()
    cur_key = (
        variant,
        tuple(sorted(params_choice.items())),
        tuple(pool_list),
        tuple(sorted(env_dict.items())),
    )
    rng.shuffle(neigh)  # randomize before slicing

    for nv, nparams, npool, nenv in neigh:
        key = (
            nv,
            tuple(sorted(nparams.items())),
            tuple(npool),
            tuple(sorted(nenv.items())),
        )
        if key == cur_key:  # no-op, skip
            continue
        if key in seen:     # duplicate, skip
            continue
        seen.add(key)
        uniq.append((nv, nparams, npool, nenv))
        if len(uniq) >= max(1, num):   # respect requested neighborhood size
            break

    return uniq



# -----------------------------
# Evaluate a (flags, env) config
# -----------------------------
def _compile_and_measure(
    evaluator: Evaluator,
    cfg: Config,
    flags_str: str,
    env: Dict[str, str],
    work: Path,
) -> Tuple[float, MetricDict, str]:
    evaluation = evaluator.evaluate(flags_str, env, work)
    metrics = evaluation.metrics
    binary = evaluation.binary

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
    if not env_list:
        env_list = [{}]
    print(f"[tabu] env_mode={tabu.env_mode} env_combos={len(env_list)}")

    workroot = Path(tempfile.mkdtemp(prefix="SCOuT_tabu_"))
    evaluator = Evaluator(cfg, workroot)
    print(f"[tabu] workdir: {workroot}")

    # Initial state: start small (base + first variant; params unset; pool empty)
    base = cfg.compiler_flags_base or ""
    variants = list(cfg.compiler_flags or [])
    variant0: Optional[str] = variants[0] if variants else None

    params_choice: Dict[str, Any] = {}
    pool_list: List[str] = []
    env0 = env_list[0] if env_list else {}

    sel = getattr(cfg, "compiler_params_select", {}) or {}
    if "k" in sel:
        PARAM_MIN = PARAM_MAX = int(sel["k"])
    else:
        PARAM_MIN = int(sel.get("min", 0))
        PARAM_MAX = int(sel.get("max", len((cfg.compiler_params or {}))))
    always = [k for k in sel.get("always", []) if k in (cfg.compiler_params or {})]
    PARAM_MIN = max(PARAM_MIN, len(always))
    if PARAM_MAX < PARAM_MIN:
        PARAM_MAX = PARAM_MIN

    params_schema = cfg.compiler_params or {}
    if PARAM_MIN > 0 and params_schema:
        keys = always + [k for k in params_schema if k not in always]
        rng.shuffle(keys)
        if always:
            keys = always + [k for k in keys if k not in always]
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

    no_improve = 0

    def score(v: float) -> float:
        return v if goal_min else -v

    print("[tabu] evaluating baseline across environments…")
    for i, e in enumerate(env_list, 1):
        key, flags_str = _render_flags(cfg, base, variant0, params_choice, pool_list)
        try:
            v, mets, binp = _compile_and_measure(evaluator, cfg, flags_str, e, workroot / "baseline" / f"env{i:03d}")
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

    tabu_q.append(cur_key + "|" + json.dumps(best_env, sort_keys=True))

    no_improve = 0
    iters = 0

    # Prepare CSV like Optuna
    if tabu.results_csv:
        out_csv = unique_csv_path(tabu.results_csv)
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    elif getattr(cfg, "csv_log", None):
        out_csv = unique_csv_path(cfg.csv_log)
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    else:
        out_csv = workroot / "tabu_results.csv"
    print(f"[tabu] writing CSV → {out_csv}")

    extra_keys: set[str] = set(best_metrics.keys())
    result_rows: List[Tuple[int, float, str, Dict[str, str], str, Dict[str, float]]] = [
        (0, best_val, best_key, dict(best_env), best_binary, dict(best_metrics))
    ]
    hdr = ["k"] + [o.metric for o in cfg.objectives] + ["compiler_flags", "env", "binary"]
    while iters < tabu.max_iters and no_improve < tabu.max_no_improve:
            iters += 1

            # Generate neighbors for the current state (flag-side)
            neigh = _neighbors(cfg, (variant, params_choice, pool_list, env), tabu, rng, tabu.neighborhood, PARAM_MIN, PARAM_MAX,)


            # Add environment neighbors if allowed
            if tabu.allow_env_moves and len(env_list) > 1:
                cur_env_key = tuple(sorted(env.items()))
                env_unique = [e for e in env_list if tuple(sorted(e.items())) != cur_env_key]
                k = min(8, len(env_unique))
                env_additions = [(variant, dict(params_choice), list(pool_list), dict(e))
                                 for e in (rng.sample(env_unique, k) if k > 0 else [])]
                # dedup against already-built neighbors
                seen_keys = {
                    (nv, tuple(sorted(np.items())), tuple(npl), tuple(sorted(ne.items())))
                    for nv, np, npl, ne in neigh
                }
                for cand in env_additions:
                    key = (cand[0], tuple(sorted(cand[1].items())), tuple(cand[2]), tuple(sorted(cand[3].items())))
                    if key not in seen_keys:
                        neigh.append(cand)
                        seen_keys.add(key)

            # Evaluate neighbors, pick the best admissible (or aspirated)
            cand_best = None
            cand_best_sc = math.inf
            cand_best_pkg = None  # (v, mets, binp, key, flags_str, env)

            for nv, nparams, npool, nenv in neigh:
                nkey, nflags = _render_flags(cfg, base, nv, nparams, npool)
                tabu_key = nkey + "|" + json.dumps(nenv, sort_keys=True)
                # admissible if not tabu OR aspirates (improves global best)
                is_tabu = (tabu_key in tabu_q)

                try:
                    v, mets, binp = _compile_and_measure(
                        evaluator, cfg, nflags, nenv,
                        workroot / f"iter{iters:04d}" / f"candidate_{len(result_rows):05d}",
                    )
                    sc = score(v)
                except Exception as ex:
                    v, mets, binp, sc = (math.inf, {"error": str(ex)}, "", math.inf)
                if math.isinf(sc):
                    continue

                aspirates = sc < score(best_val)
                if (not is_tabu) or aspirates:
                    if sc < cand_best_sc:
                        cand_best_sc = sc
                        cand_best = (nv, nparams, npool, nenv)
                        cand_best_pkg = (v, mets, binp, nkey, nflags, nenv)

                extra_keys.update(mets.keys())
                result_rows.append((iters, v, nkey, dict(nenv), binp, dict(mets)))

            if cand_best is None:
                print(f"[tabu] iter {iters}: no admissible neighbor; stopping.")
                break

            # Move to the best candidate
            variant, params_choice, pool_list, env = cand_best
            v, mets, binp, k, fstr, e = cand_best_pkg  # type: ignore[misc]
            tabu_q.append(k + "|" + json.dumps(e, sort_keys=True))
            
            improved = score(v) < score(best_val)
            meaningful = is_significant_improvement(
                best_val, v, cfg.objectives[0].goal,
                cfg.significance.min_rel_gain, cfg.significance.min_abs_gain,
            )
            if improved:
                best_val = v
                best_key = k
                best_flags = fstr
                best_env = dict(e)
                best_metrics = dict(mets)
                best_binary = binp
                print(f"[tabu] iter {iters}: IMPROVED → {cfg.objectives[0].metric}={best_val:.6g}")
            if meaningful:
                no_improve = 0
            else:
                no_improve += 1
                if not improved:
                    print(f"[tabu] iter {iters}: best={best_val:.6g} (no_improve={no_improve})")

    extra_cols = sorted(extra_keys - {o.metric for o in cfg.objectives})
    with open(out_csv, "w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(hdr + extra_cols)
        for iteration, value, flags_key, row_env, binary, metrics in result_rows:
            if math.isinf(value):
                continue
            w.writerow([iteration, value, flags_key, json.dumps(row_env), binary] + [metrics.get(k, "") for k in extra_cols])

    print("\n[tabu] ===== Summary =====")
    print(f"best: {cfg.objectives[0].metric}={best_val:.6g}")
    print(f"flags: {best_key}")
    print(f"env: {json.dumps(best_env)}")
    print(f"[tabu] results → {out_csv}")
    print(f"[tabu] unique builds: {evaluator.build_count}")
