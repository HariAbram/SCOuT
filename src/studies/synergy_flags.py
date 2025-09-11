# src/studies/synergy_flags.py
from __future__ import annotations

import csv
import json
import math
import random
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence, Set

import optuna  # only for TrialState enum reuse if you like; safe to remove

from src.build import compile_project, compile_single_source
from src.metrics import measure_likwid, measure_perf
from src.misc import unique_csv_path
from src.config import Config

Number = float
MetricDict = Dict[str, Number]


@dataclass
class _SynergyParams:
    # Budget & sizes
    eval_budget: int = 400          # total compile+run evaluations allowed
    max_seq_len: int = 12           # max length of chained sequence (in flags)
    min_seq_len: int = 4
    # Discovery throttles
    max_flags: Optional[int] = None # if set, random-subselect from flag atoms
    per_b_max_A: int = 25           # when probing (A,B), cap number of A per B
    # Strategy
    strategy: str = "ga"            # "ga" | "random" | "greedy"
    # GA params
    ga_pop: int = 24
    ga_elite: int = 4
    ga_gens: int = 20
    ga_mut: float = 0.25
    # Objective
    epsilon: float = 0.0            # require improvement > epsilon to count as synergy
    # Files
    out_dir: Optional[str] = None
    pairs_csv: str = "synergy_pairs.csv"
    results_csv: str = "synergy_results.csv"


def _get_synergy_params(cfg: Config) -> _SynergyParams:
    sy = getattr(cfg, "synergy", None)
    p = _SynergyParams()
    if sy:
        for k in _SynergyParams().__dict__.keys():
            if hasattr(sy, k): setattr(p, k, getattr(sy, k))
    return p


def _choose_objective(cfg: Config) -> Tuple[str, str]:
    """
    Returns (metric_name, direction) with direction in {"min","max"}.
    Uses the first objective if multiple provided.
    """
    if not cfg.objectives:
        raise ValueError("No objectives in config; cannot run synergy study.")
    obj = cfg.objectives[0]
    metric = obj.metric
    direction = "min" if obj.goal == "min" else "max"
    return metric, direction


def _collect_flag_atoms(cfg: Config, params: _SynergyParams) -> List[str]:
    """
    Pulls a flat list of atomic flags to toggle/combine.
    You can supply cfg.synergy.flag_atoms (preferred). Otherwise we try fallbacks.
    """
    # Preferred explicit list
    explicit = getattr(getattr(cfg, "synergy", object()), "flag_atoms", None)
    if explicit:
        atoms = list(dict.fromkeys(explicit))  # de-dup, keep order
    else:
        # Fallbacks: try pools the project likely already has.
        #  - cfg.compiler_flag_pool: iterable of flag strings
        #  - cfg.compiler_flags: dict name->flag-string (use values)
        atoms: List[str] = []
        pool = getattr(cfg, "compiler_flag_pool", None)
        if pool:
            atoms.extend(list(pool))
        cf = getattr(cfg, "compiler_flags", None)
        if cf and isinstance(cf, dict):
            atoms.extend(list(cf.values()))
        atoms = list(dict.fromkeys(atoms))
    if not atoms:
        raise ValueError(
            "No flag atoms found. Provide cfg.synergy.flag_atoms (recommended) "
            "or ensure compiler_flag_pool / compiler_flags are set."
        )
    if params.max_flags and len(atoms) > params.max_flags:
        atoms = random.sample(atoms, params.max_flags)
    return atoms


def _compile_and_measure(cfg: Config, flags: Sequence[str], env: Dict[str, str], work: Path) -> Tuple[float, MetricDict, str]:
    """
    Builds with `flags` (ordered), measures metrics, and returns (target_value, metrics, binary_path_str).
    """
    work.mkdir(parents=True, exist_ok=True)
    if cfg.source:
        bin_path = compile_single_source(cfg.compiler, cfg.source, list(flags), work / "a.out", None)
    else:
        bin_path = compile_project(cfg.project, cfg.compiler, list(flags), work, None)
    if not bin_path:
        raise RuntimeError("build failed")
    if cfg.backend == "perf":
        metrics = measure_perf(cfg.perf, bin_path, cfg.program_args, env)  # type: ignore[arg-type]
    else:
        metrics = measure_likwid(cfg.likwid, bin_path, cfg.program_args, env, cfg.runs)  # type: ignore[arg-type]
    metric_name, direction = _choose_objective(cfg)
    val = metrics[metric_name]
    if direction == "max":
        val = -val  # internally convert to "minimize"
    return float(val), metrics, str(bin_path)


def _best(a: float, b: float) -> float:
    return a if a <= b else b


def _flag_tuple(flags: Sequence[str]) -> Tuple[str, ...]:
    # keep order; dedup consecutive duplicates if any
    out: List[str] = []
    for f in flags:
        if not out or out[-1] != f:
            out.append(f)
    return tuple(out)


def _seq_value(cache: Dict[Tuple[str, ...], float],
               eval_fn,
               flags: Sequence[str],
               env: Dict[str, str],
               workroot: Path) -> float:
    key = _flag_tuple(flags)
    if key in cache:
        return cache[key]
    trial_dir = workroot / ("seq_" + "_".join(f.replace("-", "").replace("=", "")[:8] for f in key))  # short-ish
    try:
        val, _metrics, _bin = eval_fn(flags, env, trial_dir)
    except Exception:
        val = float("inf")
    cache[key] = val
    return val


def _discover_beneficial_and_pairs(cfg: Config,
                                   atoms: List[str],
                                   params: _SynergyParams,
                                   base_env: Dict[str, str],
                                   workroot: Path):
    """
    1) baseline value
    2) check each B alone -> keep beneficial Bs
    3) for each beneficial B, probe (A,B) improvements (cap per_b_max_A)
    Returns:
      baseline_val, beneficial_Bs, pairs(list of (A,B)), single_vals(dict B->val), pair_vals(dict (A,B)->val)
    """
    eval_cache: Dict[Tuple[str, ...], float] = {}
    def eval_fn(flags, env, wd): return _compile_and_measure(cfg, flags, env, wd)

    baseline_flags: List[str] = list(getattr(getattr(cfg, "synergy", object()), "base_flags", []))
    base_val = _seq_value(eval_cache, eval_fn, baseline_flags, base_env, workroot)

    # Step 2: check B alone
    single_vals: Dict[str, float] = {}
    beneficial_Bs: List[str] = []
    for b in atoms:
        val_b = _seq_value(eval_cache, eval_fn, baseline_flags + [b], base_env, workroot)
        single_vals[b] = val_b
        if val_b + params.epsilon < base_val:
            beneficial_Bs.append(b)

    # Step 3: probe (A,B)
    pairs: List[Tuple[str, str]] = []
    pair_vals: Dict[Tuple[str, str], float] = {}
    for b in beneficial_Bs:
        # Shuffle A’s; limit to per_b_max_A
        As = [a for a in atoms if a != b]
        random.shuffle(As)
        As = As[: params.per_b_max_A] if params.per_b_max_A and params.per_b_max_A < len(As) else As
        for a in As:
            val_ab = _seq_value(eval_cache, eval_fn, baseline_flags + [a, b], base_env, workroot)
            if val_ab + params.epsilon < single_vals[b]:
                pairs.append((a, b))
                pair_vals[(a, b)] = val_ab

    return base_val, beneficial_Bs, pairs, single_vals, pair_vals, baseline_flags, eval_cache


def _pairs_to_graph(pairs: List[Tuple[str, str]]):
    out_edges: Dict[str, List[str]] = {}
    in_edges: Dict[str, List[str]] = {}
    for a, b in pairs:
        out_edges.setdefault(a, []).append(b)
        in_edges.setdefault(b, []).append(a)
        out_edges.setdefault(b, [])
        in_edges.setdefault(a, [])
    # de-dup adjacency
    for k in out_edges: out_edges[k] = sorted(list(set(out_edges[k])))
    for k in in_edges: in_edges[k] = sorted(list(set(in_edges[k])))
    return out_edges, in_edges


def _pairs_to_sequence(chain: List[Tuple[str, str]]) -> List[str]:
    """(A,B),(B,C),(C,D) -> [A,B,C,D]"""
    if not chain: return []
    seq = [chain[0][0], chain[0][1]]
    for i in range(1, len(chain)):
        _, b = chain[i]
        seq.append(b)
    return seq


def _random_chain(out_edges: Dict[str, List[str]], max_len: int) -> List[Tuple[str, str]]:
    if not out_edges: return []
    start = random.choice(list(out_edges.keys()))
    chain: List[Tuple[str, str]] = []
    cur = start
    while len(chain) < max_len - 1 and out_edges.get(cur):
        nxt = random.choice(out_edges[cur])
        chain.append((cur, nxt))
        cur = nxt
    if not chain and out_edges.get(start):
        # If we picked a start without outgoing edges, try to salvage one hop
        nxts = out_edges[start]
        if nxts:
            chain.append((start, random.choice(nxts)))
    return chain


def _greedy_search(out_edges, eval_cache, eval_fn, base_env, workroot, base_flags, budget, max_len):
    tried = 0
    best_val = float("inf")
    best_seq: List[str] = []
    # Pre-score single edges by one-step value
    scored_edges: List[Tuple[float, Tuple[str, str]]] = []
    for a, bs in out_edges.items():
        for b in bs:
            val = _seq_value(eval_cache, eval_fn, base_flags + [a, b], base_env, workroot)
            tried += 1
            scored_edges.append((val, (a, b)))
            if tried >= budget: break
        if tried >= budget: break
    scored_edges.sort(key=lambda t: t[0])
    # expand greedily from top-k seeds
    seeds = [e for _, e in scored_edges[: min(8, len(scored_edges))]]
    for a, b in seeds:
        chain = [(a, b)]
        while len(chain) < max_len - 1 and out_edges.get(chain[-1][1]):
            cand = [(chain[-1][1], c) for c in out_edges[chain[-1][1]]]
            best_local = None
            best_local_val = float("inf")
            for p in cand:
                seq = _pairs_to_sequence(chain + [p])
                val = _seq_value(eval_cache, eval_fn, base_flags + seq, base_env, workroot)
                tried += 1
                if val < best_local_val:
                    best_local_val, best_local = val, p
                if tried >= budget: break
            if not best_local:
                break
            chain.append(best_local)
            if tried >= budget: break
        seq = _pairs_to_sequence(chain)
        val = _seq_value(eval_cache, eval_fn, base_flags + seq, base_env, workroot)
        tried += 1
        if val < best_val:
            best_val, best_seq = val, seq
        if tried >= budget: break
    return best_seq, best_val, tried


def _ga_search(out_edges, eval_cache, eval_fn, base_env, workroot, base_flags, params: _SynergyParams):
    # population are *chains* (list of (A,B)), but fitness evaluated on flattened flag sequence
    def mk_individual() -> List[Tuple[str, str]]:
        return _random_chain(out_edges, params.max_seq_len)

    def fitness(chain: List[Tuple[str, str]]) -> float:
        seq = _pairs_to_sequence(chain)
        if len(seq) < max(2, params.min_seq_len):
            return float("inf")
        return _seq_value(eval_cache, eval_fn, base_flags + seq, base_env, workroot)

    def crossover(p1: List[Tuple[str, str]], p2: List[Tuple[str, str]]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
        # Find common node; swap suffixes after it (as in paper). If none, return parents.
        if not p1 or not p2: return p1, p2
        nodes1 = [p1[0][0]] + [b for _, b in p1]
        nodes2 = [p2[0][0]] + [b for _, b in p2]
        commons = set(nodes1).intersection(nodes2)
        if not commons: return p1, p2
        k = random.choice(list(commons))
        # cut after first occurrence of k in each
        def cut_at(chain, k):
            nodes = [chain[0][0]] + [b for _, b in chain]
            idx = nodes.index(k)  # node index
            # chain edges up to idx-1 remain; suffix starts at edge whose src is node k
            # find edge pos whose src node equals k (which is nodes[idx])
            pos = 0
            cur_src = chain[0][0]
            while pos < len(chain) and cur_src != k:
                cur_src = chain[pos][1]
                pos += 1
            prefix = chain[:pos]
            suffix = chain[pos:]
            return prefix, suffix
        p1pre, p1suf = cut_at(p1, k)
        p2pre, p2suf = cut_at(p2, k)
        c1 = p1pre + p2suf
        c2 = p2pre + p1suf
        return c1[: params.max_seq_len - 1], c2[: params.max_seq_len - 1]

    def mutate(ind: List[Tuple[str, str]]) -> None:
        if not ind:  # turn a fresh random
            ind[:] = _random_chain(out_edges, params.max_seq_len)
            return
        # pick a position; try to re-route from there using out_edges
        pos = random.randrange(0, len(ind))
        src = ind[pos][0] if pos == 0 else ind[pos - 1][1]
        options = out_edges.get(src, [])
        if options:
            nxt = random.choice(options)
            ind[pos] = (src, nxt)
            # truncate or try to rebuild suffix
            cur = nxt
            new_tail: List[Tuple[str, str]] = []
            while len(ind[:pos + 1] + new_tail) < params.max_seq_len - 1 and out_edges.get(cur):
                c = random.choice(out_edges[cur])
                new_tail.append((cur, c))
                cur = c
            ind[pos + 1:] = new_tail

    # init
    pop = [mk_individual() for _ in range(params.ga_pop)]
    tried = 0
    best_val = float("inf")
    best_seq: List[str] = []
    metric_name, direction = _choose_objective  # noqa: keep for context

    def eval_and_update(chain):
        nonlocal tried, best_val, best_seq
        val = fitness(chain)
        tried += 1
        if val < best_val:
            best_val = val
            best_seq = _pairs_to_sequence(chain)
        return val

    # score initial pop
    scores = [eval_and_update(ind) for ind in pop]
    if tried >= params.eval_budget:
        return best_seq, best_val, tried

    for _g in range(params.ga_gens):
        # selection: tournament of size 3
        def pick():
            ks = random.sample(range(len(pop)), min(3, len(pop)))
            ks.sort(key=lambda i: scores[i])
            return pop[ks[0]]
        # elites
        ranked = list(range(len(pop)))
        ranked.sort(key=lambda i: scores[i])
        new_pop = [pop[i][:] for i in ranked[: params.ga_elite]]
        # offspring
        while len(new_pop) < params.ga_pop:
            p1 = pick()
            p2 = pick()
            c1, c2 = crossover(p1[:], p2[:])
            if random.random() < params.ga_mut: mutate(c1)
            if random.random() < params.ga_mut: mutate(c2)
            new_pop.append(c1)
            if len(new_pop) < params.ga_pop:
                new_pop.append(c2)
        pop = new_pop
        scores = [eval_and_update(ind) for ind in pop]
        if tried >= params.eval_budget:
            break

    return best_seq, best_val, tried


def _random_search(out_edges, eval_cache, eval_fn, base_env, workroot, base_flags, budget, max_len, min_len):
    tried = 0
    best_val = float("inf")
    best_seq: List[str] = []
    while tried < budget:
        chain = _random_chain(out_edges, max_len)
        seq = _pairs_to_sequence(chain)
        if len(seq) < max(2, min_len):
            tried += 1
            continue
        val = _seq_value(eval_cache, eval_fn, base_flags + seq, base_env, workroot)
        tried += 1
        if val < best_val:
            best_val, best_seq = val, seq
    return best_seq, best_val, tried


def run_synergy_study(cfg: Config) -> None:
    """
    End-to-end: discover chained synergy flag pairs, form graph, run search.
    Logs:
      - synergy_pairs.csv : (A,B) pairs discovered
      - synergy_results.csv: running log of candidate sequences and values
    """
    params = _get_synergy_params(cfg)
    metric_name, direction = _choose_objective(cfg)
    print(f"[synergy] objective: {metric_name} ({direction})")

    atoms = _collect_flag_atoms(cfg, params)
    print(f"[synergy] candidate flag atoms: {len(atoms)}")

    base_env: Dict[str, str] = {}
    workroot = Path(tempfile.mkdtemp(prefix="SCOuT_synergy_"))
    out_dir = Path(params.out_dir or workroot)
    out_dir.mkdir(parents=True, exist_ok=True)

    # === DISCOVERY ===
    base_val, beneficial_Bs, pairs, single_vals, pair_vals, base_flags, eval_cache = _discover_beneficial_and_pairs(
        cfg, atoms, params, base_env, workroot
    )

    print(f"[synergy] baseline value: {base_val:.6g}")
    print(f"[synergy] beneficial single flags: {len(beneficial_Bs)}")
    print(f"[synergy] discovered pairs: {len(pairs)}")

    # write pairs CSV
    pairs_csv = out_dir / params.pairs_csv
    with pairs_csv.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["A", "B", "val(B)", "val(A,B)", "improvement_vs_B"])
        for (a, b) in sorted(set(pairs)):
            v_b = single_vals[b]
            v_ab = pair_vals.get((a, b), math.inf)
            w.writerow([a, b, v_b, v_ab, v_b - v_ab])

    # === GRAPH + SEARCH ===
    out_edges, _in_edges = _pairs_to_graph(pairs)

    def eval_fn(flags, env, wd): return _compile_and_measure(cfg, flags, env, wd)
    results_csv = out_dir / params.results_csv
    with results_csv.open("w", newline="") as fp:
        w = csv.writer(fp)
        w.writerow(["strategy", "seq_len", "value", "flags"])
        # choose strategy
        if params.strategy == "greedy":
            seq, val, tried = _greedy_search(out_edges, eval_cache, eval_fn, base_env, workroot, base_flags, params.eval_budget, params.max_seq_len)
            w.writerow(["greedy", len(seq), val, json.dumps(base_flags + seq)])
        elif params.strategy == "random":
            seq, val, tried = _random_search(out_edges, eval_cache, eval_fn, base_env, workroot, base_flags, params.eval_budget, params.max_seq_len, params.min_seq_len)
            w.writerow(["random", len(seq), val, json.dumps(base_flags + seq)])
        else:
            seq, val, tried = _ga_search(out_edges, eval_cache, eval_fn, base_env, workroot, base_flags, params)
            w.writerow(["ga", len(seq), val, json.dumps(base_flags + seq)])

    print(f"[synergy] results → {results_csv}")
    print(f"[synergy] pairs   → {pairs_csv}")
