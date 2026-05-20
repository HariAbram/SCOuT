from __future__ import annotations

from typing import Any, Dict, List


JsonDict = Dict[str, Any]


def _classification(candidate: JsonDict) -> JsonDict:
    return dict(candidate.get("scop_classification", {}) or {})


def _labels(candidate: JsonDict) -> set[str]:
    labels = _classification(candidate).get("labels", []) or []
    return {str(label) for label in labels}


def static_prune_candidate(candidate: JsonDict, constraints: JsonDict | None = None) -> List[str]:
    constraints = constraints or {}
    features = candidate.get("features", {}) or {}
    classification = _classification(candidate)
    labels = _labels(candidate)
    name = str(candidate.get("tr", ""))
    args = list(candidate.get("args", []))
    reasons: List[str] = []

    max_tile = int(constraints.get("max_tile_size", 256))
    max_tile_volume = int(constraints.get("max_tile_volume", 65536))
    min_tile = int(constraints.get("min_tile_size", 1))
    max_shift = int(constraints.get("max_abs_shift", 128))

    if name.startswith("TILE"):
        numeric_args = [int(x) for x in args if isinstance(x, (int, float))]
        loop_rank = int(classification.get("loop_rank", 0) or 0)
        tile_rank = len(numeric_args)
        if not numeric_args:
            reasons.append("tile transform has no numeric tile sizes")
        if loop_rank and tile_rank > loop_rank:
            reasons.append(f"{name} rank {tile_rank} exceeds loop_rank={loop_rank}")
        if any(x < min_tile for x in numeric_args):
            reasons.append(f"tile size below min_tile_size={min_tile}")
        if any(x > max_tile for x in numeric_args):
            reasons.append(f"tile size above max_tile_size={max_tile}")
        if float(features.get("tile_volume", 0.0)) > max_tile_volume:
            reasons.append(f"tile volume above max_tile_volume={max_tile_volume}")
        if "small_scop" in labels and bool(constraints.get("prune_tiling_small_scops", False)):
            reasons.append("tiling small SCoP disabled by prune_tiling_small_scops")

    if name in {"FUSE", "FULL_FUSE"} and "sequence" not in labels:
        reasons.append(f"{name} requires a sequence-like SCoP")

    if name == "INTERCHANGE" and int(classification.get("loop_rank", 0) or 0) < 2:
        reasons.append("INTERCHANGE requires loop_rank >= 2")

    if "SHIFT" in name and float(features.get("max_abs_shift", 0.0)) > max_shift:
        reasons.append(f"shift magnitude above max_abs_shift={max_shift}")

    if name in {"FUSE", "SPLIT", "PARTIAL_SHIFT_VAL"} and not args:
        reasons.append(f"{name} requires explicit args")

    yaml_bytes = int(features.get("yaml_bytes", 0) or 0)
    max_yaml_bytes = int(constraints.get("max_yaml_bytes", 0) or 0)
    if max_yaml_bytes and yaml_bytes > max_yaml_bytes:
        reasons.append(f"schedule YAML larger than max_yaml_bytes={max_yaml_bytes}")

    return reasons


def static_prune_sequence(specs: List[JsonDict], constraints: JsonDict | None = None) -> List[str]:
    constraints = constraints or {}
    reasons: List[str] = []
    seen = set()
    max_same_transform = int(constraints.get("max_same_transform_per_sequence", 2))
    counts: Dict[tuple[int, int, str], int] = {}
    scop_counts: Dict[int, int] = {}
    max_per_scop = int(constraints.get("max_transforms_per_scop", 0) or 0)

    for spec in specs:
        scop = int(spec.get("scop", -1))
        node = int(spec.get("node", -1))
        name = str(spec.get("tr", ""))
        key = (
            scop,
            node,
            name,
            tuple(spec.get("args", [])),
        )
        if key in seen:
            reasons.append("duplicate transform in sequence")
        seen.add(key)
        target_key = (scop, node, name)
        counts[target_key] = counts.get(target_key, 0) + 1
        scop_counts[scop] = scop_counts.get(scop, 0) + 1

    for (scop, node, name), count in counts.items():
        if count > max_same_transform:
            reasons.append(f"{name} appears {count} times on scop={scop}, node={node}")
    if max_per_scop > 0:
        for scop, count in scop_counts.items():
            if count > max_per_scop:
                reasons.append(f"{count} transforms target scop={scop}; max_transforms_per_scop={max_per_scop}")

    return reasons


def analytical_score(candidate: JsonDict, constraints: JsonDict | None = None) -> JsonDict:
    constraints = constraints or {}
    features = candidate.get("features", {}) or {}
    classification = _classification(candidate)
    labels = _labels(candidate)
    name = str(candidate.get("tr", ""))
    score = 0.5
    risk = 0.1
    reasons: List[str] = []
    loop_rank = int(classification.get("loop_rank", 0) or 0)
    sequence_child_count = int(classification.get("sequence_child_count", 0) or 0)
    pluto_max_distance = float(features.get("pluto_max_distance", 0.0) or 0.0)
    pluto_distance_sum = float(features.get("pluto_distance_sum", 0.0) or 0.0)
    pluto_reuse_pairs = int(features.get("pluto_reuse_pair_count", 0) or 0)
    pluto_exact_pairs = int(features.get("pluto_exact_reuse_pair_count", pluto_reuse_pairs) or 0)
    pluto_stream_pairs = int(features.get("pluto_stream_pair_count", 0) or 0)
    stream_pair_weight = max(0.0, min(1.0, float(constraints.get("pluto_stream_pair_weight", 0.15))))
    pluto_effective_pairs = float(pluto_exact_pairs) + stream_pair_weight * float(pluto_stream_pairs)
    pluto_carried_dims = int(features.get("pluto_carried_dims", 0) or 0)
    pluto_by_dim = [
        float(value)
        for value in (features.get("pluto_distance_by_dim", []) or [])
        if isinstance(value, (int, float))
    ]
    pluto_enabled = bool(constraints.get("pluto_cost_model", True))
    pluto_weight = float(constraints.get("pluto_cost_weight", 1.0) or 1.0)

    if features.get("is_tile"):
        tile_volume = float(features.get("tile_volume", 0.0) or 0.0)
        max_tile_size = float(features.get("max_tile_size", 0.0) or 0.0)
        tile_rank = int(features.get("numeric_arg_count", 0) or 0)
        preferred = float(constraints.get("preferred_tile_size", 32))
        if pluto_enabled and pluto_effective_pairs > 0.0 and max_tile_size > 0:
            covered_distance = min(1.0, max_tile_size / max(pluto_max_distance + 1.0, 1.0))
            rank_coverage = min(tile_rank, max(pluto_carried_dims, 1)) / max(pluto_carried_dims, 1)
            confidence = min(1.0, pluto_effective_pairs / max(float(pluto_reuse_pairs), 1.0))
            pluto_bonus = confidence * min(0.22, 0.08 + 0.10 * covered_distance + 0.06 * rank_coverage)
            score += pluto_weight * pluto_bonus
            if pluto_exact_pairs > 0:
                reasons.append(
                    "Pluto-style tiling prior: tile bounds estimated affine reuse/dependence distance"
                )
            else:
                risk += 0.06
                reasons.append(
                    "weak Pluto tiling prior: only stream-shaped access pairs were found"
                )
            if tile_rank < pluto_carried_dims:
                risk += 0.08
                reasons.append(
                    f"tile rank {tile_rank} covers fewer carried dimensions than Pluto estimate {pluto_carried_dims}"
                )
            if max_tile_size > 0 and pluto_max_distance > 0 and max_tile_size > preferred * 2:
                risk += 0.05
                reasons.append("tile size exceeds preferred size despite Pluto distance coverage")
        elif max_tile_size:
            fallback_weight = float(
                constraints.get(
                    "preferred_tile_size_fallback_weight",
                    0.20 if pluto_enabled else 1.0,
                )
                or 0.0
            )
            if fallback_weight > 0.0:
                distance = abs(max_tile_size - preferred) / max(preferred, 1.0)
                score += fallback_weight * max(0.0, 0.14 - 0.08 * distance)
                reasons.append("weak preferred tile-size fallback heuristic")
            elif pluto_enabled:
                score += 0.03
                reasons.append("Pluto evidence absent; tile size left mostly exploratory")
        if loop_rank and tile_rank > loop_rank:
            score -= 0.35
            risk += 0.35
            reasons.append(f"tile rank {tile_rank} exceeds classified loop_rank={loop_rank}")
        elif loop_rank >= 2 and tile_rank >= 2:
            score += 0.04
            reasons.append("multi-dimensional tile matches loop nest rank")
        elif loop_rank <= 1 and tile_rank == 1:
            score += 0.02
            reasons.append("1D tile matches 1D loop classification")
        if "small_scop" in labels:
            score -= 0.10
            risk += 0.08
            reasons.append("small SCoP may not amortize tiling overhead")
        if "memory_heavy_hint" in labels:
            score += 0.05
            reasons.append("memory-heavy SCoP may benefit from locality tiling")
        if tile_volume > float(constraints.get("large_tile_volume", 4096)):
            risk += 0.25
            reasons.append("large tile volume may increase register/cache pressure")

    if features.get("is_interchange"):
        if loop_rank >= 2:
            if pluto_enabled and len(pluto_by_dim) >= 2 and max(pluto_by_dim) > 0:
                outer_distance = pluto_by_dim[0]
                inner_distance = pluto_by_dim[-1]
                if outer_distance > inner_distance:
                    score += pluto_weight * 0.20
                    risk += 0.06
                    reasons.append(
                        "Pluto-style interchange prior: larger estimated distance is carried by an outer dimension"
                    )
                else:
                    score += 0.06
                    risk += 0.10
                    reasons.append(
                        "interchange has limited Pluto distance-reduction opportunity"
                    )
            else:
                score += 0.16
                reasons.append("interchange matches multi-dimensional loop nest")
            risk += 0.08
        else:
            score -= 0.20
            risk += 0.20
            reasons.append("interchange on low-rank loop is unlikely to help")

    if features.get("is_fuse"):
        if "sequence" in labels and sequence_child_count >= 2:
            score += 0.14
            risk += 0.12
            reasons.append("sequence SCoP has adjacent children that may benefit from fusion")
            if pluto_enabled and pluto_exact_pairs > 0:
                score += min(0.08, 0.02 * pluto_exact_pairs)
                reasons.append("Pluto-style fusion prior: repeated affine accesses indicate reuse across statements")
        else:
            score -= 0.18
            risk += 0.20
            reasons.append("fusion without sequence structure is unlikely to be useful")

    if features.get("is_split"):
        if loop_rank >= 2:
            score += 0.02
            risk += 0.12
            reasons.append("split may expose structure in deeper loop nests")
        else:
            score -= 0.08
            risk += 0.14
            reasons.append("split on shallow loop may increase overhead")

    if features.get("is_shift"):
        score -= 0.02
        risk += min(0.25, 0.02 * float(features.get("max_abs_shift", 0.0)))
        if pluto_enabled and pluto_max_distance > 0:
            shift = float(features.get("max_abs_shift", 0.0) or 0.0)
            if 0 < shift <= pluto_max_distance:
                score += min(0.06, 0.02 * shift)
                reasons.append("Pluto-style shift prior: shift magnitude is within estimated dependence distance")
        if "small_scop" in labels:
            score -= 0.03
            risk += 0.03
        reasons.append("shift has uncertain locality impact")

    if features.get("is_parallel_hint"):
        score += 0.18
        risk += 0.05
        if "small_scop" in labels:
            score -= 0.05
            reasons.append("small SCoP may not benefit from extra parallel hint overhead")
        reasons.append("parallel hint may expose hardware occupancy")

    access_hints = float(features.get("access_hint_count", 0.0) or 0.0)
    loop_hints = float(features.get("loop_hint_count", 0.0) or 0.0)
    if loop_hints and access_hints / loop_hints > 2.0:
        risk += 0.05
        reasons.append("memory-access-heavy loop summary")
    if pluto_enabled and pluto_effective_pairs > 0:
        reasons.append(
            f"Pluto distance summary: max={pluto_max_distance:.1f}, sum={pluto_distance_sum:.1f}, "
            f"exact_pairs={pluto_exact_pairs}, stream_pairs={pluto_stream_pairs}, effective_pairs={pluto_effective_pairs:.1f}"
        )

    return {
        "score": max(0.0, min(1.0, score)),
        "risk": max(0.0, min(1.0, risk)),
        "reasons": reasons,
    }


def violates_constraints(candidate: JsonDict, constraints: JsonDict | None = None) -> List[str]:
    constraints = constraints or {}
    reasons: List[str] = []
    predictions = candidate.get("predictions", {}) or {}
    risk = float(predictions.get("risk", 0.0) or 0.0)
    score = float(predictions.get("score", 0.0) or 0.0)

    max_risk = constraints.get("max_predicted_risk")
    if max_risk is not None and risk > float(max_risk):
        reasons.append(f"predicted risk {risk:.3f} exceeds max_predicted_risk={max_risk}")

    min_score = constraints.get("min_predicted_score")
    if min_score is not None and score < float(min_score):
        reasons.append(f"predicted score {score:.3f} below min_predicted_score={min_score}")

    return reasons
