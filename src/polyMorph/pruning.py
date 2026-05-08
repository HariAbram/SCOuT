from __future__ import annotations

from typing import Any, Dict, List


JsonDict = Dict[str, Any]


def static_prune_candidate(candidate: JsonDict, constraints: JsonDict | None = None) -> List[str]:
    constraints = constraints or {}
    features = candidate.get("features", {}) or {}
    name = str(candidate.get("tr", ""))
    args = list(candidate.get("args", []))
    reasons: List[str] = []

    max_tile = int(constraints.get("max_tile_size", 256))
    max_tile_volume = int(constraints.get("max_tile_volume", 65536))
    min_tile = int(constraints.get("min_tile_size", 1))
    max_shift = int(constraints.get("max_abs_shift", 128))

    if name.startswith("TILE"):
        numeric_args = [int(x) for x in args if isinstance(x, (int, float))]
        if not numeric_args:
            reasons.append("tile transform has no numeric tile sizes")
        if any(x < min_tile for x in numeric_args):
            reasons.append(f"tile size below min_tile_size={min_tile}")
        if any(x > max_tile for x in numeric_args):
            reasons.append(f"tile size above max_tile_size={max_tile}")
        if float(features.get("tile_volume", 0.0)) > max_tile_volume:
            reasons.append(f"tile volume above max_tile_volume={max_tile_volume}")

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

    for (scop, node, name), count in counts.items():
        if count > max_same_transform:
            reasons.append(f"{name} appears {count} times on scop={scop}, node={node}")

    return reasons


def analytical_score(candidate: JsonDict, constraints: JsonDict | None = None) -> JsonDict:
    constraints = constraints or {}
    features = candidate.get("features", {}) or {}
    name = str(candidate.get("tr", ""))
    score = 0.5
    risk = 0.1
    reasons: List[str] = []

    if features.get("is_tile"):
        tile_volume = float(features.get("tile_volume", 0.0) or 0.0)
        max_tile_size = float(features.get("max_tile_size", 0.0) or 0.0)
        preferred = float(constraints.get("preferred_tile_size", 32))
        if max_tile_size:
            distance = abs(max_tile_size - preferred) / max(preferred, 1.0)
            score += max(0.0, 0.25 - 0.15 * distance)
            reasons.append("tile size close to preferred cache/vector heuristic")
        if tile_volume > float(constraints.get("large_tile_volume", 4096)):
            risk += 0.25
            reasons.append("large tile volume may increase register/cache pressure")

    if features.get("is_interchange"):
        score += 0.12
        risk += 0.08
        reasons.append("interchange can improve stride but may affect dependences")

    if features.get("is_fuse"):
        score += 0.10
        risk += 0.15
        reasons.append("fusion can improve reuse but may increase live ranges")

    if features.get("is_split"):
        score -= 0.03
        risk += 0.12
        reasons.append("split may expose parallelism but can increase traffic")

    if features.get("is_shift"):
        score -= 0.02
        risk += min(0.25, 0.02 * float(features.get("max_abs_shift", 0.0)))
        reasons.append("shift has uncertain locality impact")

    if features.get("is_parallel_hint"):
        score += 0.18
        risk += 0.05
        reasons.append("parallel hint may expose hardware occupancy")

    access_hints = float(features.get("access_hint_count", 0.0) or 0.0)
    loop_hints = float(features.get("loop_hint_count", 0.0) or 0.0)
    if loop_hints and access_hints / loop_hints > 2.0:
        risk += 0.05
        reasons.append("memory-access-heavy loop summary")

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
