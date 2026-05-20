from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Dict, List


JsonDict = Dict[str, Any]


def stable_json_hash(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, default=str)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _count_yaml_items(text: str, keys: List[str]) -> int:
    total = 0
    for key in keys:
        total += len(re.findall(rf"\b{re.escape(key)}\b", text, flags=re.IGNORECASE))
    return total


def node_features(scop_idx: int, node_idx: int, node: Any) -> JsonDict:
    yaml_text = str(getattr(node, "yaml_str", "") or "")
    available = getattr(node, "available_transformations", []) or []
    available_names = [getattr(item, "name", str(item)) for item in available]
    node_type = str(getattr(node, "node_type", "<unknown>"))
    pluto_cost = simplified_pluto_cost_features(yaml_text)

    return {
        "scop": scop_idx,
        "node": node_idx,
        "node_type": node_type,
        "available_transform_count": len(available_names),
        "available_transformations": available_names,
        "yaml_bytes": len(yaml_text.encode("utf-8")),
        "yaml_lines": len(yaml_text.splitlines()),
        "loop_hint_count": _count_yaml_items(yaml_text, ["for", "loop", "schedule", "band"]),
        "statement_hint_count": _count_yaml_items(yaml_text, ["statement", "stmt", "domain"]),
        "access_hint_count": _count_yaml_items(yaml_text, ["read", "write", "access", "array"]),
        **pluto_cost,
    }


def transform_features(candidate: JsonDict) -> JsonDict:
    name = str(candidate.get("tr", ""))
    args = list(candidate.get("args", []))
    numeric_args = [float(x) for x in args if isinstance(x, (int, float))]
    tile_args = numeric_args if name.startswith("TILE") else []
    shift_args = numeric_args if "SHIFT" in name else []

    return {
        "transform": name,
        "arg_count": len(args),
        "numeric_arg_count": len(numeric_args),
        "max_numeric_arg": max(numeric_args) if numeric_args else 0.0,
        "tile_volume": _product(tile_args) if tile_args else 0.0,
        "max_tile_size": max(tile_args) if tile_args else 0.0,
        "max_abs_shift": max((abs(x) for x in shift_args), default=0.0),
        "is_tile": name.startswith("TILE"),
        "is_interchange": name == "INTERCHANGE",
        "is_fuse": "FUSE" in name,
        "is_split": "SPLIT" in name,
        "is_shift": "SHIFT" in name,
        "is_parallel_hint": name in {"SET_PARALLEL", "SET_LOOP_OPT"},
    }


def enrich_candidate(candidate: JsonDict, node: Any) -> JsonDict:
    enriched = dict(candidate)
    features = {}
    features.update(node_features(int(candidate["scop"]), int(candidate["node"]), node))
    features.update(transform_features(candidate))
    enriched["features"] = features
    enriched.setdefault("predictions", {})
    enriched.setdefault("prune_reasons", [])
    return enriched


def sequence_signature(specs: List[JsonDict]) -> str:
    compact = [
        {
            "scop": int(spec["scop"]),
            "node": int(spec["node"]),
            "tr": str(spec["tr"]),
            "args": list(spec.get("args", [])),
        }
        for spec in specs
    ]
    return stable_json_hash(compact)


def candidate_key(candidate: JsonDict) -> str:
    return sequence_signature([candidate])


def structural_signature(candidates: List[JsonDict]) -> JsonDict:
    transform_counts: Dict[str, int] = {}
    node_type_counts: Dict[str, int] = {}
    available_counts: Dict[str, int] = {}
    scop_ids: set[int] = set()
    node_ids: set[tuple[int, int]] = set()

    for candidate in candidates:
        features = candidate.get("features", {}) or {}
        transform = str(candidate.get("tr", ""))
        transform_counts[transform] = transform_counts.get(transform, 0) + 1
        node_type = str(features.get("node_type", ""))
        if node_type:
            node_type_counts[node_type] = node_type_counts.get(node_type, 0) + 1
        for name in features.get("available_transformations", []) or []:
            name = str(name)
            available_counts[name] = available_counts.get(name, 0) + 1
        scop_ids.add(int(candidate.get("scop", -1)))
        node_ids.add((int(candidate.get("scop", -1)), int(candidate.get("node", -1))))

    return {
        "scop_count": len(scop_ids),
        "node_count": len(node_ids),
        "transform_counts": transform_counts,
        "node_type_counts": node_type_counts,
        "available_transform_counts": available_counts,
    }


def sequence_feature_summary(specs: List[JsonDict]) -> JsonDict:
    transform_counts: Dict[str, int] = {}
    max_tile_size = 0.0
    tile_volume_sum = 0.0
    max_abs_shift = 0.0
    scop_ids: set[int] = set()
    node_ids: set[tuple[int, int]] = set()

    for spec in specs:
        name = str(spec.get("tr", ""))
        transform_counts[name] = transform_counts.get(name, 0) + 1
        features = spec.get("features") or transform_features(spec)
        max_tile_size = max(max_tile_size, float(features.get("max_tile_size", 0.0) or 0.0))
        tile_volume_sum += float(features.get("tile_volume", 0.0) or 0.0)
        max_abs_shift = max(max_abs_shift, float(features.get("max_abs_shift", 0.0) or 0.0))
        scop_ids.add(int(spec.get("scop", -1)))
        node_ids.add((int(spec.get("scop", -1)), int(spec.get("node", -1))))

    return {
        "length": len(specs),
        "scop_count": len(scop_ids),
        "node_count": len(node_ids),
        "transform_counts": transform_counts,
        "max_tile_size": max_tile_size,
        "tile_volume_sum": tile_volume_sum,
        "max_abs_shift": max_abs_shift,
    }


def structural_similarity(left: JsonDict | None, right: JsonDict | None) -> float:
    if not left or not right:
        return 0.0
    scores = [
        _counter_similarity(
            dict(left.get("transform_counts", {}) or {}),
            dict(right.get("transform_counts", {}) or {}),
        ),
        _counter_similarity(
            dict(left.get("node_type_counts", {}) or {}),
            dict(right.get("node_type_counts", {}) or {}),
        ),
        _counter_similarity(
            dict(left.get("available_transform_counts", {}) or {}),
            dict(right.get("available_transform_counts", {}) or {}),
        ),
    ]
    for key in ["scop_count", "node_count"]:
        a = float(left.get(key, 0) or 0)
        b = float(right.get(key, 0) or 0)
        if max(a, b) > 0:
            scores.append(1.0 - abs(a - b) / max(a, b))
    return sum(scores) / len(scores) if scores else 0.0


def _counter_similarity(left: Dict[str, int], right: Dict[str, int]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 0.0
    intersection = sum(min(int(left.get(key, 0)), int(right.get(key, 0))) for key in keys)
    union = sum(max(int(left.get(key, 0)), int(right.get(key, 0))) for key in keys)
    return intersection / union if union else 0.0


def _product(values: List[float]) -> float:
    result = 1.0
    for value in values:
        result *= value
    return result


_ACCESS_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\[([^\]]+)\]")
_LOOP_VAR_RE = re.compile(r"\bi(\d+)\b")
_INT_RE = re.compile(r"(?<![A-Za-z_])[-+]?\d+(?![A-Za-z_])")


def _split_indices(text: str) -> List[str]:
    parts: List[str] = []
    current: List[str] = []
    depth = 0
    for char in text:
        if char in "([{":
            depth += 1
        elif char in ")]}" and depth > 0:
            depth -= 1
        if char == "," and depth == 0:
            part = "".join(current).strip()
            if part:
                parts.append(part)
            current = []
            continue
        current.append(char)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


def _affine_signature(expr: str, loop_rank: int) -> tuple[tuple[int, ...], int]:
    normalized = expr.replace("-", "+-")
    coeffs = [0 for _ in range(loop_rank)]
    for match in _LOOP_VAR_RE.finditer(expr):
        dim = int(match.group(1))
        if dim >= loop_rank:
            continue
        prefix = expr[: match.start()]
        sign = -1 if prefix.rstrip().endswith("-") else 1
        coeffs[dim] += sign
    constant = 0
    for raw in _INT_RE.findall(normalized):
        try:
            constant += int(raw)
        except ValueError:
            continue
    return tuple(coeffs), constant


def simplified_pluto_cost_features(yaml_text: str) -> JsonDict:
    """Estimate local affine reuse/dependence distances from access text.

    This is intentionally conservative: without exact dependence analysis, we
    only compare repeated accesses to the same array and summarize constant
    offset distances across loop dimensions. The resulting quantities are used
    as a Pluto-inspired prior, not as a legality proof.
    """
    dims = {int(match.group(1)) for match in _LOOP_VAR_RE.finditer(yaml_text)}
    loop_rank = max(dims) + 1 if dims else 0
    if loop_rank <= 0:
        return {
            "pluto_loop_rank": 0,
            "pluto_access_count": 0,
            "pluto_reuse_pair_count": 0,
            "pluto_max_distance": 0.0,
            "pluto_distance_sum": 0.0,
            "pluto_distance_by_dim": [],
            "pluto_carried_dims": 0,
            "pluto_reduction_opportunity": 0.0,
        }

    accesses: Dict[str, List[List[tuple[tuple[int, ...], int]]]] = {}
    for array_name, indices_text in _ACCESS_RE.findall(yaml_text):
        index_exprs = _split_indices(indices_text)
        if not index_exprs:
            continue
        signatures = [_affine_signature(expr, loop_rank) for expr in index_exprs]
        accesses.setdefault(array_name, []).append(signatures)

    distance_by_dim = [0 for _ in range(loop_rank)]
    reuse_pairs = 0
    distance_sum = 0
    max_distance = 0
    for refs in accesses.values():
        if len(refs) < 2:
            continue
        for left_idx in range(len(refs)):
            for right_idx in range(left_idx + 1, len(refs)):
                left = refs[left_idx]
                right = refs[right_idx]
                for l_sig, r_sig in zip(left, right):
                    l_coeffs, l_const = l_sig
                    r_coeffs, r_const = r_sig
                    if l_coeffs != r_coeffs:
                        distance = 2
                    else:
                        distance = abs(l_const - r_const)
                    if distance <= 0:
                        continue
                    reuse_pairs += 1
                    distance_sum += distance
                    max_distance = max(max_distance, distance)
                    active_dims = [idx for idx, coeff in enumerate(l_coeffs) if coeff]
                    if active_dims:
                        for dim in active_dims:
                            distance_by_dim[dim] = max(distance_by_dim[dim], distance)
                    else:
                        distance_by_dim[0] = max(distance_by_dim[0], distance)

    carried_dims = sum(1 for distance in distance_by_dim if distance > 0)
    reduction_opportunity = (
        max_distance * max(1, carried_dims) + 0.25 * distance_sum
        if reuse_pairs else 0.0
    )
    return {
        "pluto_loop_rank": loop_rank,
        "pluto_access_count": sum(len(refs) for refs in accesses.values()),
        "pluto_reuse_pair_count": reuse_pairs,
        "pluto_max_distance": float(max_distance),
        "pluto_distance_sum": float(distance_sum),
        "pluto_distance_by_dim": [float(distance) for distance in distance_by_dim],
        "pluto_carried_dims": carried_dims,
        "pluto_reduction_opportunity": float(reduction_opportunity),
    }
