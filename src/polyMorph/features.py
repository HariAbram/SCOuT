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


def _product(values: List[float]) -> float:
    result = 1.0
    for value in values:
        result *= value
    return result
