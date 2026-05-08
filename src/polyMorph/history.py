from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from src.polyMorph.features import sequence_signature


JsonDict = Dict[str, Any]


def append_history(path: str | None, record: JsonDict) -> None:
    if not path:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(record, sort_keys=True, default=str) + "\n")


def load_history(path: str | None) -> List[JsonDict]:
    if not path:
        return []
    src = Path(path)
    if not src.exists():
        return []
    records: List[JsonDict] = []
    with src.open("r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                records.append(item)
    return records


def retrieve_sequences(
    *,
    history: Iterable[JsonDict],
    source_hash: str,
    candidates: List[JsonDict],
    limit: int,
) -> List[List[JsonDict]]:
    candidate_keys = {
        (
            int(c["scop"]),
            int(c["node"]),
            str(c["tr"]),
            tuple(c.get("args", [])),
        )
        for c in candidates
    }
    sequences: List[tuple[float, List[JsonDict]]] = []
    seen_signatures: set[str] = set()

    for record in history:
        status = record.get("status")
        if status not in {"complete", "success"}:
            continue
        transforms = record.get("transforms") or []
        if not isinstance(transforms, list) or not transforms:
            continue
        if not _sequence_available(transforms, candidate_keys):
            continue
        signature = sequence_signature(transforms)
        if signature in seen_signatures:
            continue
        seen_signatures.add(signature)

        source_bonus = 1.0 if record.get("source_hash") == source_hash else 0.0
        speedup = float(record.get("speedup", 0.0) or 0.0)
        score = source_bonus + speedup
        sequences.append((score, [dict(item) for item in transforms]))

    sequences.sort(key=lambda item: item[0], reverse=True)
    return [seq for _, seq in sequences[: max(0, limit)]]


def _sequence_available(specs: List[JsonDict], candidate_keys: set[tuple[Any, ...]]) -> bool:
    for spec in specs:
        key = (
            int(spec.get("scop", -1)),
            int(spec.get("node", -1)),
            str(spec.get("tr", "")),
            tuple(spec.get("args", [])),
        )
        if key not in candidate_keys:
            return False
    return True
