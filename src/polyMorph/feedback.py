from __future__ import annotations

import re
from statistics import mean
from typing import Any, Dict, List


JsonDict = Dict[str, Any]

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_REGISTER_RE = re.compile(r"\bregister(?:s)?\b", re.IGNORECASE)
_OCCUPANCY_RE = re.compile(r"\boccupancy\b", re.IGNORECASE)
_TIMEOUT_RE = re.compile(r"\btimeout|timed out\b", re.IGNORECASE)


def parse_compiler_feedback(stdout: str = "", stderr: str = "") -> JsonDict:
    text = "\n".join(part for part in [stdout, stderr] if part)
    lower = text.lower()
    missed_vectorization = []
    vectorized = []
    remarks = []

    for line in text.splitlines():
        line_lower = line.lower()
        if "vectorized" in line_lower and "not vectorized" not in line_lower:
            vectorized.append(line.strip())
        if "not vectorized" in line_lower or "missed" in line_lower and "vector" in line_lower:
            missed_vectorization.append(line.strip())
        if "remark:" in line_lower or "warning:" in line_lower:
            remarks.append(line.strip())

    return {
        "vectorized_count": len(vectorized),
        "missed_vectorization_count": len(missed_vectorization),
        "register_pressure_hint": bool(_REGISTER_RE.search(text)),
        "occupancy_hint": bool(_OCCUPANCY_RE.search(text)),
        "timeout_hint": bool(_TIMEOUT_RE.search(text)),
        "remark_count": len(remarks),
        "sample_vectorized": vectorized[:5],
        "sample_missed_vectorization": missed_vectorization[:5],
        "sample_remarks": remarks[:8],
        "raw_bytes": len(text.encode("utf-8")),
        "raw_excerpt": text[:4000],
    }


def feedback_penalty(feedback: JsonDict) -> float:
    penalty = 0.0
    penalty += 0.08 * int(feedback.get("missed_vectorization_count", 0) or 0)
    if feedback.get("register_pressure_hint"):
        penalty += 0.15
    if feedback.get("occupancy_hint"):
        penalty += 0.10
    if feedback.get("timeout_hint"):
        penalty += 0.30
    return min(1.0, penalty)


def parse_adaptivecpp_runtime_feedback(stdout: str = "", stderr: str = "", mask: str = "") -> JsonDict:
    raw_text = "\n".join(part for part in [stdout, stderr] if part)
    text = _ANSI_RE.sub("", raw_text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    interesting_patterns = [
        "backend_loader:",
        "Registering backend:",
        "kernel_cache:",
        "hcf_cache:",
        "adaptivity_engine:",
        "Successfully compiled SSCP",
        "Load module:",
        "Submitting kernel",
        "Submitting memcpy",
        "AdaptiveCpp Warning",
        "AdaptiveCpp Error",
    ]
    interesting = [
        line
        for line in lines
        if any(pattern in line for pattern in interesting_patterns)
    ]

    return {
        "mask": mask,
        "raw_bytes": len(raw_text.encode("utf-8")),
        "line_count": len(lines),
        "error_count": _count(text, r"AdaptiveCpp Error|\bError\b"),
        "warning_count": _count(text, r"AdaptiveCpp Warning|\bWarning\b"),
        "registered_kernel_count": _count(text, r"kernel_cache: Registering kernel"),
        "hcf_object_count": _count(text, r"hcf_cache: Registering HCF object"),
        "kernel_info_count": _count(text, r"hcf_cache: Registering kernel info"),
        "backend_plugin_count": _count(text, r"backend_loader: Successfully opened plugin"),
        "registered_backend_count": _count(text, r"Registering backend:"),
        "cache_hit_count": _count(text, r"Cache hit|Persistent cache hit"),
        "persistent_cache_hit_count": _count(text, r"Persistent cache hit"),
        "cache_miss_count": _count(text, r"Cache miss"),
        "jit_compile_hint_count": _count(text, r"JIT|jit|Successfully compiled SSCP"),
        "module_load_count": _count(text, r"Load module:"),
        "pointer_alignment_count": _count(text, r"Inferred pointer alignment"),
        "noalias_count": _count(text, r"Inferred noalias pointer semantics"),
        "kernel_submit_count": _count(text, r"Submitting kernel"),
        "memcpy_submit_count": _count(text, r"Submitting memcpy"),
        "rt_shutdown": "runtime: ******* rt shutdown ********" in text,
        "sample_lines": interesting[:40],
        "raw_excerpt": text[:4000],
    }


def analyze_runtime_feedback(
    runtime_feedback: Dict[str, JsonDict],
    *,
    baseline: Dict[str, JsonDict] | None = None,
    constraints: JsonDict | None = None,
) -> JsonDict:
    constraints = constraints or {}
    baseline = baseline or {}
    per_backend: Dict[str, JsonDict] = {}
    penalties: List[float] = []
    bonuses: List[float] = []
    reasons: List[str] = []

    for mask, data in runtime_feedback.items():
        base = baseline.get(mask) or {}
        analysis = _analyze_one_runtime_backend(data, base, constraints)
        per_backend[mask] = analysis
        penalties.append(float(analysis["penalty"]))
        bonuses.append(float(analysis["bonus"]))
        reasons.extend(f"{mask}: {reason}" for reason in analysis.get("reasons", []))

    backend_sensitive, sensitivity_reasons = _backend_sensitivity(runtime_feedback, constraints)
    reasons.extend(sensitivity_reasons)

    penalty = mean(penalties) if penalties else 0.0
    bonus = mean(bonuses) if bonuses else 0.0
    score = max(0.0, min(1.0, 1.0 - penalty + bonus))
    return {
        "score": score,
        "penalty": max(0.0, min(1.0, penalty)),
        "bonus": max(0.0, min(1.0, bonus)),
        "backend_sensitive": backend_sensitive,
        "backend_sensitivity_reasons": sensitivity_reasons,
        "per_backend": per_backend,
        "reasons": reasons[:40],
    }


def merge_feedback(compile_feedback: JsonDict, runtime_feedback: Dict[str, JsonDict]) -> JsonDict:
    merged = dict(compile_feedback or {})
    if runtime_feedback:
        merged["runtime"] = runtime_feedback
        merged["runtime_error_count"] = sum(
            int(item.get("error_count", 0) or 0) for item in runtime_feedback.values()
        )
        merged["runtime_warning_count"] = sum(
            int(item.get("warning_count", 0) or 0) for item in runtime_feedback.values()
        )
        merged["runtime_cache_hit_count"] = sum(
            int(item.get("cache_hit_count", 0) or 0) for item in runtime_feedback.values()
        )
        merged["runtime_jit_compile_hint_count"] = sum(
            int(item.get("jit_compile_hint_count", 0) or 0) for item in runtime_feedback.values()
        )
    return merged


def _count(text: str, pattern: str) -> int:
    return len(re.findall(pattern, text, flags=re.IGNORECASE))


def _analyze_one_runtime_backend(data: JsonDict, baseline: JsonDict, constraints: JsonDict) -> JsonDict:
    penalty = 0.0
    bonus = 0.0
    reasons: List[str] = []

    errors = int(data.get("error_count", 0) or 0)
    warnings = int(data.get("warning_count", 0) or 0)
    cache_hits = int(data.get("cache_hit_count", 0) or 0)
    cache_misses = int(data.get("cache_miss_count", 0) or 0)
    jit_hints = int(data.get("jit_compile_hint_count", 0) or 0)
    kernel_submits = int(data.get("kernel_submit_count", 0) or 0)
    memcpy_submits = int(data.get("memcpy_submit_count", 0) or 0)
    rt_shutdown = bool(data.get("rt_shutdown"))

    error_weight = float(constraints.get("runtime_error_weight", 0.35))
    warning_weight = float(constraints.get("runtime_warning_weight", 0.03))
    jit_weight = float(constraints.get("runtime_jit_weight", 0.01))
    cache_miss_weight = float(constraints.get("runtime_cache_miss_weight", 0.08))
    memcpy_weight = float(constraints.get("runtime_memcpy_weight", 0.10))

    if errors:
        penalty += min(0.7, errors * error_weight)
        reasons.append(f"{errors} AdaptiveCpp runtime error(s)")
    if warnings:
        warning_threshold = int(constraints.get("runtime_warning_threshold", 2))
        excess = max(0, warnings - warning_threshold)
        if excess:
            penalty += min(0.3, excess * warning_weight)
            reasons.append(f"{warnings} runtime warning(s), threshold={warning_threshold}")

    if cache_misses:
        penalty += min(0.4, cache_misses * cache_miss_weight)
        reasons.append(f"{cache_misses} runtime cache miss(es)")

    baseline_jit = int(baseline.get("jit_compile_hint_count", 0) or 0)
    jit_threshold = int(constraints.get("runtime_jit_threshold", baseline_jit + 8))
    if jit_hints > jit_threshold:
        penalty += min(0.4, (jit_hints - jit_threshold) * jit_weight)
        reasons.append(f"{jit_hints} JIT/module hints exceed threshold={jit_threshold}")

    kernel_denominator = max(1, kernel_submits)
    memcpy_ratio = memcpy_submits / kernel_denominator
    baseline_memcpy = int(baseline.get("memcpy_submit_count", 0) or 0)
    baseline_kernels = int(baseline.get("kernel_submit_count", 0) or 0)
    baseline_ratio = baseline_memcpy / max(1, baseline_kernels) if baseline_kernels else None
    ratio_threshold = float(constraints.get("runtime_memcpy_per_kernel_threshold", 3.0))
    if baseline_ratio is not None:
        ratio_threshold = max(ratio_threshold, baseline_ratio * 1.25)
    if memcpy_ratio > ratio_threshold:
        penalty += min(0.35, (memcpy_ratio - ratio_threshold) * memcpy_weight)
        reasons.append(
            f"memcpy/kernel ratio {memcpy_ratio:.3f} exceeds threshold={ratio_threshold:.3f}"
        )

    if not rt_shutdown:
        penalty += float(constraints.get("runtime_missing_shutdown_penalty", 0.15))
        reasons.append("runtime shutdown marker missing")
    else:
        bonus += float(constraints.get("runtime_shutdown_bonus", 0.03))
        reasons.append("runtime shutdown completed")

    cache_total = cache_hits + cache_misses + jit_hints
    if cache_total:
        hit_rate = cache_hits / cache_total
        stable_threshold = float(constraints.get("runtime_cache_hit_rate_bonus_threshold", 0.75))
        if hit_rate >= stable_threshold:
            bonus += float(constraints.get("runtime_cache_hit_bonus", 0.06))
            reasons.append(f"stable cache hit rate {hit_rate:.3f}")
    else:
        hit_rate = 0.0

    return {
        "penalty": max(0.0, min(1.0, penalty)),
        "bonus": max(0.0, min(1.0, bonus)),
        "score": max(0.0, min(1.0, 1.0 - penalty + bonus)),
        "cache_hit_rate": hit_rate,
        "memcpy_per_kernel": memcpy_ratio,
        "reasons": reasons,
    }


def _backend_sensitivity(runtime_feedback: Dict[str, JsonDict], constraints: JsonDict) -> tuple[bool, List[str]]:
    if len(runtime_feedback) < 2:
        return False, []

    reasons: List[str] = []
    metrics = [
        "error_count",
        "warning_count",
        "cache_hit_count",
        "cache_miss_count",
        "jit_compile_hint_count",
        "memcpy_submit_count",
        "kernel_submit_count",
    ]
    relative_threshold = float(constraints.get("runtime_backend_sensitivity_rel_threshold", 0.35))
    absolute_threshold = int(constraints.get("runtime_backend_sensitivity_abs_threshold", 10))

    for metric in metrics:
        values = [
            float(data.get(metric, 0) or 0)
            for data in runtime_feedback.values()
        ]
        value_range = max(values) - min(values)
        value_mean = mean(values) if values else 0.0
        if value_range >= absolute_threshold and value_range / max(1.0, value_mean) >= relative_threshold:
            reasons.append(f"{metric} varies across backend masks: {values}")

    shutdown_values = {
        mask: bool(data.get("rt_shutdown"))
        for mask, data in runtime_feedback.items()
    }
    if len(set(shutdown_values.values())) > 1:
        reasons.append(f"runtime shutdown differs across backend masks: {shutdown_values}")

    return bool(reasons), reasons[:20]
