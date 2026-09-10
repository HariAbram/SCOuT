"""Optional PolyMorph integration.

Importing this package is intentionally lightweight.  The runner (and its
Tadashi integration) is loaded only when PolyMorph is actually invoked.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.config import Config


class PolyMorphUnavailableError(RuntimeError):
    """Raised when the optional PolyMorph prerequisites are unavailable."""


def run_poly_morph(cfg: "Config", trials_override: int | None = None) -> int:
    from .runner import run_poly_morph as _run_poly_morph

    return _run_poly_morph(cfg, trials_override)

__all__ = ["PolyMorphUnavailableError", "run_poly_morph"]
