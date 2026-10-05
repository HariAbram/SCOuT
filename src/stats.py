from __future__ import annotations

###############################################################################
# Standard library imports                                                    #
###############################################################################

import atexit
import contextlib
import csv
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterator, Optional, Union

###############################################################################
# Type helpers                                                                #
###############################################################################

PathLike = Union[str, Path]

###############################################################################
# Runtime statistics                                                          #
###############################################################################

DEFAULT_FILENAME = "scout_stats.csv"
DEFAULT_FLUSH_INTERVAL = 5.0

# Columns written on every periodic snapshot. Values are cumulative totals.
COLUMNS = (
    "timestamp",
    "elapsed_s",
    "build_time_s",
    "run_time_s",
    "scout_time_s",
    "build_count",
    "run_count",
)


def resolve_stats_path(
    cli_path: Optional[PathLike] = None,
    config_path: Optional[PathLike] = None,
    env: Optional[Dict[str, str]] = None,
) -> Path:
    """
    Decide where the stats CSV lives.
    """
    env = os.environ if env is None else env
    for candidate in (cli_path, config_path, env.get("SCOUT_STATS_FILE")):
        if candidate:
            return Path(candidate).expanduser()
    return Path.cwd() / DEFAULT_FILENAME


class RuntimeStats:
    """
    Accumulate how long SCOuT spends building, running targets, and in its own
    bookkeeping.
    """

    def __init__(self, flush_interval: float = DEFAULT_FLUSH_INTERVAL):
        self.flush_interval = float(flush_interval)
        self.path: Optional[Path] = None
        self._start: Optional[float] = None
        self._build_time = 0.0
        self._run_time = 0.0
        self._build_count = 0
        self._run_count = 0
        self._last_flush: Optional[float] = None
        self._finished = False
        self._atexit_registered = False

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self, path: Optional[PathLike] = None) -> "RuntimeStats":
        """Begin a tracked session, optionally (re)setting the output path."""
        if path is not None:
            self.path = Path(path).expanduser()
        self._start = time.perf_counter()
        self._build_time = 0.0
        self._run_time = 0.0
        self._build_count = 0
        self._run_count = 0
        self._last_flush = None
        self._finished = False
        if not self._atexit_registered:
            atexit.register(self.finish)
            self._atexit_registered = True
        return self

    @property
    def started(self) -> bool:
        return self._start is not None

    @property
    def elapsed(self) -> float:
        return (time.perf_counter() - self._start) if self._start is not None else 0.0

    def snapshot(self) -> Dict[str, Any]:
        """Current cumulative totals; own time is elapsed minus build and run."""
        elapsed = self.elapsed
        scout = max(0.0, elapsed - self._build_time - self._run_time)
        return {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "elapsed_s": elapsed,
            "build_time_s": self._build_time,
            "run_time_s": self._run_time,
            "scout_time_s": scout,
            "build_count": self._build_count,
            "run_count": self._run_count,
        }

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------
    @contextlib.contextmanager
    def _record(self, attribute: str, counter: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            if self._start is not None:
                setattr(self, attribute, getattr(self, attribute) + (time.perf_counter() - t0))
                setattr(self, counter, getattr(self, counter) + 1)
                self.maybe_flush()

    def time_build(self) -> Iterator[None]:
        """Context manager that records time spent building a target."""
        return self._record("_build_time", "_build_count")

    def time_run(self) -> Iterator[None]:
        """Context manager that records time spent running a target."""
        return self._record("_run_time", "_run_count")

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------
    def maybe_flush(self) -> None:
        """Append a snapshot when at least ``flush_interval`` seconds elapsed."""
        if self._start is None or self.path is None:
            return
        now = time.perf_counter()
        if self._last_flush is not None and (now - self._last_flush) < self.flush_interval:
            return
        self._write_row(now)

    def flush(self) -> None:
        """Append a snapshot immediately (used for the final row)."""
        if self._start is None or self.path is None:
            return
        self._write_row(time.perf_counter())

    def _write_row(self, now: float) -> None:
        assert self.path is not None
        row = self.snapshot()
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            new = not self.path.is_file() or self.path.stat().st_size == 0
            with open(self.path, "a", newline="", encoding="utf-8") as fp:
                writer = csv.writer(fp, lineterminator="\n")
                if new:
                    writer.writerow(COLUMNS)
                writer.writerow([row[column] for column in COLUMNS])
        except OSError:
            # Statistics must never break a run.
            pass
        self._last_flush = now

    def finish(self) -> None:
        """Write a final snapshot; safe to call more than once."""
        if self._start is None or self._finished:
            return
        self._finished = True
        self.flush()

    def summary_line(self) -> str:
        """One-line human-readable summary of the current totals."""
        row = self.snapshot()
        return (
            f"[stats] build={row['build_time_s']:.3f}s ({row['build_count']}), "
            f"run={row['run_time_s']:.3f}s ({row['run_count']}), "
            f"scout={row['scout_time_s']:.3f}s, total={row['elapsed_s']:.3f}s"
        )


# Process-wide tracker shared by the CLI entry point and the evaluator.
STATS = RuntimeStats()
