from __future__ import annotations

###############################################################################
# Standard library imports                                                    #
###############################################################################

from pathlib import Path
from typing import Optional, Union

###############################################################################
# Type helpers                                                                #
###############################################################################

PathLike = Union[str, Path]

###############################################################################
# Logging                                                                     #
###############################################################################

DEFAULT_FILENAME = "scout.log"


class ScoutLogger:
    """
    Append SCOuT's activity log to a file in the current working directory.
    """

    def __init__(self, path: Optional[PathLike] = None):
        self._path: Optional[Path] = Path(path).expanduser() if path is not None else None

    @property
    def path(self) -> Path:
        """Effective log file path (``./scout.log`` when no path was set)."""
        return self._path if self._path is not None else Path.cwd() / DEFAULT_FILENAME

    def start(
        self,
        config_path: Optional[PathLike] = None,
        *,
        path: Optional[PathLike] = None,
    ) -> "ScoutLogger":
        if path is not None:
            self._path = Path(path).expanduser()
        elif config_path is not None:
            self._path = Path.cwd() / f"{Path(config_path).stem}.log"
        return self

    def write(self, message: str = "") -> None:
        """Append ``message`` (plus a newline) to the log file."""
        target = self.path
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8") as log_file:
            log_file.write(message + "\n")

    def log(self, message: str = "") -> None:
        """Echo ``message`` to stdout and append it to the log file."""
        print(message)
        self.write(message)


# Process-wide logger shared by the CLI entry point and the modules it drives.
LOG = ScoutLogger()
