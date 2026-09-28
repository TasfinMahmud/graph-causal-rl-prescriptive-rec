"""Logging configuration shared by every entry point.

Each run writes to its own timestamped file alongside console output, so runs
never append into one another's logs and a result can always be traced back to
the run that produced it.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-28s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def configure_logging(
    log_dir: Path | None = None,
    run_name: str = "run",
    level: int = logging.INFO,
) -> Path | None:
    """Attach a console handler and, when ``log_dir`` is given, a file handler.

    Returns:
        Path to the log file, or ``None`` when logging only to the console.
    """
    root = logging.getLogger()
    root.setLevel(level)
    for handler in list(root.handlers):
        root.removeHandler(handler)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    console = logging.StreamHandler(stream=sys.stdout)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_dir is None:
        return None

    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{run_name}_{stamp}.log"

    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    return log_path


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
