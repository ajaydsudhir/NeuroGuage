from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


def configure_logging(run_name: str, log_dir: str | Path = "logs", level: int = logging.INFO) -> Path:
    """Configure shared console and file logging for all workflows."""
    log_directory = Path(log_dir)
    log_directory.mkdir(parents=True, exist_ok=True)

    log_path = log_directory / f"{run_name}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Reset handlers to keep repeated CLI calls from duplicating log output.
    root_logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)

    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    return log_path
