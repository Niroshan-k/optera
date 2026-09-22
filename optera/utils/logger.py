"""
Optera Centralized Logging Manager
Configures clean terminal output and dedicated persistent file logging (optera.log).
"""

import logging
import sys
from pathlib import Path
from typing import Optional, Union


def setup_optera_logging(
    output_dir: Optional[Union[str, Path]] = None,
    log_filename: str = "optera.log",
    console_level: int = logging.WARNING,
    file_level: int = logging.INFO
) -> logging.Logger:
    """
    Configures Optera framework logging:
    - Console output remains clean (WARNING / ERROR only), allowing clean executive summaries.
    - Persistent file logging writes ALL operational trace logs to output_dir/optera.log.
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # Clear existing handlers to avoid duplicates
    if root_logger.hasHandlers():
        root_logger.handlers.clear()

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-7s | %(name)-24s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 1. Clean Console Handler (WARNING & ERROR only to prevent terminal clutter)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(console_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # 2. Workspace File Handler (optera.log - All INFO/DEBUG logs saved here)
    if output_dir:
        output_path = Path(output_dir).resolve()
        output_path.mkdir(parents=True, exist_ok=True)
        log_file = output_path / log_filename

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    return root_logger
