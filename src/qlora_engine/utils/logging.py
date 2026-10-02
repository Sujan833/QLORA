"""
Logging utility for Universal QLoRA Fine-Tuner.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "qlora_engine",
    level: str = "INFO",
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Sets up a clean console and optional file logger.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "qlora_engine") -> logging.Logger:
    """
    Retrieves an existing logger or initializes a default one.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logger(name)
    return logger
