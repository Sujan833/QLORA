"""
Utilities package for hardware detection, logging, and metadata.
"""

from qlora_engine.utils.hardware import (
    detect_hardware,
    get_optimal_compute_dtype,
    print_hardware_summary,
)
from qlora_engine.utils.logging import get_logger, setup_logger
from qlora_engine.utils.metadata import save_run_metadata

__all__ = [
    "detect_hardware",
    "get_optimal_compute_dtype",
    "print_hardware_summary",
    "setup_logger",
    "get_logger",
    "save_run_metadata",
]
