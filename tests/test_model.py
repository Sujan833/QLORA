"""
Unit tests for hardware detection and model manager utilities.
"""

import pytest

torch = pytest.importorskip("torch")

from qlora_engine.utils.hardware import detect_hardware, get_optimal_compute_dtype


def test_hardware_detection():
    hw = detect_hardware()
    assert "cuda_available" in hw
    assert "device_count" in hw
    assert "optimal_dtype" in hw


def test_compute_dtype_selection():
    if torch.cuda.is_available():
        # Turing T4 (major cc 7) -> float16
        assert get_optimal_compute_dtype(7) == torch.float16
        # Ampere A100 (major cc 8) -> bfloat16
        assert get_optimal_compute_dtype(8) == torch.bfloat16
    else:
        # CPU Fallback
        assert get_optimal_compute_dtype(7) == torch.float32
