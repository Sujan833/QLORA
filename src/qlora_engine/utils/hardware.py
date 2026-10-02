"""
Hardware detection and compute dtype selection for Universal QLoRA Fine-Tuner.
"""

from typing import Any, Dict

import torch

from qlora_engine.utils.logging import get_logger

logger = get_logger("qlora_engine.hardware")


def detect_hardware() -> Dict[str, Any]:
    """
    Inspects PyTorch CUDA environment, GPU availability, VRAM, and compute capability.

    Returns:
        Dictionary containing hardware information.
    """
    cuda_available = torch.cuda.is_available()
    device_count = torch.cuda.device_count() if cuda_available else 0

    info: Dict[str, Any] = {
        "cuda_available": cuda_available,
        "device_count": device_count,
        "gpu_name": "None",
        "compute_capability": (0, 0),
        "total_vram_gb": 0.0,
        "free_vram_gb": 0.0,
        "optimal_dtype": torch.float32,
        "optimal_dtype_str": "float32",
        "single_gpu_enforced": True,
    }

    if cuda_available and device_count > 0:
        device_id = 0
        props = torch.cuda.get_device_properties(device_id)
        info["gpu_name"] = props.name
        info["compute_capability"] = (props.major, props.minor)
        info["total_vram_gb"] = round(props.total_memory / (1024**3), 2)

        try:
            free_mem, total_mem = torch.cuda.mem_get_info(device_id)
            info["free_vram_gb"] = round(free_mem / (1024**3), 2)
        except Exception:
            info["free_vram_gb"] = info["total_vram_gb"]

        optimal_dtype = get_optimal_compute_dtype(props.major)
        info["optimal_dtype"] = optimal_dtype
        info["optimal_dtype_str"] = "bfloat16" if optimal_dtype == torch.bfloat16 else "float16"

    return info


def get_optimal_compute_dtype(major_cc: int) -> torch.dtype:
    """
    Determines optimal compute dtype based on CUDA compute capability.

    NVIDIA Turing (e.g. Tesla T4, cc 7.5) does not natively support bfloat16
    in cuBLAS operations, leading to CUBLAS_STATUS_EXECUTION_FAILED during training.
    For cc >= 8 (Ampere, Hopper, Ada Lovelace), bfloat16 is supported and preferred.
    For cc < 8 (Turing T4, Volta V100), float16 is selected.
    """
    if not torch.cuda.is_available():
        return torch.float32

    if major_cc >= 8:
        return torch.bfloat16
    else:
        return torch.float16


def print_hardware_summary(info: Dict[str, Any]) -> None:
    """
    Prints a formatted summary of the hardware environment.
    """
    logger.info("=" * 60)
    logger.info("HARDWARE & EXECUTION ENVIRONMENT AUDIT")
    logger.info("=" * 60)
    logger.info(f"CUDA Available      : {info['cuda_available']}")
    logger.info(f"GPU Device Count    : {info['device_count']}")

    if info["cuda_available"]:
        logger.info(f"Selected GPU        : {info['gpu_name']} (Device 0)")
        logger.info(f"Compute Capability  : {info['compute_capability'][0]}.{info['compute_capability'][1]}")
        logger.info(f"Total VRAM          : {info['total_vram_gb']} GB")
        logger.info(f"Free VRAM           : {info['free_vram_gb']} GB")
        logger.info(f"Selected Compute Dtype: {info['optimal_dtype_str'].upper()}")
        logger.info("Execution Mode      : SINGLE GPU (DataParallel / Multi-GPU Multi-Replica Disabled)")
    else:
        logger.info("Execution Mode      : CPU ONLY (CUDA not available)")
    logger.info("=" * 60)
