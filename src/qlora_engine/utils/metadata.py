"""
Metadata persistence utility for tracking training runs and experiments.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


def save_run_metadata(
    output_dir: Path,
    config: Dict[str, Any],
    hardware_info: Dict[str, Any],
    training_stats: Dict[str, Any] = None,
) -> Path:
    """
    Saves metadata about the run, model configuration, hardware, and training execution stats to JSON.
    """
    output_dir = Path(output_dir)
    metadata_dir = output_dir / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "project_name": config.get("project", {}).get("name", "Universal-QLoRA-FineTuner"),
        "experiment_name": config.get("project", {}).get("experiment_name", "experiment"),
        "model_name": config.get("model", {}).get("name"),
        "dataset_name": config.get("dataset", {}).get("name"),
        "hardware": {
            "gpu_name": hardware_info.get("gpu_name"),
            "total_vram_gb": hardware_info.get("total_vram_gb"),
            "optimal_dtype": str(hardware_info.get("optimal_dtype_str")),
        },
        "lora_config": config.get("lora", {}),
        "training_config": config.get("training", {}),
        "training_stats": training_stats or {},
    }

    target_file = metadata_dir / "run_metadata.json"
    with target_file.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    return target_file
