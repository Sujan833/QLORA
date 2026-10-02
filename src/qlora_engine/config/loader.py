"""
Configuration loader and validator for Universal QLoRA Fine-Tuner.
"""

import math
from pathlib import Path
from typing import Any, Dict, Union

import yaml


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""
    pass


REQUIRED_TOP_LEVEL_SECTIONS = [
    "project",
    "model",
    "dataset",
    "split",
    "lora",
    "training",
]


def load_config(config_input: Union[str, Path, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Loads and validates a YAML configuration file or dictionary.

    Args:
        config_input: Path to YAML config file or dictionary.

    Returns:
        Validated configuration dictionary.

    Raises:
        ConfigurationError: If file is missing, invalid YAML, or fails validation checks.
    """
    if isinstance(config_input, dict):
        config = config_input
    else:
        path = Path(config_input)
        if not path.exists():
            raise ConfigurationError(f"Configuration file not found: {path.absolute()}")
        try:
            with path.open("r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"Failed to parse YAML configuration: {exc}") from exc

    if not isinstance(config, dict):
        raise ConfigurationError("Configuration root must be a dictionary.")

    validate_config(config)
    return config


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validates configuration structure, required sections, and value ranges.
    """
    # 1. Required top-level sections
    for section in REQUIRED_TOP_LEVEL_SECTIONS:
        if section not in config or not isinstance(config[section], dict):
            raise ConfigurationError(
                f"Missing or invalid required configuration section: '{section}'"
            )

    # 2. Model section validation
    model_cfg = config["model"]
    if not model_cfg.get("name") or not isinstance(model_cfg.get("name"), str):
        raise ConfigurationError("model.name must be a non-empty string.")

    # 3. Dataset section validation
    dataset_cfg = config["dataset"]
    source = str(dataset_cfg.get("source", "")).lower()
    valid_sources = {"huggingface", "json", "jsonl", "csv", "parquet"}
    if source not in valid_sources:
        raise ConfigurationError(
            f"dataset.source '{source}' is invalid. Must be one of {sorted(valid_sources)}."
        )

    if source == "huggingface" and not dataset_cfg.get("name"):
        raise ConfigurationError("dataset.name is required when source is 'huggingface'.")

    if source != "huggingface" and not dataset_cfg.get("path"):
        raise ConfigurationError("dataset.path is required when source is a local file type.")

    # 4. Split ratios validation
    split_cfg = config["split"]
    train_r = float(split_cfg.get("train", 0.90))
    val_r = float(split_cfg.get("validation", 0.05))
    test_r = float(split_cfg.get("test", 0.05))
    total_ratio = train_r + val_r + test_r

    if not math.isclose(total_ratio, 1.0, abs_tol=1e-5):
        raise ConfigurationError(
            f"Split ratios (train: {train_r}, validation: {val_r}, test: {test_r}) "
            f"must sum to 1.0 (current sum: {total_ratio:.6f})."
        )

    # 5. LoRA validation
    lora_cfg = config["lora"]
    if int(lora_cfg.get("r", 16)) <= 0:
        raise ConfigurationError("lora.r must be a positive integer.")
    if int(lora_cfg.get("alpha", 32)) <= 0:
        raise ConfigurationError("lora.alpha must be a positive integer.")
    if not lora_cfg.get("target_modules") or not isinstance(lora_cfg.get("target_modules"), list):
        raise ConfigurationError("lora.target_modules must be a non-empty list of module names.")

    # 6. Training validation
    training_cfg = config["training"]
    if int(training_cfg.get("batch_size", 1)) <= 0:
        raise ConfigurationError("training.batch_size must be a positive integer.")
    if int(training_cfg.get("gradient_accumulation_steps", 1)) <= 0:
        raise ConfigurationError("training.gradient_accumulation_steps must be a positive integer.")
    if float(training_cfg.get("learning_rate", 2e-4)) <= 0:
        raise ConfigurationError("training.learning_rate must be greater than 0.")
