"""
Unit tests for configuration loader and validator.
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from qlora_engine.config.loader import ConfigurationError, load_config


def test_load_valid_config_dict():
    raw_cfg = {
        "project": {"name": "TestProject"},
        "model": {"name": "Qwen/Qwen2.5-3B-Instruct"},
        "dataset": {"source": "huggingface", "name": "Tiamz/cybersecurity-instruction-dataset"},
        "split": {"train": 0.90, "validation": 0.05, "test": 0.05},
        "lora": {"r": 16, "alpha": 32, "target_modules": ["q_proj", "v_proj"]},
        "training": {"batch_size": 1, "gradient_accumulation_steps": 8, "learning_rate": 0.0002},
    }
    cfg = load_config(raw_cfg)
    assert cfg["project"]["name"] == "TestProject"


def test_invalid_split_ratios():
    raw_cfg = {
        "project": {"name": "TestProject"},
        "model": {"name": "Qwen/Qwen2.5-3B-Instruct"},
        "dataset": {"source": "huggingface", "name": "Tiamz/cybersecurity-instruction-dataset"},
        "split": {"train": 0.50, "validation": 0.05, "test": 0.05},  # sum = 0.60
        "lora": {"r": 16, "alpha": 32, "target_modules": ["q_proj"]},
        "training": {"batch_size": 1, "gradient_accumulation_steps": 8, "learning_rate": 0.0002},
    }
    with pytest.raises(ConfigurationError, match="must sum to 1.0"):
        load_config(raw_cfg)


def test_load_yaml_file():
    raw_cfg = {
        "project": {"name": "YAMLTest"},
        "model": {"name": "Qwen/Qwen2.5-3B-Instruct"},
        "dataset": {"source": "huggingface", "name": "Tiamz/cybersecurity-instruction-dataset"},
        "split": {"train": 0.90, "validation": 0.05, "test": 0.05},
        "lora": {"r": 8, "alpha": 16, "target_modules": ["q_proj"]},
        "training": {"batch_size": 1, "gradient_accumulation_steps": 4, "learning_rate": 0.0002},
    }
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        yaml.dump(raw_cfg, f)
        temp_path = f.name

    cfg = load_config(temp_path)
    assert cfg["project"]["name"] == "YAMLTest"
    Path(temp_path).unlink()
