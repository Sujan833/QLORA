"""
Unit tests for QLoraTrainer initialization and backend selection.
"""

import tempfile
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from qlora_engine.training.backend import TransformersPEFTBackend, get_training_backend
from qlora_engine.training.trainer import QLoraTrainer


def test_trainer_initialization():
    cfg = {
        "project": {"name": "Test"},
        "model": {"name": "Qwen/Qwen2.5-3B-Instruct"},
        "dataset": {"source": "huggingface", "name": "Tiamz/cybersecurity-instruction-dataset"},
        "split": {"train": 0.90, "validation": 0.05, "test": 0.05},
        "lora": {"r": 16, "alpha": 32, "target_modules": ["q_proj"]},
        "training": {"batch_size": 1, "gradient_accumulation_steps": 8, "learning_rate": 0.0002},
    }
    with tempfile.TemporaryDirectory() as tmp_dir:
        cfg["output"] = {"directory": tmp_dir}
        trainer = QLoraTrainer(cfg)
        assert trainer.output_dir == Path(tmp_dir)


def test_backend_selection():
    cfg = {"training": {"backend": "transformers"}}
    hw = {"cuda_available": False}
    backend = get_training_backend(cfg, hw)
    assert isinstance(backend, TransformersPEFTBackend)
