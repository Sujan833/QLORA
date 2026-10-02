"""
Unit tests for dataset loading, validation, cleaning, and splitting.
"""

import pytest

datasets = pytest.importorskip("datasets")
Dataset = datasets.Dataset

from qlora_engine.datasets.cleaner import DatasetCleaner
from qlora_engine.datasets.splitter import DatasetSplitter
from qlora_engine.datasets.validator import DatasetValidator


def test_validator_valid_and_invalid_messages():
    valid_msgs = [
        {"role": "user", "content": "What is phishing?"},
        {"role": "assistant", "content": "Phishing is social engineering."},
    ]
    invalid_msgs = [{"role": "user", "content": ""}]

    assert DatasetValidator.is_valid_message_list(valid_msgs) is True
    assert DatasetValidator.is_valid_message_list(invalid_msgs) is False


def test_dataset_cleaning():
    raw_data = {
        "text": ["   hello   world   \n  ", "test"],
    }
    ds = Dataset.from_dict(raw_data)
    cfg = {
        "cleaning": {
            "enabled": True,
            "trim_whitespace": True,
            "normalize_whitespace": True,
            "normalize_unicode": True,
            "remove_exact_duplicates": True,
        }
    }
    cleaned_ds = DatasetCleaner.clean(ds, cfg)
    assert cleaned_ds[0]["text"] == "hello world"


def test_dataset_splitter():
    data = {"col": list(range(100))}
    ds = Dataset.from_dict(data)

    splits = DatasetSplitter.split(ds, train_ratio=0.80, validation_ratio=0.10, test_ratio=0.10)
    assert len(splits["train"]) == 80
    assert len(splits["validation"]) == 10
    assert len(splits["test"]) == 10
