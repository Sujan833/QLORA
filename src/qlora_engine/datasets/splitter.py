"""
Deterministic Dataset Splitter for Universal QLoRA Fine-Tuner.
"""

import math
from typing import Dict, Union

from datasets import Dataset, DatasetDict

from qlora_engine.datasets.validator import DatasetValidationError


class DatasetSplitter:
    """
    Splits a Dataset into deterministic train, validation, and test splits.
    """

    @staticmethod
    def split(
        dataset: Dataset,
        train_ratio: float = 0.90,
        validation_ratio: float = 0.05,
        test_ratio: float = 0.05,
        seed: int = 3407,
    ) -> DatasetDict:
        """
        Splits dataset dynamically into train, validation, and test datasets.
        """
        total_ratio = train_ratio + validation_ratio + test_ratio
        if not math.isclose(total_ratio, 1.0, abs_tol=1e-5):
            raise DatasetValidationError(
                f"Split ratios must sum to 1.0 (train: {train_ratio}, "
                f"validation: {validation_ratio}, test: {test_ratio}, sum: {total_ratio:.6f})."
            )

        if len(dataset) < 3:
            raise DatasetValidationError("Dataset is too small to split into 3 splits (minimum 3 records).")

        temp_ratio = validation_ratio + test_ratio

        # Step 1: Split into train and temporary (validation + test)
        first_split = dataset.train_test_split(test_size=temp_ratio, seed=seed)
        train_ds = first_split["train"]
        temp_ds = first_split["test"]

        # Step 2: Split temporary into validation and test
        relative_test_ratio = test_ratio / temp_ratio
        second_split = temp_ds.train_test_split(test_size=relative_test_ratio, seed=seed)
        val_ds = second_split["train"]
        test_ds = second_split["test"]

        return DatasetDict(
            {
                "train": train_ds,
                "validation": val_ds,
                "test": test_ds,
            }
        )
