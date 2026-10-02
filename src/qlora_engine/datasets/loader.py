"""
Universal Dataset Loader for Universal QLoRA Fine-Tuner.
"""

from pathlib import Path
from typing import Any, Dict, Union

from datasets import Dataset, DatasetDict, load_dataset


class DatasetLoadError(Exception):
    """Raised when loading dataset fails."""
    pass


class DatasetLoader:
    """
    Handles loading datasets from Hugging Face Hub, local JSON, JSONL, CSV, or Parquet files.
    """

    SUPPORTED_SOURCES = {"huggingface", "json", "jsonl", "csv", "parquet"}

    def load(self, config: Dict[str, Any]) -> Union[Dataset, DatasetDict]:
        """
        Loads a dataset based on configuration parameters.
        """
        source = str(config.get("source", "")).lower().strip()

        if source not in self.SUPPORTED_SOURCES:
            raise DatasetLoadError(
                f"Unsupported dataset source '{source}'. Must be one of {sorted(self.SUPPORTED_SOURCES)}."
            )

        if source == "huggingface":
            name = config.get("name")
            if not name:
                raise DatasetLoadError("Hugging Face dataset name 'dataset.name' is required.")

            split = config.get("split")
            try:
                if split:
                    return load_dataset(name, split=split)
                return load_dataset(name)
            except Exception as e:
                raise DatasetLoadError(f"Failed to load Hugging Face dataset '{name}': {e}") from e

        path_str = config.get("path")
        if not path_str:
            raise DatasetLoadError(f"Local dataset path 'dataset.path' is required for source '{source}'.")

        path = Path(path_str)
        if not path.exists():
            raise DatasetLoadError(f"Dataset file does not exist: {path.absolute()}")

        loader_format = "json" if source in {"json", "jsonl"} else source

        try:
            split = config.get("split", "train")
            return load_dataset(loader_format, data_files=str(path), split=split)
        except Exception as e:
            raise DatasetLoadError(f"Failed to load local {source} dataset from '{path}': {e}") from e

    @staticmethod
    def inspect(dataset: Union[Dataset, DatasetDict]) -> Dict[str, Any]:
        """
        Returns metadata about columns, rows, and feature types of loaded dataset.
        """
        if isinstance(dataset, DatasetDict):
            first_split = next(iter(dataset.keys()))
            ds = dataset[first_split]
        else:
            ds = dataset

        return {
            "num_rows": len(ds),
            "columns": list(ds.column_names),
            "features": {name: str(feature) for name, feature in ds.features.items()},
        }
