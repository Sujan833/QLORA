"""
Dataset Preparation Script.
Loads dataset, validates schema, applies cleaning, normalizes to canonical messages, and creates train/val/test splits.
"""

import argparse
import sys
from pathlib import Path

# Ensure src is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

from qlora_engine.config.loader import load_config
from qlora_engine.datasets.cleaner import DatasetCleaner
from qlora_engine.datasets.formatter import DatasetFormatter
from qlora_engine.datasets.loader import DatasetLoader
from qlora_engine.datasets.splitter import DatasetSplitter
from qlora_engine.datasets.validator import DatasetValidator
from qlora_engine.utils.logging import get_logger

logger = get_logger("scripts.prepare_dataset")


def main():
    parser = argparse.ArgumentParser(description="Prepare and preprocess dataset for QLoRA fine-tuning.")
    parser.add_argument("--config", type=str, default="configs/cybersecurity.yaml", help="Path to config YAML file")
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info(f"Loaded configuration from: {args.config}")

    # 1. Load dataset
    loader = DatasetLoader()
    raw_ds = loader.load(config["dataset"])
    logger.info(f"Loaded raw dataset ({len(raw_ds)} records).")

    # 2. Schema detection & conversion
    schema = DatasetFormatter.detect_schema(raw_ds)
    logger.info(f"Detected schema: '{schema}'")

    cleaned_raw_ds, val_summary = DatasetValidator.analyze_dataset(
        raw_ds, schema, max_invalid_ratio=config["dataset"]["validation"]["max_invalid_ratio"]
    )
    logger.info(f"Validation summary: {val_summary}")

    cleaned_ds = DatasetCleaner.clean(cleaned_raw_ds, config["dataset"])
    canonical_ds = DatasetFormatter.format_to_canonical(cleaned_ds, schema)
    logger.info(f"Converted to canonical messages format ({len(canonical_ds)} valid records).")

    # 3. Dataset split
    split_cfg = config["split"]
    dataset_splits = DatasetSplitter.split(
        canonical_ds,
        train_ratio=float(split_cfg.get("train", 0.90)),
        validation_ratio=float(split_cfg.get("validation", 0.05)),
        test_ratio=float(split_cfg.get("test", 0.05)),
        seed=int(split_cfg.get("seed", 3407)),
    )

    for split_name, ds in dataset_splits.items():
        logger.info(f"  Split '{split_name}': {len(ds):,} rows")

    logger.info("Dataset Preparation Successfully Completed!")


if __name__ == "__main__":
    main()
