"""
Dataset Validation Script.
Inspects raw dataset, runs schema detection, analyzes invalid rows and exact duplicates.
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
from qlora_engine.datasets.formatter import DatasetFormatter
from qlora_engine.datasets.loader import DatasetLoader
from qlora_engine.datasets.validator import DatasetValidator
from qlora_engine.utils.logging import get_logger

logger = get_logger("scripts.validate_dataset")


def main():
    parser = argparse.ArgumentParser(description="Validate dataset quality, schema, and duplicates.")
    parser.add_argument("--config", type=str, default="configs/cybersecurity.yaml", help="Path to config YAML file")
    args = parser.parse_args()

    config = load_config(args.config)
    logger.info(f"Loaded configuration from: {args.config}")

    loader = DatasetLoader()
    raw_ds = loader.load(config["dataset"])
    inspect_info = loader.inspect(raw_ds)
    logger.info(f"Raw Dataset Inspection: {inspect_info}")

    schema = DatasetFormatter.detect_schema(raw_ds)
    logger.info(f"Detected Dataset Schema: '{schema}'")

    _, val_summary = DatasetValidator.analyze_dataset(
        raw_ds, schema, max_invalid_ratio=config["dataset"]["validation"]["max_invalid_ratio"]
    )

    logger.info("=" * 60)
    logger.info("DATASET VALIDATION SUMMARY")
    logger.info("=" * 60)
    for key, val in val_summary.items():
        logger.info(f"  {key:<22}: {val}")
    logger.info("=" * 60)
    logger.info("Dataset Validation Passed Successfully!")


if __name__ == "__main__":
    main()
