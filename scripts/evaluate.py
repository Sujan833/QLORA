"""
Evaluation Script for Fine-Tuned Model.
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
from qlora_engine.evaluation.evaluator import ModelEvaluator
from qlora_engine.inference.engine import InferenceEngine
from qlora_engine.utils.logging import get_logger

logger = get_logger("scripts.evaluate")


def main():
    parser = argparse.ArgumentParser(description="Evaluate fine-tuned LoRA model on test set.")
    parser.add_argument("--config", type=str, default="configs/cybersecurity.yaml", help="Path to config YAML file")
    parser.add_argument("--samples", type=int, default=10, help="Number of test samples to evaluate")
    args = parser.parse_args()

    config = load_config(args.config)
    output_dir = Path(config.get("output", {}).get("directory", "outputs/cybersecurity_qwen3b"))
    logger.info(f"Loaded configuration from: {args.config}")

    # 1. Load dataset & prepare test split
    loader = DatasetLoader()
    raw_ds = loader.load(config["dataset"])
    schema = DatasetFormatter.detect_schema(raw_ds)
    valid_ds, _ = DatasetValidator.analyze_dataset(
        raw_ds, schema, max_invalid_ratio=config["dataset"]["validation"]["max_invalid_ratio"]
    )
    cleaned_ds = DatasetCleaner.clean(valid_ds, config["dataset"])
    canonical_ds = DatasetFormatter.format_to_canonical(cleaned_ds, schema)

    split_cfg = config["split"]
    splits = DatasetSplitter.split(
        canonical_ds,
        train_ratio=float(split_cfg.get("train", 0.90)),
        validation_ratio=float(split_cfg.get("validation", 0.05)),
        test_ratio=float(split_cfg.get("test", 0.05)),
        seed=int(split_cfg.get("seed", 3407)),
    )
    test_ds = splits["test"]

    # 2. Instantiate inference engine with adapter
    adapter_path = output_dir
    engine = InferenceEngine(
        base_model_name=config["model"]["name"],
        adapter_path=adapter_path if adapter_path.exists() else None,
    )

    # 3. Evaluate model
    evaluator = ModelEvaluator(config)
    report = evaluator.evaluate(
        model=engine.model,
        tokenizer=engine.tokenizer,
        test_dataset=test_ds,
        output_dir=output_dir,
        max_eval_samples=args.samples,
    )

    logger.info("Evaluation Script Completed Successfully!")


if __name__ == "__main__":
    main()
