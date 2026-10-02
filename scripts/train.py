"""
End-to-End QLoRA Fine-Tuning Execution Script.
"""

import argparse
import sys
import types
from pathlib import Path

# Ensure src is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_PATH = PROJECT_ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Automatic guard against broken torchvision C++ binary extensions in Kaggle/Colab
try:
    import torchvision
except Exception:
    class DummyInterpolationMode:
        NEAREST = "nearest"
        BILINEAR = "bilinear"
        BICUBIC = "bicubic"

    class DummyImageReadMode:
        UNCHANGED = 0

    tv_mock = types.ModuleType("torchvision")
    tv_transforms_mock = types.ModuleType("torchvision.transforms")
    tv_transforms_mock.InterpolationMode = DummyInterpolationMode
    tv_io_mock = types.ModuleType("torchvision.io")
    tv_io_mock.ImageReadMode = DummyImageReadMode
    tv_io_mock.decode_image = None

    tv_mock.transforms = tv_transforms_mock
    tv_mock.io = tv_io_mock

    sys.modules["torchvision"] = tv_mock
    sys.modules["torchvision.transforms"] = tv_transforms_mock
    sys.modules["torchvision.io"] = tv_io_mock

from qlora_engine.config.loader import load_config
from qlora_engine.datasets.analyzer import TokenAnalyzer
from qlora_engine.datasets.cleaner import DatasetCleaner
from qlora_engine.datasets.formatter import DatasetFormatter
from qlora_engine.datasets.loader import DatasetLoader
from qlora_engine.datasets.splitter import DatasetSplitter
from qlora_engine.datasets.validator import DatasetValidator
from qlora_engine.models.manager import ModelManager
from qlora_engine.training.trainer import QLoraTrainer
from qlora_engine.utils.hardware import detect_hardware
from qlora_engine.utils.logging import get_logger

logger = get_logger("scripts.train")


def main():
    parser = argparse.ArgumentParser(description="Run complete QLoRA fine-tuning pipeline.")
    parser.add_argument("--config", type=str, default="configs/cybersecurity.yaml", help="Path to config YAML file")
    args = parser.parse_args()

    # 1. Load config & audit hardware
    config = load_config(args.config)
    hardware_info = detect_hardware()
    logger.info(f"Loaded configuration from: {args.config}")

    # 2. Dataset loading, validation, cleaning, formatting, and splitting
    loader = DatasetLoader()
    raw_ds = loader.load(config["dataset"])
    schema = DatasetFormatter.detect_schema(raw_ds)

    valid_ds, _ = DatasetValidator.analyze_dataset(
        raw_ds, schema, max_invalid_ratio=config["dataset"]["validation"]["max_invalid_ratio"]
    )
    cleaned_ds = DatasetCleaner.clean(valid_ds, config["dataset"])
    canonical_ds = DatasetFormatter.format_to_canonical(cleaned_ds, schema)

    split_cfg = config["split"]
    dataset_splits = DatasetSplitter.split(
        canonical_ds,
        train_ratio=float(split_cfg.get("train", 0.90)),
        validation_ratio=float(split_cfg.get("validation", 0.05)),
        test_ratio=float(split_cfg.get("test", 0.05)),
        seed=int(split_cfg.get("seed", 3407)),
    )

    # 3. Tokenizer & model loading
    model_mgr = ModelManager(config, hardware_info)
    tokenizer = model_mgr.load_tokenizer()

    # 4. Format dataset splits using tokenizer chat template
    formatted_splits = {}
    for split_name, ds in dataset_splits.items():
        formatted_splits[split_name] = DatasetFormatter.apply_chat_template(ds, tokenizer)

    # 5. Token analysis
    token_stats = TokenAnalyzer.analyze(
        formatted_splits["train"],
        tokenizer,
        max_seq_length=int(config["training"].get("max_seq_length", 1024)),
    )
    logger.info(f"Token Length Analysis: {token_stats}")

    # 6. Load base model & attach LoRA
    model = model_mgr.load_model()
    model = model_mgr.attach_lora(model)

    # 7. Execute training pipeline (includes safety audit & smoke test)
    trainer = QLoraTrainer(config)
    trainer.train(model, tokenizer, formatted_splits)

    logger.info("QLoRA Fine-Tuning Script Completed Successfully!")


if __name__ == "__main__":
    main()
