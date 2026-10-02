"""
Main QLoRA Trainer Orchestrator for Universal QLoRA Fine-Tuner.
"""

from pathlib import Path
from typing import Any, Dict, Optional

from qlora_engine.models.manager import ModelManager
from qlora_engine.training.backend import get_training_backend
from qlora_engine.utils.hardware import detect_hardware, print_hardware_summary
from qlora_engine.utils.logging import get_logger
from qlora_engine.utils.metadata import save_run_metadata

logger = get_logger("qlora_engine.trainer")


class TrainingError(Exception):
    """Raised when training or smoke test fails."""
    pass


class QLoraTrainer:
    """
    Orchestrates hardware audit, smoke testing, full training execution, and adapter saving.
    """

    def __init__(self, config: Dict[str, Any], output_dir: Optional[Path] = None):
        self.config = config
        self.output_dir = Path(
            output_dir or config.get("output", {}).get("directory", "outputs/default_experiment")
        )
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.hardware_info = detect_hardware()

    def print_safety_audit(
        self, model: Any, tokenizer: Any, dataset_splits: Any
    ) -> None:
        """
        Prints detailed pre-training safety information.
        """
        trainable_params, all_params, percentage = ModelManager.get_trainable_parameters(model)

        logger.info("=" * 70)
        logger.info("PRE-TRAINING SAFETY & CONFIGURATION AUDIT")
        logger.info("=" * 70)
        logger.info(f"Model ID            : {self.config.get('model', {}).get('name')}")
        logger.info(f"Compute Dtype       : {self.hardware_info.get('optimal_dtype_str').upper()}")
        logger.info(f"Train Dataset Size  : {len(dataset_splits['train']):,} rows")
        if "validation" in dataset_splits:
            logger.info(f"Val Dataset Size    : {len(dataset_splits['validation']):,} rows")
        if "test" in dataset_splits:
            logger.info(f"Test Dataset Size   : {len(dataset_splits['test']):,} rows")

        training_cfg = self.config.get("training", {})
        logger.info(f"Batch Size          : {training_cfg.get('batch_size', 1)}")
        logger.info(f"Grad Accumulation   : {training_cfg.get('gradient_accumulation_steps', 8)}")
        logger.info(f"Learning Rate       : {training_cfg.get('learning_rate', 2e-4)}")
        logger.info(f"Max Seq Length      : {training_cfg.get('max_seq_length', 1024)}")
        logger.info(f"Epochs              : {training_cfg.get('epochs', 2)}")
        logger.info(f"Optimizer           : {training_cfg.get('optimizer', 'adamw_8bit')}")
        logger.info(f"Trainable Params    : {trainable_params:,} / {all_params:,} ({percentage:.4f}%)")
        logger.info(
            f"Execution Strategy  : SINGLE GPU ONLY ({self.hardware_info.get('gpu_name', 'CPU')})"
        )
        logger.info("=" * 70)

    def run_smoke_test(
        self, model: Any, tokenizer: Any, dataset_splits: Any
    ) -> bool:
        """
        Performs a tiny 2-step training smoke test on a 4-sample subset before full training.
        """
        logger.info("Executing Pre-Training Smoke Test (max_steps=2 on sample subset)...")
        train_ds = dataset_splits["train"]
        sample_subset = train_ds.select(range(min(4, len(train_ds))))

        smoke_splits = {"train": sample_subset}
        backend = get_training_backend(self.config, self.hardware_info)

        try:
            smoke_output_dir = self.output_dir / "smoke_test"
            smoke_stats = backend.train(
                model=model,
                tokenizer=tokenizer,
                dataset_splits=smoke_splits,
                output_dir=smoke_output_dir,
                config=self.config,
                hardware_info=self.hardware_info,
                max_steps=2,
            )
            logger.info(
                f"Smoke Test Succeeded! (Global step: {smoke_stats.get('global_step')}, "
                f"Loss: {smoke_stats.get('train_loss')})"
            )
            return True
        except Exception as e:
            raise TrainingError(f"Pre-training smoke test failed: {e}") from e

    def train(
        self, model: Any, tokenizer: Any, dataset_splits: Any
    ) -> Dict[str, Any]:
        """
        Runs safety audit, smoke test, full training loop, adapter saving, and metadata writing.
        """
        print_hardware_summary(self.hardware_info)
        self.print_safety_audit(model, tokenizer, dataset_splits)

        # Step 1: Run Smoke Test
        self.run_smoke_test(model, tokenizer, dataset_splits)

        # Step 2: Execute Full Training
        logger.info("Commencing Full QLoRA Fine-Tuning Run...")
        backend = get_training_backend(self.config, self.hardware_info)

        try:
            training_stats = backend.train(
                model=model,
                tokenizer=tokenizer,
                dataset_splits=dataset_splits,
                output_dir=self.output_dir,
                config=self.config,
                hardware_info=self.hardware_info,
                max_steps=None,
            )
        except Exception as e:
            raise TrainingError(f"Full QLoRA training run failed: {e}") from e

        # Step 3: Save Adapter & Tokenizer
        ModelManager.save_adapter(model, tokenizer, self.output_dir)

        # Step 4: Save Metadata
        save_run_metadata(self.output_dir, self.config, self.hardware_info, training_stats)

        logger.info(
            f"QLoRA Training Run Complete. Artifacts stored in: {self.output_dir.absolute()}"
        )
        return training_stats
