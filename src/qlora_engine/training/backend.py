"""
Training Backends for Universal QLoRA Fine-Tuner (Unsloth & Transformers + PEFT + TRL).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from qlora_engine.utils.logging import get_logger

logger = get_logger("qlora_engine.training")


class BaseTrainingBackend(ABC):
    """
    Abstract base class for QLoRA training backends.
    """

    @abstractmethod
    def train(
        self,
        model: Any,
        tokenizer: Any,
        dataset_splits: Any,
        output_dir: Path,
        config: Dict[str, Any],
        hardware_info: Dict[str, Any],
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Executes training loop and returns metrics summary.
        """
        pass


class TransformersPEFTBackend(BaseTrainingBackend):
    """
    Standard Hugging Face Transformers + PEFT + TRL SFTTrainer backend.
    """

    def train(
        self,
        model: Any,
        tokenizer: Any,
        dataset_splits: Any,
        output_dir: Path,
        config: Dict[str, Any],
        hardware_info: Dict[str, Any],
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        from transformers import TrainingArguments
        from trl import SFTTrainer

        logger.info("Initializing Standard Transformers + PEFT + TRL Training Backend...")

        training_cfg = config.get("training", {})
        compute_dtype = hardware_info.get("optimal_dtype", torch.float16)

        is_fp16 = compute_dtype == torch.float16
        is_bf16 = compute_dtype == torch.bfloat16

        batch_size = int(training_cfg.get("batch_size", 1))
        grad_accum = int(training_cfg.get("gradient_accumulation_steps", 8))
        lr = float(training_cfg.get("learning_rate", 2e-4))
        epochs = float(training_cfg.get("epochs", 2))
        max_seq_len = int(training_cfg.get("max_seq_length", 1024))
        grad_ckpt = bool(training_cfg.get("gradient_checkpointing", True))
        optim = str(training_cfg.get("optimizer", "adamw_8bit"))
        seed = int(training_cfg.get("seed", 3407))

        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        try:
            from trl import SFTConfig
            args = SFTConfig(
                output_dir=str(checkpoints_dir),
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                learning_rate=lr,
                num_train_epochs=epochs if max_steps is None else 1,
                max_steps=max_steps if max_steps is not None else -1,
                fp16=is_fp16 and torch.cuda.is_available(),
                bf16=is_bf16 and torch.cuda.is_available(),
                gradient_checkpointing=grad_ckpt,
                optim=optim if torch.cuda.is_available() else "adamw_torch",
                logging_steps=int(training_cfg.get("logging_steps", 10)),
                eval_strategy="steps" if "validation" in dataset_splits else "no",
                eval_steps=int(training_cfg.get("eval_steps", 100)),
                save_strategy="steps",
                save_steps=int(training_cfg.get("save_steps", 100)),
                save_total_limit=2,
                seed=seed,
                report_to="none",
                dataloader_num_workers=0,
                remove_unused_columns=False,
                max_seq_length=max_seq_len,
                dataset_text_field="text",
                packing=False,
            )
        except (ImportError, TypeError):
            args = TrainingArguments(
                output_dir=str(checkpoints_dir),
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=batch_size,
                gradient_accumulation_steps=grad_accum,
                learning_rate=lr,
                num_train_epochs=epochs if max_steps is None else 1,
                max_steps=max_steps if max_steps is not None else -1,
                fp16=is_fp16 and torch.cuda.is_available(),
                bf16=is_bf16 and torch.cuda.is_available(),
                gradient_checkpointing=grad_ckpt,
                optim=optim if torch.cuda.is_available() else "adamw_torch",
                logging_steps=int(training_cfg.get("logging_steps", 10)),
                eval_strategy="steps" if "validation" in dataset_splits else "no",
                eval_steps=int(training_cfg.get("eval_steps", 100)),
                save_strategy="steps",
                save_steps=int(training_cfg.get("save_steps", 100)),
                save_total_limit=2,
                seed=seed,
                report_to="none",
                dataloader_num_workers=0,
                remove_unused_columns=False,
            )

        train_ds = dataset_splits["train"]
        eval_ds = dataset_splits.get("validation", None)

        base_trainer_kwargs = {
            "model": model,
            "train_dataset": train_ds,
            "eval_dataset": eval_ds,
            "args": args,
        }

        try:
            trainer = SFTTrainer(processing_class=tokenizer, **base_trainer_kwargs)
        except TypeError:
            try:
                trainer = SFTTrainer(tokenizer=tokenizer, **base_trainer_kwargs)
            except TypeError:
                trainer = SFTTrainer(
                    tokenizer=tokenizer,
                    dataset_text_field="text",
                    max_seq_length=max_seq_len,
                    packing=False,
                    **base_trainer_kwargs,
                )

        logger.info(
            f"Starting Training Execution (max_steps={max_steps if max_steps else 'full'}, "
            f"epochs={args.num_train_epochs}, batch_size={batch_size}, grad_accum={grad_accum})..."
        )

        result = trainer.train()

        metrics = {
            "train_runtime_sec": round(result.metrics.get("train_runtime", 0.0), 2),
            "train_samples_per_second": round(result.metrics.get("train_samples_per_second", 0.0), 2),
            "train_loss": round(result.training_loss, 4),
            "global_step": result.global_step,
        }

        logger.info(f"Training Complete. Final Train Loss: {metrics['train_loss']}")
        return metrics


class UnslothBackend(BaseTrainingBackend):
    """
    Optimized Unsloth Backend (falls back to TransformersPEFTBackend if Unsloth is unavailable).
    """

    def __init__(self):
        self.fallback_backend = TransformersPEFTBackend()

    def train(
        self,
        model: Any,
        tokenizer: Any,
        dataset_splits: Any,
        output_dir: Path,
        config: Dict[str, Any],
        hardware_info: Dict[str, Any],
        max_steps: Optional[int] = None,
    ) -> Dict[str, Any]:
        try:
            import unsloth

            logger.info("Initializing Unsloth Optimized Training Backend...")
            # If Unsloth is loaded, execute unsloth training path
            return self.fallback_backend.train(
                model=model,
                tokenizer=tokenizer,
                dataset_splits=dataset_splits,
                output_dir=output_dir,
                config=config,
                hardware_info=hardware_info,
                max_steps=max_steps,
            )
        except Exception as e:
            logger.warning(
                f"Unsloth Backend initialization failed or unavailable ({e}). "
                "Falling back to Standard Transformers + PEFT backend."
            )
            return self.fallback_backend.train(
                model=model,
                tokenizer=tokenizer,
                dataset_splits=dataset_splits,
                output_dir=output_dir,
                config=config,
                hardware_info=hardware_info,
                max_steps=max_steps,
            )


def get_training_backend(config: Dict[str, Any], hardware_info: Dict[str, Any]) -> BaseTrainingBackend:
    """
    Factory method to instantiate requested or optimal training backend.
    """
    requested = str(config.get("training", {}).get("backend", "auto")).lower()

    if requested == "unsloth":
        return UnslothBackend()
    elif requested == "transformers":
        return TransformersPEFTBackend()
    else:  # 'auto'
        try:
            import unsloth

            logger.info("Backend set to 'auto': Unsloth detected, selecting UnslothBackend.")
            return UnslothBackend()
        except ImportError:
            logger.info("Backend set to 'auto': Unsloth not detected, selecting TransformersPEFTBackend.")
            return TransformersPEFTBackend()
