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


import inspect


def instantiate_sft_trainer(
    model: Any,
    tokenizer: Any,
    train_ds: Any,
    eval_ds: Any,
    checkpoints_dir: Path,
    training_cfg: Dict[str, Any],
    hardware_info: Dict[str, Any],
    max_steps: Optional[int] = None,
) -> tuple[Any, Any]:
    """
    Dynamically creates SFTConfig / TrainingArguments and instantiates SFTTrainer,
    handling version differences across TRL and Transformers seamlessly via signature inspection.
    """
    from transformers import TrainingArguments
    from trl import SFTTrainer

    try:
        from trl import SFTConfig

        config_cls = SFTConfig
    except ImportError:
        config_cls = TrainingArguments

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

    # Inspect config_cls.__init__ signature for supported parameters
    config_sig = inspect.signature(config_cls.__init__)
    config_params = config_sig.parameters

    eval_key = "eval_strategy" if "eval_strategy" in config_params else "evaluation_strategy"

    base_args_dict = {
        "output_dir": str(checkpoints_dir),
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": lr,
        "num_train_epochs": epochs if max_steps is None else 1,
        "max_steps": max_steps if max_steps is not None else -1,
        "fp16": is_fp16 and torch.cuda.is_available(),
        "bf16": is_bf16 and torch.cuda.is_available(),
        "gradient_checkpointing": grad_ckpt,
        "optim": optim if torch.cuda.is_available() else "adamw_torch",
        "logging_steps": int(training_cfg.get("logging_steps", 10)),
        eval_key: "steps" if eval_ds is not None else "no",
        "eval_steps": int(training_cfg.get("eval_steps", 100)),
        "save_strategy": "steps",
        "save_steps": int(training_cfg.get("save_steps", 100)),
        "save_total_limit": 2,
        "seed": seed,
        "report_to": "none",
        "dataloader_num_workers": 0,
        "remove_unused_columns": False,
    }

    trl_extra_args = {
        "max_seq_length": max_seq_len,
        "dataset_text_field": "text",
        "packing": False,
    }

    has_var_kwarg = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in config_params.values())

    args_to_pass = dict(base_args_dict)
    if config_cls is not TrainingArguments:
        for k, v in trl_extra_args.items():
            if has_var_kwarg or k in config_params:
                args_to_pass[k] = v

    try:
        args = config_cls(**args_to_pass)
    except TypeError as e:
        logger.warning(
            f"Failed creating {config_cls.__name__} with kwargs {list(args_to_pass.keys())}: {e}. "
            "Retrying with base TrainingArguments."
        )
        args = TrainingArguments(**base_args_dict)

    if torch.cuda.is_available():
        args._n_gpu = 1

    # Inspect SFTTrainer signature
    trainer_sig = inspect.signature(SFTTrainer.__init__)
    trainer_params = trainer_sig.parameters

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "args": args,
    }

    if "processing_class" in trainer_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in trainer_params or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in trainer_params.values()):
        trainer_kwargs["tokenizer"] = tokenizer

    for extra_key, extra_val in [("dataset_text_field", "text"), ("max_seq_length", max_seq_len), ("packing", False)]:
        if extra_key in trainer_params and getattr(args, extra_key, None) is None:
            trainer_kwargs[extra_key] = extra_val

    try:
        return SFTTrainer(**trainer_kwargs), args
    except TypeError as e:
        logger.warning(
            f"SFTTrainer instantiation failed with primary kwargs ({e}). "
            "Falling back to multi-tiered kwarg resolution..."
        )

        for tok_key in ["processing_class", "tokenizer"]:
            kw = {
                "model": model,
                "train_dataset": train_ds,
                "eval_dataset": eval_ds,
                "args": args,
                tok_key: tokenizer,
            }
            try:
                return SFTTrainer(**kw), args
            except TypeError:
                pass

            try:
                kw["dataset_text_field"] = "text"
                kw["max_seq_length"] = max_seq_len
                kw["packing"] = False
                return SFTTrainer(**kw), args
            except TypeError:
                pass

        return SFTTrainer(model=model, train_dataset=train_ds, eval_dataset=eval_ds, args=args), args


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
        logger.info("Initializing Standard Transformers + PEFT + TRL Training Backend...")

        training_cfg = config.get("training", {})
        batch_size = int(training_cfg.get("batch_size", 1))
        grad_accum = int(training_cfg.get("gradient_accumulation_steps", 8))

        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        train_ds = dataset_splits["train"]
        eval_ds = dataset_splits.get("validation", None)

        trainer, args = instantiate_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            train_ds=train_ds,
            eval_ds=eval_ds,
            checkpoints_dir=checkpoints_dir,
            training_cfg=training_cfg,
            hardware_info=hardware_info,
            max_steps=max_steps,
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
