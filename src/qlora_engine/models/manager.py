"""
Model Manager for Universal QLoRA Fine-Tuner.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple, Union

import torch

from qlora_engine.utils.hardware import get_optimal_compute_dtype
from qlora_engine.utils.logging import get_logger

logger = get_logger("qlora_engine.models")


class ModelLoadError(Exception):
    """Raised when model loading or quantization fails."""
    pass


class ModelManager:
    """
    Manages loading causal language models, tokenizers, 4-bit quantization, and attaching LoRA adapters.
    """

    def __init__(self, config: Dict[str, Any], hardware_info: Dict[str, Any]):
        self.config = config
        self.hardware_info = hardware_info
        self.model_cfg = config.get("model", {})
        self.lora_cfg = config.get("lora", {})
        self.model_name = self.model_cfg.get("name")
        if not self.model_name:
            raise ModelLoadError("model.name is missing in configuration.")

    def load_tokenizer(self) -> Any:
        """
        Loads model tokenizer with proper pad token and padding side settings.
        """
        from transformers import AutoTokenizer

        logger.info(f"Loading tokenizer: {self.model_name}")
        try:
            tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                trust_remote_code=True,
            )
        except Exception as e:
            raise ModelLoadError(f"Failed to load tokenizer '{self.model_name}': {e}") from e

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            logger.info(f"pad_token was None. Set pad_token = eos_token ('{tokenizer.eos_token}')")

        tokenizer.padding_side = "right"
        return tokenizer

    def load_model(self) -> Any:
        """
        Loads causal language model with 4-bit quantization and single-GPU placement.
        """
        from peft import prepare_model_for_kbit_training
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        logger.info(f"Loading model: {self.model_name}")
        cuda_available = self.hardware_info.get("cuda_available", False)

        quant_cfg = self.model_cfg.get("quantization", {})
        use_4bit = quant_cfg.get("load_in_4bit", True) and cuda_available
        compute_dtype = self.hardware_info.get("optimal_dtype", torch.float16)

        if use_4bit:
            quant_type = quant_cfg.get("quant_type", "nf4")
            double_quant = quant_cfg.get("double_quantization", True)
            logger.info(
                f"Configuring 4-bit Quantization (nf4: {quant_type}, "
                f"double_quant: {double_quant}, compute_dtype: {compute_dtype})"
            )
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=quant_type,
                bnb_4bit_use_double_quant=double_quant,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            device_map = {"": 0} if cuda_available else "auto"
            try:
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=bnb_config,
                    device_map=device_map,
                    torch_dtype=compute_dtype,
                    trust_remote_code=True,
                )
            except Exception as e:
                raise ModelLoadError(f"Failed to load 4-bit quantized model '{self.model_name}': {e}") from e
            
            model = prepare_model_for_kbit_training(model)
        else:
            logger.info(f"Loading model in full/half precision (CUDA available: {cuda_available})")
            device_map = {"": 0} if cuda_available else "auto"
            model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map=device_map,
                torch_dtype=compute_dtype if cuda_available else torch.float32,
                trust_remote_code=True,
            )

        return model

    def attach_lora(self, model: Any) -> Any:
        """
        Inspects model submodules and attaches LoRA adapters.
        """
        from peft import LoraConfig, get_peft_model

        r = int(self.lora_cfg.get("r", 16))
        alpha = int(self.lora_cfg.get("alpha", 32))
        dropout = float(self.lora_cfg.get("dropout", 0.0))
        target_modules = list(self.lora_cfg.get("target_modules", []))

        # Inspect model modules to ensure target modules exist
        all_module_names = set(name.split(".")[-1] for name, _ in model.named_modules())
        valid_targets = [m for m in target_modules if m in all_module_names]

        if not valid_targets:
            logger.warning(
                f"Configured target modules {target_modules} not found in model module names. "
                f"Available linear modules sample: {sorted(list(all_module_names))[:10]}"
            )
            valid_targets = target_modules

        logger.info(
            f"Attaching LoRA adapter (r={r}, alpha={alpha}, dropout={dropout}, "
            f"targets={valid_targets})"
        )

        peft_config = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=valid_targets,
        )

        try:
            model = get_peft_model(model, peft_config)
        except Exception as e:
            raise ModelLoadError(f"Failed to attach LoRA adapter to model: {e}") from e

        trainable_params, all_params, percentage = self.get_trainable_parameters(model)
        logger.info(
            f"LoRA Attached Successfully: {trainable_params:,} trainable parameters out of "
            f"{all_params:,} total ({percentage:.4f}%)"
        )
        if trainable_params == 0:
            raise ModelLoadError("No trainable parameters found after LoRA attachment.")

        return model

    @staticmethod
    def get_trainable_parameters(model: Any) -> Tuple[int, int, float]:
        """
        Returns (trainable_params, all_params, trainable_percentage).
        """
        trainable = 0
        total = 0
        for _, param in model.named_parameters():
            numel = param.numel()
            total += numel
            if param.requires_grad:
                trainable += numel

        percentage = (100.0 * trainable / max(1, total))
        return trainable, total, percentage

    @staticmethod
    def save_adapter(model: Any, tokenizer: Any, output_dir: Union[str, Path]) -> Path:
        """
        Saves LoRA adapter weights and tokenizer to output directory.
        """
        output_dir = Path(output_dir)
        adapter_dir = output_dir / "adapter"
        tokenizer_dir = output_dir / "tokenizer"

        adapter_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Saving LoRA adapter to {adapter_dir}")
        model.save_pretrained(str(adapter_dir))

        logger.info(f"Saving tokenizer to {tokenizer_dir}")
        tokenizer.save_pretrained(str(tokenizer_dir))

        return adapter_dir
