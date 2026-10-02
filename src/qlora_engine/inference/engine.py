"""
Standalone Inference Engine for Universal QLoRA Fine-Tuner.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch

from qlora_engine.utils.hardware import detect_hardware
from qlora_engine.utils.logging import get_logger

logger = get_logger("qlora_engine.inference")


class InferenceEngine:
    """
    Independent inference engine capable of reloading saved base models and LoRA adapters.
    """

    def __init__(
        self,
        base_model_name: str,
        adapter_path: Optional[Union[str, Path]] = None,
        quantize_4bit: bool = True,
    ):
        self.base_model_name = base_model_name
        self.adapter_path = Path(adapter_path) if adapter_path else None
        self.hardware_info = detect_hardware()
        self.cuda_available = self.hardware_info.get("cuda_available", False)
        self.compute_dtype = self.hardware_info.get("optimal_dtype", torch.float16)

        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model_and_adapter(quantize_4bit)

    def _load_tokenizer(self) -> Any:
        from transformers import AutoTokenizer

        logger.info(f"Loading inference tokenizer from '{self.base_model_name}'...")
        tokenizer_src = self.adapter_path / "tokenizer" if (self.adapter_path and (self.adapter_path / "tokenizer").exists()) else self.base_model_name
        tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_src), trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        return tokenizer

    def _load_model_and_adapter(self, quantize_4bit: bool) -> Any:
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig

        logger.info(f"Loading base model '{self.base_model_name}'...")
        device_map = {"": 0} if self.cuda_available else "auto"

        if quantize_4bit and self.cuda_available:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=self.compute_dtype,
            )
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                quantization_config=bnb_config,
                device_map=device_map,
                torch_dtype=self.compute_dtype,
                trust_remote_code=True,
            )
        else:
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model_name,
                device_map=device_map,
                torch_dtype=self.compute_dtype if self.cuda_available else torch.float32,
                trust_remote_code=True,
            )

        if self.adapter_path:
            adapter_dir = self.adapter_path / "adapter" if (self.adapter_path / "adapter").exists() else self.adapter_path
            logger.info(f"Attaching saved LoRA adapter from '{adapter_dir}'...")
            model = PeftModel.from_pretrained(base_model, str(adapter_dir))
        else:
            model = base_model

        model.eval()
        return model

    def generate(
        self,
        prompt_or_messages: Union[str, List[Dict[str, str]]],
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_new_tokens: int = 256,
        repetition_penalty: float = 1.0,
    ) -> str:
        """
        Generates text response from prompt string or list of message dictionaries.
        """
        if isinstance(prompt_or_messages, list):
            try:
                formatted_input = self.tokenizer.apply_chat_template(
                    prompt_or_messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            except Exception:
                user_msg = next((m["content"] for m in prompt_or_messages if m["role"] == "user"), "")
                formatted_input = f"USER: {user_msg}\nASSISTANT:"
        else:
            formatted_input = prompt_or_messages

        inputs = self.tokenizer(formatted_input, return_tensors="pt")
        if self.cuda_available:
            inputs = {k: v.to(0) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        gen_tokens = outputs[0][inputs["input_ids"].shape[1] :]
        response = self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        return response
