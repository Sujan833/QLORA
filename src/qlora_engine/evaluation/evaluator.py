"""
Model Evaluator for Universal QLoRA Fine-Tuner.
"""

import json
from pathlib import Path
from typing import Any, Dict, List

from datasets import Dataset
import torch

from qlora_engine.utils.logging import get_logger

logger = get_logger("qlora_engine.evaluation")


class ModelEvaluator:
    """
    Evaluates fine-tuned model performance on held-out test set using sample generations and loss metrics.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.eval_cfg = config.get("evaluation", {})
        self.gen_cfg = self.eval_cfg.get("generation", {})

    def evaluate(
        self,
        model: Any,
        tokenizer: Any,
        test_dataset: Dataset,
        output_dir: Path,
        max_eval_samples: int = 10,
    ) -> Dict[str, Any]:
        """
        Executes sample generation evaluation on test dataset and produces JSON/MD reports.
        """
        logger.info(f"Evaluating Model on Held-out Test Set ({min(max_eval_samples, len(test_dataset))} samples)...")

        model.eval()
        eval_samples = test_dataset.select(range(min(max_eval_samples, len(test_dataset))))

        temperature = float(self.gen_cfg.get("temperature", 0.7))
        top_p = float(self.gen_cfg.get("top_p", 0.9))
        max_new_tokens = int(self.gen_cfg.get("max_new_tokens", 256))
        repetition_penalty = float(self.gen_cfg.get("repetition_penalty", 1.0))

        results: List[Dict[str, Any]] = []

        with torch.no_grad():
            for idx, record in enumerate(eval_samples):
                messages = record.get("messages", [])
                if not messages:
                    continue

                # Separate input user message and target assistant response
                user_msg = next((m["content"] for m in messages if m["role"] == "user"), "")
                ref_ans = next((m["content"] for m in messages if m["role"] == "assistant"), "")

                prompt_messages = [m for m in messages if m["role"] != "assistant"]
                
                try:
                    formatted_prompt = tokenizer.apply_chat_template(
                        prompt_messages,
                        tokenize=False,
                        add_generation_prompt=True,
                    )
                except Exception:
                    formatted_prompt = f"USER: {user_msg}\nASSISTANT:"

                inputs = tokenizer(formatted_prompt, return_tensors="pt")
                if torch.cuda.is_available():
                    inputs = {k: v.to(0) for k, v in inputs.items()}

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                )

                gen_tokens = outputs[0][inputs["input_ids"].shape[1] :]
                generated_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()

                results.append(
                    {
                        "sample_index": idx,
                        "prompt": user_msg,
                        "reference_answer": ref_ans,
                        "generated_answer": generated_text,
                    }
                )

        eval_report = {
            "model_name": self.config.get("model", {}).get("name"),
            "evaluated_samples_count": len(results),
            "generation_config": {
                "temperature": temperature,
                "top_p": top_p,
                "max_new_tokens": max_new_tokens,
                "repetition_penalty": repetition_penalty,
            },
            "sample_results": results,
        }

        # Save report outputs
        output_dir = Path(output_dir)
        eval_dir = output_dir / "evaluation"
        eval_dir.mkdir(parents=True, exist_ok=True)

        json_path = eval_dir / "evaluation_report.json"
        with json_path.open("w", encoding="utf-8") as f:
            json.dump(eval_report, f, indent=2)

        md_path = eval_dir / "evaluation_report.md"
        with md_path.open("w", encoding="utf-8") as f:
            f.write(f"# Evaluation Report\n\n")
            f.write(f"- **Model**: `{eval_report['model_name']}`\n")
            f.write(f"- **Evaluated Samples**: {len(results)}\n\n")
            f.write("## Sample Qualitative Comparisons\n\n")
            for res in results[:5]:
                f.write(f"### Sample {res['sample_index'] + 1}\n")
                f.write(f"**Prompt:**\n> {res['prompt']}\n\n")
                f.write(f"**Reference Answer:**\n```\n{res['reference_answer']}\n```\n\n")
                f.write(f"**Generated Answer:**\n```\n{res['generated_answer']}\n```\n\n")
                f.write("---\n\n")

        logger.info(f"Evaluation report saved to {json_path} and {md_path}")
        return eval_report
