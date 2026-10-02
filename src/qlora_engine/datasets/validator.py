"""
Dataset Validator for Universal QLoRA Fine-Tuner.
"""

import json
from collections import Counter
from typing import Any, Dict, List, Tuple

from datasets import Dataset


class DatasetValidationError(Exception):
    """Raised when dataset validation fails or invalid ratio exceeds threshold."""
    pass


class DatasetValidator:
    """
    Validates dataset structure, row completeness, message formats, invalid ratios, and duplicates.
    """

    @staticmethod
    def is_valid_message_list(messages: Any) -> bool:
        """
        Validates if an object is a list of valid chat message dictionaries.
        """
        if not isinstance(messages, (list, tuple)) or len(messages) == 0:
            return False

        valid_roles = {"system", "user", "assistant", "tool"}

        for msg in messages:
            if not isinstance(msg, dict):
                return False
            if "role" not in msg or "content" not in msg:
                return False

            role = str(msg["role"]).strip().lower()
            content = msg["content"]

            if role not in valid_roles:
                return False

            if content is None or not str(content).strip():
                return False

        return True

    @classmethod
    def validate_example(cls, example: Dict[str, Any], schema: str) -> bool:
        """
        Validates a single dataset record against the identified schema.
        """
        if schema == "messages":
            return cls.is_valid_message_list(example.get("messages"))

        if schema in {"instruction_answer", "instruction_output"}:
            inst = example.get("instruction")
            ans = example.get("answer") if "answer" in example else example.get("output")
            return bool(inst and str(inst).strip() and ans and str(ans).strip())

        if schema == "question_answer":
            q = example.get("question")
            a = example.get("answer")
            return bool(q and str(q).strip() and a and str(a).strip())

        if schema == "prompt_response":
            p = example.get("prompt")
            r = example.get("response")
            return bool(p and str(p).strip() and r and str(r).strip())

        if schema == "text":
            t = example.get("text")
            return bool(t and str(t).strip())

        return False

    @classmethod
    def analyze_dataset(
        cls, dataset: Dataset, schema: str, max_invalid_ratio: float = 0.10
    ) -> Tuple[Dataset, Dict[str, Any]]:
        """
        Analyzes invalid records and exact duplicates in dataset.

        Returns:
            Tuple of (cleaned_dataset, validation_summary_dict).
        """
        if len(dataset) == 0:
            raise DatasetValidationError("Dataset is empty.")

        invalid_indices = []
        seen_keys = set()
        duplicate_indices = []

        for idx, example in enumerate(dataset):
            # 1. Validate example structure
            if not cls.validate_example(example, schema):
                invalid_indices.append(idx)
                continue

            # 2. Check duplicate serialization
            if schema == "messages":
                key = json.dumps(example["messages"], sort_keys=True, ensure_ascii=False)
            elif schema == "text":
                key = str(example["text"]).strip()
            else:
                key = json.dumps({k: example[k] for k in sorted(example.keys())}, ensure_ascii=False)

            if key in seen_keys:
                duplicate_indices.append(idx)
            else:
                seen_keys.add(key)

        num_total = len(dataset)
        num_invalid = len(invalid_indices)
        invalid_ratio = num_invalid / max(1, num_total)

        summary = {
            "total_rows": num_total,
            "invalid_rows": num_invalid,
            "invalid_ratio": round(invalid_ratio, 4),
            "duplicate_rows": len(duplicate_indices),
            "valid_unique_rows": num_total - num_invalid - len(duplicate_indices),
        }

        # Verify invalid ratio threshold
        if invalid_ratio > max_invalid_ratio:
            raise DatasetValidationError(
                f"Dataset invalid record ratio ({invalid_ratio:.2%}) exceeds "
                f"maximum allowed threshold ({max_invalid_ratio:.2%})."
            )

        # Filter out invalid indices
        valid_set = set(invalid_indices)
        cleaned_ds = dataset.filter(
            lambda _, idx: idx not in valid_set,
            with_indices=True,
            desc="Filtering invalid records",
        )

        return cleaned_ds, summary
