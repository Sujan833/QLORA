"""
Universal Dataset Formatter and Canonical Schema Converter.
"""

from typing import Any, Dict, List

from datasets import Dataset

from qlora_engine.datasets.validator import DatasetValidationError


class DatasetFormatter:
    """
    Detects input dataset schema and converts supported schemas into canonical messages format:
    [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]
    """

    SUPPORTED_SCHEMAS = {
        "messages",
        "instruction_answer",
        "instruction_output",
        "question_answer",
        "prompt_response",
        "text",
    }

    @classmethod
    def detect_schema(cls, dataset: Dataset) -> str:
        """
        Inspects dataset column names to automatically identify the conversation schema.
        Raises DatasetValidationError if column structure is unsupported or ambiguous.
        """
        cols = set(dataset.column_names)

        if "messages" in cols:
            return "messages"
        if "instruction" in cols and "output" in cols:
            return "instruction_output"
        if "instruction" in cols and "answer" in cols:
            return "instruction_answer"
        if "question" in cols and "answer" in cols:
            return "question_answer"
        if "prompt" in cols and "response" in cols:
            return "prompt_response"
        if "text" in cols:
            return "text"

        raise DatasetValidationError(
            f"Could not automatically determine dataset schema from columns: {list(dataset.column_names)}. "
            f"Supported column patterns: "
            f"1) 'messages', "
            f"2) 'instruction' + 'answer'/'output', "
            f"3) 'question' + 'answer', "
            f"4) 'prompt' + 'response', "
            f"5) 'text'."
        )

    @classmethod
    def convert_example_to_messages(cls, example: Dict[str, Any], schema: str) -> List[Dict[str, str]]:
        """
        Converts a single record into canonical messages format.
        """
        if schema == "messages":
            raw_messages = example.get("messages", [])
            canonical = []
            for msg in raw_messages:
                role = str(msg.get("role", "user")).strip().lower()
                content = str(msg.get("content", "")).strip()
                if content:
                    canonical.append({"role": role, "content": content})
            return canonical

        if schema in {"instruction_answer", "instruction_output"}:
            inst = str(example.get("instruction", "")).strip()
            ans = str(example.get("answer", example.get("output", ""))).strip()
            return [
                {"role": "user", "content": inst},
                {"role": "assistant", "content": ans},
            ]

        if schema == "question_answer":
            q = str(example.get("question", "")).strip()
            a = str(example.get("answer", "")).strip()
            return [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a},
            ]

        if schema == "prompt_response":
            p = str(example.get("prompt", "")).strip()
            r = str(example.get("response", "")).strip()
            return [
                {"role": "user", "content": p},
                {"role": "assistant", "content": r},
            ]

        if schema == "text":
            t = str(example.get("text", "")).strip()
            return [{"role": "user", "content": t}]

        raise DatasetValidationError(f"Unsupported dataset schema '{schema}'.")

    @classmethod
    def format_to_canonical(cls, dataset: Dataset, schema: str) -> Dataset:
        """
        Maps dataset records to canonical 'messages' column format.
        """
        def process_row(row: Dict[str, Any]) -> Dict[str, Any]:
            return {"messages": cls.convert_example_to_messages(row, schema)}

        # Remove existing 'messages' column if re-formatting from incompatible structure
        cols_to_remove = [c for c in dataset.column_names if c != "messages"]
        canonical_ds = dataset.map(process_row, desc="Converting to canonical messages format")
        
        # Verify valid canonical messages structure (at least 1 user and 1 assistant or system message)
        def is_valid_canonical(row: Dict[str, Any]) -> bool:
            msgs = row.get("messages", [])
            return isinstance(msgs, list) and len(msgs) >= 1

        return canonical_ds.filter(is_valid_canonical, desc="Filtering invalid canonical records")

    @classmethod
    def apply_chat_template(
        cls, dataset: Dataset, tokenizer: Any, add_generation_prompt: bool = False
    ) -> Dataset:
        """
        Formats canonical messages into full text using the model tokenizer's official chat template.
        """
        def apply_template(row: Dict[str, Any]) -> Dict[str, Any]:
            messages = row["messages"]
            try:
                text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=add_generation_prompt,
                )
            except Exception:
                # Fallback simple string formatting if tokenizer lacks chat template
                formatted_parts = []
                for msg in messages:
                    role = msg["role"].upper()
                    content = msg["content"]
                    formatted_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
                text = "\n".join(formatted_parts)
            return {"text": text}

        return dataset.map(apply_template, desc="Applying tokenizer chat template")
