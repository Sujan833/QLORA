"""
Dataset Cleaner for Universal QLoRA Fine-Tuner.
"""

import json
import unicodedata
from typing import Any, Dict

from datasets import Dataset


class DatasetCleaner:
    """
    Cleans dataset text fields (whitespace, unicode, empty strings, duplicates).
    Never mutates original dataset in place; returns a new Dataset instance.
    """

    @staticmethod
    def normalize_text(
        text: Any,
        trim_whitespace: bool = True,
        normalize_whitespace: bool = True,
        normalize_unicode: bool = True,
    ) -> str:
        """
        Normalizes a string value according to configuration settings.
        """
        if text is None:
            return ""

        s = str(text)

        if normalize_unicode:
            s = unicodedata.normalize("NFC", s)

        if trim_whitespace:
            s = s.strip()

        if normalize_whitespace:
            s = " ".join(s.split())

        return s

    @classmethod
    def clean(cls, dataset: Dataset, config: Dict[str, Any]) -> Dataset:
        """
        Cleans text columns and removes exact duplicates according to YAML cleaning configuration.
        """
        cleaning_cfg = config.get("cleaning", {})
        if not cleaning_cfg.get("enabled", True):
            return dataset

        trim_ws = cleaning_cfg.get("trim_whitespace", True)
        norm_ws = cleaning_cfg.get("normalize_whitespace", True)
        norm_uni = cleaning_cfg.get("normalize_unicode", True)
        remove_dups = cleaning_cfg.get("remove_exact_duplicates", True)

        # 1. Clean row text fields
        def clean_record(example: Dict[str, Any]) -> Dict[str, Any]:
            cleaned = {}
            for col, val in example.items():
                if isinstance(val, str):
                    cleaned[col] = cls.normalize_text(val, trim_ws, norm_ws, norm_uni)
                elif isinstance(val, list):
                    # Clean list of message dicts if applicable
                    cleaned_list = []
                    for item in val:
                        if isinstance(item, dict):
                            c_item = {}
                            for k, v in item.items():
                                c_item[k] = (
                                    cls.normalize_text(v, trim_ws, norm_ws, norm_uni)
                                    if isinstance(v, str)
                                    else v
                                )
                            cleaned_list.append(c_item)
                        else:
                            cleaned_list.append(item)
                    cleaned[col] = cleaned_list
                else:
                    cleaned[col] = val
            return cleaned

        cleaned_ds = dataset.map(clean_record, desc="Cleaning dataset text fields")

        # 2. Remove exact duplicates if enabled
        if remove_dups:
            seen = set()
            duplicate_indices = set()

            for idx, example in enumerate(cleaned_ds):
                key = json.dumps(example, sort_keys=True, ensure_ascii=False)
                if key in seen:
                    duplicate_indices.add(idx)
                else:
                    seen.add(key)

            if duplicate_indices:
                cleaned_ds = cleaned_ds.filter(
                    lambda _, idx: idx not in duplicate_indices,
                    with_indices=True,
                    desc="Removing exact duplicate records",
                )

        return cleaned_ds
