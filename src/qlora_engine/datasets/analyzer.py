"""
Token Length Analyzer for Universal QLoRA Fine-Tuner.
"""

import math
from typing import Any, Dict, List

from datasets import Dataset


class TokenAnalyzer:
    """
    Analyzes token length distributions across a formatted dataset using the actual tokenizer.
    """

    @staticmethod
    def analyze(
        dataset: Dataset,
        tokenizer: Any,
        max_seq_length: int = 1024,
        sample_size: int = 500,
        text_column: str = "text",
    ) -> Dict[str, Any]:
        """
        Calculates token length statistics for a sample of records.
        """
        total_rows = len(dataset)
        if total_rows == 0:
            return {"sample_count": 0, "status": "empty dataset"}

        n_samples = min(sample_size, total_rows)
        sample_ds = dataset.select(range(n_samples))

        token_lengths: List[int] = []

        for record in sample_ds:
            text = record.get(text_column, "")
            if not text:
                continue
            token_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            token_lengths.append(len(token_ids))

        if not token_lengths:
            return {"sample_count": 0, "status": "no text contents found"}

        token_lengths.sort()
        count = len(token_lengths)
        min_len = token_lengths[0]
        max_len = token_lengths[-1]
        mean_len = round(sum(token_lengths) / count, 2)

        # Calculate percentiles
        def get_percentile(p: float) -> int:
            idx = int(math.ceil((p / 100.0) * count)) - 1
            return token_lengths[max(0, min(idx, count - 1))]

        median_len = get_percentile(50)
        p90 = get_percentile(90)
        p95 = get_percentile(95)
        p99 = get_percentile(99)

        exceeding_count = sum(1 for l in token_lengths if l > max_seq_length)
        pct_exceeding = round((exceeding_count / count) * 100.0, 2)

        return {
            "sample_count": count,
            "min_length": min_len,
            "max_length": max_len,
            "mean_length": mean_len,
            "median_length": median_len,
            "p90": p90,
            "p95": p95,
            "p99": p99,
            "configured_max_seq_length": max_seq_length,
            "exceeding_max_seq_count": exceeding_count,
            "percent_exceeding": pct_exceeding,
        }
