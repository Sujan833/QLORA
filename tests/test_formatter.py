"""
Unit tests for schema detection and canonical formatting.
"""

import pytest

datasets = pytest.importorskip("datasets")
Dataset = datasets.Dataset

from qlora_engine.datasets.formatter import DatasetFormatter
from qlora_engine.datasets.validator import DatasetValidationError


def test_schema_detection():
    # Schema A: instruction + answer
    ds_inst = Dataset.from_dict({"instruction": ["Hi"], "answer": ["Hello"]})
    assert DatasetFormatter.detect_schema(ds_inst) == "instruction_answer"

    # Schema B: question + answer
    ds_q = Dataset.from_dict({"question": ["Q"], "answer": ["A"]})
    assert DatasetFormatter.detect_schema(ds_q) == "question_answer"

    # Schema C: prompt + response
    ds_p = Dataset.from_dict({"prompt": ["P"], "response": ["R"]})
    assert DatasetFormatter.detect_schema(ds_p) == "prompt_response"

    # Schema D: messages
    ds_m = Dataset.from_dict({"messages": [[{"role": "user", "content": "Hi"}]]})
    assert DatasetFormatter.detect_schema(ds_m) == "messages"


def test_schema_detection_unknown_columns():
    ds_unknown = Dataset.from_dict({"foo": [1], "bar": [2]})
    with pytest.raises(DatasetValidationError, match="Could not automatically determine dataset schema"):
        DatasetFormatter.detect_schema(ds_unknown)


def test_canonical_conversion():
    ds_inst = Dataset.from_dict({"instruction": ["What is X?"], "answer": ["X is Y."]})
    canonical = DatasetFormatter.format_to_canonical(ds_inst, "instruction_answer")

    messages = canonical[0]["messages"]
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[0]["content"] == "What is X?"
    assert messages[1]["role"] == "assistant"
    assert messages[1]["content"] == "X is Y."
