"""Tests for verbatim_core.extractors."""

import importlib
import logging
import sys
import types

from verbatim_core.extractors import ModelSpanExtractor


class TestModelSpanExtractorWarnings:
    def test_detection_failure_warns_before_legacy_fallback(self, caplog, monkeypatch):
        class FailingAutoConfig:
            @staticmethod
            def from_pretrained(*args, **kwargs):
                raise RuntimeError("offline config")

        transformers = types.ModuleType("transformers")
        transformers.AutoConfig = FailingAutoConfig
        monkeypatch.setitem(sys.modules, "transformers", transformers)

        with caplog.at_level(logging.WARNING, logger="verbatim_core.extractors"):
            result = ModelSpanExtractor._detect_format("org/highlighter")

        assert result == ModelSpanExtractor._FORMAT_QA_MODEL
        assert "Highlighter detection failed for org/highlighter: offline config" in caplog.text

    def test_legacy_token_budget_warns_when_sentences_are_dropped(self, caplog, monkeypatch):
        class FakeDataset:
            pass

        class FakeTensor(list):
            pass

        class FakeTokenizer:
            sep_token_id = 2

            def encode_plus(self, text, **kwargs):
                size = 2 if kwargs.get("add_special_tokens") else len(text.split())
                return {
                    "input_ids": list(range(size)),
                    "attention_mask": [1] * size,
                    "offset_mapping": [(0, 1)] * size,
                }

        torch = types.ModuleType("torch")
        torch.long = "long"
        torch.tensor = lambda values, dtype=None: FakeTensor(values)
        torch_utils = types.ModuleType("torch.utils")
        torch_utils_data = types.ModuleType("torch.utils.data")
        torch_utils_data.Dataset = FakeDataset
        torch.utils = torch_utils
        torch_utils.data = torch_utils_data
        transformers = types.ModuleType("transformers")
        transformers.AutoTokenizer = object

        monkeypatch.setitem(sys.modules, "torch", torch)
        monkeypatch.setitem(sys.modules, "torch.utils", torch_utils)
        monkeypatch.setitem(sys.modules, "torch.utils.data", torch_utils_data)
        monkeypatch.setitem(sys.modules, "transformers", transformers)
        monkeypatch.delitem(sys.modules, "verbatim_core.extractor_models.dataset", raising=False)
        dataset = importlib.import_module("verbatim_core.extractor_models.dataset")

        sample = dataset.QASample(
            question="question",
            documents=[
                dataset.Document(
                    [
                        dataset.Sentence("one two", False, "1"),
                        dataset.Sentence("three four five", False, "2"),
                    ]
                )
            ],
            split="test",
            dataset_name="inference",
            task_type="qa",
        )

        with caplog.at_level(logging.WARNING, logger="verbatim_core.extractor_models.dataset"):
            dataset.QADataset([sample], FakeTokenizer(), max_length=7)[0]

        assert "exceeded the 7-token budget; dropping 1 sentence(s)" in caplog.text
