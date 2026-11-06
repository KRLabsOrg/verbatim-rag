"""Tests for verbatim_core.extractors."""

import importlib
import logging
import sys
import types

import pytest
import verbatim_core
from verbatim_core.extractors import ModelSpanExtractor

PACKAGE = "verbatim_core.extractor_models"
DATASET_MODULE = f"{PACKAGE}.dataset"


@pytest.fixture
def import_dataset_module():
    """Imports the dataset module against the packages a test fakes, and forgets it afterwards.

    The package imports the dataset module and binds its classes, so the package goes too.
    """
    names = (PACKAGE, DATASET_MODULE)
    saved_modules = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    saved_attribute = vars(verbatim_core).pop("extractor_models", None)
    yield lambda: importlib.import_module(DATASET_MODULE)
    for name in names:
        sys.modules.pop(name, None)
    vars(verbatim_core).pop("extractor_models", None)
    sys.modules.update(saved_modules)
    if saved_attribute is not None:
        verbatim_core.extractor_models = saved_attribute


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

    def test_legacy_token_budget_warns_when_sentences_are_dropped(
        self, caplog, monkeypatch, import_dataset_module
    ):
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
        dataset = import_dataset_module()

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
