"""Tests for verbatim_core.extractors."""

import importlib
import logging
import sys
import types
from unittest.mock import MagicMock

from verbatim_core.extractors import LLMSpanExtractor, ModelSpanExtractor


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


class TestVerifySpans:
    def setup_method(self):
        self.extractor = LLMSpanExtractor(llm_client=MagicMock())

    def test_keeps_verbatim_spans(self):
        result = self.extractor._verify_spans(["cat", "mat"], "The cat sat on the mat.")
        assert result == ["cat", "mat"]

    def test_filters_non_verbatim_spans(self):
        result = self.extractor._verify_spans(["cat", "dog"], "The cat sat on the mat.")
        assert result == ["cat"]

    def test_strips_whitespace(self):
        result = self.extractor._verify_spans(["  cat  "], "The cat sat.")
        assert result == ["cat"]

    def test_empty_span_filtered(self):
        result = self.extractor._verify_spans(["", "  "], "Some text.")
        assert result == []


class TestVerifySpansFuzzy:
    def setup_method(self):
        self.extractor = LLMSpanExtractor(
            llm_client=MagicMock(),
            span_match_mode="fuzzy",
        )

    def test_fuzzy_match_preserves_document_token_boundaries(self):
        span = (
            "The art of the movement spanned visual, literary, and sound media, "
            "including collage, sound poetry, cut - up writing, and sculpture."
        )
        document = (
            "x The art of the movement spanned visual , literary , and sound media , "
            "including collage , sound poetry , cut - up writing , and sculpture . more"
        )

        result = self.extractor._verify_spans([span], document)

        assert result == [
            (
                "The art of the movement spanned visual , literary , and sound media , "
                "including collage , sound poetry , cut - up writing , and sculpture ."
            )
        ]

    def test_fuzzy_match_normalizes_case_and_punctuation_spacing(self):
        result = self.extractor._verify_spans(
            ["THE CAT, SAT."],
            "Before the cat , sat . after",
        )

        assert result == ["the cat , sat ."]


class TestExtractSpans:
    def test_empty_results(self):
        extractor = LLMSpanExtractor(llm_client=MagicMock())
        result = extractor.extract_spans("What?", [])
        assert result == {}

    def test_batch_mode(self):
        mock_client = MagicMock()
        mock_client.extract_spans.return_value = {
            "doc_0": ["cat sat on the mat"],
        }

        extractor = LLMSpanExtractor(llm_client=mock_client, extraction_mode="batch", batch_size=5)

        result_obj = MagicMock()
        result_obj.text = "The cat sat on the mat."

        result = extractor.extract_spans("What animal?", [result_obj])
        assert "The cat sat on the mat." in result
        assert result["The cat sat on the mat."] == ["cat sat on the mat"]

    def test_individual_mode(self):
        mock_client = MagicMock()
        mock_client.extract_relevant_spans.return_value = ["The cat"]

        extractor = LLMSpanExtractor(llm_client=mock_client, extraction_mode="individual")

        result_obj = MagicMock()
        result_obj.text = "The cat sat."

        result = extractor.extract_spans("What?", [result_obj])
        assert result["The cat sat."] == ["The cat"]

    def test_auto_mode_selects_batch_for_small_input(self):
        mock_client = MagicMock()
        mock_client.extract_spans.return_value = {"doc_0": ["span"]}

        extractor = LLMSpanExtractor(llm_client=mock_client, extraction_mode="auto", batch_size=5)

        result_obj = MagicMock()
        result_obj.text = "Some text with span inside."

        extractor.extract_spans("Q?", [result_obj])
        mock_client.extract_spans.assert_called_once()

    def test_auto_mode_selects_individual_for_large_input(self):
        mock_client = MagicMock()
        mock_client.extract_relevant_spans.return_value = ["span"]

        extractor = LLMSpanExtractor(llm_client=mock_client, extraction_mode="auto", batch_size=2)

        results = []
        for i in range(5):
            r = MagicMock()
            r.text = f"Document {i} with span content."
            results.append(r)

        extractor.extract_spans("Q?", results)
        assert mock_client.extract_relevant_spans.call_count == 5

    def test_batch_fallback_on_error(self):
        mock_client = MagicMock()
        mock_client.extract_spans.side_effect = Exception("API error")
        mock_client.extract_relevant_spans.return_value = ["fallback span"]

        extractor = LLMSpanExtractor(llm_client=mock_client, extraction_mode="batch")

        result_obj = MagicMock()
        result_obj.text = "Text with fallback span."

        result = extractor.extract_spans("Q?", [result_obj])
        assert result["Text with fallback span."] == ["fallback span"]
