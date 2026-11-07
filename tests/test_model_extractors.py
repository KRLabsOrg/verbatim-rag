"""ModelSpanExtractor and SemanticHighlightExtractor, through extract with stub models."""

import asyncio
import logging
import re
import subprocess
import sys
import threading
import time
from types import SimpleNamespace

import pytest
from extractor_contract import ExtractorContract
from verbatim_core import Chunk, ConfigurationError, InvalidInputError, VerbatimDeprecationWarning
from verbatim_core.extractors import (
    ModelExtractConfig,
    ModelSpanExtractor,
    SemanticHighlightConfig,
    SemanticHighlightExtractor,
)

QUESTION = "When did the bridge open?"
OPENED = Chunk(id="opened", text="The bridge opened in March 1932. It carries eight lanes.")
TOLLS = Chunk(id="tolls", text="Tolls are collected electronically.")


def words(text):
    return {word for word in re.findall(r"\w+", text.lower()) if len(word) > 3}


def sentence_offsets(text):
    return [(match.start(), match.end()) for match in re.finditer(r"[^.!?\s][^.!?]*[.!?]?", text)]


class StubHighlighter:
    """Marks each sentence that shares words with the question. More shared words score higher."""

    def __init__(self, config=None):
        self.config = config
        self.calls = []
        self.devices = []
        self.tokenizer_loaded = False

    def _get_tokenizer(self):
        self.tokenizer_loaded = True

    def to(self, device):
        self.devices.append(device)
        return self

    def eval(self):
        return self

    def process(self, question, context, **settings):
        self.calls.append(SimpleNamespace(thread=threading.current_thread(), **settings))
        spans = []
        for start, end in sentence_offsets(context):
            shared = words(context[start:end]) & words(question)
            if shared:
                score = min(0.5 + 0.2 * len(shared), 0.95)
                spans.append(
                    {"start": start, "end": end, "text": context[start:end], "score": score}
                )
        return {"spans": spans}


class StubSemanticHighlighter:
    """Rates each token, and each sentence, by whether its words occur in the question.

    The model tokenizes question, separator and chunk together and reports its
    context_ranges over that combined tokenization, prefix count including the
    trailing separator token -- the convention the real remote model uses
    (#123). The extractor must read its offsets from the same encoding."""

    max_length = 4096
    sep_token = "</s>"

    def __init__(self, config=None):
        self.config = config
        self.calls = []
        self.tokenizer = self._Tokenizer(self.sep_token)

    def to(self, device):
        return self

    def eval(self):
        return self

    class _Tokenizer:
        def __init__(self, sep_token):
            self.sep_token = sep_token

        def __call__(self, text, **settings):
            return {"offset_mapping": [match.span() for match in re.finditer(r"\w+|[^\w\s]", text)]}

    def process(self, question, context, **settings):
        self.calls.append(settings)
        sentences = [context[start:end] for start, end in sentence_offsets(context)]
        probabilities = [
            0.9 if words(sentence) & words(question) else 0.1 for sentence in sentences
        ]
        threshold = settings["threshold"]
        return {
            "highlighted_sentences": [
                f"{sentence} " for sentence, p in zip(sentences, probabilities) if p > threshold
            ],
            "sentence_probabilities": probabilities,
        }

    def get_raw_predictions(self, question, contexts):
        context = contexts[0]
        answers = [
            (start, end)
            for start, end in sentence_offsets(context)
            if words(context[start:end]) & words(question)
        ]
        probabilities = [
            0.8 if any(start <= token.start() < end for start, end in answers) else 0.1
            for token in re.finditer(r"\w+|[^\w\s]", context)
        ]
        return combined_raw_predictions(question, self.sep_token, context, probabilities)


def combined_raw_predictions(question, sep_token, context, probabilities):
    """Package context probabilities the way the remote model does: indices
    over the combined question + separator + chunk tokenization, the prefix
    count including the trailing separator token (#123)."""

    combined = question + sep_token + context
    prefix_length = len(combined) - len(context)
    tokens = list(re.finditer(r"\w+|[^\w\s]", combined))
    prefix_count = sum(1 for match in tokens if match.end() <= prefix_length)
    return SimpleNamespace(
        pruning_probs=[0.0] * prefix_count + list(probabilities),
        context_ranges=[(prefix_count, prefix_count + len(probabilities))],
    )


def highlighter(**settings):
    return ModelSpanExtractor(
        StubHighlighter(), model_id="stub/highlighter", model_version="c0ffee", **settings
    )


def semantic_highlighter(**settings):
    return SemanticHighlightExtractor(
        StubSemanticHighlighter(), model_id="stub/semantic", model_version="c0ffee", **settings
    )


class TestHighlighterContract(ExtractorContract):
    @pytest.fixture
    def extractor(self):
        return highlighter()


class TestSemanticSentencesContract(ExtractorContract):
    @pytest.fixture
    def extractor(self):
        return semantic_highlighter()


class TestSemanticSpansContract(ExtractorContract):
    @pytest.fixture
    def extractor(self):
        return semantic_highlighter(config=SemanticHighlightConfig(output_mode="spans"))


class TestHighlighter:
    def test_spans_keep_the_offsets_and_probabilities_of_the_model(self):
        extraction = highlighter().extract(QUESTION, [OPENED, TOLLS])

        [span] = extraction.spans
        assert (span.chunk_id, span.start, span.end) == ("opened", 0, 32)
        assert span.text == "The bridge opened in March 1932."
        assert span.score == pytest.approx(0.7)

    def test_the_config_reaches_the_model(self):
        extractor = highlighter(config=ModelExtractConfig(threshold=0.3, doc_stride=64))

        extractor.extract(QUESTION, [OPENED])

        [call] = extractor.model.calls
        assert (call.threshold, call.min_span_chars, call.merge_gap_chars) == (0.3, 30, 20)
        assert (call.max_length, call.doc_stride) == (8192, 64)

    def test_a_call_config_replaces_the_default_for_that_call_only(self):
        extractor = highlighter()

        extractor.extract(QUESTION, [OPENED], config=ModelExtractConfig(threshold=0.7))
        extractor.extract(QUESTION, [OPENED])

        assert [call.threshold for call in extractor.model.calls] == [0.7, 0.2]

    def test_a_config_of_another_type_is_rejected(self):
        with pytest.raises(InvalidInputError) as raised:
            highlighter().extract(QUESTION, [OPENED], config=SemanticHighlightConfig())

        assert raised.value.param == "config"

    def test_empty_chunks_are_not_sent_to_the_model(self):
        extractor = highlighter()

        extractor.extract(QUESTION, [Chunk(id="empty", text="  "), OPENED])

        assert len(extractor.model.calls) == 1

    async def test_a_model_error_propagates(self):
        class BrokenHighlighter(StubHighlighter):
            def process(self, question, context, **settings):
                raise RuntimeError("CUDA out of memory")

        extractor = ModelSpanExtractor(BrokenHighlighter(), model_id="stub", model_version="1")

        with pytest.raises(RuntimeError):
            extractor.extract(QUESTION, [OPENED])
        with pytest.raises(RuntimeError):
            await extractor.aextract(QUESTION, [OPENED])

    async def test_aextract_runs_the_model_in_a_worker_thread(self):
        extractor = highlighter()

        await extractor.aextract(QUESTION, [OPENED])

        assert extractor.model.calls[0].thread is not threading.main_thread()

    def test_usage_names_the_model_and_counts_the_chunks(self):
        extraction = highlighter().extract(QUESTION, [OPENED, TOLLS])

        [usage] = extraction.usage
        assert (usage.model_id, usage.model_version) == ("stub/highlighter", "c0ffee")
        assert usage.units == {"chunks": 2, "characters": len(OPENED.text) + len(TOLLS.text)}


class TestSemanticSentences:
    def test_sentences_are_located_in_order_and_scored_with_their_probability(self):
        repeated = Chunk(id="r", text="The bridge opened. Tolls apply. The bridge opened.")

        extraction = semantic_highlighter().extract(QUESTION, [repeated])

        assert [(span.start, span.end) for span in extraction.spans] == [(0, 18), (32, 50)]
        assert {span.score for span in extraction.spans} == {0.9}

    def test_without_matching_probabilities_the_sentences_are_unscored(self):
        class NoProbabilities(StubSemanticHighlighter):
            def process(self, question, context, **settings):
                return {"highlighted_sentences": ["The bridge opened in March 1932."]}

        extractor = SemanticHighlightExtractor(NoProbabilities(), model_id="m", model_version="1")

        [span] = extractor.extract(QUESTION, [OPENED]).spans

        assert span.score == 1.0

    def test_a_sentence_that_is_not_in_the_chunk_becomes_a_warning(self, caplog):
        class Rewriting(StubSemanticHighlighter):
            def process(self, question, context, **settings):
                return {"highlighted_sentences": ["The bridge opened in 1932."]}

        extractor = SemanticHighlightExtractor(Rewriting(), model_id="m", model_version="1")

        extraction = extractor.extract(QUESTION, [OPENED])

        assert extraction.spans == []
        assert [(warning.code, warning.message) for warning in extraction.warnings] == [
            (
                "span_text_not_found",
                "A sentence the model highlighted for chunks[0] is not in the chunk.",
            )
        ]
        [record] = [record for record in caplog.records if record.levelno == logging.WARNING]
        assert (record.event, record.warning_code) == ("fallback_used", "span_text_not_found")

    def test_the_model_draws_no_progress_bar_and_reads_no_environment(self):
        extractor = semantic_highlighter()

        extractor.extract(QUESTION, [OPENED])

        [settings] = extractor.model.calls
        assert settings["show_progress"] is False
        assert (settings["preprocess_workers"], settings["preprocess_batch_size"]) == (0, 32)


class TestSemanticSpans:
    def config(self, **settings):
        return SemanticHighlightConfig(output_mode="spans", **settings)

    def test_token_regions_become_offsets_without_end_punctuation(self):
        extraction = semantic_highlighter(config=self.config()).extract(QUESTION, [OPENED])

        [span] = extraction.spans
        assert span.text == "The bridge opened in March 1932"
        assert span.score == pytest.approx(0.8)

    def test_regions_shorter_than_the_minimum_are_dropped(self):
        extractor = semantic_highlighter(config=self.config(min_span_tokens=8))

        assert extractor.extract(QUESTION, [OPENED]).spans == []

    def test_a_region_is_scored_with_its_mean_probability(self):
        class Uneven(StubSemanticHighlighter):
            def get_raw_predictions(self, question, contexts):
                return combined_raw_predictions(
                    question,
                    self.sep_token,
                    "Bridge opened today in March",
                    [0.6, 1.0, 0.8, 0.1, 0.1],
                )

        chunk = Chunk(id="c", text="Bridge opened today in March")
        extractor = SemanticHighlightExtractor(
            Uneven(), model_id="m", model_version="1", config=self.config()
        )

        [span] = extractor.extract(QUESTION, [chunk]).spans

        assert span.text == "Bridge opened today"
        assert span.score == pytest.approx(0.8)

    def test_regions_close_together_merge(self):
        class Gapped(StubSemanticHighlighter):
            def get_raw_predictions(self, question, contexts):
                return combined_raw_predictions(
                    question,
                    self.sep_token,
                    "one two three four five six seven",
                    [0.9, 0.9, 0.9, 0.1, 0.9, 0.9, 0.9],
                )

        chunk = Chunk(id="c", text="one two three four five six seven")

        def spans(merge_gap):
            config = self.config(merge_gap=merge_gap)
            extractor = SemanticHighlightExtractor(
                Gapped(), model_id="m", model_version="1", config=config
            )
            return [span.text for span in extractor.extract(QUESTION, [chunk]).spans]

        assert spans(merge_gap=1) == ["one two three four five six seven"]
        assert spans(merge_gap=0) == ["one two three", "five six seven"]

    def test_tokens_the_model_did_not_rate_become_a_warning(self, caplog):
        class Truncating(StubSemanticHighlighter):
            def get_raw_predictions(self, question, contexts):
                return combined_raw_predictions(question, self.sep_token, OPENED.text, [0.9] * 4)

        extractor = SemanticHighlightExtractor(
            Truncating(), model_id="m", model_version="1", config=self.config()
        )

        extraction = extractor.extract(QUESTION, [OPENED])

        assert [span.text for span in extraction.spans] == ["The bridge opened in"]
        assert [(warning.code, warning.message) for warning in extraction.warnings] == [
            ("chunk_truncated", "The model examined only 4 of 12 tokens of chunks[0].")
        ]
        [record] = [record for record in caplog.records if record.levelno == logging.WARNING]
        assert record.warning_code == "chunk_truncated"

    def test_each_probability_maps_to_its_own_token(self):
        """#123: the model's context_ranges index its combined tokenization
        (question + separator + chunk). Reading offsets from a separate
        chunk-only encoding started every mapped probability one token early,
        so a token's score landed on the previous token's characters."""

        class Alternating(StubSemanticHighlighter):
            def get_raw_predictions(self, question, contexts):
                return combined_raw_predictions(
                    question, self.sep_token, "alpha bridge gamma", [0.1, 0.9, 0.1]
                )

        chunk = Chunk(id="c", text="alpha bridge gamma")
        extractor = SemanticHighlightExtractor(
            Alternating(),
            model_id="m",
            model_version="1",
            config=self.config(threshold=0.5, min_span_tokens=1),
        )

        spans = extractor.extract(QUESTION, [chunk]).spans

        [span] = spans
        assert span.text == "bridge"
        assert span.score == pytest.approx(0.9)


class TestSentenceModelContract(ExtractorContract):
    @pytest.fixture
    def extractor(self):
        return sentence_model_extractor(pytest.importorskip("torch"))


class TestSentenceModel:
    """The v1 sentence model, which needs torch."""

    @pytest.fixture
    def torch(self):
        return pytest.importorskip("torch")

    @pytest.fixture
    def extractor(self, torch):
        return sentence_model_extractor(torch)

    def test_spans_are_sentences_scored_with_their_probability(self, extractor, torch):
        extraction = extractor.extract(QUESTION, [OPENED])

        [span] = extraction.spans
        assert (span.start, span.end) == (0, 32)
        assert span.score == pytest.approx(float(torch.sigmoid(torch.tensor(4.0))))

    def test_the_threshold_decides(self, extractor):
        config = ModelExtractConfig(threshold=0.99)

        assert extractor.extract(QUESTION, [OPENED], config=config).spans == []

    def test_sentences_beyond_the_input_limit_become_a_warning(self, extractor):
        long_chunk = Chunk(id="long", text=" ".join(["Bridge words fill this sentence."] * 120))

        extraction = extractor.extract(QUESTION, [long_chunk])

        [warning] = extraction.warnings
        assert warning.code == "chunk_truncated"
        assert warning.message.endswith(" of 120 sentences of chunks[0].")

    def test_it_needs_its_tokenizer(self, torch):
        with pytest.raises(ConfigurationError):
            ModelSpanExtractor(StubSentenceModel(torch), model_id="m", model_version="1")


class StubTokenizer:
    """Gives each lowercased word an id."""

    cls_token_id, sep_token_id = 1, 2

    def __init__(self):
        self.ids = {}

    def encode_plus(self, text, add_special_tokens=False, **settings):
        ids = [
            self.ids.setdefault(word.lower(), len(self.ids) + 3)
            for word in re.findall(r"\w+", text)
        ]
        if add_special_tokens:
            ids = [self.cls_token_id, *ids, self.sep_token_id]
        return {
            "input_ids": ids,
            "attention_mask": [1] * len(ids),
            "offset_mapping": [(0, 0)] * len(ids),
        }


class StubSentenceModel:
    """Rates a sentence highly when it shares a long word with the question."""

    def __init__(self, torch, tokenizer=None):
        self.torch = torch
        self.tokenizer = tokenizer

    def parameters(self):
        return iter([self.torch.zeros(1)])

    def __call__(self, input_ids, attention_mask, sentence_boundaries):
        ids = input_ids[0].tolist()
        boundaries = sentence_boundaries[0]
        long_words = {id for word, id in self.tokenizer.ids.items() if len(word) > 3}
        question = set(ids[1 : boundaries[0][0] - 1]) & long_words
        logits = [
            [0.0, 4.0] if question & set(ids[start : end + 1]) else [0.0, -4.0]
            for start, end in boundaries
        ]
        return [self.torch.tensor(logits)]


def sentence_model_extractor(torch):
    tokenizer = StubTokenizer()
    return ModelSpanExtractor(
        StubSentenceModel(torch, tokenizer),
        config=ModelExtractConfig(threshold=0.5),
        tokenizer=tokenizer,
        model_id="stub/v1",
        model_version="c0ffee",
    )


# Concurrent calls


class Overlaps:
    """Records the most calls that were inside a model's methods at the same time."""

    def __init__(self):
        self._lock = threading.Lock()
        self._inside = 0
        self.most = 0

    def __call__(self, method):
        def recorded(*args, **kwargs):
            with self._lock:
                self._inside += 1
                self.most = max(self.most, self._inside)
            try:
                time.sleep(0.005)  # long enough for unserialized calls to overlap
                return method(*args, **kwargs)
            finally:
                with self._lock:
                    self._inside -= 1

        return recorded


class RecordedModel:
    """A model whose calls are recorded: `model(...)` looks `__call__` up on the type."""

    def __init__(self, model, overlaps):
        self._model = model
        self._call = overlaps(model.__call__)

    def __call__(self, *args, **kwargs):
        return self._call(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._model, name)


class TestConcurrentCalls:
    """One extractor serves concurrent calls. Its model and tokenizer take one call at a time:
    a fast tokenizer changes its settings per call, and fails while another thread encodes."""

    async def run(self, extractor, configs, calls=20):
        await asyncio.gather(
            *(
                extractor.aextract(QUESTION, [OPENED, TOLLS], config=configs[i % len(configs)])
                for i in range(calls)
            )
        )

    async def test_the_highlighter_with_different_configs(self):
        overlaps, model = Overlaps(), StubHighlighter()
        model.process = overlaps(model.process)
        extractor = ModelSpanExtractor(model, model_id="m", model_version="1")

        await self.run(
            extractor, [ModelExtractConfig(), ModelExtractConfig(max_length=256, doc_stride=64)]
        )

        assert overlaps.most == 1

    @pytest.mark.parametrize("output_mode", ["sentences", "spans"])
    async def test_the_semantic_highlighter_with_one_config(self, output_mode):
        overlaps, model = Overlaps(), StubSemanticHighlighter()
        for name in ("process", "get_raw_predictions", "tokenizer"):
            setattr(model, name, overlaps(getattr(model, name)))
        extractor = SemanticHighlightExtractor(model, model_id="m", model_version="1")

        await self.run(extractor, [SemanticHighlightConfig(output_mode=output_mode)])

        assert overlaps.most == 1

    async def test_the_sentence_model(self):
        overlaps = Overlaps()
        extractor = sentence_model_extractor(pytest.importorskip("torch"))
        extractor.tokenizer.encode_plus = overlaps(extractor.tokenizer.encode_plus)
        extractor.model = RecordedModel(extractor.model, overlaps)

        await self.run(extractor, [ModelExtractConfig(threshold=0.5)])

        assert overlaps.most == 1


# Loading


COMMITS = {"org/highlighter": "c0ffee", "org/semantic": "decaf"}


@pytest.fixture
def model_packages(monkeypatch):
    """Fake torch and transformers packages that load stub models."""
    loaded = []

    def load(name, **settings):
        loaded.append((name, settings))
        stub = StubSemanticHighlighter if "semantic" in name else StubHighlighter
        return stub(config=SimpleNamespace(_commit_hash=COMMITS.get(name)))

    unavailable = SimpleNamespace(is_available=lambda: False)
    torch = SimpleNamespace(cuda=unavailable, backends=SimpleNamespace(mps=unavailable))
    transformers = SimpleNamespace(
        AutoConfig=SimpleNamespace(
            from_pretrained=lambda name, **settings: SimpleNamespace(
                auto_map={"AutoModel": "modeling.Highlighter"}
            )
        ),
        AutoModel=SimpleNamespace(from_pretrained=load),
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    return loaded


class TestFromPretrained:
    @pytest.mark.parametrize(
        "extractor_class, model_id, commit",
        [
            (ModelSpanExtractor, "org/highlighter", "c0ffee"),
            (SemanticHighlightExtractor, "org/semantic", "decaf"),
        ],
    )
    def test_records_the_resolved_commit_as_the_version(
        self, model_packages, extractor_class, model_id, commit
    ):
        extractor = extractor_class.from_pretrained(model_id)

        [usage] = extractor.extract(QUESTION, [OPENED]).usage
        assert (usage.model_id, usage.model_version) == (model_id, commit)

    def test_a_local_path_takes_the_version_from_the_caller(self, model_packages):
        extractor = ModelSpanExtractor.from_pretrained("/models/highlighter", model_version="v2.1")

        assert extractor.extract(QUESTION, [OPENED]).usage[0].model_version == "v2.1"

    def test_the_resolved_commit_wins_over_a_version_from_the_caller(self, model_packages):
        extractor = ModelSpanExtractor.from_pretrained("org/highlighter", model_version="v2.1")

        assert extractor.model_version == "c0ffee"

    def test_loads_the_highlighter_tokenizer_with_the_model(self, model_packages):
        extractor = ModelSpanExtractor.from_pretrained("org/highlighter")

        assert extractor.model.tokenizer_loaded

    def test_loads_onto_the_device_given_or_the_best_available(self, model_packages):
        on_gpu = ModelSpanExtractor.from_pretrained("org/highlighter", device="cuda")
        default = ModelSpanExtractor.from_pretrained("org/highlighter")

        assert (on_gpu.model.devices, default.model.devices) == (["cuda"], ["cpu"])

    def test_passes_the_token_limit_to_the_semantic_model(self, model_packages):
        SemanticHighlightExtractor.from_pretrained("org/semantic", max_tokens=2048)

        assert model_packages[0][1]["max_length"] == 2048

    def test_logs_model_loaded(self, model_packages, caplog):
        caplog.set_level(logging.INFO)

        ModelSpanExtractor.from_pretrained("org/highlighter")

        [record] = [record for record in caplog.records if getattr(record, "event", None)]
        assert (record.levelno, record.event) == (logging.INFO, "model_loaded")
        assert (record.model, record.device) == ("org/highlighter", "cpu")
        assert isinstance(record.duration_ms, int)

    @pytest.mark.parametrize("extractor_class", [ModelSpanExtractor, SemanticHighlightExtractor])
    def test_a_model_that_cannot_be_loaded_is_a_configuration_error(
        self, model_packages, monkeypatch, extractor_class
    ):
        def offline(name, **settings):
            raise OSError("We couldn't connect to 'https://huggingface.co'")

        monkeypatch.setattr(sys.modules["transformers"].AutoModel, "from_pretrained", offline)

        with pytest.raises(ConfigurationError) as raised:
            extractor_class.from_pretrained("org/model")

        assert raised.value.code == "model_not_loaded"
        assert isinstance(raised.value.__cause__, OSError)

    def test_a_package_the_model_code_needs_is_named(self, model_packages, monkeypatch):
        def needs_nltk(name, **settings):
            raise ModuleNotFoundError("No module named 'nltk'", name="nltk")

        monkeypatch.setattr(sys.modules["transformers"].AutoModel, "from_pretrained", needs_nltk)

        with pytest.raises(ConfigurationError) as raised:
            SemanticHighlightExtractor.from_pretrained("org/semantic")

        assert raised.value.code == "missing_dependency"
        assert "pip install nltk" in str(raised.value)

    @pytest.mark.parametrize("extractor_class", [ModelSpanExtractor, SemanticHighlightExtractor])
    def test_a_missing_package_names_the_extra_to_install(self, monkeypatch, extractor_class):
        monkeypatch.setitem(sys.modules, "torch", None)

        with pytest.raises(ConfigurationError) as raised:
            extractor_class.from_pretrained("org/model")

        assert raised.value.code == "missing_dependency"
        assert "verbatim-core[model]" in str(raised.value)

    def test_importing_the_extractors_loads_no_model_package(self):
        imported = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys, verbatim_core.extractors; "
                "print(sorted({'torch', 'transformers'} & set(sys.modules)))",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        assert imported.stdout.strip() == "[]"


class TestDeprecatedConstructors:
    def test_model_path_loads_with_from_pretrained_and_warns(self, model_packages):
        with pytest.warns(VerbatimDeprecationWarning, match="constructor") as warned:
            extractor = ModelSpanExtractor(model_path="org/highlighter", device="cuda")

        assert (extractor.model_id, extractor.model_version) == ("org/highlighter", "c0ffee")
        assert extractor.model.devices == ["cuda"]
        assert warned[0].filename == __file__

    def test_they_load_through_from_pretrained(self, model_packages):
        class Recording(ModelSpanExtractor):
            loaded = []

            @classmethod
            def from_pretrained(cls, model_id, device=None, **settings):
                cls.loaded.append(model_id)
                return super().from_pretrained(model_id, device, **settings)

        with pytest.warns(VerbatimDeprecationWarning):
            extractor = Recording(model_path="org/highlighter")

        assert Recording.loaded == ["org/highlighter"]
        assert isinstance(extractor.model, StubHighlighter)

    def test_no_arguments_load_the_default_model(self, model_packages):
        with pytest.warns(VerbatimDeprecationWarning):
            extractor = ModelSpanExtractor()

        assert extractor.model_id == ModelSpanExtractor.DEFAULT_MODEL

    def test_a_model_name_by_position_loads_it(self, model_packages):
        with pytest.warns(VerbatimDeprecationWarning):
            extractor = SemanticHighlightExtractor("org/semantic", output_mode="spans")

        assert extractor.model_id == "org/semantic"
        assert extractor.config == SemanticHighlightConfig(output_mode="spans")

    def test_old_settings_go_into_the_config_and_warn(self, model_packages):
        with pytest.warns(
            VerbatimDeprecationWarning, match="Passing threshold, doc_stride"
        ) as warned:
            extractor = highlighter(threshold=0.5, doc_stride=128)

        assert extractor.config == ModelExtractConfig(threshold=0.5, doc_stride=128)
        assert warned[0].filename == __file__

    @pytest.mark.parametrize("argument", ["extraction_mode", "max_display_spans"])
    def test_unused_arguments_warn(self, argument):
        with pytest.warns(VerbatimDeprecationWarning, match=f"Passing {argument}"):
            highlighter(**{argument: "individual" if argument == "extraction_mode" else 5})

    def test_an_invalid_old_setting_is_a_configuration_error(self):
        with pytest.warns(VerbatimDeprecationWarning), pytest.raises(ConfigurationError):
            semantic_highlighter(output_mode="paragraphs")

    def test_a_loaded_model_with_a_device_is_rejected(self):
        with pytest.raises(ConfigurationError):
            ModelSpanExtractor(StubHighlighter(), "cuda")
        with pytest.raises(ConfigurationError):
            SemanticHighlightExtractor(StubSemanticHighlighter(), max_tokens=2048)
