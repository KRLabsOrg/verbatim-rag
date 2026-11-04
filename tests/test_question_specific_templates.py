"""Question-specific templates, with a word-count embedder and a fake sentence-transformers."""

import json
import logging
import sys
import types
import warnings

import pytest
from verbatim_core import ConfigurationError, VerbatimDeprecationWarning
from verbatim_core.templates import TemplateManager
from verbatim_core.templates.question_specific import QuestionSpecificTemplate

TEMPLATES = [
    {"template": "Opening: [RELEVANT_SENTENCES]", "examples": ["When did it open?"]},
    {"template": "Design: [RELEVANT_SENTENCES]", "examples": ["Who designed it?", "Who built it?"]},
]


class WordEmbedder:
    """Embeds a text as the counts of a few words."""

    WORDS = ["when", "open", "who", "design", "built"]

    def __init__(self):
        self.encoded = []

    def encode(self, texts):
        self.encoded.append(list(texts))
        return [[float(text.lower().count(word)) for word in self.WORDS] for text in texts]


@pytest.fixture
def no_deprecation_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error", VerbatimDeprecationWarning)
        yield


@pytest.fixture
def sentence_transformers(monkeypatch):
    """A fake sentence-transformers. Its models record how they were loaded."""
    loaded = []

    class SentenceTransformer(WordEmbedder):
        def __init__(self, model_name, device=None):
            super().__init__()
            self.device = device or "cpu"
            loaded.append((model_name, device))

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = SentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    return loaded


class TestChoosingATemplate:
    def test_the_question_chooses_the_template_of_the_closest_example(
        self, no_deprecation_warnings
    ):
        strategy = QuestionSpecificTemplate(TEMPLATES, embedder=WordEmbedder())

        assert strategy.generate("When did the bridge open?", [], 0) == TEMPLATES[0]["template"]
        assert strategy.generate("Who designed the bridge?", [], 0) == TEMPLATES[1]["template"]

    def test_the_examples_are_embedded_at_construction(self):
        embedder = WordEmbedder()
        strategy = QuestionSpecificTemplate(TEMPLATES, embedder=embedder)
        examples = ["When did it open?", "Who designed it?", "Who built it?"]

        assert embedder.encoded == [examples]

        strategy.generate("Who built the bridge?", [], 0)

        assert embedder.encoded == [examples, ["Who built the bridge?"]]

    def test_a_longer_example_is_not_closer_for_being_longer(self):
        templates = [
            {
                "template": "Dates: [RELEVANT_SENTENCES]",
                "examples": ["When, when, when did it open?"],
            },
            {"template": "Status: [RELEVANT_SENTENCES]", "examples": ["Is it open?"]},
        ]
        strategy = QuestionSpecificTemplate(templates, embedder=WordEmbedder())

        assert strategy.generate("Is it open today?", [], 0) == "Status: [RELEVANT_SENTENCES]"

    def test_a_question_that_matches_no_example_gets_the_first_template(self):
        strategy = QuestionSpecificTemplate(TEMPLATES, embedder=WordEmbedder())

        assert strategy.generate("How tall is it?", [], 0) == TEMPLATES[0]["template"]

    def test_without_templates_the_default_template_is_used(self, no_deprecation_warnings):
        assert QuestionSpecificTemplate().generate("Q?", [], 0) == (
            QuestionSpecificTemplate.DEFAULT_TEMPLATE
        )

    def test_the_template_manager_takes_the_embedder(self, no_deprecation_warnings):
        templates = TemplateManager()

        assert templates.use_question_specific_mode(TEMPLATES, embedder=WordEmbedder()) is True
        assert templates.process("Who built the bridge?", [{"text": "Bradfield."}], []) == (
            "Design: [1] Bradfield."
        )


class TestFromPretrained:
    def test_loads_the_model_on_the_device_given(self, sentence_transformers):
        strategy = QuestionSpecificTemplate.from_pretrained(
            "org/embedder", "cuda", templates=TEMPLATES
        )

        assert sentence_transformers == [("org/embedder", "cuda")]
        assert strategy.generate("Who built it?", [], 0) == TEMPLATES[1]["template"]

    def test_without_a_device_sentence_transformers_chooses(self, sentence_transformers):
        QuestionSpecificTemplate.from_pretrained("org/embedder")

        assert sentence_transformers == [("org/embedder", None)]

    def test_logs_model_loaded(self, sentence_transformers, caplog):
        caplog.set_level(logging.INFO)

        QuestionSpecificTemplate.from_pretrained("org/embedder")

        [record] = [record for record in caplog.records if getattr(record, "event", None)]
        assert (record.levelno, record.event) == (logging.INFO, "model_loaded")
        assert (record.model, record.device) == ("org/embedder", "cpu")
        assert isinstance(record.duration_ms, int)

    def test_a_missing_package_is_a_configuration_error(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)

        with pytest.raises(ConfigurationError) as raised:
            QuestionSpecificTemplate.from_pretrained("org/embedder")

        assert raised.value.code == "missing_dependency"
        assert "pip install sentence-transformers" in str(raised.value)

    def test_a_model_that_cannot_be_loaded_is_a_configuration_error(
        self, sentence_transformers, monkeypatch
    ):
        def offline(model_name, device=None):
            raise OSError("We couldn't connect to huggingface.co")

        monkeypatch.setattr(sys.modules["sentence_transformers"], "SentenceTransformer", offline)

        with pytest.raises(ConfigurationError) as raised:
            QuestionSpecificTemplate.from_pretrained("org/embedder")

        assert raised.value.code == "model_not_loaded"
        assert isinstance(raised.value.__cause__, OSError)


class TestLoadingByName:
    """Before 0.3.0 the templates loaded their embedder by name. It still works, and warns."""

    def test_a_model_name_warns_and_loads_that_model(self, sentence_transformers):
        with pytest.warns(VerbatimDeprecationWarning, match="from_pretrained") as warned:
            strategy = QuestionSpecificTemplate(TEMPLATES, model_name="org/embedder")

        assert len(warned) == 1
        assert warned[0].filename == __file__
        assert sentence_transformers == [("org/embedder", "cpu")]
        assert strategy.generate("Who built it?", [], 0) == TEMPLATES[1]["template"]

    def test_templates_without_an_embedder_warn_and_load_the_default_model(
        self, sentence_transformers
    ):
        with pytest.warns(VerbatimDeprecationWarning, match="without an embedder") as warned:
            QuestionSpecificTemplate(TEMPLATES)

        assert warned[0].filename == __file__
        assert sentence_transformers == [("all-MiniLM-L6-v2", "cpu")]

    def test_the_template_manager_without_an_embedder_warns_at_its_caller(
        self, sentence_transformers
    ):
        with pytest.warns(VerbatimDeprecationWarning, match="without an embedder") as warned:
            TemplateManager().use_question_specific_mode(TEMPLATES)

        assert warned[0].filename == __file__


class TestSavedTemplates:
    def test_loading_a_saved_setup_reuses_the_embedder(self, tmp_path, no_deprecation_warnings):
        saved = TemplateManager()
        saved.use_question_specific_mode(TEMPLATES, embedder=WordEmbedder())
        saved.save(str(tmp_path / "templates.json"))

        loaded = TemplateManager()
        loaded.use_question_specific_mode(embedder=WordEmbedder())

        assert loaded.load(str(tmp_path / "templates.json")) is True
        assert loaded.current_mode == "question_specific"
        assert loaded.process("When did it open?", [{"text": "In 1932."}], []) == (
            "Opening: [1] In 1932."
        )

    def test_loading_without_an_embedder_loads_the_saved_model(
        self, tmp_path, sentence_transformers
    ):
        saved = QuestionSpecificTemplate.from_pretrained("org/embedder", templates=TEMPLATES)
        manager = TemplateManager()
        manager.strategies["question_specific"] = saved
        manager.save(str(tmp_path / "templates.json"))

        with pytest.warns(VerbatimDeprecationWarning, match="without an embedder") as warned:
            assert TemplateManager().load(str(tmp_path / "templates.json")) is True

        assert warned[0].filename == __file__
        assert sentence_transformers == [("org/embedder", None), ("org/embedder", "cpu")]

    def test_an_embedder_given_replaces_the_saved_model(self, tmp_path, sentence_transformers):
        saved = TemplateManager()
        saved.strategies["question_specific"] = QuestionSpecificTemplate.from_pretrained(
            "org/embedder", templates=TEMPLATES
        )
        saved.save(str(tmp_path / "templates.json"))
        loaded = TemplateManager()
        loaded.use_question_specific_mode(embedder=WordEmbedder())

        loaded.load(str(tmp_path / "templates.json"))
        loaded.save(str(tmp_path / "again.json"))

        resaved = json.loads((tmp_path / "again.json").read_text())
        assert resaved["strategies"]["question_specific"]["model_name"] is None
        assert sentence_transformers == [("org/embedder", None)]

    def test_a_setup_saved_with_an_embedder_of_its_own_needs_one_to_load(self, tmp_path):
        saved = TemplateManager()
        saved.use_question_specific_mode(TEMPLATES, embedder=WordEmbedder())
        saved.save(str(tmp_path / "templates.json"))

        with pytest.raises(ConfigurationError) as raised:
            TemplateManager().load(str(tmp_path / "templates.json"))

        assert raised.value.code == "missing_embedder"

    def test_a_saved_setup_that_needs_a_missing_package_is_a_configuration_error(
        self, tmp_path, sentence_transformers, monkeypatch
    ):
        manager = TemplateManager()
        manager.strategies["question_specific"] = QuestionSpecificTemplate.from_pretrained(
            "org/embedder", templates=TEMPLATES
        )
        manager.save(str(tmp_path / "templates.json"))
        monkeypatch.setitem(sys.modules, "sentence_transformers", None)

        with pytest.warns(VerbatimDeprecationWarning):
            with pytest.raises(ConfigurationError) as raised:
                TemplateManager().load(str(tmp_path / "templates.json"))

        assert raised.value.code == "missing_dependency"
