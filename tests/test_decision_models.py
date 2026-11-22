"""DecisionModelExtractor: span questions to a decision model, in memory or on a server (ADR 0009)."""

import json

import httpx
import pytest
from extractor_contract import ExtractorContract
from verbatim_core import Chunk, ConfigurationError, InvalidInputError, UpstreamError
from verbatim_core.decision_models import (
    DecisionModelConfig,
    DecisionModelExtractor,
    LocalDecisionModel,
    RemoteDecisionModel,
)


class WordModel:
    """A stand-in for Vela's system_one: the sentences that share a word of four letters with the request."""

    def __init__(self):
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append((state, questions))
        words = {word.strip("?.,").lower() for word in state["request"].split() if len(word) > 3}
        context = state[questions["evidence"]["over"]]
        spans, start = [], 0
        for sentence in context.split(". "):
            end = start + len(sentence)
            if words & {word.strip(".,").lower() for word in sentence.split()}:
                spans.append(
                    {
                        "label": "answer",
                        "start": start,
                        "end": end,
                        "text": context[start:end],
                        "probability": 0.9,
                    }
                )
            start = end + 2
        return {"model": "stub/vela", "spans": {"evidence": spans}, "answers": {}}


def local_extractor(model=None, **settings):
    return DecisionModelExtractor(
        LocalDecisionModel(model or WordModel(), model_id="stub/vela", model_version="c0ffee"),
        **settings,
    )


def server(model, seen=None):
    """An httpx transport that serves the model's system_one over POST /v1/systemone."""

    def handle(request):
        body = json.loads(request.content)
        if seen is not None:
            seen.append((request, body))
        return httpx.Response(200, json=model.system_one(body["state"], body["questions"]))

    return httpx.MockTransport(handle)


def remote_extractor(transport, **settings):
    remote = RemoteDecisionModel(
        "http://vela.test",
        "stub/vela",
        model_version="c0ffee",
        api_key="test-key",  # pragma: allowlist secret
        http_client=httpx.Client(transport=transport),
        async_http_client=httpx.AsyncClient(transport=transport),
    )
    return DecisionModelExtractor(remote, **settings)


class TestLocalContract(ExtractorContract):
    @pytest.fixture
    def extractor(self):
        return local_extractor()


class TestRemoteContract(ExtractorContract):
    @pytest.fixture
    def extractor(self):
        return remote_extractor(server(WordModel()))


CHUNKS = [
    Chunk(id="a", text="The arch rises 134 metres. Tolls are electronic."),
    Chunk(id="b", text="   "),
    Chunk(id="c", text="The arch is made of steel."),
]


def test_each_chunk_with_text_gets_one_span_question_with_the_question_as_request():
    model = WordModel()
    local_extractor(model).extract("How tall is the arch?", CHUNKS)

    assert [state for state, _ in model.calls] == [
        {"request": "How tall is the arch?", "context": CHUNKS[0].text},
        {"request": "How tall is the arch?", "context": CHUNKS[2].text},
    ]
    assert model.calls[0][1] == {
        "evidence": {
            "type": "span",
            "instructions": "Which spans of the context answer the request?",
            "over": "context",
            "criteria": {"answer": "text that answers the request"},
        }
    }


def test_spans_keep_the_chunk_offsets_and_the_probability_as_score():
    spans = local_extractor().extract("How tall is the arch?", CHUNKS).spans

    assert [(s.chunk_id, s.start, s.end, s.text, s.score) for s in spans] == [
        ("a", 0, 25, "The arch rises 134 metres", 0.9),
        ("c", 0, 26, "The arch is made of steel.", 0.9),
    ]


def test_the_config_sets_the_question_and_the_threshold():
    model = WordModel()
    config = DecisionModelConfig(
        instructions="Which spans name a height?",
        label="height",
        description="a height",
        threshold=0.3,
    )
    local_extractor(model, config=config).extract("arch", CHUNKS[:1])

    assert model.calls[0][1]["evidence"] == {
        "type": "span",
        "instructions": "Which spans name a height?",
        "over": "context",
        "criteria": {"height": "a height"},
        "threshold": 0.3,
    }


class WrongOffsets:
    def system_one(self, state, questions):
        context = state["context"]
        return {
            "spans": {
                "evidence": [
                    {"start": 0, "end": 8, "text": "The arch", "probability": 0.8},
                    {"start": 0, "end": 8, "text": "Another!", "probability": 0.8},
                    {"start": 4, "end": len(context) + 5, "text": None, "probability": 0.8},
                    {"start": 3, "end": 4, "probability": 0.8},
                ]
            }
        }


def test_a_span_whose_offsets_dont_hold_its_text_is_dropped_with_a_warning():
    extraction = local_extractor(WrongOffsets()).extract("q", CHUNKS[:1])

    assert [span.text for span in extraction.spans] == ["The arch"]
    assert [warning.code for warning in extraction.warnings] == ["span_offsets_mismatch"]


async def test_the_remote_model_posts_the_model_state_and_questions_with_the_key():
    seen = []
    extractor = remote_extractor(server(WordModel(), seen))

    await extractor.aextract("How tall is the arch?", CHUNKS[:1])

    request, body = seen[0]
    assert request.url == "http://vela.test/v1/systemone"
    assert request.headers["authorization"] == "Bearer test-key"
    assert body["model"] == "stub/vela"
    assert body["state"] == {"request": "How tall is the arch?", "context": CHUNKS[0].text}


@pytest.mark.parametrize(
    ("status", "error", "code"),
    [
        (401, ConfigurationError, "authentication_failed"),
        (413, InvalidInputError, "input_too_long"),
        (422, UpstreamError, "request_rejected"),
        (429, UpstreamError, "server_error"),
        (503, UpstreamError, "server_error"),
    ],
)
async def test_server_errors_become_the_library_errors(status, error, code):
    extractor = remote_extractor(
        httpx.MockTransport(lambda request: httpx.Response(status, text="secret detail"))
    )

    with pytest.raises(error) as raised:
        extractor.extract("q", CHUNKS[:1])
    with pytest.raises(error):
        await extractor.aextract("q", CHUNKS[:1])

    assert raised.value.code == code
    assert "secret detail" not in str(raised.value)


def test_an_unreadable_reply_is_an_upstream_error():
    extractor = remote_extractor(
        httpx.MockTransport(lambda request: httpx.Response(200, text="<html>"))
    )

    with pytest.raises(UpstreamError) as raised:
        extractor.extract("q", CHUNKS[:1])
    assert raised.value.code == "invalid_reply"


def test_a_timeout_is_a_retryable_upstream_error():
    def time_out(request):
        raise httpx.ReadTimeout("slow", request=request)

    with pytest.raises(UpstreamError) as raised:
        remote_extractor(httpx.MockTransport(time_out)).extract("q", CHUNKS[:1])
    assert (raised.value.code, raised.value.retryable) == ("timeout", True)


def test_a_model_without_system_one_is_rejected():
    with pytest.raises(ConfigurationError):
        LocalDecisionModel(object())
    with pytest.raises(ConfigurationError):
        DecisionModelExtractor(object())


def test_any_object_with_system_one_is_a_backend():
    class MyServer:
        model_id = "mine"
        model_version = "1"

        def system_one(self, state, questions):
            return WordModel().system_one(state, questions)

    spans = DecisionModelExtractor(MyServer()).extract("How tall is the arch?", CHUNKS[:1]).spans
    assert [span.text for span in spans] == ["The arch rises 134 metres"]
