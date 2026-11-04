"""
The Transformer: a question and chunks in, a cited answer built from the chunks' own text out.
"""

from __future__ import annotations

import logging
import re
from typing import AsyncIterator, Dict, Iterator, List, Literal, Sequence, Set, Tuple, Union

from pydantic import BaseModel, ConfigDict

from ._checks import _call_config, _default_config
from ._deprecation import warn_contextual_without_client
from ._transform_config import TransformConfig
from .errors import ConfigurationError, InvalidInputError
from .extractors import SpanExtractor
from .response_builder import ResponseBuilder
from .templates import TemplateManager
from .templates.base import TemplateResult, TemplateStrategy
from .templates.filler import TemplateFiller
from .types import Answer, AnswerCitation, Chunk, ExtractedSpan, Extraction

logger = logging.getLogger(__name__)

_STRUCTURED_MODE_UNSUPPORTED = (
    "The Transformer doesn't support structured template mode. Use VerbatimRAG for structured "
    "answers, or another template mode."
)


class SpansEvent(BaseModel):
    """The first event of a streamed transform: the spans, once they are extracted."""

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    type: Literal["spans"] = "spans"
    spans: List[ExtractedSpan]


class AnswerEvent(BaseModel):
    """The last event of a streamed transform: the answer `transform` returns for the same input."""

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    type: Literal["answer"] = "answer"
    answer: Answer


TransformEvent = Union[SpansEvent, AnswerEvent]


class Transformer:
    """Turn a question and chunks into a cited answer, built from spans of the chunks.

    The first spans fill the template of the answer, and every span gets one citation that points
    at it by chunk id and offsets. One transformer can serve concurrent calls: the settings of a
    call come in its config, and nothing is set on the transformer between calls.
    """

    def __init__(
        self,
        extractor: SpanExtractor,
        templates: TemplateManager,
        *,
        config: TransformConfig | None = None,
    ):
        """
        :param extractor: Extracts the spans of the chunks
        :param templates: The templates the answers are written with
        :param config: The settings of a call, unless the call passes its own. Without it, the
            templates' settings are the default.
        :raises ConfigurationError: If the config is not a TransformConfig, or asks for structured
            template mode
        """
        self.extractor = extractor
        self.templates = templates
        self._config = None if config is None else _default_config(config, TransformConfig)
        # Templates in structured mode raise at the call: VerbatimRAG answers structured
        # questions itself with the same templates (ADR 0004).
        if self._config is not None and self._config.template_mode == "structured":
            raise ConfigurationError(_STRUCTURED_MODE_UNSUPPORTED, code="invalid_value")
        if templates.strategy(self.config.template_mode) is None:
            warn_contextual_without_client()

    @property
    def config(self) -> TransformConfig:
        """The settings of a call that passes none: the config given, else the templates'."""
        return self._config if self._config is not None else self.templates.config

    def transform(
        self, question: str, chunks: Sequence[Chunk], *, config: TransformConfig | None = None
    ) -> Answer:
        """
        Write a cited answer to the question from the chunks.

        :param question: The question
        :param chunks: Chunks with unique ids
        :param config: Replaces the transformer's default config for this call
        :return: The answer, with its citations, spans, usage and warnings
        :raises InvalidInputError: If two chunks share an id, or the config is of the wrong type
            or asks for structured template mode
        :raises UpstreamError: If a service the extractor calls fails
        :raises ConfigurationError: If a service rejects its key, model or quota
        """
        config, strategy = self._call_settings(config)
        *_, answer_event = self._events(question, chunks, config, strategy)
        return answer_event.answer

    async def atransform(
        self, question: str, chunks: Sequence[Chunk], *, config: TransformConfig | None = None
    ) -> Answer:
        """Async version of `transform`."""
        config, strategy = self._call_settings(config)
        events = [event async for event in self._aevents(question, chunks, config, strategy)]
        return events[-1].answer

    def stream(
        self, question: str, chunks: Sequence[Chunk], *, config: TransformConfig | None = None
    ) -> Iterator[TransformEvent]:
        """
        Write the answer of `transform` as events: the spans, then the answer.

        :raises InvalidInputError: When called, if the config is of the wrong type or asks for
            structured template mode. The other errors of `transform` come from the iteration.
        """
        config, strategy = self._call_settings(config)
        return self._events(question, chunks, config, strategy)

    def astream(
        self, question: str, chunks: Sequence[Chunk], *, config: TransformConfig | None = None
    ) -> AsyncIterator[TransformEvent]:
        """Async version of `stream`."""
        config, strategy = self._call_settings(config)
        return self._aevents(question, chunks, config, strategy)

    def _call_settings(
        self, config: TransformConfig | None
    ) -> Tuple[TransformConfig, TemplateStrategy]:
        """The config and the template strategy of one call."""
        call_config = _call_config(config, self.config, TransformConfig)
        if call_config.template_mode == "structured":
            if config is None:
                # The templates' settings, switched to structured mode after construction
                raise ConfigurationError(_STRUCTURED_MODE_UNSUPPORTED, code="invalid_value")
            raise InvalidInputError(
                _STRUCTURED_MODE_UNSUPPORTED, code="invalid_value", param="config.template_mode"
            )

        strategy = self.templates.strategy(call_config.template_mode)
        if strategy is None:
            # Only contextual mode needs a client. Until the deprecation window ends, it falls
            # back to static templates.
            warn_contextual_without_client()
            strategy = self.templates.strategy("static")
        return call_config, strategy

    def _events(
        self,
        question: str,
        chunks: Sequence[Chunk],
        config: TransformConfig,
        strategy: TemplateStrategy,
    ) -> Iterator[TransformEvent]:
        logger.debug("Transforming %d chunks in %s mode", len(chunks), config.template_mode)
        extraction = self.extractor.extract(question, chunks)
        yield SpansEvent(spans=extraction.spans)
        display, references = _split(extraction.spans, config)
        template = strategy.generate_result(
            question, _texts(display + references), len(references), seed=config.seed
        )
        yield AnswerEvent(answer=_answer(extraction, template, display, references, config))

    async def _aevents(
        self,
        question: str,
        chunks: Sequence[Chunk],
        config: TransformConfig,
        strategy: TemplateStrategy,
    ) -> AsyncIterator[TransformEvent]:
        logger.debug("Transforming %d chunks in %s mode", len(chunks), config.template_mode)
        extraction = await self.extractor.aextract(question, chunks)
        yield SpansEvent(spans=extraction.spans)
        display, references = _split(extraction.spans, config)
        template = await strategy.generate_result_async(
            question, _texts(display + references), len(references), seed=config.seed
        )
        yield AnswerEvent(answer=_answer(extraction, template, display, references, config))


def _split(
    spans: List[ExtractedSpan], config: TransformConfig
) -> Tuple[List[ExtractedSpan], List[ExtractedSpan]]:
    """The spans the answer shows, and the spans it cites for reference only."""
    return spans[: config.max_display_spans], spans[config.max_display_spans :]


def _texts(spans: List[ExtractedSpan]) -> List[str]:
    return [span.text for span in spans]


def _span_data(spans: List[ExtractedSpan]) -> List[Dict[str, str]]:
    return [{"text": span.text} for span in spans]


def _answer(
    extraction: Extraction,
    template: TemplateResult,
    display: List[ExtractedSpan],
    references: List[ExtractedSpan],
    config: TransformConfig,
) -> Answer:
    filler = TemplateFiller(
        citation_mode=config.citation_mode, citation_format=config.citation_format
    )
    text = filler.fill(template.text, _span_data(display), _span_data(references))
    shown = _shown_numbers(template.text, len(display))
    citations = [
        AnswerCitation(
            number=number,
            type="display" if number in shown else "reference",
            chunk_id=span.chunk_id,
            start=span.start,
            end=span.end,
        )
        for number, span in enumerate(display + references, start=1)
    ]
    return Answer(
        text=ResponseBuilder().clean_answer(text),
        citations=citations,
        spans=extraction.spans,
        usage=[*extraction.usage, *template.usage],
        warnings=[*extraction.warnings, *template.warnings],
    )


_PER_SPAN_PLACEHOLDER = re.compile(r"\[(?:SPAN|FACT)_(\d+)\]")


def _shown_numbers(template: str, display_count: int) -> Set[int]:
    """The numbers of the spans whose text a filled template shows.

    A template with per-span placeholders shows the displayed spans it names; otherwise one with
    an aggregate placeholder shows all of them, as `TemplateFiller` fills it.
    """
    named = {int(number) for number in _PER_SPAN_PLACEHOLDER.findall(template)}
    if named:
        return {number for number in named if number <= display_count}
    if "[DISPLAY_SPANS]" in template or "[RELEVANT_SENTENCES]" in template:
        return set(range(1, display_count + 1))
    return set()
