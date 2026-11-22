"""The Transformer: a question and chunks in, a cited answer built from the chunks' own text out."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Literal, cast

from pydantic import BaseModel, ConfigDict

from verbatim_core._checks import _call_config, _default_config
from verbatim_core._deprecation import warn_contextual_without_client
from verbatim_core._transform_config import TransformConfig
from verbatim_core.errors import ConfigurationError, InvalidInputError
from verbatim_core.plan import PlanConfig, distinct_spans, fallback_plan, plan_candidates, renumber_plan
from verbatim_core.render import Rendered, RenderOptions, layout_plan, render
from verbatim_core.response_builder import ResponseBuilder
from verbatim_core.templates.contextual import ContextualTemplate
from verbatim_core.templates.filler import TemplateFiller
from verbatim_core.types import Answer, AnswerCitation, Chunk, ExtractedSpan, Extraction

if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterator, Sequence

    from verbatim_core.extractors import SpanExtractor
    from verbatim_core.templates import TemplateManager
    from verbatim_core.templates.base import TemplateResult, TemplateStrategy

logger = logging.getLogger(__name__)

_STRUCTURED_MODE_UNSUPPORTED = (
    "The Transformer doesn't support structured template mode. Use VerbatimRAG for structured "
    "answers, or another template mode."
)


class SpansEvent(BaseModel):
    """The first event of a streamed transform: the spans, once they are extracted."""

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    type: Literal["spans"] = "spans"
    spans: list[ExtractedSpan]


class AnswerEvent(BaseModel):
    """The last event of a streamed transform: the answer `transform` returns for the same input."""

    model_config = ConfigDict(frozen=True, hide_input_in_errors=True)

    type: Literal["answer"] = "answer"
    answer: Answer


TransformEvent = SpansEvent | AnswerEvent


class Transformer:
    """Turn a question and chunks into a cited answer, built from spans of the chunks.

    The first spans fill the template of the answer, and every span gets one citation that points
    at it by chunk id and offsets. One transformer can serve concurrent calls: the settings of a
    call come in its config, and nothing is set on the transformer between calls.
    """

    def __init__(
        self,
        extractor: SpanExtractor[Any],
        templates: TemplateManager,
        *,
        config: TransformConfig | None = None,
    ) -> None:
        """Build a transformer from an extractor and the templates of the answers.

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

    def transform(self, question: str, chunks: Sequence[Chunk], *, config: TransformConfig | None = None) -> Answer:
        """Write a cited answer to the question from the chunks.

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
        # The last event is always the answer.
        return cast("AnswerEvent", answer_event).answer

    async def atransform(
        self, question: str, chunks: Sequence[Chunk], *, config: TransformConfig | None = None
    ) -> Answer:
        """Async version of `transform`."""
        config, strategy = self._call_settings(config)
        events = [event async for event in self._aevents(question, chunks, config, strategy)]
        return cast("AnswerEvent", events[-1]).answer

    def stream(
        self, question: str, chunks: Sequence[Chunk], *, config: TransformConfig | None = None
    ) -> Iterator[TransformEvent]:
        """Write the answer of `transform` as events: the spans, then the answer.

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

    def _call_settings(self, config: TransformConfig | None) -> tuple[TransformConfig, TemplateStrategy]:
        """Return the config and the template strategy of one call."""
        call_config = _call_config(config, self.config, TransformConfig)
        if call_config.template_mode == "structured":
            if config is None:
                # The templates' settings, switched to structured mode after construction
                raise ConfigurationError(_STRUCTURED_MODE_UNSUPPORTED, code="invalid_value")
            raise InvalidInputError(_STRUCTURED_MODE_UNSUPPORTED, code="invalid_value", param="config.template_mode")

        strategy = self.templates.strategy(call_config.template_mode)
        if strategy is None:
            # Only contextual mode needs a client. Until the deprecation window ends, it falls
            # back to static templates.
            warn_contextual_without_client()
            # Static mode needs no client, so it always has a strategy.
            strategy = cast("TemplateStrategy", self.templates.strategy("static"))
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
        if (planner := _planner(config, strategy)) is not None:
            plan_config = cast("PlanConfig", config.answer_plan)
            candidates = _candidates(extraction.spans, plan_config)
            template = planner.generate_plan_result(question, _texts(candidates), plan_config)
            yield AnswerEvent(answer=_plan_answer(extraction, template, candidates, planner, config))
            return
        display, references = _split(_answer_spans(extraction.spans, config), config)
        template = strategy.generate_result(question, _texts(display + references), len(references), seed=config.seed)
        yield AnswerEvent(
            answer=_template_answer(extraction, template, display, references, strategy=strategy, config=config)
        )

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
        if (planner := _planner(config, strategy)) is not None:
            plan_config = cast("PlanConfig", config.answer_plan)
            candidates = _candidates(extraction.spans, plan_config)
            template = await planner.generate_plan_result_async(question, _texts(candidates), plan_config)
            yield AnswerEvent(answer=_plan_answer(extraction, template, candidates, planner, config))
            return
        display, references = _split(_answer_spans(extraction.spans, config), config)
        template = await strategy.generate_result_async(
            question, _texts(display + references), len(references), seed=config.seed
        )
        yield AnswerEvent(
            answer=_template_answer(extraction, template, display, references, strategy=strategy, config=config)
        )


def _planner(config: TransformConfig, strategy: TemplateStrategy) -> ContextualTemplate | None:
    """Return the strategy when the call writes an answer plan: contextual mode with answer_plan."""
    if config.answer_plan is not None and isinstance(strategy, ContextualTemplate):
        return strategy
    return None


def _candidates(spans: list[ExtractedSpan], plan_config: PlanConfig) -> list[ExtractedSpan]:
    """Return the spans an answer plan chooses from, without repeats, chunks taking turns."""
    order = plan_candidates(
        [span.text for span in spans], [span.chunk_id for span in spans], plan_config.max_candidates
    )
    return [spans[i] for i in order]


def _plan_answer(
    extraction: Extraction,
    template: TemplateResult,
    candidates: list[ExtractedSpan],
    strategy: ContextualTemplate,
    config: TransformConfig,
) -> Answer:
    """Render the plan of a call, numbering citations in the order they appear.

    The spans that aren't candidates are cited for reference only, after the others. The call's
    citation settings apply when it sets them, the strategy's otherwise, as for placeholder
    templates. The answer's plan numbers spans by their place in `Answer.spans`.
    """
    plan_config = cast("PlanConfig", config.answer_plan)
    plan = template.plan or fallback_plan(len(candidates), plan_config.max_items, strategy.plan_text)
    chosen = {id(span) for span in candidates}
    spans = candidates + [span for span in extraction.spans if id(span) not in chosen]
    settings = config if {"citation_mode", "citation_format"} & config.model_fields_set else strategy
    options = RenderOptions(
        citation_mode="hidden" if settings.citation_mode == "hidden" else "inline",
        citation_format=settings.citation_format,
        compact_quotes=plan_config.compact_quotes,
        references_label=strategy.plan_text.references_label if plan_config.show_references else None,
    )
    texts = _texts(spans)
    rendered = render(layout_plan(plan, texts, shown_limit=len(candidates)), texts, options)
    place = {id(span): number for number, span in enumerate(extraction.spans, start=1)}
    return Answer(
        text=rendered.text,
        citations=_citations(rendered, spans),
        spans=extraction.spans,
        usage=[*extraction.usage, *template.usage],
        warnings=[*extraction.warnings, *template.warnings],
        plan=renumber_plan(plan, [place[id(span)] for span in spans]),
    )


def _template_answer(  # noqa: PLR0913  # the parts of one call
    extraction: Extraction,
    template: TemplateResult,
    display: list[ExtractedSpan],
    references: list[ExtractedSpan],
    *,
    strategy: TemplateStrategy,
    config: TransformConfig,
) -> Answer:
    """Fill the placeholder template of a call.

    The strategy's filler fills it, with the strategy's citation settings, unless the call's config
    sets citation settings: then a filler of the call's own does, and no strategy changes. A
    strategy whose class fills its own way writes the text; the citations come from the same
    template either way.
    """
    shown, cited = _span_data(display), _span_data(references)
    rendered = _filler(strategy, config).render(template.text, shown, cited)
    text = strategy.fill(template.text, shown, cited) if _fills_its_own_way(strategy) else rendered.text
    return Answer(
        text=ResponseBuilder().clean_answer(text),
        citations=_citations(rendered, display + references),
        spans=extraction.spans,
        usage=[*extraction.usage, *template.usage],
        warnings=[*extraction.warnings, *template.warnings],
    )


def _citations(rendered: Rendered, spans: list[ExtractedSpan]) -> list[AnswerCitation]:
    return [
        AnswerCitation(
            number=citation.number,
            type=citation.type,
            chunk_id=spans[citation.span].chunk_id,
            start=spans[citation.span].start,
            end=spans[citation.span].end,
        )
        for citation in rendered.citations
    ]


def _split(spans: list[ExtractedSpan], config: TransformConfig) -> tuple[list[ExtractedSpan], list[ExtractedSpan]]:
    """Split the spans into the ones the answer shows and the ones it cites for reference only."""
    return spans[: config.max_display_spans], spans[config.max_display_spans :]


def _answer_spans(spans: list[ExtractedSpan], config: TransformConfig) -> list[ExtractedSpan]:
    """Return the spans the answer chooses from: all, or without repeats when the config asks."""
    if not config.drop_repeated_spans:
        return spans
    kept = [spans[i] for i in distinct_spans(_texts(spans))]
    if len(kept) < len(spans):
        logger.debug("Left out %d repeated spans of %d", len(spans) - len(kept), len(spans))
    return kept


def _texts(spans: list[ExtractedSpan]) -> list[str]:
    return [span.text for span in spans]


def _span_data(spans: list[ExtractedSpan]) -> list[dict[str, str]]:
    return [{"text": span.text} for span in spans]


def _filler(strategy: TemplateStrategy, config: TransformConfig) -> TemplateFiller:
    """Return the filler of a call: the strategy's, unless the config sets citation settings."""
    filler = getattr(strategy, "filler", None)
    if isinstance(filler, TemplateFiller) and not {"citation_mode", "citation_format"} & config.model_fields_set:
        return filler
    return TemplateFiller(citation_mode=config.citation_mode, citation_format=config.citation_format)


def _fills_its_own_way(strategy: TemplateStrategy) -> bool:
    """Return whether the strategy's class defines a `fill` of its own, outside the library."""
    return not type(strategy).fill.__module__.startswith("verbatim_core.")
