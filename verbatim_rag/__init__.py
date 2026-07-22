"""Provenance-first RAG with verbatim evidence spans and citations."""

__version__ = "0.2.8"

from verbatim_rag.core import VerbatimRAG as VerbatimRAG
from verbatim_rag.document import Document as Document
from verbatim_rag.extractors import LLMSpanExtractor as LLMSpanExtractor
from verbatim_rag.extractors import SpanExtractor as SpanExtractor
from verbatim_rag.index import VerbatimIndex as VerbatimIndex
from verbatim_rag.intent import IntentDecision as IntentDecision
from verbatim_rag.intent import IntentDetector as IntentDetector
from verbatim_rag.intent import LLMIntentDetector as LLMIntentDetector
from verbatim_rag.models import Citation as Citation
from verbatim_rag.models import DocumentWithHighlights as DocumentWithHighlights
from verbatim_rag.models import Highlight as Highlight
from verbatim_rag.models import QueryRequest as QueryRequest
from verbatim_rag.models import QueryResponse as QueryResponse
from verbatim_rag.models import StreamingResponse as StreamingResponse
from verbatim_rag.models import StreamingResponseType as StreamingResponseType
from verbatim_rag.models import StructuredAnswer as StructuredAnswer
from verbatim_rag.providers import IndexProvider as IndexProvider
from verbatim_rag.providers import RAGProvider as RAGProvider
from verbatim_rag.providers import VerbatimRAGProvider as VerbatimRAGProvider
from verbatim_rag.rerankers import BaseReranker as BaseReranker
from verbatim_rag.rerankers import CohereReranker as CohereReranker
from verbatim_rag.rerankers import JinaReranker as JinaReranker
from verbatim_rag.rerankers import JinaV3Reranker as JinaV3Reranker
from verbatim_rag.rerankers import Reranker as Reranker
from verbatim_rag.rerankers import SentenceTransformersReranker as SentenceTransformersReranker
from verbatim_rag.schema import DocumentSchema as DocumentSchema
from verbatim_rag.streaming import StreamingRAG as StreamingRAG
from verbatim_rag.templates import TemplateManager as TemplateManager
from verbatim_rag.transform import VerbatimTransform as VerbatimTransform
from verbatim_rag.transform import verbatim_query as verbatim_query
from verbatim_rag.transform import verbatim_query_async as verbatim_query_async
from verbatim_rag.universal_document import UniversalDocument as UniversalDocument
from verbatim_rag.verbatim_doc import VerbatimDOC as VerbatimDOC

# Optional ingestion module (requires docling + chonkie)
try:
    from verbatim_rag.ingestion import DocumentProcessor

    INGESTION_AVAILABLE = True
except ImportError:
    DocumentProcessor = None
    INGESTION_AVAILABLE = False
