"""
Dependency injection setup for FastAPI
"""

import logging
from typing import Annotated

from fastapi import Depends, HTTPException, Request
from verbatim_core import Transformer
from verbatim_core.extractors import LLMSpanExtractor
from verbatim_core.templates import TemplateManager

from api.config import APIConfig, get_config
from api.services.rag_service import APIService
from verbatim_rag.core import LLMClient, VerbatimRAG

logger = logging.getLogger(__name__)


# Global instances (initialized once)
_rag_instance: VerbatimRAG = None
_template_manager: TemplateManager = None
_api_service: APIService = None


def build_transformer(llm_client: LLMClient | None = None) -> Transformer:
    """A Transformer with LLM extraction and contextual templates. The transform route's is built
    once, at startup.

    The templates are in contextual mode themselves, not only by the transformer's config, because
    StreamingRAG fills them without the transformer until query_stream replaces it (#97).
    """
    llm_client = llm_client or LLMClient()
    return Transformer(
        LLMSpanExtractor(llm_client),
        TemplateManager(llm_client=llm_client, default_mode="contextual"),
    )


def get_transformer(request: Request) -> Transformer:
    """The Transformer that startup built for the transform route."""
    return request.app.state.transformer


def get_rag_instance(config: Annotated[APIConfig, Depends(get_config)]) -> VerbatimRAG:
    """Get or create RAG instance (singleton)"""
    global _rag_instance

    if _rag_instance is None:
        try:
            from verbatim_rag.embedding_providers import SentenceTransformersProvider
            from verbatim_rag.index import VerbatimIndex
            from verbatim_rag.vector_stores import LocalMilvusStore

            llm_client = LLMClient(
                model="moonshotai/kimi-k2-instruct-0905",
                api_base="https://api.groq.com/openai/v1/",
            )

            dense_provider = SentenceTransformersProvider(
                model_name="ibm-granite/granite-embedding-small-english-r2",
                device="cpu",
            )

            # Create vector store
            vector_store = LocalMilvusStore(
                db_path=str(config.index_path),
                collection_name="acl",
                enable_dense=True,
                enable_sparse=False,
                dense_dim=dense_provider.get_dimension(),
            )

            # Create index
            index = VerbatimIndex(vector_store=vector_store, dense_provider=dense_provider)

            # Create RAG instance with the index
            _rag_instance = VerbatimRAG(
                index, transformer=build_transformer(llm_client), llm_client=llm_client
            )
            logger.info(f"RAG instance created with index path: {config.index_path}")
        except Exception as e:
            logger.error(f"Failed to create RAG instance: {e}")
            raise HTTPException(
                status_code=500, detail=f"Failed to initialize RAG system: {str(e)}"
            )

    return _rag_instance


def get_template_manager(
    config: Annotated[APIConfig, Depends(get_config)],
) -> TemplateManager:
    """Get or create template manager (singleton)"""
    global _template_manager

    if _template_manager is None:
        try:
            _template_manager = TemplateManager()

            # Load templates if file exists
            if config.templates_path.exists():
                _template_manager.load(str(config.templates_path))
                logger.info(
                    f"Template manager created and loaded config from: {config.templates_path}"
                )
            else:
                logger.info(
                    f"Template manager created without config (file not found: {config.templates_path})"
                )
        except Exception as e:
            logger.error(f"Failed to create template manager: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to initialize template manager: {str(e)}",
            )

    return _template_manager


def get_api_service(
    rag: Annotated[VerbatimRAG, Depends(get_rag_instance)],
    template_manager: Annotated[TemplateManager, Depends(get_template_manager)],
) -> APIService:
    """Get API service instance"""
    global _api_service

    if _api_service is None:
        _api_service = APIService(rag, template_manager)
        logger.info("API service created")

    return _api_service


def check_system_ready(rag: Annotated[VerbatimRAG, Depends(get_rag_instance)]) -> bool:
    """Check if the RAG system is ready to handle requests"""
    try:
        # Check if index is loaded
        if not hasattr(rag, "index") or rag.index is None:
            raise HTTPException(
                status_code=503, detail="RAG system is not ready. Index not loaded."
            )
        return True
    except Exception as e:
        logger.error(f"System readiness check failed: {e}")
        raise HTTPException(status_code=503, detail="RAG system is not ready")
