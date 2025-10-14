"""
Dependency injection setup for FastAPI
"""

import logging
from typing import Annotated
from fastapi import Depends, HTTPException
from verbatim_rag.core import VerbatimRAG
from verbatim_core.templates import TemplateManager
from verbatim_rag.core import LLMClient

from api.config import APIConfig, get_config
from api.services.rag_service import APIService

logger = logging.getLogger(__name__)


# Global instances (initialized once)
_rag_instance: VerbatimRAG = None
_template_manager: TemplateManager = None
_api_service: APIService = None


def get_rag_instance(config: Annotated[APIConfig, Depends(get_config)]) -> VerbatimRAG:
    """Get or create RAG instance (singleton)"""
    global _rag_instance

    if _rag_instance is None:
        try:
            from verbatim_rag.index import VerbatimIndex
            from verbatim_rag.vector_stores import LocalMilvusStore
            from verbatim_rag.embedding_providers import SpladeProvider

            llm_client = LLMClient(
                model="gpt-4o-mini",
                temperature=1.0,
            )

            # Create providers
            sparse_provider = SpladeProvider(
                model_name="opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill",
                device="cpu",
            )

            # Create vector store
            vector_store = LocalMilvusStore(
                db_path=str(config.index_path),
                collection_name="verbatim_rag",
                enable_dense=False,
                enable_sparse=True,
            )

            # Create index
            index = VerbatimIndex(
                vector_store=vector_store, sparse_provider=sparse_provider
            )

            # Create RAG instance with the index
            _rag_instance = VerbatimRAG(
                index=index,
                k=5,
                template_mode="contextual",
                llm_client=llm_client,
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
