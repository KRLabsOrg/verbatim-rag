"""
Dependency injection setup for FastAPI
"""

import logging
from typing import Annotated

from fastapi import Depends, HTTPException
from verbatim_rag.core import VerbatimRAG
from verbatim_rag.template_manager import TemplateManager

from config import APIConfig, get_config
from services.rag_service import APIService

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

            # Create index with modern simplified API
            # Use config.index_path as the db_path for the Milvus database
            index = VerbatimIndex(
                db_path=str(config.index_path),
                dense_model=None,
                sparse_model="naver/splade-v3",  # Uncomment for hybrid mode
            )

            # Create RAG instance with the index
            _rag_instance = VerbatimRAG(index=index, model="gpt-4.1", k=5)
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
                _template_manager.load_templates(str(config.templates_path))
                logger.info(
                    f"Template manager created and loaded templates from: {config.templates_path}"
                )
            else:
                logger.info(
                    f"Template manager created without templates (file not found: {config.templates_path})"
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
