"""
RAG service implementation for the API layer
"""

import logging
from typing import Optional

from verbatim_rag.core import VerbatimRAG
from verbatim_rag.template_manager import TemplateManager
from verbatim_rag import QueryResponse

logger = logging.getLogger(__name__)


class APIService:
    """Service layer for RAG operations"""

    def __init__(self, rag: VerbatimRAG, template_manager: TemplateManager):
        self.rag = rag
        self.template_manager = template_manager
        logger.info("APIService initialized")

    def validate_query_request(self, question: str) -> None:
        """Validate query request parameters"""
        if not question or not question.strip():
            raise ValueError("Question cannot be empty")

        if len(question) > 1000:
            raise ValueError("Question too long (max 1000 characters)")

    def query(self, question: str, template_id: Optional[str] = None) -> QueryResponse:
        """Execute a query through the RAG system"""
        try:
            # Use the RAG system to process the query
            response = self.rag.query(question)
            return response
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            raise

    async def query_async(
        self, question: str, template_id: Optional[str] = None
    ) -> QueryResponse:
        """Execute an async query through the RAG system"""
        try:
            # For now, just call the sync version
            # In the future, this could be implemented with async RAG processing
            response = self.rag.query(question)
            return response
        except Exception as e:
            logger.error(f"Async query execution failed: {e}")
            raise

    def get_templates(self) -> list:
        """Get available templates"""
        try:
            return self.template_manager.list_templates()
        except Exception as e:
            logger.error(f"Failed to get templates: {e}")
            raise

    def health_check(self) -> dict:
        """Perform health check on the service"""
        try:
            # Check if RAG system is ready
            ready = hasattr(self.rag, "index") and self.rag.index is not None

            return {
                "status": "healthy" if ready else "initializing",
                "ready": ready,
                "components": {
                    "rag": ready,
                    "template_manager": self.template_manager is not None,
                },
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {"status": "unhealthy", "ready": False, "error": str(e)}
