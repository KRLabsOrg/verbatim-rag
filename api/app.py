"""
Clean FastAPI server for the Verbatim RAG system.
Decoupled from RAG logic with proper dependency injection.
"""

import json
import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Annotated, Any, Optional

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse as FastAPIStreamingResponse
from pydantic import BaseModel, Field

# Check for OpenAI API key
if "OPENAI_API_KEY" not in os.environ:
    print("Warning: OPENAI_API_KEY environment variable not set.")
    print("Please set your OpenAI API key using:")
    print("export OPENAI_API_KEY=your_api_key_here")

try:
    from verbatim_rag import (
        QueryResponse,
        TemplateManager,
        VerbatimRAG,
    )
except ImportError as e:
    print(f"Error importing verbatim_rag: {e}")
    sys.exit(1)

from verbatim_core import InvalidInputError, Transformer
from verbatim_core._context import _chunks_from_context
from verbatim_core.response_builder import ResponseBuilder

from api.config import APIConfig, get_config
from api.dependencies import (
    build_transformer,
    check_system_ready,
    get_api_service,
    get_rag_instance,
    get_template_manager,
    get_transformer,
)
from api.services.rag_service import APIService
from verbatim_rag.streaming import _dict_events

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Request/Response models
class QueryRequestModel(BaseModel):
    question: str
    template_id: Optional[str] = None
    k: Optional[int] = None
    hybrid_weights: Optional[dict[str, float]] = None
    rrf_k: int = 60
    filter: Optional[str] = None
    search_params: Optional[dict[str, Any]] = None


class StreamQueryRequestModel(BaseModel):
    question: str
    num_docs: int = 5
    hybrid_weights: Optional[dict[str, float]] = None
    rrf_k: int = 60
    filter: Optional[str] = None
    search_params: Optional[dict[str, Any]] = None


class StatusResponse(BaseModel):
    resources_loaded: bool
    message: str


class TemplateListResponse(BaseModel):
    templates: list[dict]


# RAG-agnostic verbatim transform models
class VerbatimContextItem(BaseModel):
    content: str = Field(..., min_length=1)
    title: str | None = ""
    source: str | None = ""
    metadata: dict | None = None


class VerbatimTransformRequest(BaseModel):
    question: str = Field(..., min_length=1)
    context: list[VerbatimContextItem] = Field(default_factory=list)
    answer: str | None = None  # ignored for now


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting Verbatim RAG API server...")

    # The transform route's Transformer serves every request. The other dependencies are
    # initialized on first request.
    app.state.transformer = build_transformer()

    yield

    logger.info("Shutting down Verbatim RAG API server...")


def create_app() -> FastAPI:
    """Create FastAPI application with proper configuration"""
    config = get_config()

    app = FastAPI(
        title="Verbatim RAG API",
        description="API for provenance-first RAG with source excerpts and citations",
        version="1.0.0",
        lifespan=lifespan,
        debug=config.debug,
    )

    # CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=config.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_app()


# Root endpoint removed to allow static files to serve React app at /


@app.get("/api/documents")
async def get_documents(
    rag: Annotated[VerbatimRAG, Depends(get_rag_instance)],
    _: Annotated[bool, Depends(check_system_ready)],
):
    """
    Get list of indexed documents

    Returns:
        List of documents with metadata
    """
    try:
        # Get documents from the vector store via the index
        if hasattr(rag, "index") and rag.index is not None:
            documents = []

            # Try to get documents from the vector store if it has the method
            if hasattr(rag.index.vector_store, "get_all_documents"):
                docs = rag.index.vector_store.get_all_documents()
                for doc in docs or []:
                    documents.append(
                        {
                            "id": doc.get("id", "unknown"),
                            "title": doc.get("title", "Unknown Document"),
                            "source": doc.get("source", "Unknown source"),
                            "content_length": len(doc.get("raw_content", "")),
                        }
                    )
            else:
                # Fallback: return a message indicating documents are indexed but not retrievable
                logger.info("Documents are indexed but document metadata retrieval not implemented")

            return {"documents": documents}
        else:
            return {"documents": []}

    except Exception as e:
        logger.error(f"Failed to get documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")


@app.get("/api/status", response_model=StatusResponse)
async def get_status(
    config: Annotated[APIConfig, Depends(get_config)],
    rag: Annotated[VerbatimRAG, Depends(get_rag_instance)],
):
    """Get system status"""
    try:
        # Check if system is ready
        ready = hasattr(rag, "index") and rag.index is not None

        return StatusResponse(
            resources_loaded=ready,
            message=f"RAG system {'ready' if ready else 'initializing'}",
        )
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        return StatusResponse(resources_loaded=False, message=f"System error: {str(e)}")


@app.post("/api/query", response_model=QueryResponse)
async def query_endpoint(
    request: QueryRequestModel,
    api_service: Annotated[APIService, Depends(get_api_service)],
    _: Annotated[bool, Depends(check_system_ready)],
):
    """
    Query the RAG system

    Args:
        request: Query request with question and optional template ID

    Returns:
        Query response with answer and supporting documents
    """
    try:
        # Validate request
        api_service.validate_query_request(request.question)

        # Execute query using the RAG package directly
        response = api_service.rag.query(
            request.question,
            k=request.k,
            hybrid_weights=request.hybrid_weights,
            rrf_k=request.rrf_k,
            filter=request.filter,
            search_params=request.search_params,
        )

        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Query failed: {e}")
        raise HTTPException(status_code=500, detail="Query failed")


@app.post("/api/query_async", response_model=QueryResponse)
async def query_async_endpoint(
    request: QueryRequestModel,
    api_service: Annotated[APIService, Depends(get_api_service)],
    _: Annotated[bool, Depends(check_system_ready)],
):
    """Async query endpoint using async RAG pipeline."""
    try:
        api_service.validate_query_request(request.question)
        response = await api_service.query_async(
            request.question,
            request.template_id,
            k=request.k,
            hybrid_weights=request.hybrid_weights,
            rrf_k=request.rrf_k,
            filter=request.filter,
            search_params=request.search_params,
        )
        return response
    except Exception as e:
        logger.error(f"Async query failed: {e}")
        raise HTTPException(status_code=500, detail="Async query failed")


@app.post("/api/transform/verbatim", response_model=QueryResponse)
async def verbatim_transform_endpoint(
    request: VerbatimTransformRequest,
    transformer: Annotated[Transformer, Depends(get_transformer)],
):
    """RAG-agnostic verbatim transform: question + context -> verbatim answer.

    The optional `answer` field is ignored.
    """
    try:
        # The same conversion as VerbatimTransform's, so a context item gets the same checks
        chunks = _chunks_from_context([item.model_dump() for item in request.context])
        answer = await transformer.atransform(request.question, chunks)
        return ResponseBuilder().build_from_answer(request.question, answer, chunks)
    except InvalidInputError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # the application boundary: any other failure is a 500
        # The error code or class only: a message may quote the request.
        logger.error("Verbatim transform failed: %s", getattr(e, "code", None) or type(e).__name__)
        raise HTTPException(status_code=500, detail="Verbatim transform failed")


@app.post("/api/query/async", response_model=QueryResponse)
async def query_async_slash_endpoint(
    request: QueryRequestModel,
    api_service: Annotated[APIService, Depends(get_api_service)],
    _: Annotated[bool, Depends(check_system_ready)],
):
    """
    Async query the RAG system

    Args:
        request: Query request with question and optional template ID

    Returns:
        Query response with answer and supporting documents
    """
    try:
        # Validate request
        api_service.validate_query_request(request.question)

        # Execute async query using the RAG package directly
        response = await api_service.rag.query_async(
            request.question,
            k=request.k,
            hybrid_weights=request.hybrid_weights,
            rrf_k=request.rrf_k,
            filter=request.filter,
            search_params=request.search_params,
        )

        return response

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Async query failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/api/templates", response_model=TemplateListResponse)
async def get_templates(
    template_manager: Annotated[TemplateManager, Depends(get_template_manager)],
):
    """Get available templates (return modes as simple records)."""
    try:
        modes = template_manager.get_available_modes()
        return TemplateListResponse(templates=[{"mode": m} for m in modes])
    except Exception as e:
        logger.error(f"Failed to get templates: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve templates")


@app.post("/api/query/stream")
async def query_stream_endpoint(
    request: StreamQueryRequestModel,
    rag: Annotated[VerbatimRAG, Depends(get_rag_instance)],
    api_service: Annotated[APIService, Depends(get_api_service)],
    _: Annotated[bool, Depends(check_system_ready)],
):
    """
    Stream a query as newline-delimited JSON events: documents, progress, highlights, then the
    answer, or an error as the last event.
    """
    try:
        api_service.validate_query_request(request.question)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    events = rag.query_stream(
        request.question,
        k=request.num_docs,
        filter=request.filter,
        hybrid_weights=request.hybrid_weights,
        rrf_k=request.rrf_k,
        search_params=request.search_params,
    )
    # StreamingRAG's conversion, which logs a failure and sends it as the last event
    lines = (json.dumps(event) + "\n" async for event in _dict_events(events))
    return FastAPIStreamingResponse(
        lines,
        media_type="application/x-ndjson",
        headers={
            "Content-Type": "application/x-ndjson",
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked",
        },
    )


# Serve static frontend files for production deployment (mounted last to not interfere with API routes)
if os.path.exists("./static"):
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory="static", html=True), name="static")


if __name__ == "__main__":
    import uvicorn

    config = get_config()
    uvicorn.run(
        "app:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level=config.log_level.lower(),
    )
