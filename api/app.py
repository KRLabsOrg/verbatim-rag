"""
Clean FastAPI server for the Verbatim RAG system.
Decoupled from RAG logic with proper dependency injection.
"""

import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Annotated, Optional

from fastapi import FastAPI, HTTPException, Depends
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
        QueryRequest,
        QueryResponse,
        VerbatimRAG,
        TemplateManager,
        StreamingRAG,
    )
except ImportError as e:
    print(f"Error importing verbatim_rag: {e}")
    sys.exit(1)

from config import APIConfig, get_config
from dependencies import (
    get_api_service,
    get_rag_instance,
    get_template_manager,
    check_system_ready,
)
from services.rag_service import APIService

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Request/Response models
class QueryRequestModel(BaseModel):
    question: str
    template_id: Optional[str] = None


class StreamQueryRequestModel(BaseModel):
    question: str
    num_docs: int = 5


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

    # Dependencies will be initialized on first request
    # No global state initialization needed

    yield

    logger.info("Shutting down Verbatim RAG API server...")


def create_app() -> FastAPI:
    """Create FastAPI application with proper configuration"""
    config = get_config()

    app = FastAPI(
        title="Verbatim RAG API",
        description="API for the Verbatim RAG system - prevents hallucination by extracting verbatim spans from documents",
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


@app.get("/")
async def root():
    """Root endpoint - basic health check"""
    return {"status": "online", "message": "Verbatim RAG API is running"}


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
                logger.info(
                    "Documents are indexed but document metadata retrieval not implemented"
                )

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
        response = api_service.rag.query(request.question)

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
        response = await api_service.query_async(request.question, request.template_id)
        return response
    except Exception as e:
        logger.error(f"Async query failed: {e}")
        raise HTTPException(status_code=500, detail="Async query failed")


@app.post("/api/transform/verbatim", response_model=QueryResponse)
async def verbatim_transform_endpoint(request: VerbatimTransformRequest):
    """RAG-agnostic verbatim transform: question + context -> verbatim answer.

    The optional `answer` field is currently ignored.
    """
    from verbatim_rag.transform import VerbatimTransform

    try:
        vt = VerbatimTransform()
        # Convert Pydantic models to dicts expected by the transform
        context_dicts = [c.model_dump() for c in request.context]
        resp = await vt.transform_async(
            question=request.question, context=context_dicts, answer=request.answer
        )
        return resp
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Verbatim transform failed: {e}")
        raise HTTPException(status_code=500, detail="Verbatim transform failed")


@app.post("/api/query/async", response_model=QueryResponse)
async def query_async_endpoint(
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
        response = await api_service.rag.query_async(request.question)

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
    Stream a query response in stages using the package's streaming interface

    Args:
        request: Stream query request with question and optional num_docs

    Returns:
        Streaming response with documents, highlights, and final answer
    """
    try:
        # Validate request
        api_service.validate_query_request(request.question)

        # Create streaming RAG instance
        streaming_rag = StreamingRAG(rag)

        async def generate_clean_response():
            """Clean response generator using the package's streaming interface"""
            import json

            logger.info(f"Starting streaming query for: {request.question}")

            try:
                stage_count = 0
                async for stage in streaming_rag.stream_query(
                    request.question, request.num_docs
                ):
                    stage_count += 1
                    logger.info(
                        f"Yielding stage {stage_count}: {stage.get('type', 'unknown')}"
                    )
                    yield json.dumps(stage) + "\n"

                if stage_count == 0:
                    logger.warning("No stages yielded from streaming query")
                    yield (
                        json.dumps(
                            {
                                "type": "error",
                                "error": "No data returned from RAG system",
                                "done": True,
                            }
                        )
                        + "\n"
                    )

            except Exception as e:
                logger.error(f"Streaming error: {e}")
                import traceback

                traceback.print_exc()
                yield (
                    json.dumps({"type": "error", "error": str(e), "done": True}) + "\n"
                )

        # Return streaming response with proper headers
        return FastAPIStreamingResponse(
            generate_clean_response(),
            media_type="application/x-ndjson",
            headers={
                "Content-Type": "application/x-ndjson",
                "Cache-Control": "no-cache, no-transform",
                "X-Accel-Buffering": "no",
                "Transfer-Encoding": "chunked",
            },
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Stream query failed: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


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
