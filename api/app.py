"""
FastAPI server for the Verbatim RAG system.
"""

import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse as FastAPIStreamingResponse
from pydantic import BaseModel, Field

if "OPENAI_API_KEY" not in os.environ:
    print("Warning: OPENAI_API_KEY environment variable not set.")
    print("Please set your OpenAI API key using:")
    print("export OPENAI_API_KEY=your_api_key_here")

try:
    from verbatim_rag import (
        Citation,
        DocumentWithHighlights,
        Highlight,
        QueryRequest,
        QueryResponse,
        StreamingResponseType,
        StructuredAnswer,
        TemplateManager,
        VerbatimIndex,
        VerbatimRAG,
    )
    from verbatim_rag import (
        StreamingResponse as VerbatimStreamingResponse,
    )
except ImportError as e:
    print(f"Error importing verbatim_rag: {e}")
    sys.exit(1)

DEFAULT_INDEX_PATH = "index/"
DEFAULT_TEMPLATES_PATH = "templates.json"

index = None
rag = None
template_manager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global index, rag, template_manager

    try:
        if Path(DEFAULT_INDEX_PATH).exists():
            print(f"Loading index from {DEFAULT_INDEX_PATH}...")
            index = VerbatimIndex.load(DEFAULT_INDEX_PATH)

            if Path(DEFAULT_TEMPLATES_PATH).exists():
                print(f"Loading templates from {DEFAULT_TEMPLATES_PATH}...")
                template_manager = TemplateManager()
                template_manager.load_templates(DEFAULT_TEMPLATES_PATH)

            print("Initializing RAG system...")
            rag = VerbatimRAG(
                index=index,
                template_manager=template_manager,
            )
            print("RAG system initialized successfully.")
        else:
            print(
                f"Index not found at {DEFAULT_INDEX_PATH}. Please load resources first."
            )
    except Exception as e:
        print(f"Error during startup: {e}")

    yield

    # Cleanup (if needed)
    print("Shutting down...")


app = FastAPI(lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


class LoadResourcesRequest(BaseModel):
    """Request model for loading resources."""

    api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key. If not provided, will use the environment variable.",
        example="sk-...",
    )

    class Config:
        schema_extra = {"example": {"api_key": "sk-your-api-key-here"}}


class LoadResourcesResponse(BaseModel):
    """Response model for loading resources."""

    message: str = Field(
        description="Status message describing the result of the operation",
        example="Resources loaded successfully.",
    )
    success: bool = Field(
        description="Whether the operation was successful", example=True
    )

    class Config:
        schema_extra = {
            "example": {"message": "Resources loaded successfully.", "success": True}
        }


class StatusResponse(BaseModel):
    """Response model for the status endpoint."""

    resources_loaded: bool = Field(
        description="Whether the RAG system resources are loaded and ready",
        example=True,
    )
    message: str = Field(description="Status message", example="RAG system is ready.")

    class Config:
        schema_extra = {
            "example": {"resources_loaded": True, "message": "RAG system is ready."}
        }


@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Check if resources are loaded and the system is ready."""
    global rag

    if rag:
        return StatusResponse(
            resources_loaded=True, message="RAG system is initialized and ready."
        )
    else:
        return StatusResponse(
            resources_loaded=False,
            message="RAG system not initialized. Please load resources first.",
        )


@app.post("/api/load-resources", response_model=LoadResourcesResponse)
async def load_resources(request: LoadResourcesRequest = None):
    """Load or reload resources for the RAG system."""
    global index, rag, template_manager

    if request and request.api_key:
        os.environ["OPENAI_API_KEY"] = request.api_key

    try:
        print("Loading index...")
        index = VerbatimIndex.load(DEFAULT_INDEX_PATH)

        print("Loading templates...")
        template_manager = TemplateManager()
        if Path(DEFAULT_TEMPLATES_PATH).exists():
            template_manager.load_templates(DEFAULT_TEMPLATES_PATH)

        print("Initializing RAG system...")
        rag = VerbatimRAG(
            index=index,
            template_manager=template_manager,
        )

        return LoadResourcesResponse(
            message="Resources loaded successfully.", success=True
        )
    except Exception as e:
        print(f"Error loading resources: {e}")
        return LoadResourcesResponse(
            message=f"Error loading resources: {str(e)}", success=False
        )


@app.post("/api/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Process a query and return the answer with highlighted documents."""
    global rag

    if not rag:
        raise HTTPException(
            status_code=400,
            detail="RAG system not initialized. Please load resources first.",
        )

    try:
        # Update the number of documents to retrieve
        rag.k = request.num_docs

        # Process the query using the core library
        result = rag.query(request.question)

        # Return the result directly as a QueryResponse
        return QueryResponse(
            question=result.question,
            answer=result.answer,
            structured_answer=result.structured_answer,
            documents=result.documents,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing query: {str(e)}")


@app.post("/api/query/stream")
async def query_stream(request: QueryRequest):
    """Process a query and stream the results in stages."""
    global rag

    if not rag:
        raise HTTPException(
            status_code=400,
            detail="RAG system not initialized. Please load resources first.",
        )

    async def generate_response():
        try:
            # Step 1: Retrieve documents
            rag.k = request.num_docs
            docs = rag.index.search(request.question, k=request.num_docs)

            # Create documents without highlights
            documents_without_highlights = [
                DocumentWithHighlights(content=doc.content, highlights=[])
                for doc in docs
            ]

            # Send all documents in a single batch
            yield (
                json.dumps(
                    {
                        "type": "documents",
                        "data": [
                            doc.model_dump() for doc in documents_without_highlights
                        ],
                    }
                )
                + "\n"
            )
            yield "\n"  # Force flush

            # Step 2: Extract relevant spans and create highlights
            relevant_spans = rag.extractor.extract_spans(request.question, docs)
            documents_with_highlights = []
            all_citations = []

            # Process each document to find highlights
            for i, doc in enumerate(docs):
                doc_content = doc.content
                highlights = []

                # Track regions that have already been highlighted to avoid duplicates
                highlighted_regions = set()

                # Find all spans in this document
                if doc_content in relevant_spans and relevant_spans[doc_content]:
                    # Sort spans by length (descending) to prioritize longer matches
                    sorted_spans = sorted(
                        relevant_spans[doc_content], key=len, reverse=True
                    )

                    for span in sorted_spans:
                        # Find all non-overlapping occurrences of this span
                        start = 0
                        while True:
                            start = doc_content.find(span, start)
                            if start == -1:
                                break

                            # Check if this region overlaps with an already highlighted region
                            end = start + len(span)
                            overlap = False

                            for region_start, region_end in highlighted_regions:
                                # Check for any kind of overlap
                                if start <= region_end and end >= region_start:
                                    overlap = True
                                    break

                            if not overlap:
                                # Create a highlight for this non-overlapping region
                                highlight = Highlight(text=span, start=start, end=end)
                                highlights.append(highlight)

                                # Add to tracked regions
                                highlighted_regions.add((start, end))

                                # Create a citation
                                all_citations.append(
                                    Citation(
                                        text=span,
                                        doc_index=i,
                                        highlight_index=len(highlights) - 1,
                                    )
                                )

                            # Move past this occurrence
                            start = end

                # Add document with its highlights
                documents_with_highlights.append(
                    DocumentWithHighlights(content=doc_content, highlights=highlights)
                )

            # Send documents with highlights
            yield (
                json.dumps(
                    {
                        "type": "highlights",
                        "data": [doc.model_dump() for doc in documents_with_highlights],
                    }
                )
                + "\n"
            )
            yield "\n"  # Force flush

            # Step 3: Generate answer
            template = rag._generate_template(request.question)
            answer = rag._fill_template(template, relevant_spans.values())

            # Clean up the answer
            if answer.startswith('"') and answer.endswith('"'):
                answer = answer[1:-1]
            answer = answer.replace("\\n", "\n")

            # Create structured answer with citations
            structured_answer = StructuredAnswer(text=answer, citations=all_citations)

            # Create the final result
            result = QueryResponse(
                question=request.question,
                answer=answer,
                structured_answer=structured_answer,
                documents=documents_with_highlights,
            )

            # Send final answer
            yield (
                json.dumps(
                    {"type": "answer", "data": result.model_dump(), "done": True}
                )
                + "\n"
            )

        except Exception as e:
            yield json.dumps({"error": str(e), "type": "error", "done": True}) + "\n"

    # Return streaming response with headers to prevent buffering
    return FastAPIStreamingResponse(
        generate_response(),
        media_type="application/x-ndjson",
        headers={
            "Content-Type": "application/x-ndjson",
            "Cache-Control": "no-cache, no-transform",
            "X-Accel-Buffering": "no",  # Prevents nginx buffering
            "Transfer-Encoding": "chunked",  # Force chunked encoding
        },
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
