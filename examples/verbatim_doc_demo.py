#!/usr/bin/env python3
"""
VerbatimDOC Demo: Document generation with embedded RAG queries

This example shows how to use VerbatimDOC to create documents with embedded
queries that get automatically filled with content from your knowledge base.

Usage:
    python verbatim_doc_demo.py
"""

import asyncio
from verbatim_rag import VerbatimRAG, VerbatimIndex, Document
from verbatim_rag.verbatim_doc import VerbatimDOC, VerbatimRAGAdapter
from verbatim_rag.document import DocumentType, Chunk, ProcessedChunk, ChunkType
from verbatim_rag.vector_stores import LocalMilvusStore
from verbatim_rag.embedding_providers import SentenceTransformersProvider


async def demo_verbatim_doc():
    """Demonstrate VerbatimDOC document generation"""

    print("🚀 VerbatimDOC Demo: Document Generation with Embedded Queries")
    print("=" * 65)

    # Step 1: Create knowledge base
    print("📚 Creating knowledge base...")

    doc = Document(
        title="AI Research Paper",
        source="research.pdf",
        content_type=DocumentType.PDF,
        raw_content="""
        This research introduces three main contributions to artificial intelligence: 
        novel attention mechanisms that improve model interpretability, 
        contrastive learning methodology for better representation learning, 
        and comprehensive benchmarking across multiple tasks.
        
        The methodology employs transformer-based models with enhanced attention mechanisms,
        multi-task training on diverse datasets, and contrastive learning for robust representations.
        
        Key findings show that our approach achieves 92% accuracy on GLUE benchmark,
        representing a 18% improvement over previous state-of-the-art methods.
        The model demonstrates superior performance across all evaluated tasks.
        
        Experimental results include 20% improvement in sentiment analysis,
        25% better named entity recognition, and 15% enhanced question answering performance.
        The approach shows consistent gains across different domains and languages.
        
        The main conclusion is that combining enhanced attention mechanisms with contrastive learning
        creates a robust framework for language understanding with broad applicability.
        """,
    )

    # Process into chunks
    paragraphs = [p.strip() for p in doc.raw_content.split("\n") if p.strip()]
    for i, paragraph in enumerate(paragraphs):
        chunk = Chunk(
            document_id=doc.id,
            content=paragraph,
            chunk_number=i,
            chunk_type=ChunkType.PARAGRAPH,
        )
        processed_chunk = ProcessedChunk(chunk_id=chunk.id, enhanced_content=paragraph)
        chunk.add_processed_chunk(processed_chunk)
        doc.add_chunk(chunk)

    print(f"✅ Knowledge base created with {len(doc.chunks)} chunks")

    # Step 2: Set up VerbatimDOC system
    print("\n🔧 Setting up VerbatimDOC system...")

    # Create index and RAG
    dense_provider = SentenceTransformersProvider(
        model_name="sentence-transformers/all-MiniLM-L6-v2", device="cpu"
    )
    vector_store = LocalMilvusStore(
        db_path="./verbatim_doc_demo.db",
        collection_name="verbatim_rag",
        dense_dim=384,
        enable_dense=True,
        enable_sparse=False,
    )
    index = VerbatimIndex(vector_store=vector_store, dense_provider=dense_provider)
    index.add_documents([doc])

    rag = VerbatimRAG(index)
    # Configure for clean extraction (no template wrapper)
    rag.template_manager.use_single_mode("[RELEVANT_SENTENCES]")

    # Create VerbatimDOC processor
    adapter = VerbatimRAGAdapter(rag)
    doc_processor = VerbatimDOC(adapter)

    print("✅ VerbatimDOC system ready")

    # Step 3: Define template with embedded queries
    print("\n📝 Document template with embedded queries:")

    template = """# AI Research Executive Summary

## Overview
This research presents [!query=what are the main contributions|max_length=100] using advanced machine learning techniques.

## Methodology  
The approach [!query=what methodology was used|format=short] with focus on scalability and performance.

## Key Results
Our findings demonstrate:
[!query=what were the experimental results|format=bullet]

## Performance Metrics
The system achieved [!query=what accuracy was achieved] showing significant advancement in the field.

## Conclusion
[!query=what is the main conclusion|max_length=150]

---
*Generated using VerbatimDOC with embedded RAG queries*
"""

    print("-" * 60)
    print(template)
    print("-" * 60)

    # Step 4: Auto-generate document
    print("\n⚙️ Auto-generating document...")

    try:
        generated_doc = await doc_processor.process(template, auto_approve=True)

        print("\n📄 Generated Document:")
        print("=" * 60)
        print(generated_doc)
        print("=" * 60)

        # Save result
        with open("examples/generated_research_summary.md", "w") as f:
            f.write(generated_doc)
        print("\n💾 Saved to: examples/generated_research_summary.md")

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return

    # Step 5: Interactive mode demo
    print("\n🎯 Interactive Mode Demo:")
    print(
        "In interactive mode, you can review and edit each query result before finalizing."
    )

    try:
        original_text, query_results = await doc_processor.process_interactive(template)

        print(f"\n🔍 Found {len(query_results)} embedded queries:")

        for i, qr in enumerate(query_results, 1):
            print(f"\n  {i}. Query: '{qr.query.text}'")
            if qr.query.params:
                print(f"     Parameters: {qr.query.params}")
            print(f'     Extracted content: "{qr.result[:100]}..."')

            # In a real application, you'd present this to the user for approval
            qr.approved = True  # Auto-approve for demo

        # Generate final document
        final_doc = doc_processor.finalize(original_text, query_results)

        approved_count = sum(1 for r in query_results if r.approved)
        print(
            f"\n✅ Interactive mode completed: {approved_count}/{len(query_results)} queries approved"
        )

    except Exception as e:
        print(f"❌ Interactive mode failed: {e}")

    print("\n🎉 VerbatimDOC Demo Complete!")
    print("💡 Key features demonstrated:")
    print("   • Embedded query syntax: [!query=your question]")
    print("   • Parameter support: |format=bullet, |max_length=100")
    print("   • Auto-generation and interactive review modes")
    print("   • Clean extraction without conversational templates")


if __name__ == "__main__":
    asyncio.run(demo_verbatim_doc())
