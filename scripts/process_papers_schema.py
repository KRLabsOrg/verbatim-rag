#!/usr/bin/env python3
"""
Process papers from JSON metadata using DocumentSchema.from_url
Much simpler approach - no manual downloads, leverages docling integration
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
from tqdm import tqdm

# Import verbatim-rag components
from verbatim_rag import VerbatimIndex
from verbatim_rag.schema import DocumentSchema
from verbatim_rag.ingestion.document_processor import DocumentProcessor


class DocumentSchemaProcessor:
    """Processes papers using DocumentSchema.from_url approach"""

    def __init__(
        self,
        papers_json: str,
        doc_dir: str = "downloads/schema_documents",
        status_file: str = "downloads/schema_status.json",
    ):
        self.papers_json = papers_json
        self.doc_dir = Path(doc_dir)
        self.status_file = Path(status_file)

        # Create directories
        self.doc_dir.mkdir(parents=True, exist_ok=True)
        self.status_file.parent.mkdir(parents=True, exist_ok=True)

        # Load or initialize status
        self.status = self.load_status()

        self.processor = DocumentProcessor()

    def load_status(self) -> Dict:
        """Load processing status from file"""
        if self.status_file.exists():
            with open(self.status_file, "r") as f:
                return json.load(f)
        return {"processed": [], "failed": [], "last_run": None}

    def save_status(self):
        """Save processing status to file"""
        self.status["last_run"] = time.time()
        with open(self.status_file, "w") as f:
            json.dump(self.status, f, indent=2)

    def load_papers_json(self) -> List[Dict]:
        """Load papers from JSON file"""
        print(f"Loading papers from: {self.papers_json}")

        with open(self.papers_json, "r", encoding="utf-8") as f:
            papers = json.load(f)

        print(f"Loaded {len(papers)} papers")
        return papers

    def is_processed(self, paper_id: str) -> bool:
        """Check if paper is already processed"""
        return paper_id in self.status["processed"]

    def is_document_saved(self, paper_id: str) -> bool:
        """Check if document is already saved"""
        doc_path = self.doc_dir / f"{paper_id}_schema.json"
        return doc_path.exists()

    def process_paper(self, paper: Dict) -> Optional[DocumentSchema]:
        """Process single paper using DocumentSchema.from_url"""
        paper_id = paper.get("id", "")
        doc_path = self.doc_dir / f"{paper_id}_schema.json"

        try:
            # Get paper URL
            url = f"{paper.get('url').strip('/')}.pdf" if paper.get("url", "") else None
            if not url:
                raise Exception("No URL found")

            # Create DocumentSchema from URL - this handles all downloading and parsing
            document = DocumentSchema.from_url(
                url=url,
                title=paper.get("title", ""),
                doc_type="academic_paper",
                processor=self.processor,
                authors=paper.get("authors", [])[:5],
                conference=paper.get("booktitle", ""),
                year=paper.get("year", ""),
                publisher=paper.get("publisher", ""),
                doi=paper.get("doi", ""),
                pages=paper.get("pages", ""),
                address=paper.get("address", ""),
                month=paper.get("month", ""),
                paper_id=paper_id,
                category="acl_anthology",  # Assuming NLP papers from ACL Anthology
            )

            # Save document schema as JSON
            doc_data = {
                "id": document.id,
                "content": document.content,  # Include content for processing
                "title": document.title,
                "source": document.source,
                "doc_type": document.doc_type,
                "content_type": document.content_type.value,
                "created_at": document.created_at.isoformat(),
                "metadata": document.metadata,
            }

            with open(doc_path, "w", encoding="utf-8") as f:
                json.dump(doc_data, f, indent=2, ensure_ascii=False)

            # Update status
            if paper_id not in self.status["processed"]:
                self.status["processed"].append(paper_id)

            return document

        except Exception as e:
            print(f"Failed to process {paper_id}: {e}")
            if paper_id not in self.status["failed"]:
                self.status["failed"].append(paper_id)
            return None

    def load_processed_documents(self) -> List[DocumentSchema]:
        """Load all processed DocumentSchemas from files"""
        documents = []

        for doc_file in self.doc_dir.glob("*_schema.json"):
            try:
                with open(doc_file, "r", encoding="utf-8") as f:
                    doc_data = json.load(f)

                # Reconstruct DocumentSchema
                from datetime import datetime
                from verbatim_rag.document import DocumentType

                # Convert datetime back
                if "created_at" in doc_data:
                    doc_data["created_at"] = datetime.fromisoformat(
                        doc_data["created_at"]
                    )

                # Convert content_type back to enum
                if "content_type" in doc_data:
                    doc_data["content_type"] = DocumentType(doc_data["content_type"])

                document = DocumentSchema(**doc_data)
                documents.append(document)

            except Exception as e:
                print(f"Failed to load {doc_file}: {e}")

        return documents

    def create_vector_index(self, index_path: str) -> VerbatimIndex:
        """Create vector index from all processed documents"""
        print(f"Creating vector index: {index_path}")

        # Ensure index path ends with .db
        if not index_path.endswith(".db"):
            index_path = index_path + ".db"

        # Load all processed documents
        documents = self.load_processed_documents()
        print(f"Loaded {len(documents)} processed documents")

        if not documents:
            print("No processed documents found!")
            return None

        # Create index - using google/embeddinggemma-300m as in the modified script
        index = VerbatimIndex(
            dense_model="google/embeddinggemma-300m", db_path=index_path
        )

        # Add documents to index
        print("Adding documents to index...")
        index.add_documents(documents)

        print(f"Vector index created: {index_path}")
        return index

    def get_stats(self) -> Dict:
        """Get processing statistics"""
        return {
            "total_processed": len(self.status["processed"]),
            "total_failed": len(self.status["failed"]),
            "documents_on_disk": len(list(self.doc_dir.glob("*_schema.json"))),
        }

    def process_all_papers(
        self,
        max_papers: Optional[int] = None,
        skip_processing: bool = False,
        index_path: Optional[str] = None,
    ):
        """Main processing pipeline"""

        papers = self.load_papers_json()

        if max_papers:
            papers = papers[:max_papers]
            print(f"Processing first {max_papers} papers")

        print(f"\nStarting processing of {len(papers)} papers...")
        print(f"Processing: {'SKIP' if skip_processing else 'YES'}")
        print(f"Indexing: {'YES' if index_path else 'NO'}")

        if not skip_processing:
            # Filter papers that need processing
            papers_to_process = []
            for paper in papers:
                paper_id = paper.get("id", "")
                if not self.is_processed(paper_id) and not self.is_document_saved(
                    paper_id
                ):
                    papers_to_process.append(paper)

            print(f"Papers needing processing: {len(papers_to_process)}")

            # Process papers
            for paper in tqdm(papers_to_process, desc="Processing papers"):
                try:
                    self.process_paper(paper)

                    # Save status periodically
                    if len(self.status["processed"]) % 10 == 0:
                        self.save_status()

                except Exception as e:
                    print(f"Error processing {paper.get('id', 'unknown')}: {e}")
                    continue

            # Final save
            self.save_status()

        # Create vector index if requested
        if index_path:
            self.create_vector_index(index_path)

        # Final statistics
        stats = self.get_stats()
        print(f"\n{'=' * 50}")
        print("PROCESSING COMPLETE")
        print(f"{'=' * 50}")
        print(f"Documents processed: {stats['total_processed']}")
        print(f"Failed: {stats['total_failed']}")
        print(f"Files on disk: {stats['documents_on_disk']}")
        print(f"Status file: {self.status_file}")
        if index_path:
            print(f"Vector index: {index_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Process papers using DocumentSchema.from_url"
    )
    parser.add_argument("--papers-json", required=True, help="Path to papers JSON file")
    parser.add_argument(
        "--doc-dir",
        default="downloads/schema_documents",
        help="Directory for processed documents",
    )
    parser.add_argument(
        "--status-file", default="downloads/schema_status.json", help="Status file path"
    )
    parser.add_argument("--index-path", help="Path for vector index (optional)")
    parser.add_argument(
        "--max-papers", type=int, help="Maximum papers to process (for testing)"
    )
    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Skip document processing, only create index",
    )

    args = parser.parse_args()

    processor = DocumentSchemaProcessor(
        papers_json=args.papers_json, doc_dir=args.doc_dir, status_file=args.status_file
    )

    processor.process_all_papers(
        max_papers=args.max_papers,
        skip_processing=args.skip_processing,
        index_path=args.index_path,
    )


if __name__ == "__main__":
    main()
