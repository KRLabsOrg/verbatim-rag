"""
Bulk-ingest Markdown files with paired JSON metadata and add them to the index.

Each Markdown file in --md-dir is expected to have a corresponding JSON file in
--meta-dir with the same stem (e.g., article.md -> article.json). The JSON can
contain any fields; known fields like title/doc_type will be used explicitly and
all other keys will be stored in metadata via DocumentSchema.

Example:
    python examples/bulk_ingest_markdown.py \
        --md-dir ./notes/md \
        --meta-dir ./notes/metadata \
        --vector-db milvus-cloud \
        --collection-name verbatim_rag \
        --uri "$MILVUS_CLOUD_URI" \
        --api-key "$MILVUS_CLOUD_API_KEY"
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

from tqdm import tqdm

from verbatim_rag.document import DocumentType
from verbatim_rag.index import VerbatimIndex
from verbatim_rag.schema import DocumentSchema
from verbatim_rag.vector_stores import LocalMilvusStore, CloudMilvusStore
from verbatim_rag.embedding_providers import SpladeProvider


def discover_markdown_files(md_dir: Path) -> List[Path]:
    return sorted([p for p in md_dir.rglob("*.md") if p.is_file()])


def load_metadata_for(md_path: Path, meta_dir: Path) -> dict:
    meta_path = meta_dir / f"{md_path.stem}.json"
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception as exc:  # be resilient to malformed JSON
        print(f"Warning: failed to parse JSON metadata for {md_path.name}: {exc}")
        return {}


def build_schema(md_path: Path, metadata: dict) -> DocumentSchema:
    # Read Markdown content directly
    content = md_path.read_text(encoding="utf-8")

    # Known fields (with defaults); all other keys go to metadata automatically
    title = metadata.get("title", md_path.stem)
    doc_type = metadata.get("doc_type", "custom")

    return DocumentSchema(
        content=content,
        source=str(md_path),
        title=title,
        doc_type=doc_type,
        content_type=DocumentType.MARKDOWN,
        **metadata,
    )


def build_index(
    vector_db: str,
    collection_name: str,
    uri: str | None,
    api_key: str | None,
    db_path: str | None,
) -> VerbatimIndex:
    # Create sparse embedding provider
    sparse_provider = SpladeProvider(
        model_name="opensearch-project/opensearch-neural-sparse-encoding-doc-v3-distill",
        device="cpu",
    )

    # Vector DB selection
    if vector_db.lower() == "milvus-cloud":
        vector_store = CloudMilvusStore(
            collection_name=collection_name,
            uri=uri or os.getenv("VERBATIM_MILVUS_URI", ""),
            token=api_key or os.getenv("VERBATIM_MILVUS_API_KEY", ""),
            enable_dense=False,
            enable_sparse=True,
        )
    elif vector_db.lower() == "milvus-local":
        vector_store = LocalMilvusStore(
            db_path=db_path or "./milvus_verbatim.db",
            collection_name=collection_name,
            enable_dense=False,
            enable_sparse=True,
        )
    else:
        raise ValueError("--vector-db must be one of: milvus-cloud, milvus-local")

    return VerbatimIndex(vector_store=vector_store, sparse_provider=sparse_provider)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--md-dir", required=True, help="Directory of Markdown files")
    parser.add_argument(
        "--meta-dir",
        required=True,
        help="Directory containing JSON metadata files matching Markdown stems",
    )
    parser.add_argument(
        "--vector-db",
        default="milvus-cloud",
        choices=["milvus-cloud", "milvus-local"],
        help="Vector DB backend",
    )
    parser.add_argument(
        "--collection-name",
        default=os.getenv("VERBATIM_COLLECTION_NAME", "verbatim_rag"),
        help="Collection name",
    )
    parser.add_argument(
        "--uri", default=os.getenv("VERBATIM_MILVUS_URI"), help="Milvus Cloud URI"
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("VERBATIM_MILVUS_API_KEY"),
        help="Milvus Cloud API key",
    )
    parser.add_argument(
        "--db-path",
        default=os.getenv("VERBATIM_DB_PATH", "./milvus_verbatim.db"),
        help="Local Milvus DB path (for milvus-local)",
    )

    args = parser.parse_args()

    md_dir = Path(args.md_dir).expanduser().resolve()
    meta_dir = Path(args.meta_dir).expanduser().resolve()
    if not md_dir.exists() or not md_dir.is_dir():
        raise FileNotFoundError(f"Markdown directory not found: {md_dir}")
    if not meta_dir.exists() or not meta_dir.is_dir():
        raise FileNotFoundError(f"Metadata directory not found: {meta_dir}")

    print(f"Discovering Markdown files in: {md_dir}")
    md_files = discover_markdown_files(md_dir)
    if not md_files:
        print("No Markdown files found. Nothing to ingest.")
        return

    print(
        f"Initializing index (backend={args.vector_db}, collection={args.collection_name})..."
    )
    index = build_index(
        vector_db=args.vector_db,
        collection_name=args.collection_name,
        uri=args.uri,
        api_key=args.api_key,
        db_path=args.db_path,
    )

    print(f"Building DocumentSchemas for {len(md_files)} files...")
    documents: List[DocumentSchema] = []
    for md_path in tqdm(md_files, desc="Reading files"):
        metadata = load_metadata_for(md_path, meta_dir)
        schema = build_schema(md_path, metadata)
        documents.append(schema)

    print("Adding documents to index...")
    index.add_documents(documents)

    print("Done.")


if __name__ == "__main__":
    main()
