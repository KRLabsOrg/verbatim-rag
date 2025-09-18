#!/usr/bin/env python3
"""
Script to process all 2025 papers from anthology+abstracts-2.bib
Downloads papers, extracts metadata, and indexes them using VerbatimRAG
"""

import argparse
import json
import os
import re
import requests
import time
from pathlib import Path
from typing import Dict, List, Optional

import bibtexparser
from tqdm import tqdm

# Import verbatim-rag components
from verbatim_rag import VerbatimIndex
from verbatim_rag.ingestion import DocumentProcessor


class AnthologyProcessor:
    """Processes ACL Anthology bibliography entries"""

    def __init__(
        self, download_dir: str = "downloads/2025_papers", max_retries: int = 3
    ):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(parents=True, exist_ok=True)
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update(
            {"User-Agent": "Mozilla/5.0 (compatible; AnthologyProcessor/1.0)"}
        )

    def parse_bib_file(self, bib_path: str) -> List[Dict]:
        """Parse bibliography file and extract 2025 entries"""
        print(f"Parsing bibliography file: {bib_path}")

        with open(bib_path, "r", encoding="utf-8") as bib_file:
            bib_database = bibtexparser.load(bib_file)

        # Filter for 2025 papers
        # papers_2025 = []
        # for entry in bib_database.entries:
        #     if entry.get('year') == '2025' and entry.get('ENTRYTYPE') == 'inproceedings':
        #         papers_2025.append(entry)

        # print(f"Found {len(papers_2025)} papers from 2025")
        print(f"Found {len(bib_database.entries)} papers")
        return bib_database.entries

    def extract_metadata(self, entry: Dict) -> Dict:
        """Extract relevant metadata from bibliography entry"""
        metadata = {
            "id": entry.get("ID", ""),
            "title": entry.get("title", "").replace("{", "").replace("}", ""),
            "authors": self._parse_authors(entry.get("author", "")),
            "year": entry.get("year", ""),
            "booktitle": entry.get("booktitle", "").replace("{", "").replace("}", ""),
            "publisher": entry.get("publisher", ""),
            "url": entry.get("url", ""),
            "doi": entry.get("doi", ""),
            "pages": entry.get("pages", ""),
            "address": entry.get("address", ""),
            "month": entry.get("month", ""),
            "abstract": entry.get("abstract", "") if entry.get("abstract") else None,
        }

        # Clean up editor field if present
        if "editor" in entry:
            metadata["editor"] = self._parse_authors(entry["editor"])

        return metadata

    def _parse_authors(self, authors_str: str) -> List[str]:
        """Parse author string into list of individual authors"""
        if not authors_str:
            return []

        # Remove LaTeX formatting
        authors_str = re.sub(r"{\\\w+\s*([^}]*)}", r"\1", authors_str)
        authors_str = authors_str.replace("{", "").replace("}", "")

        # Split by 'and' and clean up
        authors = [author.strip() for author in authors_str.split(" and ")]
        return [author for author in authors if author]

    def download_paper(self, metadata: Dict) -> Optional[str]:
        """Download paper PDF from URL or DOI"""
        paper_id = metadata["id"]
        pdf_path = self.download_dir / f"{paper_id}.pdf"

        # Skip if already downloaded
        if pdf_path.exists():
            print(f"Already downloaded: {paper_id}")
            return str(pdf_path)

        # Try URL first, then DOI
        download_urls = []
        if metadata["url"]:
            # For ACL Anthology URLs, append .pdf
            if "aclanthology.org" in metadata["url"]:
                download_urls.append(metadata["url"].replace(".html", "") + ".pdf")
            download_urls.append(metadata["url"])

        if metadata["doi"]:
            # Try DOI resolution
            download_urls.append(f"https://doi.org/{metadata['doi']}")

        for url in download_urls:
            if self._download_from_url(url, pdf_path, paper_id):
                return str(pdf_path)

        print(f"Failed to download: {paper_id}")
        return None

    def _download_from_url(self, url: str, pdf_path: Path, paper_id: str) -> bool:
        """Attempt to download PDF from a specific URL"""
        for attempt in range(self.max_retries):
            try:
                print(f"Downloading {paper_id} from {url} (attempt {attempt + 1})")

                response = self.session.get(url, timeout=30, stream=True)
                response.raise_for_status()

                # Check if it's actually a PDF
                content_type = response.headers.get("content-type", "").lower()
                if "pdf" not in content_type and not url.endswith(".pdf"):
                    # Try to find PDF link in HTML
                    if "text/html" in content_type:
                        pdf_url = self._extract_pdf_link(response.text, url)
                        if pdf_url:
                            return self._download_from_url(pdf_url, pdf_path, paper_id)
                    continue

                # Download PDF
                with open(pdf_path, "wb") as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)

                print(f"Successfully downloaded: {paper_id}")
                return True

            except Exception as e:
                print(f"Attempt {attempt + 1} failed for {paper_id}: {e}")
                time.sleep(2**attempt)  # Exponential backoff

        return False

    def _extract_pdf_link(self, html: str, base_url: str) -> Optional[str]:
        """Extract PDF download link from HTML page"""
        # Common patterns for PDF links
        pdf_patterns = [
            r'href=["\']([^"\']*\.pdf)["\']',
            r'href=["\']([^"\']*download[^"\']*)["\']',
        ]

        for pattern in pdf_patterns:
            matches = re.findall(pattern, html, re.IGNORECASE)
            if matches:
                pdf_url = matches[0]
                if pdf_url.startswith("http"):
                    return pdf_url
                else:
                    # Relative URL
                    from urllib.parse import urljoin

                    return urljoin(base_url, pdf_url)

        return None

    def process_papers(
        self,
        bib_path: str,
        index_path: str,
        max_papers: Optional[int] = None,
        skip_download: bool = False,
    ) -> None:
        """Main processing pipeline"""

        # Parse bibliography
        papers = self.parse_bib_file(bib_path)

        if max_papers:
            papers = papers[:max_papers]
            print(f"Processing first {max_papers} papers")

        # Initialize components
        if not skip_download:
            processor = DocumentProcessor()

        # Ensure index path ends with .db for Milvus
        if not index_path.endswith(".db"):
            index_path = index_path + ".db"

        index = VerbatimIndex(
            dense_model="sentence-transformers/all-MiniLM-L6-v2", db_path=index_path
        )

        # Track progress
        successful_downloads = 0
        processed_documents = 0
        failed_papers = []

        print("\nStarting paper processing...")

        processed_papers = []

        for paper in tqdm(papers, desc="Processing papers"):
            try:
                metadata = self.extract_metadata(paper)

                processed_papers.append(metadata)

            except Exception as e:
                print(f"Error processing paper: {e}")
                continue

        with open(self.download_dir / "papers.json", "w", encoding="utf-8") as f:
            json.dump(processed_papers, f, indent=2, ensure_ascii=False)

        # Summary
        print(f"\n{'=' * 50}")
        print("PROCESSING SUMMARY")
        print(f"{'=' * 50}")
        print(f"Total papers found: {len(papers)}")
        if not skip_download:
            print(f"Successfully downloaded: {successful_downloads}")
            print(f"Successfully processed: {processed_documents}")
            print(f"Failed papers: {len(failed_papers)}")

            if failed_papers:
                print(f"\nFailed papers: {failed_papers[:10]}")  # Show first 10
                if len(failed_papers) > 10:
                    print(f"... and {len(failed_papers) - 10} more")
        else:
            print("Metadata extracted for all papers")

        # Save failed papers list
        if failed_papers:
            failed_path = self.download_dir / "failed_papers.json"
            with open(failed_path, "w") as f:
                json.dump(failed_papers, f, indent=2)
            print(f"\nFailed papers list saved to: {failed_path}")


def main():
    parser = argparse.ArgumentParser(description="Process 2025 ACL Anthology papers")
    parser.add_argument("--bib-file", required=True, help="Path to bibliography file")
    parser.add_argument("--index-path", required=True, help="Path for the output index")
    parser.add_argument(
        "--download-dir",
        default="downloads/2025_papers",
        help="Directory for downloaded papers",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        help="Maximum number of papers to process (for testing)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Only extract metadata, don't download papers",
    )

    args = parser.parse_args()

    # Check dependencies
    if not args.skip_download:
        if not os.getenv("OPENAI_API_KEY"):
            print("Warning: OPENAI_API_KEY not set. Some features may not work.")

    processor = AnthologyProcessor(download_dir=args.download_dir)
    processor.process_papers(
        bib_path=args.bib_file,
        index_path=args.index_path,
        max_papers=args.max_papers,
        skip_download=args.skip_download,
    )


if __name__ == "__main__":
    main()
