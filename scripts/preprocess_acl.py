import argparse
import json
import logging
import os
from collections import Counter

import concurrent.futures

from docling.document_converter import DocumentConverter
from tqdm import tqdm


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - %(message)s",
)
# log = logging.getLogger(__name__)


class AnthologyPreprocessor:
    def __init__(self, input_dir, output_dir, metadata_file):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.metadata_file = metadata_file
        self._load_metadata()
        self.converter = DocumentConverter()

    def _load_metadata(self):
        papers = {}
        logging.info(f"loading metadata from {self.metadata_file}")
        with open(self.metadata_file) as f:
            metadata = json.load(f)
        for paper in metadata:
            fn = paper["url"].split("/")[-2]
            papers[fn] = paper

        self.papers = papers
        logging.info(f"loaded metadata for {len(papers)} papers")

    def pdf_to_md(self, input_fn):
        result = self.converter.convert(input_fn)
        return result.document.export_to_markdown()

    def process_paper(self, input_fn, output_fn, paper_metadata):
        with open(output_fn, "w") as f:
            f.write(self.pdf_to_md(input_fn))
        return True

    def _process_all_parallel(self, to_process, max_workers):
        logging.info(
            f"will process {len(to_process)} papers in parallel ({max_workers=})"
        )
        stats = Counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_fn = {
                executor.submit(
                    self.process_paper,
                    data["input_fn"],
                    data["output_fn"],
                    data["metadata"],
                ): fn
                for fn, data in to_process.items()
            }
            for future in concurrent.futures.as_completed(future_to_fn):
                fn = future_to_fn[future]
                try:
                    success = future.result()
                    stats["success"] += 1
                except Exception as exc:
                    print("%r generated an exception: %s" % (fn, exc))
                    stats["failure"] += 1

        logging.info(
            f'finished parallel processing, {stats["success"]} papers processed, {stats["failed"]} failed'
        )
        return success

    def process_all(self, max_workers):
        files = os.listdir(self.input_dir)
        to_process = {}
        for raw_fn in tqdm(files):
            if not raw_fn.endswith(".pdf"):
                logging.warning(f"skipping file with unknown extension: {raw_fn}")
                continue
            fn = raw_fn.replace(".pdf", "")
            if fn not in self.papers:
                logging.warning(f"skipping file not in metadata file: {raw_fn}")
                continue
            paper_metadata = self.papers[fn]
            input_fn = os.path.join(self.input_dir, raw_fn)
            output_fn = os.path.join(self.output_dir, f"{fn}.md")
            to_process[fn] = {
                "input_fn": input_fn,
                "output_fn": output_fn,
                "metadata": paper_metadata,
            }

        # self._process_all_parallel(to_process, max_workers=max_workers)
        self._process_all_simple(to_process)

    def _process_all_simple(self, to_process):
        for fn, data in to_process.items():
            if os.path.exists(data['output_fn']):
                continue
            self.process_paper(
                data["input_fn"], data["output_fn"], data["metadata"]
            )


def get_args():
    parser = argparse.ArgumentParser(description="Preprocess ACL Anthology papers")
    parser.add_argument(
        "--metadata-file", required=True, help="Path to paper metadata file"
    )
    parser.add_argument(
        "--input-dir", required=True, help="Directory for downloaded papers"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for storing markdown papers"
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        help="Maximum number of papers to process (for testing)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        help="Maximum number of parallel workers",
    )

    return parser.parse_args()


def main():
    args = get_args()
    # Check dependencies

    preprocessor = AnthologyPreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        metadata_file=args.metadata_file,
    )
    preprocessor.process_all(args.max_workers)


if __name__ == "__main__":
    main()
