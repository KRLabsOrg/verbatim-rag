import argparse
import json
import logging
import os
from collections import Counter

import concurrent.futures

from docling.datamodel.base_models import ConversionStatus
from docling.document_converter import DocumentConverter


logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s : %(module)s (%(lineno)s) - %(levelname)s - %(message)s",
)
# log = logging.getLogger(__name__)

TO_SKIP = {"2025.acl-long.1427", "D18-1021", "2024.emnlp-main.85"}


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
        try:
            md = self.pdf_to_md(input_fn)
        except Exception as exception:
            logging.error(f"error parsing {input_fn=}, {exception=}")
            return False
        else:
            with open(output_fn, "w") as f:
                f.write(md)
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

    def _process_all(self, input_dir, output_dir, max_workers, dry_run):
        print(f"processing {input_dir=} to {output_dir=}")
        files_and_paths, subdirs_and_paths = [], []
        for child in os.listdir(input_dir):
            path = os.path.join(input_dir, child)
            if os.path.isdir(path):
                subdirs_and_paths.append((child, path))
            else:
                files_and_paths.append((child, path))

        to_process = {}
        for file, file_path in files_and_paths:
            if not file.endswith(".pdf"):
                logging.warning(f"skipping file with unknown extension: {file}")
                continue
            fn = file.replace(".pdf", "")
            if fn not in self.papers:
                logging.warning(f"skipping file not in metadata file: {file}")
                continue
            if fn in TO_SKIP:
                logging.warning(f"skipping file specified in TO_SKIP: {file}")
                continue

            paper_metadata = self.papers[fn]
            output_fn = os.path.join(output_dir, f"{fn}.md")
            if os.path.exists(output_fn):
                continue
            to_process[fn] = {
                "input_fn": file_path,
                "output_fn": output_fn,
                "metadata": paper_metadata,
            }

        print(
            f"will process {len(to_process)} files and then recurse in {len(subdirs_and_paths)} subdirectories"
        )
        if to_process:
            if not os.path.exists(output_dir):
                if dry_run:
                    print(f"would create {output_dir=}")
                else:
                    print(f"creating output directory {output_dir}")
                    os.makedirs(output_dir)
            # self._process_all_parallel(to_process, max_workers=max_workers)
            # self._process_all_simple(to_process)
            if dry_run:
                for fn, item in to_process.items():
                    print(f'would process {item["input_fn"]} to {item["output_fn"]}')
            else:
                self._process_all_batch(to_process)

        for dirname, dir_path in subdirs_and_paths:
            print(f"recursing into subdirectory {dir_path}")
            out_dir = os.path.join(output_dir, dirname)
            self._process_all(dir_path, out_dir, max_workers, dry_run)

    def process_all(self, max_workers, dry_run=False):
        self._process_all(self.input_dir, self.output_dir, max_workers, dry_run)

    def _process_all_batch(self, to_process):
        paths = [data["input_fn"] for fn, data in to_process.items()]
        conv_results = self.converter.convert_all(paths, raises_on_error=False)
        for conv_res in conv_results:
            if conv_res.status == ConversionStatus.SUCCESS:
                doc_filename = conv_res.input.file.stem
                output_fn = to_process[doc_filename]["output_fn"]
                with open(output_fn, "w") as f:
                    f.write(conv_res.document.export_to_markdown())
            elif conv_res.status == ConversionStatus.PARTIAL_SUCCESS:
                logging.warning(
                    f"Document {conv_res.input.file} was partially converted with the following errors:"
                )
                for item in conv_res.errors:
                    logging.error(f"\t{item.error_message}")
            else:
                logging.warning(f"Document {conv_res.input.file} failed to convert.")

    def _process_all_simple(self, to_process):
        for fn, data in to_process.items():
            self.process_paper(data["input_fn"], data["output_fn"], data["metadata"])


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
    parser.add_argument("--dry-run", action="store_true", help="Dry run")

    parser.add_argument(
        "--doc-batch-size",
        required=True,
        type=int,
        help="Number of documents processed in one batch",
    )
    parser.add_argument(
        "--page-batch-size",
        required=True,
        type=int,
        help="Number of pages processed in one batch",
    )
    parser.add_argument(
        "--max-papers",
        type=int,
        help="Maximum number of papers to process (for testing)",
    )
    parser.add_argument(
        "--max-workers", type=int, help="Maximum number of parallel workers"
    )

    return parser.parse_args()


def main():
    args = get_args()
    # Check dependencies
    from docling.datamodel.settings import settings

    settings.perf.page_batch_size = args.page_batch_size
    settings.perf.doc_batch_size = args.doc_batch_size
    preprocessor = AnthologyPreprocessor(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        metadata_file=args.metadata_file,
    )
    preprocessor.process_all(args.max_workers, args.dry_run)


if __name__ == "__main__":
    main()
