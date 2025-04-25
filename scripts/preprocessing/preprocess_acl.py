# scripts/preprocess_acl.py

import argparse
import logging
from pathlib import Path

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2] / "src"))

from verbatim_rag.data.preprocessing import build_dataset, aggregate_cases
from verbatim_rag.util.text_processing_util import clean_text_df

logging.basicConfig(level=logging.INFO)


def main(args):
    df = build_dataset(args.xml, args.key, args.mapping)
    # clean text columns
    df = clean_text_df(df, text_columns=[
        "patient_narrative", "patient_question", "clinician_question",
        "note_excerpt", "sentence_text"], list_columns=[])
    agg = aggregate_cases(df)
    out = Path(args.output)
    agg.to_csv(out, index=False)
    logging.info(" Wrote %d cases to %s", len(agg), out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()

    DATA_DIR = Path("../../data/dev/")

    p.add_argument("--xml", type=Path, default=Path(DATA_DIR / "raw" / "archehr-qa.xml"))
    p.add_argument("--key", type=Path, default=Path(DATA_DIR / "raw" / "archehr-qa_key.json"))
    p.add_argument("--mapping", type=Path, default=Path(DATA_DIR / "raw" / "archehr-qa_mapping.json"))
    p.add_argument("--output", type=Path, default=Path(DATA_DIR / "processed" / "arch-dev.csv"))

    args = p.parse_args()
    main(args)
