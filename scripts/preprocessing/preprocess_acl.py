# scripts/preprocess_acl.py

import argparse
import logging
from pathlib import Path
import pandas as pd

from verbatim_rag.data.processing import build_dataset, aggregate_cases
import util.preprocessing_util as util

logging.basicConfig(level=logging.INFO)


def main(args):
    df = build_dataset(args.xml, args.key, args.mapping)
    # clean text columns
    df = util.clean_text_df(df, text_columns=[
        "patient_narrative","patient_question","clinician_question",
        "note_excerpt","sentence_text"], list_columns=[])
    agg = aggregate_cases(df)
    out = Path(args.output)
    agg.to_csv(out, index=False)
    logging.info("Wrote %d cases to %s", len(agg), out)


if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--xml",       type=Path, required=True)
    p.add_argument("--key",       type=Path, required=True)
    p.add_argument("--mapping",   type=Path, required=True)
    p.add_argument("--output",    type=Path, required=True)
    args=p.parse_args()
    main(args)