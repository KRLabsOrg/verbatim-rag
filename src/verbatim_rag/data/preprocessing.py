# src/verbatim_rag/data/processing.py

import pandas as pd
from .loaders import load_key_from_arch_data, load_mapping_from_arch_data, parse_xml_from_arch


def build_dataset(xml_path, key_path, map_path) -> pd.DataFrame:
    """
    Build a merged DataFrame from XML, key, and mapping files.

    Args:
        xml_path: Path to the XML file or archive.
        key_path: Path to the key data file or archive.
        map_path: Path to the mapping data file or archive.

    Returns:
        Merged pandas DataFrame with proper integer types.
    """
    # Load raw data
    data = parse_xml_from_arch(xml_path)
    keys = load_key_from_arch_data(key_path)
    mappings = load_mapping_from_arch_data(map_path)

    # Ensure integer dtypes in one step
    data = data.astype({"case_id": int, "sentence_id": int})
    keys = keys.astype({"case_id": int, "sentence_id": int})
    mappings = mappings.astype({"case_id": int})

    # Perform merges
    result = (
        data
        .merge(keys, on=["case_id", "sentence_id"], how="left")
        .merge(mappings, on="case_id", how="left")
    )

    return result


def aggregate_cases(df: pd.DataFrame) -> pd.DataFrame:
    def agg(grp):
        sents = grp["sentence_text"].tolist()
        labs = [1 if r in ("essential", "relevant") else 0
                for r in grp["relevance"]]
        return pd.Series({
            "patient_question": grp["patient_question"].iat[0],
            "clinician_question": grp["clinician_question"].iat[0],
            "note_excerpt": grp["note_excerpt"].iat[0],
            "sentences": sents,
            "labels": labs,
        })

    return df.groupby("case_id").apply(agg).reset_index(drop=True)
