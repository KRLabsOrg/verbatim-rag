# src/verbatim_rag/data/loaders.py

import json
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET


# ------------ ACL Data ------------ #

def load_key_from_arch_data(path: Path) -> pd.DataFrame:
    with open(path) as f:
        jd = json.load(f)
    rows = []
    for case in jd:
        cid = case["case_id"]
        for ans in case["answers"]:
            rows.append({
                "case_id": cid,
                "sentence_id": ans["sentence_id"],
                "relevance": ans["relevance"],
            })
    return pd.DataFrame(rows)


def load_mapping_from_arch_data(path: Path) -> pd.DataFrame:
    with open(path) as f:
        md = json.load(f)
    rows = [{"case_id": c["case_id"],
             "document_id": c["document_id"],
             "document_source": c["document_source"]}
            for c in md]
    return pd.DataFrame(rows)


def parse_xml_from_arch(path: Path) -> pd.DataFrame:
    tree = ET.parse(path)
    root = tree.getroot()
    rows = []
    for case in root.findall("case"):
        cid = int(case.attrib["id"])
        narrative = case.findtext("patient_narrative", None) or ""
        pq = case.findtext("patient_question/phrase", None) or ""
        cq = case.findtext("clinician_question", None) or ""
        note = case.findtext("note_excerpt", None) or ""
        for sent in case.findall("note_excerpt_sentences/sentence"):
            rows.append({
                "case_id": cid,
                "patient_narrative": narrative,
                "patient_question": pq,
                "clinician_question": cq,
                "note_excerpt": note,
                "sentence_id": int(sent.attrib["id"]),
                "sentence_text": sent.text or "",
                "start_char_index": int(sent.attrib["start_char_index"]),
                "length": int(sent.attrib["length"]),
            })
    return pd.DataFrame(rows)
