from pathlib import Path
import argparse
import json
from dataclasses import dataclass
import xml.etree.ElementTree as ET
from typing import Optional


@dataclass
class KeyData:
    case_id: int
    sentence_id: int
    relevance: str


@dataclass
class MappingData:
    case_id: int
    document_id: int
    document_source: str


@dataclass
class CaseData:
    case_id: int
    patient_narrative: str
    patient_question: str
    clinician_question: str
    note_excerpt: str
    sentence_id: int
    sentence_text: str
    paragraph_id: int
    start_char_index: int
    length: int


@dataclass
class Sentence:
    sentence_id: int
    sentence_text: str
    paragraph_id: int
    start_char_index: int
    length: int
    relevance: Optional[str] = None

    @classmethod
    def from_json(cls, json_data: dict) -> "Sentence":
        return cls(
            sentence_id=json_data["sentence_id"],
            sentence_text=json_data["sentence_text"],
            paragraph_id=json_data["paragraph_id"],
            start_char_index=json_data["start_char_index"],
            length=json_data["length"],
            relevance=json_data["relevance"],
        )

    def to_json(self) -> dict:
        return {
            "sentence_id": self.sentence_id,
            "sentence_text": self.sentence_text,
            "paragraph_id": self.paragraph_id,
            "start_char_index": self.start_char_index,
            "length": self.length,
            "relevance": self.relevance,
        }


@dataclass
class Case:
    case_id: int
    patient_narrative: str
    patient_question: str
    clinician_question: str
    note_excerpt: str
    sentences: list[Sentence]
    document_id: Optional[int] = None
    document_source: Optional[str] = None

    @classmethod
    def from_json(cls, json_data: dict) -> "Case":
        return cls(
            case_id=json_data["case_id"],
            patient_narrative=json_data["patient_narrative"],
            patient_question=json_data["patient_question"],
            clinician_question=json_data["clinician_question"],
            note_excerpt=json_data["note_excerpt"],
            sentences=[
                Sentence.from_json(sentence) for sentence in json_data["sentences"]
            ],
            document_id=json_data["document_id"],
            document_source=json_data["document_source"],
        )

    def to_json(self) -> dict:
        return {
            "case_id": self.case_id,
            "patient_narrative": self.patient_narrative,
            "patient_question": self.patient_question,
            "clinician_question": self.clinician_question,
            "note_excerpt": self.note_excerpt,
            "sentences": [sentence.to_json() for sentence in self.sentences],
            "document_id": self.document_id,
            "document_source": self.document_source,
        }


@dataclass
class ArchehrData:
    cases: list[Case]

    def to_json(self) -> dict:
        return {"cases": [case.to_json() for case in self.cases]}

    @classmethod
    def from_json(cls, json_data: dict) -> "ArchehrData":
        return cls(cases=[Case.from_json(case) for case in json_data["cases"]])


def extract_key_data(key_file_path: Path) -> dict:
    """Extract key data from the key file.

    :param key_file_path: Path to the key file.
    :return: Key data.
    """
    with open(key_file_path, "r") as f:
        key_data = json.load(f)

    structured_key_data = []

    for case in key_data:
        case_id = case["case_id"]
        for answer in case["answers"]:
            structured_key_data.append(
                KeyData(
                    case_id=int(case_id),
                    sentence_id=int(answer["sentence_id"]),
                    relevance=answer["relevance"],
                )
            )

    return structured_key_data


def extract_mapping_data(mapping_file_path: Path) -> dict:
    """Extract mapping data from the mapping file.

    :param mapping_file_path: Path to the mapping file.
    :return: Mapping data.
    """
    with open(mapping_file_path, "r") as f:
        mapping_data = json.load(f)

    structured_mapping_data = []

    for mapping in mapping_data:
        structured_mapping_data.append(
            MappingData(
                case_id=int(mapping["case_id"]),
                document_id=int(mapping["document_id"]),
                document_source=mapping["document_source"],
            )
        )

    return {mapping.case_id: mapping for mapping in structured_mapping_data}


def parse_xml_to_case_data(path_to_xml: Path) -> list[Case]:
    tree = ET.parse(path_to_xml)
    root = tree.getroot()

    cases = []

    for case in root.findall("case"):
        case_id = int(case.attrib["id"])

        # Extract patient details
        patient_narrative = (
            case.find("patient_narrative").text
            if case.find("patient_narrative") is not None
            else "No patient narrative"
        )
        patient_question = (
            case.find("patient_question/phrase").text
            if case.find("patient_question/phrase") is not None
            else "No patient question"
        )
        clinician_question = (
            case.find("clinician_question").text
            if case.find("clinician_question") is not None
            else "No clinician question"
        )

        # Extract clinical note excerpts
        note_excerpt = (
            case.find("note_excerpt").text
            if case.find("note_excerpt") is not None
            else "No note excerpt"
        )

        # Extract sentence-level details from note excerpts
        sentences = []
        for sentence in case.findall("note_excerpt_sentences/sentence"):
            sentence_id = int(sentence.attrib["id"])
            paragraph_id = int(sentence.attrib["paragraph_id"])
            start_char_index = int(sentence.attrib["start_char_index"])
            length = int(sentence.attrib["length"])
            sentence_text = (
                sentence.text if sentence.text is not None else "No sentence text"
            )

            sentences.append(
                Sentence(
                    sentence_id=sentence_id,
                    sentence_text=sentence_text,
                    paragraph_id=paragraph_id,
                    start_char_index=start_char_index,
                    length=length,
                )
            )

        cases.append(
            Case(
                case_id=case_id,
                patient_narrative=patient_narrative,
                patient_question=patient_question,
                clinician_question=clinician_question,
                note_excerpt=note_excerpt,
                sentences=sentences,
            )
        )

    return cases


def preprocess_data(data_dir: Path) -> list[Case]:
    data_file_name = "archehr-qa.xml"
    key_file_name = "archehr-qa_key.json"
    mapping_file_name = "archehr-qa_mapping.json"

    data_file_path = data_dir / data_file_name
    key_file_path = data_dir / key_file_name
    mapping_file_path = data_dir / mapping_file_name

    key_data = []

    if key_file_path.exists():
        key_data = extract_key_data(key_file_path)
    mapping_data = extract_mapping_data(mapping_file_path)
    cases = parse_xml_to_case_data(data_file_path)

    # Create a dictionary for quick lookup of relevance by case_id and sentence_id
    relevance_lookup = {}
    for item in key_data:
        if item.case_id not in relevance_lookup:
            relevance_lookup[item.case_id] = {}
        relevance_lookup[item.case_id][item.sentence_id] = item.relevance

    # Add relevance information to each sentence
    for case in cases:
        if case.case_id in relevance_lookup:
            for sentence in case.sentences:
                if sentence.sentence_id in relevance_lookup[case.case_id]:
                    sentence.relevance = relevance_lookup[case.case_id][
                        sentence.sentence_id
                    ]

        case.document_id = mapping_data[case.case_id].document_id
        case.document_source = mapping_data[case.case_id].document_source

    return cases


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/archehr/dev")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    cases = preprocess_data(data_dir)

    print(f"Processed {len(cases)} cases")

    archehr_data = ArchehrData(cases=cases)
    with open(data_dir / "archehr-qa_processed.json", "w") as f:
        json.dump(archehr_data.to_json(), f)


if __name__ == "__main__":
    main()
