"""Sentence-level extractor for selecting the sentences relevant to a question.

This subpackage provides the data structures, model and training loop for a
sentence-level binary classifier that decides which sentences in a document
answer a given question:
- Sentence, Document, QASample, QAData: JSON-serializable training data
  (sentences with relevance labels, grouped into documents and QA samples)
- QADataset: PyTorch dataset that tokenizes a question with a document and
  tracks sentence boundaries
- qa_collate_fn: collate function that pads QADataset items into batches
- QAModel: transformer encoder with a per-sentence classification head
- Trainer: simple training and evaluation loop for QAModel

The training CLI (``train``) and RAGBench conversion (``preprocess_ragbench``)
are run as modules and are not re-exported here.
"""

from importlib import import_module
from typing import TYPE_CHECKING

from verbatim_core.extractor_models.dataset import Document, QAData, QADataset, QASample, Sentence

if TYPE_CHECKING:
    from verbatim_core.extractor_models.model import QAModel
    from verbatim_core.extractor_models.trainer import Trainer, qa_collate_fn

# Torch-heavy names are resolved on first access so importing the data
# structures does not require ``torch.nn`` or the training loop.
# Note: once the minimum supported Python is 3.15, this can be replaced with
# PEP 810 explicit lazy imports (``lazy from ... import ...``).
_LAZY_EXPORTS = {
    "QAModel": "verbatim_core.extractor_models.model",
    "Trainer": "verbatim_core.extractor_models.trainer",
    "qa_collate_fn": "verbatim_core.extractor_models.trainer",
}

__all__ = [
    "Document",
    "QAData",
    "QADataset",
    "QAModel",
    "QASample",
    "Sentence",
    "Trainer",
    "qa_collate_fn",
]


def __getattr__(name: str) -> object:
    module = _LAZY_EXPORTS.get(name)
    if module is None:
        msg = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(msg)
    return getattr(import_module(module), name)
