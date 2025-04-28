from archehr.models.base import ArchehrModel

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


class BERTModel(ArchehrModel):
    def __init__(
        self,
        model_name: str,
        include_narrative: bool = True,
        context_size: int = 1,
        device: str = "auto",
        max_length: int = 1024,
        threshold: float = 0.5,
    ):
        """
        Initialize the BERT model.

        :param model_name: The name of the model to use, either local or remote.
        :param include_narrative: Whether to include the patient narrative in the input.
        :param context_size: The number of sentences to include as context before and after each sentence.
        :param device: The device to use for inference. If "auto", will use GPU if available, otherwise CPU.
        :param max_length: The maximum sequence length for the model.
        :param threshold: The threshold for classification.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)

        # Handle auto device selection
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model.to(self.device)
        self.model.eval()

        self.include_narrative = include_narrative
        self.context_size = context_size
        self.max_length = max_length
        self.threshold = threshold

    def _build_inputs(
        self,
        patient_narrative: str,
        clinician_question: str,
        note_excerpt: str,
        sentences: list[str],
    ) -> list[str]:
        """
        Build the inputs for the model.
        """
        inputs = []

        # sentences = [sentence.replace("\n", " ") for sentence in sentences]

        if self.include_narrative:
            question = f"Patient narrative: {patient_narrative}\n\nClinician question: {clinician_question}"
        else:
            question = clinician_question

        for i, sentence in enumerate(sentences):
            if self.context_size > 0:
                # Get context before
                start_idx = max(0, i - self.context_size)
                # Get context after
                end_idx = min(len(sentences), i + self.context_size + 1)

                context_before = " ".join(sentences[start_idx:i])
                context_after = " ".join(sentences[i + 1 : end_idx])

                marked_sentence = f"[TARGET] {sentence} [/TARGET]"

                if context_before and context_after:
                    contextual_sentence = (
                        f"{context_before} {marked_sentence} {context_after}"
                    )
                elif context_before:
                    contextual_sentence = f"{context_before} {marked_sentence}"
                elif context_after:
                    contextual_sentence = f"{marked_sentence} {context_after}"
                else:
                    contextual_sentence = marked_sentence

                inputs.append(contextual_sentence)
            else:
                inputs.append(sentence)

        return [(question, contextual_sentence) for contextual_sentence in inputs]

    def predict(
        self,
        patient_narrative: str,
        clinician_question: str,
        note_excerpt: str,
        sentences: list[str],
    ) -> list[bool]:
        """
        Predict the relevance of a sentence in a case.

        :param patient_narrative: The patient's narrative.
        :param clinician_question: The clinician's question.
        :param note_excerpt: The note excerpt.
        """
        results = []

        contextual_sentences = self._build_inputs(
            patient_narrative, clinician_question, note_excerpt, sentences
        )

        for question, contextual_sentence in contextual_sentences:
            inputs = self.tokenizer(
                question,
                contextual_sentence,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits

                if len(logits.shape) > 1 and logits.shape[1] > 1:
                    probs = torch.softmax(logits, dim=-1)
                    batch_pred = (probs[:, 1] >= self.threshold).long().cpu().tolist()
                else:
                    batch_pred = (
                        (logits.squeeze(-1) >= self.threshold).long().cpu().tolist()
                    )

                results.extend(batch_pred)

        return results
