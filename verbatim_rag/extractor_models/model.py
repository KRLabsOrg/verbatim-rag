import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModel

# Set up logger
logger = logging.getLogger(__name__)


class QAModel(nn.Module):
    def __init__(
        self, model_name="answerdotai/ModernBERT-base", hidden_dim=768, num_labels=2
    ):
        super().__init__()
        self.model_name = model_name
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        self.bert = AutoModel.from_pretrained(model_name)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        sentence_boundaries: list[list[tuple[int, int]]],
    ) -> list[torch.Tensor]:
        """
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len]
        sentence_boundaries: list of length batch_size,
            each element is a list of (start_idx, end_idx) for that doc's sentences.

        Returns:
          A list of length batch_size,
          each item is a tensor of shape [num_sentences_i, num_labels].
        """
        # Add basic validation
        batch_size = input_ids.size(0)
        if len(sentence_boundaries) != batch_size:
            # Handle mismatch gracefully by padding or truncating
            if len(sentence_boundaries) < batch_size:
                # Pad with empty lists
                logger.warning(
                    f"Sentence boundaries list length ({len(sentence_boundaries)}) is less than batch size ({batch_size}). Padding with empty lists."
                )
                sentence_boundaries = sentence_boundaries + [
                    [] for _ in range(batch_size - len(sentence_boundaries))
                ]
            else:
                # Truncate
                logger.warning(
                    f"Sentence boundaries list length ({len(sentence_boundaries)}) is greater than batch size ({batch_size}). Truncating."
                )
                sentence_boundaries = sentence_boundaries[:batch_size]

        try:
            outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            last_hidden = outputs.last_hidden_state  # [batch_size, seq_len, hidden_dim]

            seq_len = last_hidden.size(1)
            batch_logits = []

            for b_idx in range(batch_size):
                boundaries = sentence_boundaries[b_idx]
                sentence_embs = []
                for start_idx, end_idx in boundaries:
                    # Boundary validation
                    if start_idx < 0:
                        logger.warning(
                            f"Negative start index ({start_idx}) found. Setting to 0."
                        )
                        start_idx = 0
                    if end_idx >= seq_len:
                        logger.warning(
                            f"End index ({end_idx}) exceeds sequence length ({seq_len}). Truncating."
                        )
                        end_idx = seq_len - 1
                    if start_idx > end_idx:
                        # Invalid boundary, skip this sentence
                        logger.warning(
                            f"Invalid boundary: start_idx ({start_idx}) > end_idx ({end_idx}). Skipping this sentence."
                        )
                        continue

                    try:
                        token_states = last_hidden[b_idx, start_idx : end_idx + 1, :]
                        if token_states.size(0) == 0:
                            # If no tokens for that sentence
                            logger.warning(
                                f"No tokens for sentence with boundaries ({start_idx}, {end_idx}). Using zero vector."
                            )
                            pooled = torch.zeros(
                                self.classifier.in_features, device=last_hidden.device
                            )
                        else:
                            # example: mean pooling
                            pooled = token_states.mean(dim=0)
                        sentence_embs.append(pooled)
                    except Exception as e:
                        # Skip this sentence if there's an error
                        logger.error(
                            f"Error processing sentence with boundaries ({start_idx}, {end_idx}): {e}"
                        )
                        continue

                if len(sentence_embs) == 0:
                    # If no sentences
                    logger.warning(
                        f"No valid sentences found for document at index {b_idx}."
                    )
                    logits = torch.empty(
                        0, self.classifier.out_features, device=last_hidden.device
                    )
                else:
                    try:
                        logits = self.classifier(torch.stack(sentence_embs, dim=0))
                    except Exception as e:
                        # If classifier fails, return empty tensor
                        logger.error(
                            f"Error in classifier for document at index {b_idx}: {e}"
                        )
                        logits = torch.empty(
                            0, self.classifier.out_features, device=last_hidden.device
                        )

                batch_logits.append(logits)

        except Exception as e:
            # Global exception handler
            logger.error(f"Error in forward pass: {e}")
            # Return empty logits for the whole batch
            batch_logits = [
                torch.empty(0, self.classifier.out_features, device=input_ids.device)
                for _ in range(batch_size)
            ]

        return batch_logits

    def get_config(self) -> dict:
        """Get model configuration as a dictionary."""
        return {
            "model_name": self.model_name,
            "hidden_dim": self.hidden_dim,
            "num_labels": self.num_labels,
        }

    def save_pretrained(self, save_dir: str) -> None:
        """Save model to a directory in the Hugging Face format.

        :param save_dir: Directory to save the model to
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Save model config
        config = self.get_config()
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Save the underlying transformer model
        self.bert.save_pretrained(save_dir / "bert")

        # Save the classifier weights
        classifier_path = save_dir / "classifier.pt"
        torch.save(self.classifier.state_dict(), classifier_path)

        # Save the full model for easy loading
        model_path = save_dir / "pytorch_model.bin"
        torch.save(self.state_dict(), model_path)

        return save_dir

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "QAModel":
        """Load model from a directory in the Hugging Face format.

        :param model_dir: Directory containing the saved model
        :return: QAModel: Loaded model
        """
        model_dir = Path(model_dir)

        # Load config
        with open(model_dir / "config.json", "r") as f:
            config = json.load(f)

        # Create model with proper config
        model = cls(
            model_name=config.get("model_name", "answerdotai/ModernBERT-base"),
            hidden_dim=config.get("hidden_dim", 768),
            num_labels=config.get("num_labels", 2),
        )

        # Load the full model state
        model_path = model_dir / "pytorch_model.bin"
        if model_path.exists():
            model.load_state_dict(torch.load(model_path, map_location="cpu"))
        else:
            # Alternative: load parts separately if full model not available
            bert_dir = model_dir / "bert"
            if bert_dir.exists():
                model.bert = AutoModel.from_pretrained(bert_dir)

            classifier_path = model_dir / "classifier.pt"
            if classifier_path.exists():
                model.classifier.load_state_dict(
                    torch.load(classifier_path, map_location="cpu")
                )

        return model
