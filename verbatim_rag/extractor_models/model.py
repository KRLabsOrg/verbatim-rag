import json
import logging
from pathlib import Path
from datetime import datetime

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

    def save_pretrained(self, save_dir: str, tokenizer=None, metadata=None) -> None:
        """Save model to a directory in the standard Hugging Face format.

        :param save_dir: Directory to save the model to
        :param tokenizer: Optional tokenizer to save alongside the model
        :param metadata: Optional dictionary with additional metadata for the README
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        if metadata is None:
            metadata = {}

        # 1. Save model config in the root directory
        config = self.get_config()
        with open(save_dir / "config.json", "w") as f:
            json.dump(config, f, indent=2)
        logger.info(f"Config saved to {save_dir / 'config.json'}")

        # 2. Save model weights in safetensors format (preferred)
        try:
            from safetensors.torch import save_file

            model_safetensors_path = save_dir / "model.safetensors"
            # Convert state dict to CPU for safetensors compatibility
            state_dict_cpu = {k: v.cpu() for k, v in self.state_dict().items()}
            save_file(state_dict_cpu, model_safetensors_path)
            logger.info(
                f"Model weights saved to {model_safetensors_path} in safetensors format"
            )
        except ImportError:
            # Fallback to PyTorch format if safetensors is not available
            model_bin_path = save_dir / "pytorch_model.bin"
            torch.save(self.state_dict(), model_bin_path)
            logger.info(f"Model weights saved to {model_bin_path} (PyTorch format)")

        # 3. Save the tokenizer if provided
        if tokenizer is not None:
            tokenizer.save_pretrained(save_dir)
            logger.info(f"Tokenizer saved to {save_dir}")
        else:
            # Create empty tokenizer files for compatibility
            with open(save_dir / "special_tokens_map.json", "w") as f:
                json.dump({}, f)

            with open(save_dir / "tokenizer_config.json", "w") as f:
                json.dump(
                    {"model_type": "bert", "tokenizer_class": "BertTokenizer"},
                    f,
                    indent=2,
                )

            logger.info("Created placeholder tokenizer files for compatibility")

        # 4. Create a model_config.json file with additional details
        model_config = {
            "model_name": self.model_name,
            "hidden_dim": self.hidden_dim,
            "num_labels": self.num_labels,
            **metadata,
        }
        with open(save_dir / "model_config.json", "w") as f:
            json.dump(model_config, f, indent=2)
        logger.info(
            f"Extended model configuration saved to {save_dir / 'model_config.json'}"
        )

        # 5. Create a README with usage instructions
        timestamp = metadata.get(
            "timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        )
        best_f1 = metadata.get("best_f1", "N/A")

        readme_path = save_dir / "README.md"
        readme_content = f"""# QA Sentence Classifier

This model classifies sentences as relevant or not relevant to a given question.

## Model Details
- Base model: {self.model_name}
- Hidden dimension: {self.hidden_dim}
- Number of labels: {self.num_labels}
- Best validation F1: {best_f1}
- Saved on: {timestamp}

## Loading the Model

```python
from verbatim_rag.extractor_models.model import QAModel
from transformers import AutoTokenizer

# Load the model
model = QAModel.from_pretrained("{save_dir}")

# Load the tokenizer
tokenizer = AutoTokenizer.from_pretrained("{save_dir}")
```

## Format

The model weights are saved in safetensors format, which is more secure and efficient.
The safetensors format requires an additional package:

```
pip install safetensors
```

## Directory Structure

This model follows the standard Hugging Face model format:
- `config.json`: Model configuration
- `model.safetensors`: Model weights in safetensors format
- `special_tokens_map.json`, `tokenizer_config.json`, etc.: Tokenizer files
"""
        with open(readme_path, "w") as f:
            f.write(readme_content)
        logger.info(f"README with loading instructions saved to {readme_path}")

        return save_dir

    @classmethod
    def from_pretrained(cls, model_dir: str) -> "QAModel":
        """Load model from a directory in the Hugging Face format.

        :param model_dir: Directory containing the saved model
        :return: QAModel: Loaded model
        """
        model_dir = Path(model_dir)
        logger.info(f"Loading model from {model_dir}")

        # Load config
        config_path = model_dir / "config.json"
        if not config_path.exists():
            logger.warning(f"No config.json found in {model_dir}, using defaults")
            config = {}
        else:
            with open(config_path, "r") as f:
                config = json.load(f)

        # Create model with proper config
        model = cls(
            model_name=config.get("model_name", "answerdotai/ModernBERT-base"),
            hidden_dim=config.get("hidden_dim", 768),
            num_labels=config.get("num_labels", 2),
        )

        # Try loading weights from different formats
        model_safetensors_path = model_dir / "model.safetensors"
        model_pt_path = model_dir / "model.pt"
        model_hf_path = model_dir / "pytorch_model.bin"  # Standard HF name

        if model_safetensors_path.exists():
            try:
                from safetensors.torch import load_file

                state_dict = load_file(model_safetensors_path)
                model.load_state_dict(state_dict)
                logger.info(
                    f"Loaded model weights from {model_safetensors_path} (safetensors format)"
                )
                return model
            except ImportError:
                logger.warning("safetensors not available, trying other formats")

        if model_pt_path.exists():
            logger.info(f"Loading weights from {model_pt_path}")
            model.load_state_dict(torch.load(model_pt_path, map_location="cpu"))
            return model

        if model_hf_path.exists():
            logger.info(f"Loading weights from {model_hf_path}")
            model.load_state_dict(torch.load(model_hf_path, map_location="cpu"))
            return model

        # If no model weights found, try loading BERT separately
        logger.warning(
            f"No model weights found in {model_dir}, checking bert directory"
        )

        bert_dir = model_dir / "bert"
        if bert_dir.exists():
            logger.info(f"Loading bert from {bert_dir}")
            model.bert = AutoModel.from_pretrained(bert_dir)
        else:
            logger.warning(f"No bert directory found in {model_dir}, using base model")

        # Initialize a new model with the loaded bert (or default bert)
        return model
