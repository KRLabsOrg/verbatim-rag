import logging
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from verbatim_rag.extractor_models.dataset import QADataset

# Set up logger
logger = logging.getLogger(__name__)


def qa_collate_fn(batch: list[dict]) -> dict:
    """batch is a list of N items (N = batch_size), each item is the dict returned by QADataset.__getitem__.
    We need to pad input_ids and attention_mask to the max length in this batch.

    We'll keep:
      - offset_mapping: list of lists
      - sentence_boundaries: list of lists
      - sentence_offset_mappings: list of lists
      - labels: list of 1D tensors of shape [num_sentences]
    """
    input_ids_list = []
    attention_mask_list = []
    offset_mappings = []
    sentence_boundaries = []
    sentence_offset_mappings = []
    labels_list = []

    for item in batch:
        input_ids_list.append(item["input_ids"])
        attention_mask_list.append(item["attention_mask"])
        offset_mappings.append(item["offset_mapping"])
        sentence_boundaries.append(item["sentence_boundaries"])
        sentence_offset_mappings.append(item["sentence_offset_mappings"])
        labels_list.append(item["labels"])

    # Pad input_ids and attention_mask
    padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
    padded_attention_mask = pad_sequence(
        attention_mask_list, batch_first=True, padding_value=0
    )

    return {
        "input_ids": padded_input_ids,  # [batch_size, max_seq_len_in_batch]
        "attention_mask": padded_attention_mask,  # [batch_size, max_seq_len_in_batch]
        "offset_mapping": offset_mappings,  # list of length batch_size
        "sentence_boundaries": sentence_boundaries,  # list of length batch_size
        "sentence_offset_mappings": sentence_offset_mappings,
        "labels": labels_list,  # list of length batch_size (each is a 1D Tensor)
    }


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: QADataset,
        dev_dataset: QADataset = None,
        batch_size: int = 2,
        lr: float = 2e-5,
        epochs: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        output_dir: str = None,
    ) -> None:
        """
        Simple trainer class for training a model on a dataset.

        :param model: The model to train
        :param train_dataset: The training dataset
        :param dev_dataset: The development dataset (optional)
        :param batch_size: The batch size
        :param lr: The learning rate
        :param epochs: The number of epochs
        :param device: The device to use
        :param output_dir: Directory to save model checkpoints
        """
        self.model = model
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else None
        self.best_f1 = 0.0

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=qa_collate_fn,
        )

        if dev_dataset is not None:
            self.dev_dataloader = DataLoader(
                dev_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=qa_collate_fn,
            )
        else:
            self.dev_dataloader = None

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.model.to(self.device)

    def _train_one_epoch(self) -> float:
        """The training loop for the model."""
        self.model.train()
        total_loss = 0.0
        step_count = 0

        for batch in tqdm(self.train_dataloader, desc="Training"):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            sentence_boundaries = batch["sentence_boundaries"]
            labels_list = batch["labels"]

            self.optimizer.zero_grad()

            logits_list = self.model(input_ids, attention_mask, sentence_boundaries)
            # Compute loss
            batch_loss = 0.0
            doc_count = 0
            for i, logits in enumerate(logits_list):
                labels_i = labels_list[i].to(self.device)  # shape: [num_sentences_i]
                if logits.size(0) == 0:
                    # if no sentences
                    continue
                loss_i = self.criterion(logits, labels_i)
                batch_loss += loss_i
                doc_count += 1

            if doc_count > 0:
                # average the doc losses in the batch (or you can sum)
                batch_loss = batch_loss / doc_count
                batch_loss.backward()
                self.optimizer.step()

                total_loss += batch_loss.item()
                step_count += 1

        if step_count == 0:
            return 0.0

        return total_loss / step_count

    def train(self) -> None:
        """The training loop for the model."""
        for epoch in range(1, self.epochs + 1):
            logger.info(f"Epoch {epoch}/{self.epochs}")
            train_loss = self._train_one_epoch()
            logger.info(f"  Train Loss: {train_loss:.4f}")

            if self.dev_dataloader is not None:
                metrics = self._evaluate(self.dev_dataloader)
                logger.info(f"  Dev Loss: {metrics['loss']:.4f}")
                logger.info(f"  Dev Precision: {metrics['precision']:.4f}")
                logger.info(f"  Dev Recall: {metrics['recall']:.4f}")
                logger.info(f"  Dev F1: {metrics['f1']:.4f}")

                # Save best model based on F1 score
                if self.output_dir and metrics["f1"] > self.best_f1:
                    self.best_f1 = metrics["f1"]
                    checkpoint_dir = self.output_dir / "checkpoint-best"
                    checkpoint_dir.mkdir(exist_ok=True, parents=True)
                    model_path = checkpoint_dir / "model.pt"
                    self.save_model(model_path)
                    logger.info(f"  New best model saved with F1: {self.best_f1:.4f}")
            else:
                logger.info("  No validation data provided, skipping evaluation.")

    def _evaluate(self, dev_dataloader: DataLoader) -> dict:
        """Evaluate the model on the development dataset.

        Returns:
            dict: Dictionary with metrics including loss, precision, recall, f1
        """
        self.model.eval()
        total_loss = 0.0
        step_count = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in dev_dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                sentence_boundaries = batch["sentence_boundaries"]
                labels_list = batch["labels"]

                logits_list = self.model(input_ids, attention_mask, sentence_boundaries)

                batch_loss = 0.0
                doc_count = 0
                for i, logits in enumerate(logits_list):
                    labels_i = labels_list[i].to(self.device)
                    if logits.size(0) == 0:
                        continue

                    # Calculate loss
                    loss_i = self.criterion(logits, labels_i)
                    batch_loss += loss_i
                    doc_count += 1

                    # Get predictions for metrics
                    preds_i = torch.argmax(logits, dim=1).cpu().numpy()
                    labels_i_np = labels_i.cpu().numpy()

                    # Extend lists with batch predictions and labels
                    all_preds.extend(preds_i)
                    all_labels.extend(labels_i_np)

                if doc_count > 0:
                    batch_loss = batch_loss / doc_count
                    total_loss += batch_loss.item()
                    step_count += 1

        # Calculate metrics
        metrics = {}
        if step_count == 0:
            metrics["loss"] = 0.0
            metrics["precision"] = 0.0
            metrics["recall"] = 0.0
            metrics["f1"] = 0.0
            metrics["accuracy"] = 0.0
            return metrics

        metrics["loss"] = total_loss / step_count

        if len(all_preds) > 0:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average="binary", zero_division=0
            )
            accuracy = accuracy_score(all_labels, all_preds)

            metrics["precision"] = precision
            metrics["recall"] = recall
            metrics["f1"] = f1
            metrics["accuracy"] = accuracy
        else:
            metrics["precision"] = 0.0
            metrics["recall"] = 0.0
            metrics["f1"] = 0.0
            metrics["accuracy"] = 0.0

        return metrics

    def save_model(self, save_path) -> None:
        """Save the model to the given path."""
        logger.info(f"Saving model to {save_path}")
        torch.save(self.model.state_dict(), save_path)
        return save_path
