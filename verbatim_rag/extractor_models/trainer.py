import logging
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
from datetime import timedelta

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
    if not batch:
        logger.warning("Empty batch passed to qa_collate_fn")
        return {}

    input_ids_list = []
    attention_mask_list = []
    offset_mappings = []
    sentence_boundaries = []
    sentence_offset_mappings = []
    labels_list = []

    for item in batch:
        try:
            required_keys = [
                "input_ids",
                "attention_mask",
                "offset_mapping",
                "sentence_boundaries",
                "sentence_offset_mappings",
                "labels",
            ]
            missing_keys = [k for k in required_keys if k not in item]
            if missing_keys:
                logger.warning(f"Item missing keys: {missing_keys}")
                # Create empty tensors for missing keys
                if "input_ids" not in item:
                    item["input_ids"] = torch.tensor([0], dtype=torch.long)
                if "attention_mask" not in item:
                    item["attention_mask"] = torch.tensor([0], dtype=torch.long)
                if "offset_mapping" not in item:
                    item["offset_mapping"] = []
                if "sentence_boundaries" not in item:
                    item["sentence_boundaries"] = []
                if "sentence_offset_mappings" not in item:
                    item["sentence_offset_mappings"] = []
                if "labels" not in item:
                    item["labels"] = torch.tensor([], dtype=torch.long)

            input_ids_list.append(item["input_ids"])
            attention_mask_list.append(item["attention_mask"])
            offset_mappings.append(item["offset_mapping"])
            sentence_boundaries.append(item["sentence_boundaries"])
            sentence_offset_mappings.append(item["sentence_offset_mappings"])
            labels_list.append(item["labels"])
        except Exception as e:
            logger.error(f"Error processing item in collate_fn: {e}")
            # Skip this item or add placeholder
            # Add an empty placeholder to keep batch size consistent
            input_ids_list.append(torch.tensor([0], dtype=torch.long))
            attention_mask_list.append(torch.tensor([0], dtype=torch.long))
            offset_mappings.append([])
            sentence_boundaries.append([])
            sentence_offset_mappings.append([])
            labels_list.append(torch.tensor([], dtype=torch.long))

    # If all lists are empty after processing, return empty dict
    if not input_ids_list:
        logger.warning("All items in batch were invalid")
        return {}

    try:
        # Pad input_ids and attention_mask
        padded_input_ids = pad_sequence(
            input_ids_list, batch_first=True, padding_value=0
        )
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
    except Exception as e:
        logger.error(f"Error padding sequences in collate_fn: {e}")
        return {
            "input_ids": torch.zeros((len(input_ids_list), 1), dtype=torch.long),
            "attention_mask": torch.zeros(
                (len(attention_mask_list), 1), dtype=torch.long
            ),
            "offset_mapping": [[] for _ in input_ids_list],
            "sentence_boundaries": [[] for _ in input_ids_list],
            "sentence_offset_mappings": [[] for _ in input_ids_list],
            "labels": [torch.tensor([], dtype=torch.long) for _ in input_ids_list],
        }


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: QADataset,
        dev_dataset: QADataset = None,
        tokenizer=None,
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
        :param tokenizer: Tokenizer to save with the model
        :param batch_size: The batch size
        :param lr: The learning rate
        :param epochs: The number of epochs
        :param device: The device to use
        :param output_dir: Directory to save model checkpoints
        """
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else None
        self.best_f1 = 0.0
        self.current_epoch = 0

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=qa_collate_fn,
            num_workers=0,  # Disable multiprocessing to avoid fork-related issues
            pin_memory=torch.cuda.is_available(),  # Speed up data transfer to GPU if available
        )

        if dev_dataset is not None:
            self.dev_dataloader = DataLoader(
                dev_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=qa_collate_fn,
                num_workers=0,  # Disable multiprocessing to avoid fork-related issues
                pin_memory=torch.cuda.is_available(),  # Speed up data transfer to GPU if available
            )
        else:
            self.dev_dataloader = None

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.model.to(self.device)

    def _train_one_epoch(self) -> float:
        """Train the model for one epoch.

        :return: Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        step_count = 0

        progress_bar = tqdm(self.train_dataloader, desc="Training", leave=True)

        try:
            for batch in progress_bar:
                try:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    sentence_boundaries = batch["sentence_boundaries"]
                    labels_list = batch["labels"]

                    self.optimizer.zero_grad()

                    # Forward pass
                    logits_list = self.model(
                        input_ids, attention_mask, sentence_boundaries
                    )

                    # Compute loss
                    batch_loss = 0.0
                    doc_count = 0
                    for i, logits in enumerate(logits_list):
                        # Make sure we have labels and logits of the same length
                        if i >= len(labels_list):
                            logger.warning(
                                f"Mismatch between logits and labels list lengths: {len(logits_list)} vs {len(labels_list)}"
                            )
                            continue

                        labels_i = labels_list[i].to(
                            self.device
                        )  # shape: [num_sentences_i]

                        if logits.size(0) == 0:
                            # if no sentences in the document, skip
                            continue

                        # Make sure we have enough labels for all logits
                        if logits.size(0) > labels_i.size(0):
                            logger.warning(
                                f"Mismatch between logits and labels sizes: {logits.size(0)} vs {labels_i.size(0)}"
                            )
                            logits = logits[: labels_i.size(0), :]

                        loss_i = self.criterion(logits, labels_i[: logits.size(0)])
                        batch_loss += loss_i
                        doc_count += 1

                    if doc_count > 0:
                        # average the doc losses in the batch
                        batch_loss = batch_loss / doc_count
                        batch_loss.backward()
                        self.optimizer.step()

                        total_loss += batch_loss.item()
                        step_count += 1

                        progress_bar.set_postfix(
                            {
                                "loss": f"{batch_loss.item():.4f}",
                                "avg_loss": f"{total_loss / step_count:.4f}"
                                if step_count > 0
                                else "N/A",
                            }
                        )

                except RuntimeError as e:
                    if "out of memory" in str(e).lower():
                        logger.error(f"CUDA out of memory error: {e}")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        logger.error(f"Runtime error in training loop: {e}")
                        raise
                except Exception as e:
                    logger.error(f"Error processing batch: {e}")
                    continue

        except Exception as e:
            logger.error(f"Error during training: {e}")
            if step_count == 0:
                # If we haven't successfully completed any steps, re-raise the error
                raise

        if step_count == 0:
            logger.warning("No steps completed in this epoch")
            return 0.0

        return total_loss / step_count

    def train(self) -> float:
        """Train the model for multiple epochs.

        :return: Best F1 score achieved during training
        """
        start_time = time.time()

        print(f"\nStarting training on {self.device}")
        print(f"Training samples: {len(self.train_dataloader.dataset)}")
        if self.dev_dataloader:
            print(f"Validation samples: {len(self.dev_dataloader.dataset)}")

        for epoch in range(self.epochs):
            self.current_epoch = epoch + 1  # Update current epoch
            epoch_start = time.time()
            print(f"\nEpoch {self.current_epoch}/{self.epochs}")

            # Train for one epoch
            train_loss = self._train_one_epoch()

            epoch_time = time.time() - epoch_start
            print(
                f"Epoch {self.current_epoch} completed in {timedelta(seconds=int(epoch_time))}. Average loss: {train_loss:.4f}"
            )

            if self.dev_dataloader is not None:
                print("\nEvaluating...")
                metrics = self._evaluate(self.dev_dataloader)
                print("Validation metrics:")
                print(f"  Loss:      {metrics['loss']:.4f}")
                print(f"  Precision: {metrics['precision']:.4f}")
                print(f"  Recall:    {metrics['recall']:.4f}")
                print(f"  F1 Score:  {metrics['f1']:.4f}")
                print(f"  Accuracy:  {metrics['accuracy']:.4f}")

                # Save metrics to a JSON file
                if self.output_dir:
                    metrics_path = self.output_dir / "metrics.json"
                    metrics_data = {
                        "loss": float(metrics["loss"]),
                        "precision": float(metrics["precision"]),
                        "recall": float(metrics["recall"]),
                        "f1": float(metrics["f1"]),
                        "accuracy": float(metrics["accuracy"]),
                        "epoch": self.current_epoch,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    with open(metrics_path, "w") as f:
                        json.dump(metrics_data, f, indent=2)

                # Save best model based on F1 score
                if self.output_dir and metrics["f1"] > self.best_f1:
                    self.best_f1 = metrics["f1"]
                    # Save directly to output_dir with standard names
                    model_path = self.output_dir
                    self.save_model(model_path)

                    # Save best metrics separately
                    best_metrics_path = self.output_dir / "best_metrics.json"
                    with open(best_metrics_path, "w") as f:
                        json.dump(metrics_data, f, indent=2)

                    print(f"New best model saved with F1: {self.best_f1:.4f}")
            else:
                print("No validation data provided, skipping evaluation.")

                # If no validation data, save the latest model after each epoch
                if self.output_dir:
                    model_path = self.output_dir
                    self.save_model(model_path)
                    print(f"Model saved after epoch {self.current_epoch}")

        # Final training summary
        total_time = time.time() - start_time
        hours, remainder = divmod(int(total_time), 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"\nTraining completed in {hours:02}:{minutes:02}:{seconds:02}")
        print(
            f"Best validation F1: {self.best_f1:.4f}"
            if self.best_f1 > 0
            else "No validation performed"
        )

        return self.best_f1

    def _evaluate(self, dev_dataloader: DataLoader) -> dict:
        """Evaluate the model on the development dataset.

        :return: Dictionary with metrics including loss, precision, recall, f1
        """
        self.model.eval()
        total_loss = 0.0
        step_count = 0

        all_preds = []
        all_labels = []

        try:
            with torch.no_grad():
                progress_bar = tqdm(dev_dataloader, desc="Evaluating", leave=False)
                for batch in progress_bar:
                    try:
                        input_ids = batch["input_ids"].to(self.device)
                        attention_mask = batch["attention_mask"].to(self.device)
                        sentence_boundaries = batch["sentence_boundaries"]
                        labels_list = batch["labels"]

                        logits_list = self.model(
                            input_ids, attention_mask, sentence_boundaries
                        )

                        batch_loss = 0.0
                        doc_count = 0
                        for i, logits in enumerate(logits_list):
                            # Skip if we have a mismatch in lists
                            if i >= len(labels_list):
                                logger.warning(
                                    f"Mismatch between logits and labels list lengths: {len(logits_list)} vs {len(labels_list)}"
                                )
                                continue

                            labels_i = labels_list[i].to(self.device)
                            if logits.size(0) == 0:
                                continue

                            # Make sure sizes match
                            effective_size = min(logits.size(0), labels_i.size(0))
                            if logits.size(0) != labels_i.size(0):
                                logger.warning(
                                    f"Mismatch between logits and labels sizes: {logits.size(0)} vs {labels_i.size(0)}"
                                )
                                logits = logits[:effective_size]
                                labels_i = labels_i[:effective_size]

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

                            # Update progress bar with loss
                            progress_bar.set_postfix(
                                {"loss": f"{batch_loss.item():.4f}"}
                            )
                    except Exception as e:
                        logger.error(f"Error evaluating batch: {e}")
                        continue
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            # If we have no results, still try to return partial metrics

        # Calculate metrics
        metrics = {}
        if step_count == 0:
            logger.warning("No evaluation steps completed")
            metrics["loss"] = 0.0
            metrics["precision"] = 0.0
            metrics["recall"] = 0.0
            metrics["f1"] = 0.0
            metrics["accuracy"] = 0.0
            return metrics

        metrics["loss"] = total_loss / step_count

        try:
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
                logger.warning("No predictions collected during evaluation")
                metrics["precision"] = 0.0
                metrics["recall"] = 0.0
                metrics["f1"] = 0.0
                metrics["accuracy"] = 0.0
        except Exception as e:
            logger.error(f"Error calculating metrics: {e}")
            metrics["precision"] = 0.0
            metrics["recall"] = 0.0
            metrics["f1"] = 0.0
            metrics["accuracy"] = 0.0

        return metrics

    def save_model(self, save_path) -> None:
        """Save the model to the given path with metadata.

        :param save_path: Path to save the model to
        :return: None
        """
        if isinstance(save_path, str) or isinstance(save_path, Path):
            save_dir = Path(save_path)
            if save_dir.suffix:
                save_dir = save_dir.parent
        else:
            save_dir = save_path

        # Create metadata
        metadata = {
            "best_f1": float(self.best_f1),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "epochs_trained": self.current_epoch,
            "learning_rate": self.lr,
            "batch_size": self.batch_size,
            "device": str(self.device),
        }

        # Use the model's save_pretrained method
        self.model.save_pretrained(
            save_dir, tokenizer=self.tokenizer, metadata=metadata
        )

        return save_dir
