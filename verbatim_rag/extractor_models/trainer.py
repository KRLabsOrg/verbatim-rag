import logging
from pathlib import Path
import json
import time
from datetime import timedelta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

from verbatim_rag.extractor_models.dataset import QADataset

# Set up logger
logger = logging.getLogger(__name__)


def qa_collate_fn(batch: list[dict]) -> dict:
    # drop any None samples
    batch = [item for item in batch if item is not None]
    if not batch:
        logger.warning("Empty batch passed to qa_collate_fn (all items were None)")
        return {}

    input_ids_list = []
    attention_mask_list = []
    offset_mappings = []
    sentence_boundaries = []
    sentence_offset_mappings = []
    labels_list = []

    for item in batch:
        try:
            # ensure all keys exist
            required = [
                "input_ids",
                "attention_mask",
                "offset_mapping",
                "sentence_boundaries",
                "sentence_offset_mappings",
                "labels",
            ]
            missing = [k for k in required if k not in item]
            for k in missing:
                logger.warning(f"Item missing key {k}, inserting dummy")
                if k in ("input_ids", "attention_mask"):
                    item[k] = torch.tensor([0], dtype=torch.long)
                else:
                    item[k] = []

            input_ids_list.append(item["input_ids"])
            attention_mask_list.append(item["attention_mask"])
            offset_mappings.append(item["offset_mapping"])
            sentence_boundaries.append(item["sentence_boundaries"])
            sentence_offset_mappings.append(item["sentence_offset_mappings"])
            labels_list.append(item["labels"])
        except Exception as e:
            logger.error(f"Error in collate_fn item: {e}, inserting placeholders")
            input_ids_list.append(torch.tensor([0], dtype=torch.long))
            attention_mask_list.append(torch.tensor([0], dtype=torch.long))
            offset_mappings.append([])
            sentence_boundaries.append([])
            sentence_offset_mappings.append([])
            labels_list.append(torch.tensor([], dtype=torch.long))

    # pad sequences
    try:
        padded_input_ids = pad_sequence(input_ids_list, batch_first=True, padding_value=0)
        padded_attention_mask = pad_sequence(attention_mask_list, batch_first=True, padding_value=0)
        return {
            "input_ids": padded_input_ids,
            "attention_mask": padded_attention_mask,
            "offset_mapping": offset_mappings,
            "sentence_boundaries": sentence_boundaries,
            "sentence_offset_mappings": sentence_offset_mappings,
            "labels": labels_list,
        }
    except Exception as e:
        logger.error(f"Error padding sequences: {e}")
        # fallback to minimal tensors
        bs = len(input_ids_list)
        return {
            "input_ids": torch.zeros((bs, 1), dtype=torch.long),
            "attention_mask": torch.zeros((bs, 1), dtype=torch.long),
            "offset_mapping": [[] for _ in range(bs)],
            "sentence_boundaries": [[] for _ in range(bs)],
            "sentence_offset_mappings": [[] for _ in range(bs)],
            "labels": [torch.tensor([], dtype=torch.long) for _ in range(bs)],
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
        device: str = None,
        output_dir: str = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir) if output_dir else None
        self.best_f1 = 0.0
        self.current_epoch = 0

        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=qa_collate_fn,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )
        self.dev_dataloader = (
            DataLoader(
                dev_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=qa_collate_fn,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            )
            if dev_dataset is not None
            else None
        )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        self.model.to(self.device)

    def _train_one_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        steps = 0
        pbar = tqdm(self.train_dataloader, desc="Training", leave=True)

        for batch in pbar:
            try:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                sentence_boundaries = batch["sentence_boundaries"]
                labels_list = batch["labels"]

                self.optimizer.zero_grad()
                logits_list = self.model(input_ids, attention_mask, sentence_boundaries)

                batch_loss = 0.0
                doc_count = 0
                for i, logits in enumerate(logits_list):
                    # skip any Nones
                    if logits is None:
                        continue
                    if i >= len(labels_list):
                        logger.warning(
                            f"Logits/labels length mismatch {len(logits_list)} vs {len(labels_list)}"
                        )
                        continue

                    labels_i = labels_list[i].to(self.device)
                    if logits.size(0) == 0:
                        continue

                    if logits.size(0) > labels_i.size(0):
                        logits = logits[: labels_i.size(0)]

                    loss_i = self.criterion(logits, labels_i[: logits.size(0)])
                    batch_loss += loss_i
                    doc_count += 1

                if doc_count:
                    batch_loss = batch_loss / doc_count
                    batch_loss.backward()
                    self.optimizer.step()
                    total_loss += batch_loss.item()
                    steps += 1
                    pbar.set_postfix(
                        {
                            "loss": f"{batch_loss.item():.4f}",
                            "avg_loss": f"{total_loss/steps:.4f}",
                        }
                    )

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.error(f"OOM during training: {e}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
            except Exception as e:
                logger.error(f"Error processing batch: {e}")
                continue

        return total_loss / steps if steps else 0.0

    def train(self) -> float:
        start = time.time()
        print(f"\nStarting training on {self.device}")
        print(f"Training samples: {len(self.train_dataloader.dataset)}")
        if self.dev_dataloader:
            print(f"Validation samples: {len(self.dev_dataloader.dataset)}")

        for ep in range(self.epochs):
            self.current_epoch = ep + 1
            print(f"\nEpoch {self.current_epoch}/{self.epochs}")
            loss = self._train_one_epoch()
            elapsed = time.time() - start
            print(
                f"Epoch {self.current_epoch} completed in "
                f"{timedelta(seconds=int(elapsed))}. Average loss: {loss:.4f}"
            )

            if self.dev_dataloader:
                print("\nEvaluating...")
                metrics = self._evaluate(self.dev_dataloader)
                print(
                    f"Validation ⇒ loss: {metrics['loss']:.4f}  "
                    f"f1: {metrics['f1']:.4f}  acc: {metrics['accuracy']:.4f}"
                )

                if self.output_dir and metrics["f1"] > self.best_f1:
                    self.best_f1 = metrics["f1"]
                    self.save_model(self.output_dir)
                    print(f"New best model (F1={self.best_f1:.4f}) saved.")
            else:
                # no dev: save after each epoch
                if self.output_dir:
                    self.save_model(self.output_dir)
                    print(f"Model saved after epoch {self.current_epoch}")

        total = time.time() - start
        print(f"\nTraining finished in {timedelta(seconds=int(total))}")
        return self.best_f1

    def _evaluate(self, dataloader: DataLoader) -> dict:
        self.model.eval()
        total_loss = 0.0
        steps = 0
        all_preds, all_labels = [], []

        with torch.no_grad():
            pbar = tqdm(dataloader, desc="Evaluating", leave=False)
            for batch in pbar:
                try:
                    input_ids = batch["input_ids"].to(self.device)
                    attention_mask = batch["attention_mask"].to(self.device)
                    sentence_boundaries = batch["sentence_boundaries"]
                    labels_list = batch["labels"]

                    logits_list = self.model(input_ids, attention_mask, sentence_boundaries)
                    batch_loss = 0.0
                    doc_count = 0

                    for i, logits in enumerate(logits_list):
                        if logits is None:
                            continue
                        if i >= len(labels_list):
                            continue

                        labels_i = labels_list[i].to(self.device)
                        if logits.size(0) == 0:
                            continue

                        eff = min(logits.size(0), labels_i.size(0))
                        logits, labels_i = logits[:eff], labels_i[:eff]

                        batch_loss += self.criterion(logits, labels_i)
                        doc_count += 1

                        preds = torch.argmax(logits, dim=1).cpu().numpy()
                        labs = labels_i.cpu().numpy()
                        all_preds.extend(preds)
                        all_labels.extend(labs)

                    if doc_count:
                        total_loss += (batch_loss / doc_count).item()
                        steps += 1
                        pbar.set_postfix({"loss": f"{(batch_loss/doc_count).item():.4f}"})

                except Exception as e:
                    logger.error(f"Error during eval batch: {e}")
                    continue

        loss = total_loss / steps if steps else 0.0
        if all_preds:
            prec, rec, f1, _ = precision_recall_fscore_support(
                all_labels, all_preds, average="binary", zero_division=0
            )
            acc = accuracy_score(all_labels, all_preds)
        else:
            prec = rec = f1 = acc = 0.0

        return {"loss": loss, "precision": prec, "recall": rec, "f1": f1, "accuracy": acc}
    
    def save_model(self, save_path) -> Path:
        # Determine the directory to save into
        save_dir = Path(save_path)
        if save_dir.suffix:
            save_dir = save_dir.parent
        save_dir.mkdir(parents=True, exist_ok=True)
    
        # Build your metadata
        metadata = {
            "best_f1": float(self.best_f1),
            "epochs_trained": self.current_epoch,
            "learning_rate": self.lr,
            "batch_size": self.batch_size,
            "device": str(self.device),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    
        # 1) Save the model + config + metadata (write config.json + model weights file)
        self.model.save_pretrained(save_dir, metadata=metadata)
    
        # 2) Save the tokenizer files
        if self.tokenizer is not None:
            # writes tokenizer.json, tokenizer_config.json, special_tokens_map.json
            self.tokenizer.save_pretrained(save_dir)
    
        return save_dir
