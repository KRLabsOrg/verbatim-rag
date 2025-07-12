import logging
from pathlib import Path
import time
from datetime import timedelta

import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.nn.utils.rnn import pad_sequence
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from tqdm import tqdm

from verbatim_rag.extractor_models.dataset import QADataset

# Set up logger
logger = logging.getLogger(__name__)

def qa_collate_fn(batch: list[dict]) -> dict:
    # Drop any None samples
    batch = [item for item in batch if item is not None]
    if not batch:
        logger.warning("Empty batch passed to qa_collate_fn (all items were None)")
        return {}

    input_ids_list, attention_mask_list = [], []
    offset_mappings, sentence_boundaries = [], []
    sentence_offset_mappings, labels_list = [], []

    for item in batch:
        try:
            required = [
                "input_ids", "attention_mask",
                "offset_mapping", "sentence_boundaries",
                "sentence_offset_mappings", "labels"
            ]
            for k in required:
                if k not in item:
                    logger.warning(f"Missing {k}, inserting dummy")
                    item[k] = (torch.tensor([0], dtype=torch.long)
                               if k in ("input_ids", "attention_mask") else [])

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
        bs = len(input_ids_list)
        return {
            k: (torch.zeros((bs,1), dtype=torch.long)
                if k in ("input_ids","attention_mask") else [[]]*bs)
            for k in ["input_ids","attention_mask",
                      "offset_mapping","sentence_boundaries",
                      "sentence_offset_mappings","labels"]
        }

def compute_metrics_from_arrays(all_labels, all_preds):
    """
    Compute accuracy, binary & per-class precision/recall/f1, plus weighted f1.
    """
    acc = accuracy_score(all_labels, all_preds)
    prec, rec, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    prc, recc, f1c, _ = precision_recall_fscore_support(
        all_labels, all_preds, average=None, zero_division=0
    )
    _, _, f1w, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="weighted", zero_division=0
    )

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "f1_weighted": f1w,
        "precision_cls0": prc[0],
        "recall_cls0": recc[0],
        "f1_cls0": f1c[0],
        "precision_cls1": prc[1],
        "recall_cls1": recc[1],
        "f1_cls1": f1c[1],
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
        class_weight: float = 1.0,
        dynamic_weighting: bool = False,
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
        self.patience = 1  # stop after 1 epoch without improvement

        # Build dataloaders
        self.train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=qa_collate_fn,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        self.dev_dataloader = (
            DataLoader(
                dev_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=qa_collate_fn,
                num_workers=0,
                pin_memory=torch.cuda.is_available(),
            ) if dev_dataset else None
        )

        # Compute class weight if requested
        if dynamic_weighting:
            pos = sum(s.relevant for _, doc in train_dataset.samples for s in doc.sentences)
            neg = sum(not s.relevant for _, doc in train_dataset.samples for s in doc.sentences)
            class_weight_val = neg / pos if pos > 0 else 1.0
        else:
            class_weight_val = class_weight

        # Loss with class weights
        weight = torch.tensor([1.0, class_weight_val], dtype=torch.float).to(self.device)
        self.criterion = nn.CrossEntropyLoss(weight=weight)

        # Optimizer & scheduler
        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.lr)
        total_steps = len(self.train_dataloader) * self.epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

        self.model.to(self.device)

        # Initialize WandB
        wandb.init(
            project="sentence-relevance",
            config={
                "lr": self.lr,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "model_name": getattr(self.model, "model_name", None),
                "class_weight": class_weight_val,
                "patience": self.patience,
            }
        )

    def _train_one_epoch(self) -> float:
        self.model.train()
        total_loss, steps = 0.0, 0

        for batch in tqdm(self.train_dataloader, desc="Training", leave=True):
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            boundaries, labels_list = batch["sentence_boundaries"], batch["labels"]

            self.optimizer.zero_grad()
            logits_list = self.model(input_ids, attention_mask, boundaries)

            batch_loss, doc_count = 0.0, 0
            for i, logits in enumerate(logits_list):
                if logits is None or logits.size(0) == 0:
                    continue
                labels_i = labels_list[i].to(self.device)
                s = min(logits.size(0), labels_i.size(0))
                batch_loss += self.criterion(logits[:s], labels_i[:s])
                doc_count += 1

            if doc_count == 0:
                continue

            batch_loss = batch_loss / doc_count
            batch_loss.backward()

            # Clip gradients, optimizer & scheduler step
            clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            # Log loss and current LR
            current_lr = self.scheduler.get_last_lr()[0]
            global_step = self.current_epoch * len(self.train_dataloader) + steps
            wandb.log({
                "train/loss": batch_loss.item(),
                "train/lr": current_lr,
                "step": global_step,
            })

            total_loss += batch_loss.item()
            steps += 1

        return total_loss / steps if steps else 0.0

    def train(self) -> float:
        start = time.time()
        print(f"Starting training on {self.device} ({len(self.train_dataloader.dataset)} samples)")

        self.no_improve_epochs = 0

        for ep in range(1, self.epochs + 1):
            self.current_epoch = ep
            print(f"\nEpoch {ep}/{self.epochs}")
            loss = self._train_one_epoch()
            print(f"Epoch {ep} loss={loss:.4f} time={(time.time() - start):.0f}s")

            if self.dev_dataloader:
                metrics = self._evaluate(self.dev_dataloader)
                print(f"Dev f1={metrics['f1']:.4f} acc={metrics['accuracy']:.4f}")
                
                wandb.log({
                    "epoch":        ep,
                    "dev/loss":     metrics["loss"],
                    "dev/accuracy": metrics["accuracy"],
                    "dev/precision":metrics["precision"],
                    "dev/recall":   metrics["recall"],
                    "dev/f1":       metrics["f1"],
                })

                if metrics["f1"] > self.best_f1:
                    self.best_f1 = metrics["f1"]
                    self.no_improve_epochs = 0
                    if self.output_dir:
                        self.save_model(self.output_dir)
                        print(f"New best model (f1={self.best_f1:.4f}) saved")
                else:
                    self.no_improve_epochs += 1
                    print(f"No improvement for {self.no_improve_epochs}/{self.patience} epochs")
                    if self.no_improve_epochs >= self.patience:
                        print(f"No improvement in {self.patience} epochs → stopping early.")
                        break

        print(f"\nTraining completed in {timedelta(seconds=int(time.time() - start))}")
        return self.best_f1

    def _evaluate(self, dataloader: DataLoader) -> dict:
        self.model.eval()
        total_loss, steps, all_preds, all_labels = 0.0, 0, [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                boundaries, labels_list = batch["sentence_boundaries"], batch["labels"]
                logits_list = self.model(input_ids, attention_mask, boundaries)

                batch_loss, doc_count = 0.0, 0
                for i, logits in enumerate(logits_list):
                    if logits is None or logits.size(0) == 0:
                        continue
                    labels_i = labels_list[i].to(self.device)
                    s = min(logits.size(0), labels_i.size(0))
                    batch_loss += self.criterion(logits[:s], labels_i[:s])
                    doc_count += 1
                    preds = torch.argmax(logits[:s], dim=1).cpu().numpy()
                    labs = labels_i[:s].cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labs)

                if doc_count:
                    total_loss += (batch_loss / doc_count).item()
                    steps += 1

        loss = total_loss / steps if steps else 0.0
        metrics = (compute_metrics_from_arrays(all_labels, all_preds)
                   if all_preds else compute_metrics_from_arrays([], []))
        return {"loss": loss, **metrics}

    def save_model(self, save_path) -> Path:
        save_dir = Path(save_path)
        if save_dir.suffix:
            save_dir = save_dir.parent
        save_dir.mkdir(parents=True, exist_ok=True)

        metadata = {
            "best_f1": float(self.best_f1),
            "epochs_trained": self.current_epoch,
            "learning_rate": self.lr,
            "batch_size": self.batch_size,
            "device": self.device,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        self.model.save_pretrained(save_dir, metadata=metadata)
        if self.tokenizer:
            self.tokenizer.save_pretrained(save_dir)
        return save_dir
