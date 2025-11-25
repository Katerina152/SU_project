import lightning as L 
import time 
import torch
import os 
from torchmetrics.classification import BinaryAccuracy
from torchmetrics.regression import MeanSquaredError
from lightning.pytorch.callbacks import BasePredictionWriter
from cool_project.backbones_heads import custom_loss # is this okk or but specific functions?
import numpy as np 
import torch.nn.functional as F 
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from typing import Dict
from pathlib import Path
from torchmetrics.classification import MulticlassAccuracy
from .metrics import build_metrics_for_task


class LightningModel(L.LightningModule):
    def __init__(self, model: torch.nn.Module, cfg: Dict):
        """
        model: VisionTransformerWithHead instance
        cfg:   dict with things like:
               {
                 "task": "single_label_classification" | "multi_label_classification" | "regression",
                 "lr": 1e-4,
                 "weight_decay": 0.0,
                 "experiment_name": "isic_vit_test",
                 "output_dir": "runs",
                 "top_k": 5
               }
        """
        super().__init__()
        self.model = model
        self.cfg = cfg

        
        # Let the model be the source of truth for task
        self.task = getattr(self.model, "task", cfg.get("task", "single_label_classification"))
        self.top_k = int(cfg.get("top_k", 5))

        # Optimizer hyperparameters
        self.lr = cfg.get("lr", 1e-4)
        self.weight_decay = cfg.get("weight_decay", 0.0)

        # Where to save outputs
        self.exp_name = cfg.get("experiment_name", "default_exp")
        self.output_dir = Path(cfg.get("output_dir", "runs")) / self.exp_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- NEW: metrics from your helper ---
        num_classes = getattr(self.model, "num_classes", cfg.get("num_classes", None))
        self.num_classes = num_classes

        self.train_metrics = build_metrics_for_task(
            task=self.task,
            num_classes=num_classes,
            top_k=self.top_k,
        )
        # clone for val / test so they have independent states
        self.val_metrics = {k: m.clone() for k, m in self.train_metrics.items()}
        self.test_metrics = {k: m.clone() for k, m in self.train_metrics.items()}

    # ----------------------
    # Forward
    # ----------------------
    def forward(self, pixel_values, labels=None, **kwargs):
        """
        We expect the underlying model (VisionTransformerWithHead)
        to return an object with fields:
          - loss
          - logits
          - embeddings
        """
        return self.model(pixel_values=pixel_values, labels=labels, **kwargs)

    # ----------------------
    # Training Step
    # ----------------------
    def training_step(self, batch, batch_idx):
        pixel_values, labels = batch
        pixel_values = pixel_values.to(self.device)
        labels = labels.to(self.device)

        out = self(pixel_values, labels=labels)
        loss = out.loss
        logits = out.logits

        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # --- NEW: update torchmetrics ---
        if self.task == "single_label_classification":
            for name, metric in self.train_metrics.items():
                # For MulticlassAccuracy / top-k, passing logits is fine
                metric.update(logits, labels)

        elif self.task in ["multi_label_classification", "multi_label", "multilabel"]:
            probs = torch.sigmoid(logits)
            for name, metric in self.train_metrics.items():
                metric.update(probs, labels)

        elif self.task == "regression":
            for name, metric in self.train_metrics.items():
                metric.update(logits.squeeze(), labels.squeeze())

        return loss

    def on_train_epoch_end(self):
        for name, metric in self.train_metrics.items():
            value = metric.compute()
            self.log(f"train_{name}", value, prog_bar=(name == "accuracy"), on_epoch=True)
            metric.reset()


    # ----------------------
    # Validation Step
    # ----------------------
    def validation_step(self, batch, batch_idx):
        pixel_values, labels = batch
        pixel_values = pixel_values.to(self.device)
        labels = labels.to(self.device)

        out = self(pixel_values, labels=labels)
        loss = out.loss
        logits = out.logits

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # --- NEW: update torchmetrics ---
        if self.task == "single_label_classification":
            for name, metric in self.val_metrics.items():
                # For MulticlassAccuracy / top-k, passing logits is fine
                metric.update(logits, labels)

        elif self.task in ["multi_label_classification", "multi_label", "multilabel"]:
            probs = torch.sigmoid(logits)
            for name, metric in self.val_metrics.items():
                metric.update(probs, labels)

        elif self.task == "regression":
            for name, metric in self.val_metrics.items():
                metric.update(logits.squeeze(), labels.squeeze())

        return loss
    
    def on_val_epoch_end(self):
        for name, metric in self.train_metrics.items():
            value = metric.compute()
            self.log(f"train_{name}", value, prog_bar=(name == "accuracy"), on_epoch=True)
            metric.reset()



    # ----------------------
    # Test Step
    # ----------------------
    def test_step(self, batch, batch_idx):
        pixel_values, labels = batch
        pixel_values = pixel_values.to(self.device)
        labels = labels.to(self.device)

        out = self(pixel_values, labels=labels)
        loss = out.loss
        logits = out.logits

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # --- NEW: update torchmetrics ---
        if self.task == "single_label_classification":
            for name, metric in self.test_metrics.items():
                # For MulticlassAccuracy / top-k, passing logits is fine
                metric.update(logits, labels)

        elif self.task in ["multi_label_classification", "multi_label", "multilabel"]:
            probs = torch.sigmoid(logits)
            for name, metric in self.test_metrics.items():
                metric.update(probs, labels)

        elif self.task == "regression":
            for name, metric in self.test_metrics.items():
                metric.update(logits.squeeze(), labels.squeeze())

        return loss
    
    def on_test_epoch_end(self):
        for name, metric in self.test_metrics.items():
            value = metric.compute()
            self.log(f"train_{name}", value, prog_bar=(name == "accuracy"), on_epoch=True)
            metric.reset()

    # ----------------------
    # Save embeddings
    # ----------------------
    def save_embeddings(self, dataloader, filename: str = "embeddings.pt"):
        self.eval()
        all_embs = []

        with torch.no_grad():
            for batch in dataloader:
                pixel_values, _ = batch
                pixel_values = pixel_values.to(self.device)

                out = self.model(pixel_values=pixel_values)
                all_embs.append(out.embeddings.cpu())

        final = torch.cat(all_embs, dim=0)
        save_path = self.output_dir / filename
        torch.save(final, save_path)
        print(f"[LightningModel] Saved embeddings to {save_path}")

    # ----------------------
    # Optimizer
    # ----------------------
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            (p for p in self.parameters() if p.requires_grad),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )
        return optimizer
