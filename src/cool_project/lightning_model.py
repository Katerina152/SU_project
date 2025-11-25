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

        # Metric name for logging
        if self.task == "single_label_classification":
            self.metric_name = "accuracy"
        elif self.task == "multi_label_classification":
            self.metric_name = "accuracy"
        elif self.task == "regression":
            self.metric_name = "rmse"
        else:
            raise ValueError(f"Unsupported task: {self.task}")

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

        if self.task == "single_label_classification":
            preds = torch.argmax(logits, dim=1)
            batch_acc = (preds == labels).float().mean()
            self.log(
                f"train_{self.metric_name}",
                batch_acc,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

            # Top-k accuracy
            num_classes = logits.size(1)
            if num_classes > 2 and self.top_k > 1:
                k = min(self.top_k, num_classes)
                topk = torch.topk(logits, k=k, dim=1).indices  # [B, k]
                correct_topk = (topk == labels.unsqueeze(1)).any(dim=1).float().mean()
                self.log(
                    f"train_top{k}_accuracy",
                    correct_topk,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                )

        elif self.task == "multi_label_classification":
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            batch_acc = (preds == labels).float().mean()
            self.log("train_accuracy", batch_acc, prog_bar=True, on_epoch=True)

        elif self.task == "regression":
            batch_rmse = torch.sqrt(F.mse_loss(logits.squeeze(), labels.squeeze()))
            self.log(
                f"train_{self.metric_name}",
                batch_rmse,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

        return loss

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

        if self.task == "single_label_classification":
            preds = torch.argmax(logits, dim=1)
            batch_acc = (preds == labels).float().mean()
            self.log(
                f"val_{self.metric_name}",
                batch_acc,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

            num_classes = logits.size(1)
            if num_classes > 2 and self.top_k > 1:
                k = min(self.top_k, num_classes)
                topk = torch.topk(logits, k=k, dim=1).indices
                correct_topk = (topk == labels.unsqueeze(1)).any(dim=1).float().mean()
                self.log(
                    f"val_top{k}_accuracy",
                    correct_topk,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                )

        elif self.task == "multi_label_classification":
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            batch_acc = (preds == labels).float().mean()
            self.log("val_accuracy", batch_acc, prog_bar=True, on_epoch=True)

        elif self.task == "regression":
            batch_rmse = torch.sqrt(F.mse_loss(logits.squeeze(), labels.squeeze()))
            self.log(
                f"val_{self.metric_name}",
                batch_rmse,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

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

        if self.task == "single_label_classification":
            preds = torch.argmax(logits, dim=1)
            batch_acc = (preds == labels).float().mean()
            self.log(
                f"test_{self.metric_name}",
                batch_acc,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

            num_classes = logits.size(1)
            if num_classes > 2 and self.top_k > 1:
                k = min(self.top_k, num_classes)
                topk = torch.topk(logits, k=k, dim=1).indices
                correct_topk = (topk == labels.unsqueeze(1)).any(dim=1).float().mean()
                self.log(
                    f"test_top{k}_accuracy",
                    correct_topk,
                    prog_bar=False,
                    on_step=False,
                    on_epoch=True,
                )

        elif self.task == "multi_label_classification":
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            batch_acc = (preds == labels).float().mean()
            self.log("test_accuracy", batch_acc, prog_bar=True, on_epoch=True)

        elif self.task == "regression":
            batch_rmse = torch.sqrt(F.mse_loss(logits.squeeze(), labels.squeeze()))
            self.log(
                f"test_{self.metric_name}",
                batch_rmse,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

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
