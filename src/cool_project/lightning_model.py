import lightning as L 
import time 
import torch
import os 
from lightning.pytorch.callbacks import BasePredictionWriter
from cool_project.backbones_heads import custom_loss 
import numpy as np 
import random
import torch.nn.functional as F 
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from typing import Dict
from pathlib import Path
from .metrics import build_metrics_for_task
import psutil
from fvcore.nn import FlopCountAnalysis
from cool_project.backbones_heads.custom_loss import EmbeddingDistillationLoss
import torch.nn as nn


try:
    from deepspeed.ops.adam import DeepSpeedCPUAdam
    from lightning.pytorch.strategies import DeepSpeedStrategy
    from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler
    HAVE_DEEPSPEED = True
except ImportError:
    DeepSpeedCPUAdam = None
    DeepSpeedStrategy = None
    FlopsProfiler = None
    HAVE_DEEPSPEED = False

class LightningModel(L.LightningModule):
    def __init__(self, model: torch.nn.Module, cfg: Dict, distill_on: bool = False):
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
        self.distill_on = distill_on
        
        self.save_hyperparameters(cfg)

        if self.distill_on:
            self.distill_loss = EmbeddingDistillationLoss(
                lamda_feat=cfg.get("lamda_feat", 1.0),
                lamda_cos=cfg.get("lamda_cos", 1.0),
            )

        # FLOPs profiling config
        self.profile_flops = bool(cfg.get("profile_flops", False))

        # Which batch to profile (e.g. 10th batch)
        self.flops_batch = int(cfg.get("flops_batch", 10))
        self.profiler = None
        self._flops_profiled = False  

        
        # Let the model be the source of truth for task
        self.task = getattr(self.model, "task", cfg.get("task", "single_label_classification"))
        if self.task in ["single_label_classification"]:
            self.top_k = int(cfg.get("top_k", 5))
        else:
            # segmentation, multilabel, regression usually don’t use top-k
            self.top_k = None

        self.loss_type = getattr(self.model, "loss_type", cfg.get("loss_type", "auto"))
        self.class_weights = getattr(self.model, "class_weights", cfg.get("class_weights", None))
        print("[LightningModel] class_weights:", None if self.class_weights is None else self.class_weights)

        # ---- DEBUG: init sanity ----
        print("[DEBUG:init] task =", self.task, "loss_type =", self.loss_type)
        print("[DEBUG:init] num_classes =", getattr(self.model, "num_classes", None))
        print("[DEBUG:init] cfg keys include class_weights?", "class_weights" in cfg)
        print("[DEBUG:init] cfg.class_weights type:", type(cfg.get("class_weights", None)))


        # check trainable params (useful to know if backbone frozen)
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"[DEBUG:init] trainable params = {trainable:,} / {total:,}")

        # check class weight sanity (if present)
        if self.class_weights is not None:
            w = torch.as_tensor(self.class_weights).float()
            print("[DEBUG:init] class_weights (cpu) =", w.detach().cpu().tolist())
            print("[DEBUG:init] class_weights min/max/mean =", float(w.min()), float(w.max()), float(w.mean()))
            assert torch.isfinite(w).all(), "class_weights has NaN/Inf"
            assert float(w.max()) > 1.0 or float(w.min()) < 1.0, "class_weights look uninformative (all ~1?)"
            assert float(w.min()) > 1e-4, "class_weights are tiny (bug: inverse-counts not scaled?)"



        # Optimizer hyperparameters
        self.lr = cfg.get("lr", 1e-4)
        self.weight_decay = cfg.get("weight_decay", 0.0)
        

        # Where to save outputs
        self.exp_name = cfg.get("experiment_name", "default_exp")
        self.output_dir = Path(cfg.get("output_dir", "runs")) 
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # --- metrics from your helper ---
        num_classes = getattr(self.model, "num_classes", cfg.get("num_classes", None))
        self.num_classes = num_classes

        self.train_metrics = build_metrics_for_task(
            task=self.task,
            num_classes=num_classes,
            top_k=self.top_k,
        )
        self.val_metrics = {k: m.clone() for k, m in self.train_metrics.items()}
        self.test_metrics = {k: m.clone() for k, m in self.train_metrics.items()}

    
    def _log_memory(self, tag: str):
        process = psutil.Process(os.getpid())
        mem_gb = process.memory_info().rss / 1024**3
        self.log(f"{tag}_cpu_memory_gb", mem_gb, prog_bar=False, on_step=False, on_epoch=True)

        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
            self.log(f"{tag}_gpu_memory_gb", gpu_mem, prog_bar=False, on_step=False, on_epoch=True)
            # optional: reset peak stats so next epoch is fresh
            torch.cuda.reset_peak_memory_stats()
    
    def _debug_once(self, tag: str, msg: str):
        flag = f"_did_debug_{tag}"
        if getattr(self, flag, False):
            return
        print(msg)
        setattr(self, flag, True)

    
    def _compute_loss(self, logits, labels):
        if labels is None:
            return None

        # --- segmentation branch ---
        if self.task == "segmentation":
            weight = self.class_weights.to(logits.device) if getattr(self, "class_weights", None) is not None else None
            loss_fn = nn.CrossEntropyLoss(weight=weight)

            if logits.ndim == 4 and labels.ndim == 3 and logits.shape[2:] != labels.shape[-2:]:
                logits = F.interpolate(logits, size=labels.shape[-2:], mode="bilinear", align_corners=False)

            return loss_fn(logits, labels)

        # --- classification/regression ---
        loss_type = self.loss_type
        if loss_type == "auto":
            if self.task == "regression":
                loss_type = "mse"
            elif self.task == "single_label_classification":
                loss_type = "ce"
            elif self.task == "multi_label_classification":
                loss_type = "bce"
            else:
                raise ValueError(f"Unknown task: {self.task}")

        if loss_type == "mse":
            return nn.MSELoss()(logits, labels)

        if loss_type == "ce":
            weight = self.class_weights.to(logits.device) if getattr(self, "class_weights", None) is not None else None
            loss_fn = nn.CrossEntropyLoss(weight=weight)
            return loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))

        if loss_type == "bce":
            return nn.BCEWithLogitsLoss()(logits, labels)

        if loss_type == "custom":
            raise ValueError("loss_type='custom' is not supported in this logits-only LightningModel.")

        raise ValueError(f"Unknown loss_type: {self.loss_type}")
    
    def _update_classification_metrics(self, metric_dict, logits, labels):
        preds = torch.argmax(logits, dim=1)  # [B]
        C = logits.size(-1)

        probs = None
        probs_pos = None
        if C == 2:
            probs_pos = torch.softmax(logits, dim=1)[:, 1]  # [B]
        else:
            probs = torch.softmax(logits, dim=1)            # [B, C]

        for name, metric in metric_dict.items():
            n = name.lower()
            if n == "auroc":
                if C == 2:
                    metric.update(probs_pos, labels)        # [B]
                else:
                    metric.update(probs, labels)            # [B, C]
            elif n.startswith("top"):                       # e.g. "top5_accuracy"
                metric.update(logits, labels)               # [B, C]
            else:
                metric.update(preds, labels)                # [B]


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

        pixel_values, labels, image_ids = batch
        pixel_values = pixel_values.to(self.device)
        labels = labels.to(self.device)

        # ---- DEBUG: first batch sanity ----
        self._debug_once(
            "train_batch0",
            (
                f"[DEBUG:train_batch0] pixel_values={tuple(pixel_values.shape)} dtype={pixel_values.dtype} device={pixel_values.device}\n"
                f"[DEBUG:train_batch0] labels={tuple(labels.shape)} dtype={labels.dtype} device={labels.device} "
                f"min={int(labels.min())} max={int(labels.max())}\n"
                f"[DEBUG:train_batch0] configured num_classes={self.num_classes}"
            ),
        )

        # labels must be int64 for CE and in range
        if self.task == "single_label_classification":
            assert labels.dtype in (torch.int64, torch.long), f"CE expects int64 labels, got {labels.dtype}"
            assert labels.min() >= 0, "negative label found"
            if self.num_classes is not None:
                assert labels.max() < self.num_classes, f"label out of range: max={int(labels.max())} num_classes={self.num_classes}"


        #out = self(pixel_values, labels=labels)
        out = self(pixel_values)
        loss = self._compute_loss(out.logits, labels)
        logits = out.logits

        self._debug_once(
            "train_forward0",
            (
                f"[DEBUG:train_forward0] logits={tuple(logits.shape)} dtype={logits.dtype} device={logits.device}\n"
                f"[DEBUG:train_forward0] loss={float(loss.detach().cpu())}\n"
                f"[DEBUG:train_forward0] model.task={self.task} loss_type={self.loss_type}"
            ),
        )

        # logits must match num_classes for single-label classification
        if self.task == "single_label_classification" and self.num_classes is not None:
            assert logits.ndim == 2, f"expected [B,C] logits, got shape {tuple(logits.shape)}"
            assert logits.size(1) == self.num_classes, f"logits C={logits.size(1)} != num_classes={self.num_classes}"

        # confirm CE is actually using weights (prints once)
        if self.task == "single_label_classification":
            w = self.class_weights
            self._debug_once(
                "loss_weight_check",
                "[DEBUG:loss_weight_check] CE weight is "
                + ("None" if w is None else str(torch.as_tensor(w).detach().cpu().tolist())),
            )


        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        
        # --- update torchmetrics ---
        if self.task == "single_label_classification":
            self._update_classification_metrics(self.train_metrics, logits, labels)


        elif self.task in ["multi_label_classification", "multi_label", "multilabel"]:
            probs = torch.sigmoid(logits)
            for name, metric in self.train_metrics.items():
                metric.update(probs, labels)

        elif self.task == "regression":
            for name, metric in self.train_metrics.items():
                metric.update(logits.squeeze(), labels.squeeze())
        
        elif self.task == "segmentation":
            # logits: [B, C, h, w], labels: [B, H, W]
            if logits.ndim == 4 and labels.ndim == 3 and logits.shape[2:] != labels.shape[-2:]:
                logits_for_metrics = F.interpolate(
                    logits,
                    size=labels.shape[-2:],  # (H, W), e.g. (224, 224)
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                logits_for_metrics = logits

            preds = torch.argmax(logits_for_metrics, dim=1)  # [B, H, W]

            for name, metric in self.train_metrics.items():
                metric.update(preds, labels)


        return loss

    def on_train_epoch_end(self):

        for name, metric in self.train_metrics.items():
            value = metric.compute()
            self.log(f"train_{name}", value, prog_bar=(name == "accuracy"), on_epoch=True)
            metric.reset()

        self._log_memory("train")


    # ----------------------
    # Validation Step
    # ----------------------
    def validation_step(self, batch, batch_idx):
        
        pixel_values, labels, image_ids = batch
        pixel_values = pixel_values.to(self.device)
        labels = labels.to(self.device)

        #out = self(pixel_values, labels=labels)
        out = self(pixel_values)
        loss = self._compute_loss(out.logits, labels)
        logits = out.logits

        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # --- update torchmetrics ---
        if self.task == "single_label_classification":
            self._update_classification_metrics(self.val_metrics, logits, labels)
        
        elif self.task in ["multi_label_classification", "multi_label", "multilabel"]:
            probs = torch.sigmoid(logits)
            for name, metric in self.val_metrics.items():
                metric.update(probs, labels)

        elif self.task == "regression":
            for name, metric in self.val_metrics.items():
                metric.update(logits.squeeze(), labels.squeeze())
        
        elif self.task == "segmentation":
            if logits.ndim == 4 and labels.ndim == 3 and logits.shape[2:] != labels.shape[-2:]:
                logits_for_metrics = F.interpolate(
                    logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                logits_for_metrics = logits

            preds = torch.argmax(logits_for_metrics, dim=1)  # [B, H, W]

            for name, metric in self.val_metrics.items():
                metric.update(preds, labels)

        return loss
    
    def on_val_epoch_end(self):

        for name, metric in self.val_metrics.items():
            value = metric.compute()
            self.log(f"eval_{name}", value, prog_bar=(name == "accuracy"), on_epoch=True)
            metric.reset()

        self._log_memory("eval")



    # ----------------------
    # Test Step
    # ----------------------
    def test_step(self, batch, batch_idx):

        pixel_values, labels, image_ids = batch
        pixel_values = pixel_values.to(self.device)
        labels = labels.to(self.device)

        #out = self(pixel_values, labels=labels)
        out = self(pixel_values)
        loss = self._compute_loss(out.logits, labels)
        logits = out.logits

        self.log("test_loss", loss, prog_bar=True, on_step=False, on_epoch=True)

        # ✅ ADD THIS BLOCK RIGHT HERE
        if self.task == "single_label_classification" and batch_idx == 0:
            preds = torch.argmax(logits, dim=1)
            print("[DEBUG:test_pred_stats0] true counts:",
                torch.bincount(labels, minlength=logits.size(1)).detach().cpu().tolist())
            print("[DEBUG:test_pred_stats0] pred counts:",
                torch.bincount(preds, minlength=logits.size(1)).detach().cpu().tolist())


        # --- update torchmetrics ---
        if self.task == "single_label_classification":
            self._update_classification_metrics(self.test_metrics, logits, labels)

        elif self.task in ["multi_label_classification", "multi_label", "multilabel"]:
            probs = torch.sigmoid(logits)
            for name, metric in self.test_metrics.items():
                metric.update(probs, labels)

        elif self.task == "regression":
            for name, metric in self.test_metrics.items():
                metric.update(logits.squeeze(), labels.squeeze())
        
        elif self.task == "segmentation":
            if logits.ndim == 4 and labels.ndim == 3 and logits.shape[2:] != labels.shape[-2:]:
                logits_for_metrics = F.interpolate(
                    logits,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            else:
                logits_for_metrics = logits

            preds = torch.argmax(logits_for_metrics, dim=1)  # [B, H, W]

            for name, metric in self.test_metrics.items():
                metric.update(preds, labels)


        return loss
    
    def on_test_epoch_end(self):
        for name, metric in self.test_metrics.items():
            value = metric.compute()
            self.log(f"test_{name}", value, prog_bar=(name == "accuracy"), on_epoch=True)
            metric.reset()

        self._log_memory("test")
    

    # ----------------------
    # Save embeddings
    # ----------------------
    def save_embeddings(self, dataloader, filename: str = "embeddings.pt"):
        self.eval()
        all_embs = []

        with torch.no_grad():
            for batch in dataloader:
                pixel_values = batch[0]
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


    def _move_metrics_to_device(self):
        """Ensure all torchmetrics live on the same device as the model."""
        device = self.device
        for metric_dict in [self.train_metrics, self.val_metrics, self.test_metrics]:
            for m in metric_dict.values():
                m.to(device)
    
    class _EmbedOnly(torch.nn.Module):
        """
        Wrapper so fvcore calls the model with the correct signature
        and excludes loss computation.
        """
        def __init__(self, model: torch.nn.Module):
            super().__init__()
            self.model = model

        def forward(self, x: torch.Tensor):
            out = self.model(pixel_values=x, labels=None, return_dict=True)
            return out.embeddings


#===== CHANGE FOR THE NEW OUTPUT ==== TO DO =====
    
    def on_fit_start(self):
        self._move_metrics_to_device()
        if not self.profile_flops:
            return

        if not HAVE_DEEPSPEED or FlopsProfiler is None:
            print(
                "[LightningModel] profile_flops=True but DeepSpeed/FlopsProfiler not "
                "available. Will use fvcore-based FLOPs estimation instead."
            )
            self.profiler = None
            return

        # If using DeepSpeedStrategy, pass its engine, else just pass the model
        strategy = getattr(self.trainer, "strategy", None)
        if isinstance(strategy, DeepSpeedStrategy):
            self.profiler = FlopsProfiler(self, ds_engine=strategy.model)
        else:
            self.profiler = FlopsProfiler(self.model)



    def on_train_batch_start(self, batch, batch_idx):
        if not self.profile_flops or self._flops_profiled:
            return

        if batch_idx != self.flops_batch:
            return

        # ---------------- DeepSpeed path ----------------
        if self.profiler is not None:
            self.profiler.start_profile()
            self._batch_profile_start_time = time.time()
            return

        # ---------------- fvcore fallback ----------------
        # batch = (pixel_values, labels, image_ids)
        pixel_values = batch[0].to(self.device, non_blocking=True)

        embed_model = self._EmbedOnly(self.model).to(self.device)

        was_training = self.model.training
        try:
            self.model.eval()
            with torch.no_grad():
                flops = FlopCountAnalysis(embed_model, pixel_values)
                total_flops = flops.total()
        finally:
            if was_training:
                self.model.train()

        gflops = total_flops / 1e9

        self.log(
            "flops_g",
            gflops,
            prog_bar=False,
            on_step=False,
            on_epoch=True,
        )

        flops_path = self.output_dir / "flops_fvcore.log"
        with open(flops_path, "w") as f:
            f.write(f"Batch index: {batch_idx}\n")
            f.write(f"Batch size: {pixel_values.shape[0]}\n")
            f.write(f"Total FLOPs: {total_flops}\n")
            f.write(f"GFLOPs: {gflops:.3f}\n")

        print(f"[LightningModel] Saved fvcore FLOPs to {flops_path}")
        self._flops_profiled = True



    def on_train_batch_end(self, outputs, batch, batch_idx):
        if not self.profile_flops or self.profiler is None or self._flops_profiled:
            return

        if batch_idx == self.flops_batch:
            self.profiler.end_profile()

            flops_path = self.output_dir / "flops.log"
            self.profiler.print_model_profile(
                profile_step=batch_idx,
                module_depth=2,
                top_modules=3,
                detailed=True,
                output_file=str(flops_path),
            )

            self.profiler.reset()
            self._flops_profiled = True

class LightningEmbeddingExtractor(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def predict_step(self, batch, batch_idx):
        pixel_values, labels, image_ids = batch

        out = self.model(pixel_values=pixel_values)

        return {
            "embeddings": out.embeddings.detach().cpu(),  # [B, D]
            "labels": labels.detach().cpu(),              # [B] or [B, C]
            "image_ids": list(image_ids),                 # list[str], length B
        }
