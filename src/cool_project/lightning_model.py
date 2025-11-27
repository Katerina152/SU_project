import lightning as L 
import time 
import torch
import os 
from lightning.pytorch.callbacks import BasePredictionWriter
from cool_project.backbones_heads import custom_loss # is this okk or but specific functions?
import numpy as np 
import random
import torch.nn.functional as F 
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from typing import Dict
from pathlib import Path
from .metrics import build_metrics_for_task
import psutil
#from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler
#from lightning.pytorch.strategies import DeepSpeedStrategy

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
        
        self.save_hyperparameters(cfg)

        def _log_memory(self, tag: str):
            process = psutil.Process(os.getpid())
            mem_gb = process.memory_info().rss / 1024**3
            self.log(f"{tag}_cpu_memory_gb", mem_gb, prog_bar=False, on_step=False, on_epoch=True)

            if torch.cuda.is_available():
                gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
                self.log(f"{tag}_gpu_memory_gb", gpu_mem, prog_bar=False, on_step=False, on_epoch=True)
                # optional: reset peak stats so next epoch is fresh
                torch.cuda.reset_peak_memory_stats()

        # FLOPs profiling config
        self.profile_flops = bool(cfg.get("profile_flops", False))
        # Which batch to profile (e.g. 10th batch)
        self.flops_batch = int(cfg.get("flops_batch", 10))
        self.profiler = None
        self._flops_profiled = False  

        
        # Let the model be the source of truth for task
        self.task = getattr(self.model, "task", cfg.get("task", "single_label_classification"))
        self.top_k = int(cfg.get("top_k", 5))


        # Optimizer hyperparameters
        self.lr = cfg.get("lr", 1e-4)
        self.weight_decay = cfg.get("weight_decay", 0.0)
        

        # Where to save outputs
        self.exp_name = cfg.get("experiment_name", "default_exp")
        self.output_dir = Path(cfg.get("output_dir", "runs")) 
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
        

        #process = psutil.Process(os.getpid())
        #mem_gb = process.memory_info().rss / 1024**3
        #self.log("cpu_memory_gb", mem_gb, prog_bar=True)

        #if torch.cuda.is_available():
            #gpu_mem = torch.cuda.memory_allocated() / 1024**3
            #self.log("gpu_memory_gb", gpu_mem)


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

        #log memory at end of epoch
        self._log_memory("train")


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

        # --- update torchmetrics ---
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
        for name, metric in self.val_metrics.items():
            value = metric.compute()
            self.log(f"eval_{name}", value, prog_bar=(name == "accuracy"), on_epoch=True)
            metric.reset()
        
        # log memory at end of epoch
        self._log_memory("eval")




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

        # --- update torchmetrics ---
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
            self.log(f"test_{name}", value, prog_bar=(name == "accuracy"), on_epoch=True)
            metric.reset()
        
        # log memory at end of epoch
        self._log_memory("test")

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

    def on_fit_start(self):
        if not self.profile_flops:
            return

        if not HAVE_DEEPSPEED or FlopsProfiler is None:
            print("[LightningModel] profile_flops=True but DeepSpeed/FlopsProfiler not available. Skipping FLOPs profiling.")
            return

        # If using DeepSpeedStrategy, pass its engine, else just pass the model
        strategy = getattr(self.trainer, "strategy", None)
        if isinstance(strategy, DeepSpeedStrategy):
            self.profiler = FlopsProfiler(self, ds_engine=strategy.model)
        else:
            self.profiler = FlopsProfiler(self.model)

    def on_train_batch_start(self, batch, batch_idx):
        if not self.profile_flops or self.profiler is None or self._flops_profiled:
            return

        if batch_idx == self.flops_batch:
            self.profiler.start_profile()
            self._batch_profile_start_time = time.time()


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