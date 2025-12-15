import lightning as L
import time
import torch
import os
import psutil
from typing import Dict
from pathlib import Path

from fvcore.nn import FlopCountAnalysis
from cool_project.backbones_heads.custom_loss import EmbeddingDistillationLoss

try:
    from lightning.pytorch.strategies import DeepSpeedStrategy
    from deepspeed.profiling.flops_profiler.profiler import FlopsProfiler
    HAVE_DEEPSPEED = True
except ImportError:
    DeepSpeedStrategy = None
    FlopsProfiler = None
    HAVE_DEEPSPEED = False


class LightningDistillModel(L.LightningModule):
    """
    Distillation-only LightningModule.

    Expects batches like:
      batch[0] = pixel_values: Tensor [B, C, H, W]
      batch[1] = teacher_embs: Tensor [B, D]

    Underlying model is expected to return an object with:
      - embeddings: Tensor [B, D]
    """

    def __init__(self, model: torch.nn.Module, cfg: Dict):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.save_hyperparameters(cfg)

        teacher_dim = int(cfg["teacher"]["embedding_dim"])   # add this to config
        student_dim = int(getattr(model, "embed_dim"))

        self.proj = torch.nn.Linear(student_dim, teacher_dim) if student_dim != teacher_dim else torch.nn.Identity()
        train_cfg = cfg.get("train", {})
        self.distill_loss = EmbeddingDistillationLoss(
            lamda_feat=train_cfg.get("lamda_feat", 1.0),
            lamda_cos=train_cfg.get("lamda_cos", 1.0),
        )

        # Optimizer hyperparameters
        self.lr = cfg.get("lr", 1e-4)
        self.weight_decay = cfg.get("weight_decay", 0.0)

        # Where to save outputs (logs, flops files, etc.)
        self.exp_name = cfg.get("experiment_name", "default_exp")
        self.output_dir = Path(cfg.get("output_dir", "runs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # FLOPs profiling config
        self.profile_flops = bool(cfg.get("profile_flops", False))
        self.flops_batch = int(cfg.get("flops_batch", 10))
        self.profiler = None
        self._flops_profiled = False

    def _log_memory(self, tag: str):
        process = psutil.Process(os.getpid())
        mem_gb = process.memory_info().rss / 1024**3
        self.log(f"{tag}_cpu_memory_gb", mem_gb, prog_bar=False, on_step=False, on_epoch=True)

        if torch.cuda.is_available():
            gpu_mem = torch.cuda.max_memory_allocated() / 1024**3
            self.log(f"{tag}_gpu_memory_gb", gpu_mem, prog_bar=False, on_step=False, on_epoch=True)
            torch.cuda.reset_peak_memory_stats()

    def forward(self, pixel_values, **kwargs):
        # labels=None ensures we don't compute supervised losses accidentally
        return self.model(pixel_values=pixel_values, labels=None, **kwargs)

    def training_step(self, batch, batch_idx):
        pixel_values, teacher_embs = batch

        # Safe even if Lightning already moved them
        pixel_values = pixel_values.to(self.device, non_blocking=True)
        teacher_embs = teacher_embs.to(self.device, non_blocking=True)

        out = self.model(pixel_values=pixel_values, labels=None)
        #student_embs = out.embeddings
        student_embs = self.proj(out.embeddings)

        loss = self.distill_loss(student_embs, teacher_embs)

        # Use self.print (plays nicer with multi-GPU than print)
        if batch_idx == 0:
            self.print(f"[DISTILL] pixel_values={tuple(pixel_values.shape)} teacher={tuple(teacher_embs.shape)} student={tuple(student_embs.shape)}")

        self.log("train_distill_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values, teacher_embs = batch
        pixel_values = pixel_values.to(self.device, non_blocking=True)
        teacher_embs = teacher_embs.to(self.device, non_blocking=True)

        out = self.model(pixel_values=pixel_values, labels=None)
        #student_embs = out.embeddings
        student_embs = self.proj(out.embeddings)

        loss = self.distill_loss(student_embs, teacher_embs)
        self.log("val_distill_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def test_step(self, batch, batch_idx):
        pixel_values, teacher_embs = batch
        pixel_values = pixel_values.to(self.device, non_blocking=True)
        teacher_embs = teacher_embs.to(self.device, non_blocking=True)

        out = self.model(pixel_values=pixel_values, labels=None)
        #student_embs = out.embeddings
        student_embs = self.proj(out.embeddings)

        loss = self.distill_loss(student_embs, teacher_embs)
        self.log("test_distill_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def on_train_epoch_end(self):
        self._log_memory("train")

    def on_validation_epoch_end(self):
        self._log_memory("val")

    def on_test_epoch_end(self):
        self._log_memory("test")

    def configure_optimizers(self):
        return torch.optim.AdamW(
            (p for p in self.parameters() if p.requires_grad),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

    # ---------------------- FLOPs profiling ----------------------
    def on_fit_start(self):
        if not self.profile_flops:
            return

        if not HAVE_DEEPSPEED or FlopsProfiler is None:
            self.print(
                "[LightningDistillModel] profile_flops=True but DeepSpeed FlopsProfiler not available. "
                "Will use fvcore FLOPs estimation instead."
            )
            self.profiler = None
            return

        strategy = getattr(self.trainer, "strategy", None)
        if DeepSpeedStrategy is not None and isinstance(strategy, DeepSpeedStrategy):
            self.profiler = FlopsProfiler(self, ds_engine=strategy.model)
        else:
            self.profiler = FlopsProfiler(self.model)

    def on_train_batch_start(self, batch, batch_idx):
        if not self.profile_flops or self._flops_profiled:
            return
        if batch_idx != self.flops_batch:
            return

        # If DeepSpeed profiler is available, use it
        if self.profiler is not None:
            self.profiler.start_profile()
            self._batch_profile_start_time = time.time()
            return

        # ---------- fvcore fallback ----------
        # IMPORTANT: don't unpack the batch; distill batch != supervised batch
        pixel_values = batch[0].to(self.device, non_blocking=True)

        self.model.eval()
        with torch.no_grad():
            flops_analysis = FlopCountAnalysis(self.model, pixel_values)
            total_flops = flops_analysis.total()

        gflops = total_flops / 1e9
        self.log("flops_g", gflops, prog_bar=False, on_step=False, on_epoch=True)

        flops_path = self.output_dir / "flops_fvcore.log"
        with open(flops_path, "w") as f:
            f.write(f"Batch index: {batch_idx}\n")
            f.write(f"Batch size: {pixel_values.shape[0]}\n")
            f.write(f"Total FLOPs (this batch): {total_flops}\n")
            f.write(f"GFLOPs (this batch): {gflops:.3f}\n")

        self.print(f"[LightningDistillModel] Saved fvcore FLOPs info to {flops_path}")
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

