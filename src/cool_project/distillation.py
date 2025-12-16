"""
Distillation-only training entrypoint.

Assumes you already generated teacher embeddings using the extraction script, saved as:
  <...>/embeddings/train_embeddings.pt
  <...>/embeddings/val_embeddings.pt
  <...>/embeddings/test_embeddings.pt

Each file should be a torch-saved dict with keys:
  - "embeddings": FloatTensor [N, D]
  - "image_ids": list[str] length N
  - "labels": (optional) ignored

This script wraps the base dataset so each __getitem__ returns:
  (pixel_values, teacher_embedding)

IMPORTANT:
- Your LightningModel must implement distillation logic when batch[1] is an embedding tensor.
"""

import json
import os
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import lightning as L
from torch.utils.data import Dataset, DataLoader
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping
from cool_project.dataloader import DistillDatasetById, load_teacher_id_map
from cool_project.backbones_heads.models import build_model_from_config
from cool_project.lightning_model_distill import LightningDistillModel
from cool_project.lightning_model_distill import LightningDistillModel
from cool_project.data_domain import create_domain_loaders, DOMAIN_DATASET_MAP


logger = logging.getLogger(__name__)


# ----------------------------
# Utilities
# ----------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


def setup_logging(seed_dir: Path):
    seed_dir.mkdir(parents=True, exist_ok=True)
    log_file = seed_dir / "train_distill.log"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root_logger.addHandler(ch)

    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)

    logger.info(f"[distill] Logging to {log_file}")



# ----------------------------
# Main distillation training
# ----------------------------
def run_distillation(config_path: str):
    cfg = load_config(config_path)

    exp_name = cfg.get("experiment_name", "distill_experiment")
    seed = cfg["train"].get("seed", 42)
    L.seed_everything(seed)

    data_cfg = cfg["data"]
    domain = data_cfg["domain"]

    dataset_name = data_cfg.get("dataset_name") or DOMAIN_DATASET_MAP.get(domain)
    if dataset_name is None:
        raise ValueError(f"Unknown domain '{domain}'. Please add it to DOMAIN_DATASET_MAP or set data.dataset_name")

    # Output dir layout matches your project convention
    dataset_root = Path(dataset_name)
    out_dir = Path(cfg.get("output_dir", "runs"))
    exp_root = out_dir / dataset_root / exp_name
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    seed_dir = exp_root / f"seed_{seed}" / f"job_{job_id}"
    #seed_dir = exp_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(seed_dir)
    logger.info(f"[distill] Domain={domain} dataset_name={dataset_name}")
    logger.info(f"[distill] seed_dir={seed_dir}")

    # Teacher embeddings path (directory)
    teacher_cfg = cfg.get("teacher", {})
    embeddings_dir = teacher_cfg.get("embeddings_dir", None)
    if embeddings_dir is None:
        raise ValueError(
            "Distillation-only script requires teacher.embeddings_dir in config "
            "(directory containing train_embeddings.pt etc.)."
        )
    embeddings_dir = Path(embeddings_dir)

    # Build base loaders (we only need images; labels ignored)
    base_loaders = create_domain_loaders(
        domain=domain,
        dataset_name=dataset_name,
        resolution=data_cfg["resolution"],
        batch_size=data_cfg.get("batch_size", 32),
        num_workers=data_cfg.get("num_workers", 4),
        balanced_train=data_cfg.get("balanced_train", False),
        val_split=data_cfg.get("val_split", 0.0),
        return_one_hot=False,
    )

    # Wrap datasets with teacher embeddings
    def wrap_split(split: str, base_loader: Optional[DataLoader], shuffle: bool) -> Optional[DataLoader]:
        if base_loader is None:
            return None
        id_to_emb = load_teacher_id_map(embeddings_dir, split)
        wrapped_ds = DistillDatasetById(base_loader.dataset, id_to_emb)
        return DataLoader(
            wrapped_ds,
            batch_size=data_cfg.get("batch_size", 32),
            num_workers=data_cfg.get("num_workers", 4),
            shuffle=shuffle,
            pin_memory=True,
            drop_last=False,
        )

    train_loader = wrap_split("train", base_loaders.get("train"), shuffle=True)
    val_loader   = wrap_split("val",   base_loaders.get("val"),   shuffle=False)
    test_loader  = wrap_split("test",  base_loaders.get("test"),  shuffle=False)

    # Quick sanity check
    xb, tb = next(iter(train_loader))
    logger.info(f"[distill] First batch x.shape={getattr(xb, 'shape', None)} teacher.shape={getattr(tb, 'shape', None)}")

    # Build student model
    student_model = build_model_from_config(cfg)

    # Lightning wrapper config (keep minimal)
    exp_cfg = cfg["train"].copy()
    exp_cfg["task"] = "distillation"
    exp_cfg["experiment_name"] = exp_name
    exp_cfg["output_dir"] = str(seed_dir)
    #exp_cfg["distill_on"] = True  # make it explicit for LightningModel
    exp_cfg["teacher"] = cfg["teacher"]

    lit_model = LightningDistillModel(student_model, exp_cfg)

    # Loggers
    tb_logger = TensorBoardLogger(save_dir=str(seed_dir), name="", version="")
    csv_logger = CSVLogger(save_dir=str(seed_dir), name="", version="")

    ckpt_dir = seed_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks = []
    if val_loader is not None:
        callbacks.append(ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best",
            monitor="val_distill_loss",
            mode="min",
            save_top_k=1,
        ))
        callbacks.append(EarlyStopping(
            monitor="val_distill_loss",
            mode="min",
            patience=exp_cfg.get("early_stopping_patience", 5),
        ))
    else:
        callbacks.append(ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="last",
            save_last=True,
            save_top_k=0,
        ))

    # Hardware (same pattern you used)
    num_visible = torch.cuda.device_count()
    num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES", os.environ.get("SLURM_NNODES", "1")))
    use_gpu = num_visible > 0
    strategy = "ddp" if use_gpu and (num_visible > 1 or num_nodes > 1) else "auto"

    trainer_kwargs = dict(
        max_epochs=exp_cfg.get("max_epochs", 10),
        logger=[tb_logger, csv_logger],
        callbacks=callbacks,
        log_every_n_steps=10,
        enable_checkpointing=True,
    )

    if use_gpu:
        trainer_kwargs.update(accelerator="gpu", devices=num_visible, num_nodes=num_nodes, strategy=strategy)
    else:
        trainer_kwargs.update(accelerator="cpu", devices=1, num_nodes=1)

    trainer = L.Trainer(**trainer_kwargs)

    logger.info("[distill] Starting distillation training...")
    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if test_loader is not None:
        logger.info("[distill] Starting test...")
        trainer.test(lit_model, dataloaders=test_loader)

    logger.info("[distill] Done.")

