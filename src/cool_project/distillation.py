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
from lightning.pytorch.callbacks import Callback
from cool_project.dataloader import DistillDatasetById, load_teacher_id_map, distill_two_view_collate, distill_one_view_collate, distill_k_view_collate
from cool_project.backbones_heads.models import build_model_from_config
from cool_project.lightning_model_distill import LightningDistillModel
from cool_project.lightning_model_distill import LightningDistillModel
from cool_project.data_domain import create_domain_loaders, DOMAIN_DATASET_MAP
from cool_project.preprocessing import build_pipeline_for_model, build_aug_pipeline_for_model



logger = logging.getLogger(__name__)


# ----------------------------
# Utilities
# ----------------------------
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return json.load(f)


class StopReasonCallback(Callback):
    def on_fit_end(self, trainer, pl_module):
        # epoch info
        max_epochs = trainer.max_epochs
        ended_epoch = trainer.current_epoch + 1  # current_epoch is 0-based

        logger.info(f"[distill] FIT_END: ended_epoch={ended_epoch} max_epochs={max_epochs}")
        logger.info(f"[distill] FIT_END: global_step={trainer.global_step}")
        logger.info(f"[distill] FIT_END: should_stop={trainer.should_stop} interrupted={trainer.interrupted}")

        # early stopping info (if present)
        es = next((cb for cb in trainer.callbacks if isinstance(cb, EarlyStopping)), None)
        if es is not None:
            logger.info(
                f"[distill] FIT_END: EarlyStopping monitor={es.monitor} mode={es.mode} "
                f"patience={es.patience} stopped_epoch={getattr(es,'stopped_epoch',None)} "
                f"wait_count={getattr(es,'wait_count',None)} best_score={getattr(es,'best_score',None)}"
            )

        # checkpoint info (best model)
        ckpt = next((cb for cb in trainer.callbacks if isinstance(cb, ModelCheckpoint)), None)
        if ckpt is not None:
            logger.info(
                f"[distill] FIT_END: Checkpoint monitor={ckpt.monitor} best_model_score={ckpt.best_model_score} "
                f"best_model_path={ckpt.best_model_path}"
            )


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
        model_type=cfg["model"].get("type", "vit").lower(),
        backbone_type=cfg["model"].get("backbone", {}).get("type", None),
        model_name=cfg["model"].get("backbone", {}).get("model_name", None),
    )

    def wrap_split(split: str, base_loader, shuffle: bool):
        if base_loader is None:
            return None

        id_to_emb = load_teacher_id_map(embeddings_dir, split)

        model_type = cfg["model"].get("type", "vit").lower()
        backbone_type = cfg["model"].get("backbone", {}).get("type", None)
        model_name = cfg["model"].get("backbone", {}).get("model_name", None)

        base_tf = build_pipeline_for_model(
            model_type=model_type,
            size=data_cfg["resolution"],
            mode="train" if split == "train" else "test",
            backbone_type=backbone_type,
            model_name=model_name,
        )

        if split == "train":
            #K = 5
            K = int(cfg.get("teacher", {}).get("num_views", 5))

            aug_transforms = [
                build_aug_pipeline_for_model(
                    model_type=model_type,
                    size=data_cfg["resolution"],
                    mode="train",
                    backbone_type=backbone_type,
                    model_name=model_name,
                    aug_id=i,
                )
                for i in range(K)
            ]
        else:
            aug_transforms = None

        logger.info(f"[distill] split={split} base_tf:\n{base_tf}")
        if aug_transforms is not None:
            for i, tf in enumerate(aug_transforms):
                logger.info(f"[distill] split={split} aug_tf[{i}]:\n{tf}")

        wrapped_ds = DistillDatasetById(
            base_dataset=base_loader.dataset,
            id_to_embedding=id_to_emb,
            base_transform=base_tf,
            aug_transforms=aug_transforms,
        )

        dl = DataLoader(
            wrapped_ds,
            batch_size=data_cfg.get("batch_size", 32),
            num_workers=data_cfg.get("num_workers", 4),
            shuffle=shuffle,
            pin_memory=True,
            drop_last=False,
            collate_fn=distill_k_view_collate,
        )

        return dl


    train_loader = wrap_split("train", base_loaders.get("train"), shuffle=True)
    val_loader   = wrap_split("val",   base_loaders.get("val"),   shuffle=False)
    test_loader  = wrap_split("test",  base_loaders.get("test"),  shuffle=False)

    X, T = next(iter(train_loader))
    logger.info(f"[distill] Train batch X={X.shape} T={T.shape}")

    if val_loader is not None:
        Xv, Tv = next(iter(val_loader))
        logger.info(f"[distill] Val batch X={Xv.shape} T={Tv.shape}")

    if test_loader is not None:
        Xt, Tt = next(iter(test_loader))
        logger.info(f"[distill] Test batch X={Xt.shape} T={Tt.shape}")



    # Quick sanity check
    #xb, tb = next(iter(train_loader))
    #logger.info(f"[distill] First batch x.shape={getattr(xb, 'shape', None)} teacher.shape={getattr(tb, 'shape', None)}")

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
    
    callbacks.append(StopReasonCallback())


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

    trainer = L.Trainer(**trainer_kwargs, precision="16-mixed")


    logger.info("[distill] Starting distillation training...")
    #assert X.shape[0] == T.shape[0], "Mismatch: batch images vs teacher embeddings"
    #assert X.ndim == 4, "Images must be [B,C,H,W]"
    #assert T.ndim == 2, "Embeddings must be [B,D]"

    #K = 5
    K = int(cfg.get("teacher", {}).get("num_views", 5))

    assert X.ndim == 4 and T.ndim == 2
    assert X.shape[0] == T.shape[0]

    # Infer batch size from data
    assert T.shape[0] % (K + 1) == 0, \
        f"Total views {T.shape[0]} not divisible by K+1={K+1}"

    B_actual = T.shape[0] // (K + 1)

    logger.info(
        f"[distill] Detected batch_size={B_actual}, num_views={K}, total_images={X.shape[0]}"
    )

    trainer.fit(lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if test_loader is not None:
        logger.info("[distill] Starting test...")
        trainer.test(lit_model, dataloaders=test_loader)

    logger.info("[distill] Done.")


"""
    # Wrap datasets with teacher embeddings
    def wrap_split(split: str, base_loader: Optional[DataLoader], shuffle: bool) -> Optional[DataLoader]:
        if base_loader is None:
            return None

        id_to_emb = load_teacher_id_map(embeddings_dir, split)

        model_type = cfg["model"].get("type", "vit").lower()
        backbone_type = cfg["model"].get("backbone", {}).get("type", None)
        model_name = cfg["model"].get("backbone", {}).get("model_name", None)

        # Base transform: whatever policy you want for the "clean" view
        # IMPORTANT: for your code, mode should be "train" or not; you used "test" elsewhere,
        # but build_pipeline_for_model treats anything != "train" as "else".
        base_tf = build_pipeline_for_model(
            model_type=model_type,
            size=data_cfg["resolution"],
            mode="train" if split == "train" else "test",
            backbone_type=backbone_type,
            model_name=model_name,
        )

        if split == "train":
            # Aug transform only for training
            aug_tf = build_aug_pipeline_for_model(
                model_type=model_type,
                size=data_cfg["resolution"],
                mode="train",
                backbone_type=backbone_type,
                model_name=model_name,
            )

            wrapped_ds = DistillDatasetById(
                base_dataset=base_loader.dataset,
                id_to_embedding=id_to_emb,
                base_transform=base_tf,
                aug_transform=aug_tf,
            )
            collate = distill_two_view_collate

        else:
            # val/test: single view only (no augmentation)
            wrapped_ds = DistillDatasetById(
                base_dataset=base_loader.dataset,
                id_to_embedding=id_to_emb,
                base_transform=base_tf,
                aug_transform=None,
            )
            collate = distill_one_view_collate
        
        logger.info(
            f"[distill] split={split}\n"
            f"  base_tf={base_tf}\n"
            f"  aug_tf={'None' if split != 'train' else aug_tf}\n"
            f"  collate={'two_view' if split == 'train' else 'one_view'}"
        )


        return DataLoader(
            wrapped_ds,
            batch_size=data_cfg.get("batch_size", 32),
            num_workers=data_cfg.get("num_workers", 4),
            shuffle=shuffle,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate,
        )

"""