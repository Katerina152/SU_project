"""
Training entry point for ViT + head.

- Loads config JSON
- Creates domain-specific dataloaders
- Builds model from config
- Wraps in LightningModel
- Trains, validates, tests
- Saves everything under:
    <DATASET_NAME>/<experiment_name>/seed_<seed>/

Example for dermatology + seed=0:
    ISIC2019/isic_vit_binary_224/seed_0/
"""

import json
import argparse
from pathlib import Path

import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from cool_project.backbones_heads.models import build_model_from_config
from cool_project.lightning_model import LightningModel
from cool_project.data_domain import create_domain_loaders, DOMAIN_DATASET_MAP
from cool_project.dataloader import is_multilabel_from_config

import logging

logger = logging.getLogger(__name__)

# ----------------------------
# Config loader
# ----------------------------
def load_config(path: str):
    with open(path, "r") as f:
        return json.load(f)


def setup_logging(seed_dir: Path):
    """
    Log to console + seed_dir/train.log

    seed_dir example:
        ISIC2019/isic_vit_binary_224/seed_0
    """
    seed_dir.mkdir(parents=True, exist_ok=True)
    log_file = seed_dir / "train.log"

    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Console
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    logger.info(f"Logging to {log_file}")


# ----------------------------
# Main training function
# ----------------------------
def main(config_path: str):
    cfg = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    # ----------------------------
    # Task type / multilabel
    # ----------------------------
    is_multilabel = is_multilabel_from_config(cfg)
    return_one_hot = is_multilabel
    logger.info(f"is_multilabel_from_config(cfg) = {is_multilabel}")
    logger.info(f"return_one_hot = {return_one_hot}")

    # Top-level config fields
    exp_name = cfg.get("experiment_name", "default_experiment")
    logger.info(f"experiment_name = {exp_name}")

    # Seed (top-level, or default 0)
    seed = cfg["train"].get("seed", 42)
    L.seed_everything(seed)
    logger.info(f"Seed = {seed}")

    # Optional but explicit
    #torch.manual_seed(seed)
    #np.random.seed(seed)
    #random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # ----------------------------------------------------
    # 1. DATA
    # ----------------------------------------------------
    data_cfg = cfg["data"]
    domain = data_cfg["domain"]         
    logger.info(f"Domain = {domain}")

    # Map domain 
    dataset_name = DOMAIN_DATASET_MAP.get(domain)
    if dataset_name is None:
        raise ValueError(
            f"Unknown domain '{domain}'. Please add it to DOMAIN_DATASET_MAP."
        )

    logger.info(f"Dataset name (from domain) = {dataset_name}")

    # Final directory for this run:
    #   <DATASET_NAME>/<experiment_name>/seed_<seed>/
    dataset_root = Path(dataset_name)
    exp_root = dataset_root / exp_name
    seed_dir = exp_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(seed_dir)  # now logger writes to train.log in seed_dir
    logger.info(f"Experiment root = {exp_root}")
    logger.info(f"Seed directory  = {seed_dir}")

    # Create domain loaders (your existing API)
    loaders = create_domain_loaders(
        domain=domain,
        resolution=data_cfg["resolution"],
        batch_size=data_cfg.get("batch_size", 32),
        num_workers=data_cfg.get("num_workers", 4),
        balanced_train=data_cfg.get("balanced_train", False),
        val_split=data_cfg.get("val_split", 0.0),
        return_one_hot=return_one_hot,
    )

    # Your create_domain_loaders returns a dict
    train_loader = loaders["train"]
    val_loader = loaders.get("val")
    test_loader = loaders["test"]

    logger.info(f"Dataloaders created with keys: {list(loaders.keys())}")
    try:
        logger.info(f"#train batches per epoch: {len(train_loader)}")
        if val_loader is not None:
            logger.info(f"#val batches per epoch:   {len(val_loader)}")
        logger.info(f"#test batches per epoch:  {len(test_loader)}")
    except TypeError:
        logger.warning("One of the dataloaders has no __len__ (infinite / iterable).")

    # --- Inspect first batch shapes for sanity ---
    first_batch = next(iter(train_loader))
    pixel_values, labels = first_batch
    logger.info(f"First train batch pixel_values.shape = {pixel_values.shape}")
    logger.info(f"First train batch labels.shape       = {labels.shape}")
    logger.info(
        f"pixel_values dtype = {pixel_values.dtype}, device = {pixel_values.device}"
    )
    logger.info(
        f"labels dtype       = {labels.dtype}, device = {labels.device}"
    )

    # --- num_classes from data & consistency check ---
    base_train_ds = getattr(train_loader.dataset, "dataset", train_loader.dataset)
    num_classes_data = getattr(base_train_ds, "num_classes", None)

    cfg_head = cfg["model"].get("head", {})
    num_classes_cfg = cfg_head.get("output_dim")

    if num_classes_data is not None and num_classes_cfg is not None:
        if num_classes_data != num_classes_cfg:
            raise ValueError(
                f"num_classes from data ({num_classes_data}) != "
                f"model.head.output_dim from config ({num_classes_cfg})."
            )
    elif num_classes_data is not None and num_classes_cfg is None:
        # optional: auto-fill config if missing
        cfg["model"].setdefault("head", {})
        cfg["model"]["head"]["output_dim"] = num_classes_data
        num_classes_cfg = num_classes_data

    logger.info(f"num_classes (data) = {num_classes_data}")
    logger.info(f"num_classes (cfg)  = {num_classes_cfg}")

    # ----------------------------------------------------
    # 2. MODEL
    # ----------------------------------------------------
    model = build_model_from_config(cfg)
    logger.info("Model built from config:")
    logger.info(model.__class__.__name__)

    # ----------------------------------------------------
    # 3. LIGHTNING WRAPPER
    # ----------------------------------------------------
    # Start from train config
    exp_cfg = cfg["train"].copy()

    task = exp_cfg.get("task", "single_label_classification").lower()
    exp_cfg["task"] = task
    exp_cfg["num_classes"] = num_classes_cfg
    exp_cfg["experiment_name"] = exp_name
    exp_cfg["output_dir"] = str(seed_dir)
   

    logger.info(f"Lightning task         = {task}")
    logger.info(f"Lightning output_dir   = {seed_dir}")
    logger.info("Lightning experiment_name set to empty string for flat structure.")

    lit_model = LightningModel(
        model=model,
        cfg=exp_cfg,
    )

    # ----------------------------------------------------
    # 4. LOGGERS (TensorBoard + CSV)
    # ----------------------------------------------------
    # We want metrics.csv and events.* directly in seed_dir.
    # Lightning normally creates: save_dir / name / version
    # By setting name="" and version="", those extra levels collapse.
    tb_logger = TensorBoardLogger(
        save_dir=str(seed_dir),
        name="",
        version="",
    )

    csv_logger = CSVLogger(
        save_dir=str(seed_dir),
        name="",
        version="",
    )

    logger.info(f"TensorBoard / CSV logs will be in: {seed_dir}")

    # ----------------------------------------------------
    # 5. CALLBACKS
    # ----------------------------------------------------
    callbacks = []

    # Early stopping if val_loader exists
    if val_loader is not None:
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                patience=exp_cfg.get("early_stopping_patience", 5),
                mode="min",
            )
        )

    # Best model checkpoint
    # Save to:
    #   <DATASET_NAME>/<experiment_name>/seed_<seed>/checkpoints/
    ckpt_dir = seed_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    callbacks.append(
        ModelCheckpoint(
            dirpath=ckpt_dir,
            filename="best",
            monitor="val_loss" if val_loader is not None else None,
            save_top_k=1,
            mode="min",
        )
    )

    logger.info(f"Checkpoints will be saved under: {ckpt_dir}")

    # ----------------------------------------------------
    # 6. TRAINER
    # ----------------------------------------------------
    use_gpu = torch.cuda.is_available() and exp_cfg.get("gpus", 0) > 0
    logger.info(f"use_gpu = {use_gpu}")

    trainer_kwargs = dict(
        max_epochs=exp_cfg.get("max_epochs", 10),
        logger=[tb_logger, csv_logger],
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    if use_gpu:
        trainer_kwargs.update(
            accelerator="gpu",
            devices=exp_cfg.get("gpus", 1),
        )
    else:
        trainer_kwargs.update(
            accelerator="cpu",
            devices=1,
        )

    trainer = L.Trainer(**trainer_kwargs)

    # ----------------------------------------------------
    # 7. FIT
    # ----------------------------------------------------
    logger.info("Starting training...")
    trainer.fit(
        lit_model,
        train_dataloaders=train_loader,
        val_dataloaders=val_loader,
    )

    # ----------------------------------------------------
    # 8. TEST (optional)
    # ----------------------------------------------------
    if test_loader is not None:
        logger.info("Starting test...")
        trainer.test(
            lit_model,
            dataloaders=test_loader,
        )

    logger.info("Training script finished.")


# ----------------------------
# CLI ENTRY
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config JSON",
    )
    args = parser.parse_args()
    main(args.config)
