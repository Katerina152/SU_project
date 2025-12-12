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
from cool_project.training_utils import build_trainer_from_env
from transformers.utils import default_cache_path
import os

print("HF default cache path:", default_cache_path)
print("HF_HOME:", os.getenv("HF_HOME"))
print("TRANSFORMERS_CACHE:", os.getenv("TRANSFORMERS_CACHE"))



import logging

logger = logging.getLogger(__name__)

# ----------------------------
# Config loader
# ----------------------------
def load_config(path: str):
    with open(path, "r") as f:
        return json.load(f)


def setup_logging(seed_dir: Path):
    seed_dir.mkdir(parents=True, exist_ok=True)
    log_file = seed_dir / "train.log"

    # Get the ROOT logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    root_logger.addHandler(ch)

    # File handler
    fh = logging.FileHandler(log_file)
    fh.setFormatter(fmt)
    root_logger.addHandler(fh)

    # You can still log via the module logger
    logger.info(f"Logging to {log_file}")



# ----------------------------
# Main training function
# ----------------------------
def build_training_experiment(config_path: str):
    cfg = load_config(config_path)
    logger.info(f"Loaded config from {config_path}")

    # ----------------------------
    # Task type / multilabel / segmentation
    # ----------------------------
    train_task = cfg.get("task", "single_label_classification").lower()
    model_type = cfg["model"].get("type", "vit").lower()

    is_segmentation = (train_task == "segmentation" or model_type == "seg_vit")

    if is_segmentation:
        # segmentation: masks, not one-hot
        is_multilabel = False
        return_one_hot = False
    else:
        # classification/regression: keep old logic
        is_multilabel = is_multilabel_from_config(cfg)
        return_one_hot = is_multilabel

    logger.info(f"train_task = {train_task}")
    logger.info(f"model_type = {model_type}")
    logger.info(f"is_segmentation = {is_segmentation}")
    logger.info(f"is_multilabel_from_config(cfg) = {is_multilabel}")
    logger.info(f"return_one_hot = {return_one_hot}")

    # Top-level config fields
    exp_name = cfg.get("experiment_name", "default_experiment")
    logger.info(f"experiment_name = {exp_name}")

    # Seed (top-level, or default 0)
    seed = cfg["train"].get("seed", 42)
    L.seed_everything(seed)
    logger.info(f"Seed = {seed}")

    # Optional
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

    # Create domain loaders
    loaders = create_domain_loaders(
        domain=domain,
        dataset_variant=dataset_variant,
        resolution=data_cfg["resolution"],
        batch_size=data_cfg.get("batch_size", 32),
        num_workers=data_cfg.get("num_workers", 4),
        balanced_train=data_cfg.get("balanced_train", False),
        val_split=data_cfg.get("val_split", 0.0),
        return_one_hot=return_one_hot,
    )

    
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
    

    # ----------------------------------------------------
    # 1b. If distillation is enabled, wrap *all* splits with DistillDataset
    # ----------------------------------------------------
    distill_on = cfg["train"].get("distill", False)

    if distill_on:
        logger.info(f"Distillation flag from config: {distill_on}")
        teacher_cfg = cfg.get("teacher", {})
        emb_root = teacher_cfg.get("embeddings_path", None)

        if emb_root is None:
            raise ValueError(
                "distill=True but no teacher.embeddings_path provided in config."
            )

        # Collect base datasets for each split
        base_datasets = {}
        if train_loader is not None:
            base_datasets["train"] = train_loader.dataset
        if val_loader is not None:
            base_datasets["val"] = val_loader.dataset
        if test_loader is not None:
            base_datasets["test"] = test_loader.dataset

        # Build a temporary student model just to query embedding dim D
        student_model = build_model_from_config(cfg)
        try:
            D = student_model.vit.config.hidden_size
        except AttributeError:
            D = student_model.backbone.config.hidden_size

        teacher_embs_by_split = {}

        for split_name, base_ds in base_datasets.items():
            N = len(base_ds)

            # Derive a path per split, e.g. "path/to/emb.pt" -> "path/to/emb_train.pt" etc.
            if emb_root.endswith(".pt"):
                stem = emb_root[:-3]
                emb_path = f"{stem}_{split_name}.pt"
            else:
                # if emb_root is a directory, save inside it
                emb_path = str(Path(emb_root) / f"teacher_{split_name}.pt")

            if not os.path.exists(emb_path):
                logger.warning(f"Teacher embeddings file not found for {split_name}: {emb_path}")
                logger.warning(f"Creating RANDOM teacher embeddings for {split_name} (debug).")
                teacher_embs = torch.randn(N, D)
                torch.save(teacher_embs, emb_path)
                logger.warning(f"Saved random teacher embeddings to: {emb_path}")
                logger.warning(f"Shape: {teacher_embs.shape}")
            else:
                logger.info(f"Loading teacher embeddings for {split_name} from: {emb_path}")
                teacher_embs = torch.load(emb_path)

            if teacher_embs.ndim != 2:
                raise ValueError(
                    f"[{split_name}] Expected teacher_embs to have shape [N, D], "
                    f"but got {teacher_embs.shape}. Please regenerate teacher embeddings."
                )

            if teacher_embs.shape[0] != N:
                raise ValueError(
                    f"[{split_name}] Teacher embeddings length {teacher_embs.shape[0]} != "
                    f"{split_name} dataset length {N}"
                )

            teacher_embs_by_split[split_name] = teacher_embs

        # Now wrap each split with DistillDataset -> (image, teacher_emb)
        def make_loader(base_ds, teacher_embs, shuffle):
            return DataLoader(
                DistillDataset(base_ds, teacher_embs),
                batch_size=data_cfg.get("batch_size", 32),
                num_workers=data_cfg.get("num_workers", 4),
                shuffle=shuffle,
                pin_memory=True,
            )

        if "train" in base_datasets:
            train_loader = make_loader(base_datasets["train"], teacher_embs_by_split["train"], shuffle=True)
            loaders["train"] = train_loader
            logger.info("Distillation mode: train_loader now returns (image, teacher_emb).")

        if "val" in base_datasets:
            val_loader = make_loader(base_datasets["val"], teacher_embs_by_split["val"], shuffle=False)
            loaders["val"] = val_loader
            logger.info("Distillation mode: val_loader now returns (image, teacher_emb).")

        if "test" in base_datasets:
            test_loader = make_loader(base_datasets["test"], teacher_embs_by_split["test"], shuffle=False)
            loaders["test"] = test_loader
            logger.info("Distillation mode: test_loader now returns (image, teacher_emb).")

    # --- Inspect first batch shapes for sanity ---
    first_batch = next(iter(train_loader))

    if distill_on:
        pixel_values, teacher_embs = first_batch
        logger.info(f"First train batch pixel_values.shape = {pixel_values.shape}")
        logger.info(f"First train batch teacher_embs.shape = {teacher_embs.shape}")
        logger.info(
            f"pixel_values dtype = {pixel_values.dtype}, device = {pixel_values.device}"
        )
        logger.info(
            f"teacher_embs dtype = {teacher_embs.dtype}, device = {teacher_embs.device}"
        )
    else:
        pixel_values, labels = first_batch
        logger.info(f"First train batch pixel_values.shape = {pixel_values.shape}")
        logger.info(f"First train batch labels.shape       = {labels.shape}")
        logger.info(
            f"pixel_values dtype = {pixel_values.dtype}, device = {pixel_values.device}"
        )
        logger.info(
            f"labels dtype       = {labels.dtype}, device = {labels.device}"
        )


    base_train_ds = getattr(train_loader.dataset, "dataset", train_loader.dataset)
    num_classes_data = getattr(base_train_ds, "num_classes", None)

    cfg_head = cfg["model"].get("head", {})
    num_classes_cfg = cfg_head.get("output_dim")

    if is_segmentation:
        # For segmentation we rely on the config
        if num_classes_cfg is None:
            raise ValueError(
                "Segmentation task but model.head.output_dim is not set in config. "
                "Please set it to the number of segmentation classes."
            )
        # optional: if your SegmentationDataset has num_classes, you can still check consistency
        if num_classes_data is not None and num_classes_data != num_classes_cfg:
            raise ValueError(
                f"[segmentation] num_classes from data ({num_classes_data}) != "
                f"model.head.output_dim from config ({num_classes_cfg})."
            )
    else:
        # Original classification / regression logic
        if num_classes_data is not None and num_classes_cfg is not None:
            if num_classes_data != num_classes_cfg:
                raise ValueError(
                    f"num_classes from data ({num_classes_data}) != "
                    f"model.head.output_dim from config ({num_classes_cfg})."
                )
        elif num_classes_data is not None and num_classes_cfg is None:
            # auto-fill config
            cfg["model"].setdefault("head", {})
            cfg["model"]["head"]["output_dim"] = num_classes_data
            num_classes_cfg = num_classes_data

    logger.info(f"num_classes (data) = {num_classes_data}")
    logger.info(f"num_classes (cfg)  = {num_classes_cfg}")

    # --- Inspect label distribution in the training CSV  ---
    if hasattr(base_train_ds, "df"):
        df = base_train_ds.df
        label_cols = base_train_ds.label_cols

        logger.info("First 5 rows of train CSV:")
        logger.info("\n%s", df.head().to_string())

        class_sums = df[label_cols].sum().to_dict()
        logger.info(f"Label columns: {label_cols}")
        logger.info(f"Class counts (sum over labels): {class_sums}")

        # For single-label (one-hot) this should equal num_samples
        logger.info(
            f"Total label sum across all classes: {sum(class_sums.values())} "
            f"(num_samples = {len(df)})"
        )
    
    print(type(train_loader.dataset))
    # might be Subset or GenericCSVDataset or DistillDataset

    if hasattr(train_loader.dataset, "dataset"):
        print("Underlying dataset type:", type(train_loader.dataset.dataset))


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

    #task = exp_cfg.get("task", "single_label_classification").lower()
    exp_cfg["task"] =  train_task
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
    # 6. TRAINER (SLURM + hardware driven, no gpus in config)
    # ----------------------------------------------------
    # Number of GPUs SLURM (or the environment) made visible to this process
    num_visible = torch.cuda.device_count()

    # Number of nodes from SLURM, or 1 if running locally / not under SLURM
    num_nodes = int(
        os.environ.get("SLURM_JOB_NUM_NODES",
                       os.environ.get("SLURM_NNODES", "1"))
    )

    # Decide whether we will use GPU at all
    use_gpu = num_visible > 0

    # If we have GPUs and/or multiple nodes, decide strategy
    if use_gpu and (num_visible > 1 or num_nodes > 1):
        strategy = "ddp"
    else:
        strategy = "auto"

    # ---- Debug logging so we can see what's happening ----
    logger.info("===== Trainer hardware configuration =====")
    logger.info(f"SLURM num_nodes          = {num_nodes}")
    logger.info(f"CUDA visible GPUs        = {num_visible}")
    logger.info(f"use_gpu                  = {use_gpu}")
    logger.info(f"Chosen Lightning strategy= {strategy}")
    logger.info("=========================================")

    trainer_kwargs = dict(
        max_epochs=exp_cfg.get("max_epochs", 10),
        logger=[tb_logger, csv_logger],
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    if use_gpu:
        trainer_kwargs.update(
            accelerator="gpu",
            devices=num_visible,   # use all GPUs SLURM gave us
            num_nodes=num_nodes,
            strategy=strategy,
        )
    else:
        logger.warning("No GPUs visible: falling back to CPU training.")
        trainer_kwargs.update(
            accelerator="cpu",
            devices=1,
            num_nodes=1,
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

    # Debug: see if any submodule starts in eval mode
    for name, module in lit_model.named_modules():
        if not module.training:
            print("Module in eval mode at start:", name)
            break

    # ----------------------------------------------------
    # 8. TEST 
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
#if __name__ == "__main__":
    #parser = argparse.ArgumentParser()
    #parser.add_argument(
        #"--config",
        #type=str,
        #required=True,
        #help="Path to config JSON",
    #)
    #args = parser.parse_args()
    #main(args.config)
