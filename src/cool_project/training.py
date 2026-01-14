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
import copy
import itertools
import csv
import torch
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger, CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping

from cool_project.backbones_heads.models import build_model_from_config
from cool_project.lightning_model import LightningModel
from cool_project.data_domain import create_domain_loaders, DOMAIN_DATASET_MAP
from cool_project.dataloader import is_multilabel_from_config, compute_class_weights
from cool_project.training_utils import build_trainer_from_env
from transformers.utils import default_cache_path
import os
import logging
import tempfile

print("HF default cache path:", default_cache_path)
print("HF_HOME:", os.getenv("HF_HOME"))
print("TRANSFORMERS_CACHE:", os.getenv("TRANSFORMERS_CACHE"))

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
def build_training_experiment(config_path: str, run_tag: str | None = None, run_test: bool = True):
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

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # ----------------------------------------------------
    # 1. DATA
    # ----------------------------------------------------
    data_cfg = cfg["data"]
    domain = data_cfg["domain"]         
    logger.info(f"Domain = {domain}")

    # Map domain 
    dataset_name = data_cfg.get("dataset_name") or DOMAIN_DATASET_MAP.get(domain)
    if dataset_name is None:
        raise ValueError(
            f"Unknown domain '{domain}'. Please add it to DOMAIN_DATASET_MAP."
        )

    logger.info(f"Dataset name (from domain) = {dataset_name}")

    # Final directory for this run:
    #   <DATASET_NAME>/<experiment_name>/seed_<seed>/
    dataset_root = Path(dataset_name)
    out_dir = Path(cfg.get("output_dir", "runs"))
    exp_root = out_dir / dataset_root / exp_name
    job_id = os.environ.get("SLURM_JOB_ID", "local")

    if run_tag is None:
        seed_dir = exp_root / f"seed_{seed}" / f"job_{job_id}"
    else:
        seed_dir = exp_root / f"seed_{seed}" / run_tag / f"job_{job_id}"

    #seed_dir = exp_root / f"seed_{seed}" / f"job_{job_id}"
    #seed_dir = exp_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(seed_dir)  # now logger writes to train.log in seed_dir
    logger.info(f"Experiment root = {exp_root}")
    logger.info(f"Seed directory  = {seed_dir}")

    # Create domain loaders
    loaders = create_domain_loaders(
        domain=domain,
        dataset_name=dataset_name,
        resolution=data_cfg["resolution"],
        batch_size=data_cfg.get("batch_size", 32),
        num_workers=data_cfg.get("num_workers", 4),
        balanced_train=data_cfg.get("balanced_train", False),
        val_split=data_cfg.get("val_split", 0.0),
        return_one_hot=return_one_hot,
        model_type=cfg["model"].get("type", "vit").lower(),
        backbone_type=cfg["model"].get("backbone", {}).get("type", None),
        model_name=cfg["model"].get("backbone", {}).get("model_name", None),
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

    # --- Inspect first batch shapes for sanity ---
    first_batch = next(iter(train_loader))

    pixel_values, labels, image_ids = first_batch
    logger.info(f"First train batch pixel_values.shape = {getattr(pixel_values, 'shape', None)}")
    logger.info(f"First train batch labels type        = {type(labels)}")
    logger.info(f"First train batch labels shape       = {getattr(labels, 'shape', None)}")

    if torch.is_tensor(pixel_values):
        logger.info(f"pixel_values dtype={pixel_values.dtype}, device={pixel_values.device}")

    if torch.is_tensor(labels):
        logger.info(f"labels dtype={labels.dtype}, device={labels.device}")
    else:
        logger.info(f"labels preview={labels[:10] if isinstance(labels, (list, tuple)) else labels}")

    # The dataset the model actually trains on (may be a Subset)
    train_ds = train_loader.dataset

    # Underlying dataset (for metadata + CSV inspection)
    base_train_ds = train_ds.dataset if hasattr(train_ds, "dataset") else train_ds

    num_classes_data = getattr(base_train_ds, "num_classes", None)

    # ---- Compute class weights on the TRAIN SPLIT ----
    cw, _ = compute_class_weights(train_ds)  # <- IMPORTANT: use train_ds, not base_train_ds
    cw = cw.float()

    logger.info(f"[train] computed class_weights on train split (len={len(cw)}): {cw.tolist()}")

    # ---- Inspect TRAIN-SPLIT class-index distribution ----
    try:
        from collections import Counter
        counts = Counter()
        for i in range(len(train_ds)):
            y = train_ds[i][1]
            if isinstance(y, torch.Tensor) and y.ndim > 0:
                y = int(y.argmax().item())
            else:
                y = int(y)
            counts[y] += 1
        logger.info(f"Train-split class-index counts: {dict(counts)}")

        if hasattr(base_train_ds, "class_names"):
            logger.info(f"class_names: {base_train_ds.class_names}")
    except Exception as e:
        logger.warning(f"Could not compute train-split class distribution: {e}")



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
    #exp_cfg["class_weights"] = cfg.get("class_weights", None)  # optional if you use it
    model.class_weights = cw
    #exp_cfg["class_weights"] = cw  # tensor


    #task = exp_cfg.get("task", "single_label_classification").lower()
    exp_cfg["task"] =  train_task
    exp_cfg["loss_type"] = cfg["model"].get("loss_type", "auto")
    exp_cfg["num_classes"] = num_classes_cfg
    exp_cfg["experiment_name"] = exp_name
    exp_cfg["output_dir"] = str(seed_dir)
   

    logger.info(f"Lightning task         = {train_task}")
    logger.info(f"Lightning output_dir   = {seed_dir}")
    logger.info("Lightning experiment_name set to empty string for flat structure.")

    #check that, we removed didtillation option
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
                verbose=True,
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

    # Capture best val score from ModelCheckpoint callback
    best_val = None
    for cb in trainer.callbacks:
        if isinstance(cb, ModelCheckpoint):
            if cb.best_model_score is not None:
                best_val = float(cb.best_model_score.detach().cpu().item())
            break

    # TEST (only if run_test=True)
    if run_test and test_loader is not None:
        logger.info("Starting test (best checkpoint)...")
        trainer.test(
            lit_model,
            dataloaders=test_loader,
            ckpt_path="best"  # <- important
        )

    logger.info("Training script finished.")
    return best_val

    

def set_nested(cfg, key: str, value):
    cur = cfg
    parts = key.split(".")
    for part in parts[:-1]:
        cur = cur[part]
    cur[parts[-1]] = value


def run_hparam_sweep(config_path: str):
    base_cfg = load_config(config_path)

    tune = base_cfg.get("tune", {})
    grid = tune.get("grid", {})
    metric = tune.get("metric", "val_loss")
    mode = tune.get("mode", "min")

    keys = list(grid.keys())
    values_list = [grid[k] for k in keys]

    results = []
    for values in itertools.product(*values_list):
        cfg_trial = copy.deepcopy(base_cfg)

        for k, v in zip(keys, values):
            set_nested(cfg_trial, k, v)

        # make run_tag
        parts = []
        for k, v in zip(keys, values):
            k2 = k.replace(".", "_")
            parts.append(f"{k2}_{v:g}" if isinstance(v, float) else f"{k2}_{v}")
        run_tag = "__".join(parts)

        # write trial config to temp file and call your existing function
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
            json.dump(cfg_trial, tmp, indent=2)
            tmp_path = tmp.name

        best_val = build_training_experiment(tmp_path, run_tag=run_tag, run_test=False)

        row = {k: v for k, v in zip(keys, values)}
        row[f"best_{metric}"] = best_val
        results.append(row)

    # save CSV summary into seed folder
    domain = base_cfg["data"]["domain"]
    dataset_name = base_cfg["data"].get("dataset_name") or DOMAIN_DATASET_MAP.get(domain)
    exp_name = base_cfg.get("experiment_name", "default_experiment")
    seed = base_cfg["train"].get("seed", 42)
    out_dir = Path(base_cfg.get("output_dir", "runs"))
    exp_root = out_dir / Path(dataset_name) / exp_name / f"seed_{seed}"
    exp_root.mkdir(parents=True, exist_ok=True)

    summary_path = exp_root / "hparam_sweep_results.csv"
    fieldnames = keys + [f"best_{metric}"]

    with open(summary_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(results)

    valid = [r for r in results if r[f"best_{metric}"] is not None]
    if valid:
        best = min(valid, key=lambda r: r[f"best_{metric}"]) if mode == "min" else max(valid, key=lambda r: r[f"best_{metric}"])
        print("BEST:", best)
    print("Saved sweep summary to:", summary_path)


def run_trial_by_index(config_path: str, index: int):
    base_cfg = load_config(config_path)
    tune = base_cfg.get("tune", {})
    grid = tune.get("grid", {})

    keys = list(grid.keys())
    values_list = [grid[k] for k in keys]

    combos = list(itertools.product(*values_list))
    if index < 0 or index >= len(combos):
        raise ValueError(f"trial_index {index} out of range [0, {len(combos)-1}]")

    values = combos[index]
    cfg_trial = copy.deepcopy(base_cfg)
    for k, v in zip(keys, values):
        set_nested(cfg_trial, k, v)

    # run_tag
    parts = []
    for k, v in zip(keys, values):
        k2 = k.replace(".", "_")
        parts.append(f"{k2}_{v:g}" if isinstance(v, float) else f"{k2}_{v}")
    run_tag = "__".join(parts)

    # temp config
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as tmp:
        json.dump(cfg_trial, tmp, indent=2)
        tmp_path = tmp.name

    # IMPORTANT: no test for tuning trials
    best_val = build_training_experiment(tmp_path, run_tag=run_tag, run_test=False)
    print(f"[trial {index}] {run_tag} best_val={best_val}")

'''
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--trial-index", type=int, default=None)  # <-- for SLURM arrays
    args = parser.parse_args()

    cfg = load_config(args.config)

    # If SLURM array passes a trial index, run exactly one combo
    if args.trial_index is not None:
        run_trial_by_index(args.config, args.trial_index)

    else:
        do_sweep = args.sweep or cfg.get("tune", {}).get("enabled", False)
        if do_sweep:
            run_hparam_sweep(args.config)
        else:
            build_training_experiment(args.config, run_tag=None, run_test=True)
'''