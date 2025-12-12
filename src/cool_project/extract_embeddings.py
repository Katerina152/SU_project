import json
import argparse
from pathlib import Path
import logging
import os

import torch
import lightning as L

from cool_project.backbones_heads.models import build_model_from_config
from cool_project.lightning_model import LightningEmbeddingExtractor
from cool_project.data_domain import create_domain_loaders, DOMAIN_DATASET_MAP
from cool_project.dataloader import is_multilabel_from_config

logger = logging.getLogger(__name__)


def load_config(path: str):
    with open(path, "r") as f:
        return json.load(f)


def setup_logging(seed_dir: Path):
    seed_dir.mkdir(parents=True, exist_ok=True)
    log_file = seed_dir / "extract_embeddings.log"

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

    logger.info(f"[extract] Logging to {log_file}")


def build_embedding_experiment(config_path: str):
    cfg = load_config(config_path)
    logger.info(f"[extract] Loaded config from {config_path}")

    # ----------------------------
    # Experiment / seed dirs (same pattern as training)
    # ----------------------------
    exp_name = cfg.get("experiment_name", "default_experiment")
    out_root = Path(cfg.get("output_dir", "runs"))
    seed = cfg["train"].get("seed", 42)
    L.seed_everything(seed)

    data_cfg = cfg["data"]
    domain = data_cfg["domain"]
    logger.info(f"[extract] Domain = {domain}")

    dataset_name = DOMAIN_DATASET_MAP.get(domain)
    if dataset_name is None:
        raise ValueError(
            f"Unknown domain '{domain}'. Please add it to DOMAIN_DATASET_MAP."
        )

    dataset_root = Path(dataset_name)
    exp_root = dataset_root / exp_name
    seed_dir = exp_root / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(seed_dir)
    logger.info(f"[extract] Experiment root = {exp_root}")
    logger.info(f"[extract] Seed directory  = {seed_dir}")



    # ----------------------------
    # 2. DATA (reuse create_domain_loaders)
    # ----------------------------
    loaders = create_domain_loaders(
        domain=domain,
        resolution=data_cfg["resolution"],
        batch_size=data_cfg.get("batch_size", 32),
        num_workers=data_cfg.get("num_workers", 4),
        balanced_train=False,                 # no balancing for embeddings
        val_split=data_cfg.get("val_split", 0.0),
        return_one_hot=False,
    )

    train_loader = loaders.get("train")
    val_loader   = loaders.get("val")
    test_loader  = loaders.get("test")

    logger.info(f"[extract] Dataloaders created with keys: {list(loaders.keys())}")

    # ----------------------------
    # 3. MODEL (backbone + head)
    # ----------------------------
    model = build_model_from_config(cfg)      # ViT / DINO / seg_ViT
    logger.info(f"[extract] Model built: {model.__class__.__name__}")

    # Lightning module used only for prediction (embeddings)
    extractor = LightningEmbeddingExtractor(model)

    # ----------------------------
    # 4. TRAINER (same GPU / DDP logic as training)
    # ----------------------------
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

    logger.info("===== Trainer hardware configuration (embeddings) =====")
    logger.info(f"SLURM num_nodes          = {num_nodes}")
    logger.info(f"CUDA visible GPUs        = {num_visible}")
    logger.info(f"use_gpu                  = {use_gpu}")
    logger.info(f"Chosen Lightning strategy= {strategy}")
    logger.info("======================================================")

    trainer_kwargs = dict(
        max_epochs=1,           # we don't train, but Trainer needs some value
        log_every_n_steps=50,
        logger=False,           # no TB/CSV logs needed here
        enable_checkpointing=False,
    )

    if use_gpu:
        trainer_kwargs.update(
            accelerator="gpu",
            devices=num_visible,   # use all GPUs SLURM gave us
            num_nodes=num_nodes,
            strategy=strategy,
        )
    else:
        logger.warning("[extract] No GPUs visible: using CPU for embedding extraction.")
        trainer_kwargs.update(
            accelerator="cpu",
            devices=1,
            num_nodes=1,
        )

    trainer = L.Trainer(**trainer_kwargs)

    # ----------------------------
    # 5. Embeddings directory
    # ----------------------------
    emb_dir = seed_dir / "embeddings"
    emb_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"[extract] Embeddings will be saved under: {emb_dir}")

    def run_and_save(split: str, loader):
        if loader is None:
            return None
        logger.info(f"[extract] Extracting embeddings for split: {split}")

        preds = trainer.predict(
            extractor,
            dataloaders=loader,
        )  # list of [B, D] tensors

        embs = torch.cat(preds, dim=0).cpu()
        path = emb_dir / f"{split}_embeddings.pt"
        torch.save(embs, path)
        logger.info(f"[extract] Saved {split} embeddings to {path}, shape = {embs.shape}")
        return embs

    emb_train = run_and_save("train", train_loader)
    emb_val   = run_and_save("val",   val_loader)
    emb_test  = run_and_save("test",  test_loader)

    # ----------------------------
    # 6. metadata.json
    # ----------------------------
    # pick any non-None tensor to infer embedding_dim
    if emb_train is not None:
        some_emb = emb_train
    elif emb_val is not None:
        some_emb = emb_val
    elif emb_test is not None:
        some_emb = emb_test
    else:
        raise RuntimeError("No embeddings were generated (no loaders).")

    emb_dim = some_emb.shape[-1]
    resolution = data_cfg["resolution"]

    meta = {
        "dataset": dataset_name,
        "domain": domain,
        "experiment_name": exp_name,
        "seed": seed,
        "model": cfg["model"].get("type", "vit"),
        "model_info": {
            "hf_model_name": cfg["model"]["backbone"].get("hf_model_name", None),
        },
        "resolution": resolution,
        "embedding_dim": emb_dim,
        "num_samples": {
            "train": int(emb_train.shape[0]) if emb_train is not None else 0,
            "val":   int(emb_val.shape[0])   if emb_val   is not None else 0,
            "test":  int(emb_test.shape[0])  if emb_test  is not None else 0,
        },
    }

    meta_path = emb_dir / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"[extract] Wrote metadata to {meta_path}")
    logger.info("[extract] Finished embedding extraction.")


# Optional entry point
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", type=str, required=True)
#     args = parser.parse_args()
#     build_embedding_experiment(args.config)
