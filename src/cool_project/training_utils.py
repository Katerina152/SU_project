import os
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.strategies import DeepSpeedStrategy

def build_trainer_from_env(cfg) -> Trainer:
    """
    cfg: your config dict (where you already store stuff like max_epochs, etc.)
    This will:
      - detect number of GPUs on the node
      - detect number of nodes from SLURM
      - pick a sane strategy (single GPU vs DDP vs DeepSpeed)
    """

    # How many GPUs are visible to this process?
    num_devices = torch.cuda.device_count()

    # Number of nodes from SLURM (fallback to 1 if not set)
    num_nodes = int(os.environ.get("SLURM_JOB_NUM_NODES",
                                   os.environ.get("SLURM_NNODES", "1")))

    # --- Decide strategy ---
    # Optional flags in cfg or env to force DeepSpeed
    use_deepspeed = bool(cfg.get("use_deepspeed", False)
                         or os.environ.get("USE_DEEPSPEED", "0") == "1")

    if use_deepspeed:
        # You requested DeepSpeed explicitly
        strategy = DeepSpeedStrategy(stage=cfg.get("deepspeed_stage", 2))
    else:
        # If we have more than 1 GPU or more than 1 node, use DDP
        if num_devices > 1 or num_nodes > 1:
            strategy = "ddp"
        else:
            strategy = "auto"  # single GPU / CPU case

    # Safety: never pass 0 devices
    if num_devices == 0 and cfg.get("accelerator", "gpu") == "gpu":
        raise RuntimeError("No GPUs visible, but accelerator='gpu' requested.")

    accelerator = cfg.get("accelerator", "gpu")

    trainer = Trainer(
        accelerator=accelerator,
        devices=num_devices if accelerator == "gpu" else "auto",
        num_nodes=num_nodes,
        strategy=strategy,
        max_epochs=cfg.get("max_epochs", 10),
        precision=cfg.get("precision", "16-mixed"), 
        log_every_n_steps=cfg.get("log_every_n_steps", 50),
        
    )

    return trainer

