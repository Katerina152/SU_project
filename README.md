# Cool Project
### Feature-Level Knowledge Distillation for Medical Image Models

Cool Project is a research framework for **feature-based knowledge distillation** and
**multi-resolution representation learning** in medical imaging.
The framework enables training student models using **pre-extracted teacher embeddings**,
with experiments configured via JSON files and executed locally or on **SLURM-based GPU clusters**.

---

## Table of Contents
- Installation
- Distillation
- Embedding Extraction
- Training 
- Configuration
- Running on SLURM
- Data Directory

---

## Installation

### Prerequisites
- Python **3.10–3.12**
- SLURM (optional, for cluster execution), CUDA-enabled GPU (recommended)


### Setup
```bash
pip install -r requirements.txt
pip install -e .
```

---
## Distillation

```bash
python -m cool_project \
  --function distillation \
  --config scripts/config_distillation.json
```

The core functionality of this repository is feature-level knowledge distillation,
where a student network is trained to match teacher representations extracted at higher
resolutions or from larger models.

Distillation is fully configured via JSON files
(see scripts/config_distillation.json).


### Distillation objective

Distillation is performed at the **representation level**.
For each image and augmentation, both teacher and student embeddings are first
projected into a **shared embedding space of equal dimensionality** using lightweight
projection heads.

The student is then optimized to align with the teacher using the following objective:

\[
\mathcal{L}_{\text{distill}} =
\lambda_{\text{feat}} \lVert h_t - h_s \rVert_2^2 +
\lambda_{\text{cos}} \left(1 - \cos(h_t, h_s)\right)
\]

where \( h_t \) and \( h_s \) denote the projected teacher and student embeddings.

The loss consists of:
- **Feature regression loss** (`lamda_feat`): enforces **magnitude alignment** between
  teacher and student embeddings via an ℓ2 distance.
- **Cosine similarity loss** (`lamda_cos`): enforces **directional alignment** in the
  embedding space, independent of scale.

The combined objective encourages both magnitude and directional consistency, which
helps stabilize representation-level distillation when teacher and student inputs
(e.g. resolutions or augmentations) differ substantially.



## Embedding Extraction

Teacher embeddings must be extracted before running distillation.

```bash

python -m cool_project \
  --function extract_embeddings \
  --config scripts/config_extract_embeddings.json
```

Extracted embeddings are stored on disk and reused across experiments,
enabling efficient multi-run and multi-resolution studies.


### Configuration

Experiments are controlled via JSON configuration files in `scripts/`.

### Distillation config (key parameters)

**General**
- `experiment_name`: experiment identifier used for logging and output naming
- `output_dir`: root directory where runs are saved (e.g. `runs_new/`)
- `task`: experiment type (e.g. `distillation`)

**Teacher embeddings**
- `teacher.embeddings_dir`: path to a directory containing embeddings for the same dataset splits
  used during training. Expected structure:
  <embeddings_dir>/
    train_embeddings/
    val_embeddings/
    test_embeddings/


- `teacher.num_views`: number of views/augmentations per sample stored in the teacher embeddings
(typically 1–5). The student is trained to match the corresponding view embeddings.
- `teacher.embedding_dim`: embedding dimensionality of the teacher (student projections must match this).

**Data**
- `data.dataset_name`: dataset identifier (e.g. `ISIC2017`)
- `data.domain`: dataset domain (e.g. `dermatology`)
- `data.resolution`: input image resolution used for student training
- `data.batch_size`: batch size
- `data.num_workers`: dataloader workers
- `data.val_split`: fraction of training data held out for validation (if applicable)
- `data.balanced_train`: whether to balance sampling during training

**Student model**
- `model.type`: model family (e.g. `vit`, `resnet`)
- `model.backbone.model_name`: backbone identifier (e.g. timm model string)
- `model.backbone.pooling`: embedding pooling strategy (e.g. `mean`, `cls`)
- `model.backbone.freeze_backbone`: freeze or finetune the backbone during training

**Training**
- `train.lamda_feat`: weight for ℓ2 feature regression loss
- `train.lamda_cos`: weight for cosine alignment loss
- `train.lr`, `train.weight_decay`, `train.dropout_rate`
- `train.max_epochs`, `train.early_stopping_patience`
- `train.seed`

Refer to the full JSON files in `scripts/` for additional options.

### Optional: Hyperparameter tuning (training)

Some training configs include a `tune` section to run a small grid search on SLURM array jobs.

- `tune.enabled`: enable/disable tuning
- `tune.metric`: metric to optimize (e.g. `val_loss`)
- `tune.mode`: `min` or `max`
- `tune.grid`: parameter grid (e.g. learning rate, weight decay, batch size)

Example (abbreviated):
"tune": {
  "enabled": true,
  "metric": "val_loss",
  "mode": "min",
  "grid": {
    "train.lr": [1e-4, 3e-4, 1e-3],
    "train.weight_decay": [0.0, 1e-4, 1e-3],
    "data.batch_size": [32, 64]
  }
}


### Running on SLURM (GPU cluster)

The repository includes SLURM-compatible scripts for execution on HPC clusters
(e.g. Sherlock).

## Typical workflow
1. Prepare a virtual environment in scratch
2. Configure required environment variables
3. Submit the SLURM job

For example from the rrot directory:

```bash
sbatch scripts/run_distillation.sh
```

## Modules

On our cluster we load:

```bash
module purge
module load python/3.12.1
module load cuda/12.2
```

What the SLURM scripts handle

GPU allocation (--gres=gpu:1)

Hugging Face authentication via HUGGINGFACE_HUB_TOKEN
(the script fails fast if the token is missing)

Cache setup (HF_HOME, optional TRANSFORMERS_CACHE)

Logging and diagnostics

SLURM stdout/stderr logs

optional GPU utilization logging via nvidia-smi

Execution via a scratch virtual environment, without activating it
(the script calls <venv>/bin/python directly)

Advanced usage

Some scripts additionally support:

Multi-seed experiments via SLURM array jobs

Hyperparameter sweeps using SLURM_ARRAY_TASK_ID


### Data Directory

The dataloader resolves the dataset root as:

the DATA_ROOT environment variable if set

otherwise defaults to <repo>/data

On SLURM we typically set:

export DATA_ROOT=/scratch/users/<username>/data
