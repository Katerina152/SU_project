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
- CUDA-enabled GPU (recommended)
- SLURM (optional, for cluster execution)


### Setup

From the project root directory:

```bash
pip install -r requirements.txt
pip install -e .
```

---
## Distillation

The core functionality of this repository is feature-level knowledge distillation,
where a student network is trained to match teacher representations extracted at higher
resolutions or from larger models.

```bash
python -m cool_project \
  --function distillation \
  --config scripts/config_distillation.json
```

Distillation is fully configured via JSON files
(see scripts/config_distillation.json).


### Distillation objective

Distillation is performed at the **representation level**.
For each image and augmentation, both teacher and student embeddings are first
projected into a **shared embedding space of equal dimensionality** using lightweight
projection heads.

The student is then optimized to align with the teacher using the following objective:

$$
\mathcal{L}_{\text{distill}} =
\lambda_{\text{feat}} \lVert h_t - h_s \rVert_2^2 +
\lambda_{\text{cos}} \left(1 - \cos(h_t, h_s)\right)
$$

where \( h_t \) and \( h_s \) denote the projected teacher and student embeddings.

This encourages both magnitude and directional alignment between teacher and student
representations.


## Embedding Extraction

Teacher embeddings must be extracted before running distillation.

```bash

python -m cool_project \
  --function extract_embeddings \
  --config scripts/config_extract_embeddings.json
```

Extracted embeddings are stored on disk and reused across experiments,
enabling efficient multi-run and multi-resolution studies.

## Training

After distillation (or using a frozen pretrained encoder), a linear probe or lightweight
head can be trained for downstream tasks.

```bash

python -m cool_project \
  --function training \
  --config scripts/config_training_non_HP.json
```

### Configuration

Experiments are controlled via JSON configuration files in `scripts/`.

**General**
- `experiment_name`: experiment identifier used for logging and output naming
- `output_dir`: root directory where runs are saved (e.g. `runs_distillation/`)
- `task`: experiment type (e.g. `distillation`)

**Teacher embeddings**
- `teacher.embeddings_dir`: path to a directory containing embeddings for the same dataset splits
  used during training. Expected structure:
  <embeddings_dir>/
    train_embeddings/
    val_embeddings/
    test_embeddings/


- `teacher.num_views`: number of views/augmentations per sample (student model input)
(typically 1–5). The student is trained to match the corresponding view embeddings.
- `teacher.embedding_dim`: embedding dimensionality of the teacher (student projections must match this).

**Data**
- `data.dataset_name`: dataset identifier (e.g. `ISIC2017`)
- `data.domain`: dataset domain (e.g. `dermatology`)
- `data.resolution`: input image resolution used for student training
- `data.batch_size`: batch size
- `data.num_workers`: dataloader workers
- `data.val_split`: fraction of training data held out for validation (if validation dataset is missing)
- `data.balanced_train`: whether to balance sampling during training

Class weights are also computed from the training split and passed to the loss function (e.g. CrossEntropyLoss(weight=...))

These two mechanisms are independent and can be enabled separately

**Student model**
- `model.type`: model family (e.g. `vit`, `resnet`)
- `model.backbone.type`: backbone provider (hf for Hugging Face Transformers, timm for timm models)
- `model.backbone.model_name`: backbone identifier (e.g. timm model string)
- `model.backbone.pooling`: embedding pooling strategy (e.g. `mean`, `cls`)
- `model.backbone.freeze_backbone`: freeze or finetune the backbone during training

**Model head**
- `model.head.type`: head type (e.g. linear, mlp)
- `model.head.output_dim`: number of output classes (classification) or targets
- `model.head.output_activation`: optional activation (null for logits)
- `model.head.dropout`: dropout probability applied in the head

**Loss function**
- `model.loss_type`: training loss (e.g. ce, bce)

**Training**
- `train.lamda_feat`: weight for ℓ2 feature loss
- `train.lamda_cos`: weight for cosine alignment loss
- `train.lr`, `train.weight_decay`, `train.dropout_rate`
- `train.max_epochs`, `train.early_stopping_patience`
- `train.seed`
- `train.profile.flops`, `train.flops.batch`

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

```bash
python -m cool_project \
  --function training \
  --config scripts/config_training.json
```

### Running on SLURM (GPU cluster)

The repository includes SLURM-compatible scripts for execution on HPC clusters
(e.g. Sherlock).

## Typical workflow
1. Prepare a virtual environment in scratch
2. Configure required environment variables
3. Submit the SLURM job

For example from the root directory:

```bash
sbatch scripts/run_distillation.sh
```

## Modules

Typical modules:

```bash
module purge
module load python/3.12.1
module load cuda/12.2
```

The SLURM scripts handles:

- GPU allocation (--gres=gpu:1)
- Hugging Face authentication via HUGGINGFACE_HUB_TOKEN
- Cache setup (HF_HOME, optional TRANSFORMERS_CACHE)
- Logging and diagnostics
- SLURM stdout/stderr logs
- Optional GPU utilization logging 
- Execution via a scratch virtual environment (requirements mentioned above)

Some scripts additionally support:
- Multi-seed experiments via SLURM array jobs (Bash file used for multiple seeds: run_training_final_non_HP_seeds.sh)
- Hyperparameter sweeps using SLURM_ARRAY_TASK_ID


### Data Directory

The dataset root is resolved as:

- DATA_ROOT environment variable (if set); if not, defaults to <repo>/data (repo: project root folder)

## Dataset and Embedding Layout

All stages of the pipeline (embedding extraction, training, and distillation)
assume a **consistent dataset split** into training, validation, and test sets.

### Raw image dataset structure

Datasets are expected to be organized as follows:

```text
<data_root>/<dataset_name>/
  train/
    images/
      *.jpg
    labels.csv
  val/
    images/
      *.jpg
    labels.csv
  test/
    images/
      *.jpg
    labels.csv

