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

The core functionality of this repository is feature-level knowledge distillation,
where a student network is trained to match teacher representations extracted at higher
resolutions or from larger models.

Distillation is fully configured via JSON files
(see scripts/config_distillation.json).

---

### ✅ Running distillation (copy button will appear)
```markdown
### Running distillation
```bash
python -m cool_project \
  --function distillation \
  --config scripts/config_distillation.json
```

Distillation objective

The training objective is a weighted combination of:

Feature regression loss (lamda_feat)

Cosine similarity loss (lamda_cos)

This enforces both magnitude and directional alignment between teacher and student embeddings.


Embedding Extraction

Teacher embeddings must be extracted before running distillation.

python -m cool_project \
  --function extract_embeddings \
  --config scripts/config_extract_embeddings.json


Extracted embeddings are stored on disk and reused across experiments,
enabling efficient multi-run and multi-resolution studies.


Configuration

Experiments are controlled via JSON configuration files.

Distillation config (key parameters)

experiment_name: experiment identifier used for logging and output naming

output_dir: output root for runs (e.g. runs_new/)

Teacher

teacher.embeddings_dir: path to pre-extracted teacher embeddings

teacher.num_views: number of views per sample

teacher.embedding_dim: embedding dimensionality

Data

data.dataset_name: dataset identifier (e.g. ISIC2017)

data.resolution: input resolution for student training

data.batch_size, num_workers

data.val_split, balanced_train

Model

model.backbone.model_name: timm backbone (e.g. tiny_vit_21m_224.in1k)

model.backbone.pooling: pooling strategy (e.g. mean)

model.backbone.freeze_backbone: freeze or finetune backbone

Training

train.lamda_feat: feature regression loss weight

train.lamda_cos: cosine similarity loss weight

train.lr, weight_decay, dropout_rate

train.max_epochs, early_stopping_patience

train.seed

Refer to the full JSON files in scripts/ for additional parameters.


Running on SLURM (GPU cluster)

The repository includes SLURM-compatible scripts for execution on HPC clusters
(e.g. Sherlock).

Typical workflow

Prepare a virtual environment in scratch

Configure environment variables

Submit the SLURM job

sbatch scripts/slurm/distillation.slurm

Modules

On our cluster we load:

module purge
module load python/3.12.1
module load cuda/12.2

What the SLURM scripts handle

GPU allocation (--gres=gpu:1)

Hugging Face authentication (HUGGINGFACE_HUB_TOKEN, fails fast if missing)

Cache setup (HF_HOME, optional TRANSFORMERS_CACHE)

Logging and diagnostics (stdout/stderr + optional GPU utilization logs)

Execution via a scratch virtual environment (without activating it)

Some scripts also support:

Multi-seed experiments via SLURM array jobs

Hyperparameter sweeps using SLURM_ARRAY_TASK_ID

📁 Data Directory

The dataloader resolves the dataset root as:

the DATA_ROOT environment variable if set

otherwise defaults to <repo>/data

On SLURM we typically set:

export DATA_ROOT=/scratch/users/<username>/data
