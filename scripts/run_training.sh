#!/bin/bash
#SBATCH --job-name=train_distillation
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu               
#SBATCH --gres=gpu:1                   
#SBATCH --nodes=1 
#SBATCH --ntasks=1  
#SBATCH --cpus-per-task=1  
#SBATCH --time=02:00:00              
#SBATCH --mem=32G


echo "JOB ID: $SLURM_JOBID"
echo "Running on host: $(hostname)"

# -------------------------------------------------------
# Load Sherlock modules
# -------------------------------------------------------
module purge
module load python/3.12.1
module load cuda/12.2   # You can pick the CUDA version that works with your PyTorch


#$SCRATCH/venvs/cool_project/bin/python -m huggingface_hub.login

source ~/.secrets/hf_token

# make sure both names are set
export HF_TOKEN="$HUGGINGFACE_HUB_TOKEN"
export HUGGINGFACE_HUB_TOKEN="$HUGGINGFACE_HUB_TOKEN"

# fail fast
if [ -z "${HUGGINGFACE_HUB_TOKEN:-}" ]; then
  echo "ERROR: HF token not set"
  exit 1
fi

# -------------------------------------------------------
# Activate your virtual environment
# -------------------------------------------------------
#source ~/cool_project/venv/bin/activate

# ---- venv in SCRATCH (no activate) ----
VENV=/scratch/users/ksa828/venvs/cool_project
PY=$VENV/bin/python

if [ ! -x "$PY" ]; then
  echo "ERROR: venv python not found/executable at $PY"
  echo "Did you create it with: python -m venv $VENV ?"
  exit 1
fi


export HF_HOME=/scratch/users/ksa828/hf_cache
export TRANSFORMERS_CACHE=$HF_HOME
mkdir -p "$HF_HOME"

echo "HF_HOME set to: $HF_HOME"

mkdir -p ~/cool_project/logs

cd ~/cool_project

export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=1      # turn off later when things are stable
export NCCL_DEBUG=WARN

nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.total,memory.used \
           --format=csv -l 10 > logs/gpu_usage_$SLURM_JOBID.log 2>&1 &
NVIDIA_SMI_PID=$!

export DATA_ROOT=/scratch/users/ksa828/data


# -------------------------------------------------------
# Run the training
# -------------------------------------------------------
#python -m cool_project \
   # --function distillation \
   # --config scripts/config_distillation.json

echo "DATA_ROOT=$DATA_ROOT"
env | grep DATA_ROOT

# ---- Run using venv python ----
"$PY" -m cool_project \
    --function extract_embeddings \
    --config scripts/config_extract_embeddings.json

echo "Training finished!"

kill $NVIDIA_SMI_PID

