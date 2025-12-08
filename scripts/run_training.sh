#!/bin/bash
#SBATCH --job-name=train_for_classification
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu               # <--- IMPORTANT: adjust if Sherlock uses a different GPU partition name
#SBATCH --gres=gpu:1                   # request 1 GPU
#SBATCH --nodes=1 
#SBATCH --ntasks=1  
#SBATCH --cpus-per-task=1  
#SBATCH --time=02:00:00               # 2 hours, change as needed
#SBATCH --mem=32G


echo "JOB ID: $SLURM_JOBID"
echo "Running on host: $(hostname)"
echo "GPUs allocated:"
nvidia-smi

# -------------------------------------------------------
# Load Sherlock modules
# -------------------------------------------------------
module purge
module load python/3.12.1
module load cuda/12.2   # You can pick the CUDA version that works with your PyTorch

# -------------------------------------------------------
# Activate your virtual environment
# -------------------------------------------------------
source ~/cool_project/venv/bin/activate

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



# -------------------------------------------------------
# Run the training
# -------------------------------------------------------
python -m cool_project \
    --function training \
    --config scripts/config_training.json

echo "Training finished!"

kill $NVIDIA_SMI_PID

