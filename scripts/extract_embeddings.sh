#!/bin/bash
#SBATCH --job-name=extract_embeddings_vit
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=25:00:00
#SBATCH --mem=32G

set -euo pipefail

echo "JOB ID: ${SLURM_JOBID}"
echo "Running on host: $(hostname)"

module purge
module load python/3.12.1
module load cuda/12.2

# Hugging Face token
source ~/.secrets/hf_token
export HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_HUB_TOKEN:-}"
export HF_TOKEN="${HUGGINGFACE_HUB_TOKEN}"

if [ -z "${HUGGINGFACE_HUB_TOKEN}" ]; then
  echo "ERROR: HF token not set"
  exit 1
fi

# Python
VENV=/scratch/users/ksa828/venvs/cool_project
PY="${VENV}/bin/python"
if [ ! -x "${PY}" ]; then
  echo "ERROR: venv python not found/executable at ${PY}"
  exit 1
fi

# Caches / working dir
export HF_HOME=/scratch/users/ksa828/hf_cache
export TRANSFORMERS_CACHE="${HF_HOME}"
mkdir -p "${HF_HOME}"

export DATA_ROOT=/scratch/users/ksa828/data

mkdir -p ~/cool_project/logs
cd ~/cool_project

echo "HF_HOME=${HF_HOME}"
echo "DATA_ROOT=${DATA_ROOT}"

# Debug / diagnostics
export PYTHONFAULTHANDLER=1
export CUDA_LAUNCH_BLOCKING=0
export NCCL_DEBUG=WARN
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# (Optional but helpful for tight OOM / fragmentation)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# GPU info once
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "GPUs allocated:"
  nvidia-smi
else
  echo "WARNING: nvidia-smi not found in PATH"
fi

# Background GPU logging
NVIDIA_SMI_PID=""
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=timestamp,name,utilization.gpu,memory.total,memory.used \
    --format=csv -l 10 > "logs/gpu_usage_${SLURM_JOBID}.log" 2>&1 &
  NVIDIA_SMI_PID=$!
else
  echo "WARNING: nvidia-smi not found; skipping GPU logging"
fi

cleanup() {
  if [ -n "${NVIDIA_SMI_PID:-}" ] && ps -p "${NVIDIA_SMI_PID}" >/dev/null 2>&1; then
    kill "${NVIDIA_SMI_PID}" || true
  fi
}
trap cleanup EXIT

# Run training
"${PY}" -m cool_project \
  --function extract_embeddings \
  --config scripts/config_extract_embeddings.json

echo "Done."


