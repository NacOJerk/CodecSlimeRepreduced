#!/bin/bash
#SBATCH --job-name=cs-smoke-amd-min
#SBATCH --output=/home/morg/students/dortirosh/audio_ml_tau_final/backbones/slurm/logs/%x_%j.out
#SBATCH --error=/home/morg/students/dortirosh/audio_ml_tau_final/backbones/slurm/logs/%x_%j.err
#SBATCH --partition=gpu-morgeva
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=n-210
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00

REPO=/home/morg/students/dortirosh/audio_ml_tau_final
BIGCODEC=$REPO/external/BigCodec
VENV=/home/morg/students/dortirosh/envs/cs_amd

export PATH=$VENV/bin:${PATH:-}
export VIRTUAL_ENV=$VENV
export PYTHONPATH=$REPO:$BIGCODEC:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

echo "[slurm] host=$(hostname) cuda=${CUDA_VISIBLE_DEVICES:-unset} rocr=${ROCR_VISIBLE_DEVICES:-unset} hip=${HIP_VISIBLE_DEVICES:-unset}"
cd "$REPO"
python -u backbones/scripts/smoke_amd_min.py
