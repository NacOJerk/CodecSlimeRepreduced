#!/bin/bash
#SBATCH --job-name=cs-smoke-amd
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

echo "[smoke] node: $(hostname)"
echo "[smoke] python: $(which python)"
echo "[smoke] cuda visible: ${CUDA_VISIBLE_DEVICES:-unset}, rocm visible: ${HIP_VISIBLE_DEVICES:-unset}, rocr: ${ROCR_VISIBLE_DEVICES:-unset}"

cd "$REPO"
python backbones/scripts/smoke_amd.py vq8k fsq18k
