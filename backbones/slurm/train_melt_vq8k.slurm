#!/bin/bash
#SBATCH --job-name=cs-melt-vq8k
#SBATCH --output=/home/morg/students/dortirosh/audio_ml_tau_final/backbones/slurm/logs/%x_%j.out
#SBATCH --error=/home/morg/students/dortirosh/audio_ml_tau_final/backbones/slurm/logs/%x_%j.err
#SBATCH --partition=killable
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:TERM@120

REPO=/home/morg/students/dortirosh/audio_ml_tau_final

trap 'echo "[slurm] caught TERM, resubmitting"; sbatch $0; exit 0' TERM

# Batch 16 fits comfortably in 48GB L40s (half of the 32 used on H100 80GB).
$REPO/backbones/slurm/auto_resume_melt.sh codecslime_melt_vq8k melt-vq8k-100k \
    dataset.train.batch_size=16
