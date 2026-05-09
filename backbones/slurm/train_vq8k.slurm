#!/bin/bash
#SBATCH --job-name=cs-vq8k
#SBATCH --output=/home/morg/students/dortirosh/audio_ml_tau_final/backbones/slurm/logs/%x_%j.out
#SBATCH --error=/home/morg/students/dortirosh/audio_ml_tau_final/backbones/slurm/logs/%x_%j.err
#SBATCH --partition=gpu-h100-killable
#SBATCH --account=gpu-research
#SBATCH --nodelist=n-102
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=24:00:00
#SBATCH --signal=B:TERM@120

REPO=/home/morg/students/dortirosh/audio_ml_tau_final

trap 'echo "[slurm] caught TERM, resubmitting"; sbatch $0; exit 0' TERM

$REPO/backbones/slurm/auto_resume.sh codecslime_vq8k vq8k-300k
