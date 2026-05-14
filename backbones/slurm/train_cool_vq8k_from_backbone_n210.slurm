#!/bin/bash
#SBATCH --job-name=cs-cool-vq8k-from-backbone-n210
#SBATCH --output=/home/morg/students/dortirosh/audio_ml_tau_final/backbones/slurm/logs/%x_%j.out
#SBATCH --error=/home/morg/students/dortirosh/audio_ml_tau_final/backbones/slurm/logs/%x_%j.err
#SBATCH --partition=gpu-morgeva
#SBATCH --account=gpu-research
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=n-210
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@180

REPO=/home/morg/students/dortirosh/audio_ml_tau_final

trap 'echo "[slurm] caught TERM, resubmitting"; sbatch $0; exit 0' TERM

# Cool-without-Melt ablation on n-210: starts from the FFR backbone vq8k-300k
# directly (no Melt). Same n-210 batch-64 / sqrt-scaled LR / 12.5k-step recipe
# as the from-Melt n-210 cool job.
$REPO/backbones/slurm/auto_resume_cool_amd.sh \
    codecslime_cool_vq8k_from_backbone_n210 \
    cool-vq8k-from-backbone-n210-12500
