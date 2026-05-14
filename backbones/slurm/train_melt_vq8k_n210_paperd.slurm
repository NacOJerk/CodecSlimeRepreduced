#!/bin/bash
#SBATCH --job-name=cs-melt-vq8k-n210-paperd
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

# Same config as melt-vq8k-n210-12500 but with USE_PAPER_D_ENFORCE=True via override.
# This run validates our d-vector fix: paper-literal formula vs ours.
$REPO/backbones/slurm/auto_resume_melt_amd.sh codecslime_melt_vq8k_n210 melt-vq8k-n210-paperd-12500 \
    +train.compression.use_paper_d_enforce=true
