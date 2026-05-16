#!/bin/bash
#SBATCH --job-name=cs-melt-vq8k-n210
#SBATCH --output=backbones/slurm/logs/%x_%j.out
#SBATCH --error=backbones/slurm/logs/%x_%j.err
#SBATCH --partition=gpu-morgeva
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=n-210
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --signal=B:TERM@180

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Derive REPO root (backbones/slurm/ -> root is ../../)
REPO="$(cd "$SCRIPT_DIR/../../" && pwd)"

mkdir -p "$REPO/backbones/slurm/logs"

trap 'echo "[slurm] caught TERM, resubmitting"; sbatch $0; exit 0' TERM

$REPO/backbones/slurm/auto_resume_melt_amd.sh codecslime_melt_vq8k_n210 melt-vq8k-n210-12500
