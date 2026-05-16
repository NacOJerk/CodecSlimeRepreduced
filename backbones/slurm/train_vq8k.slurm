#!/bin/bash
#SBATCH --job-name=cs-vq8k
#SBATCH --output=backbones/slurm/logs/%x_%j.out
#SBATCH --error=backbones/slurm/logs/%x_%j.err
#SBATCH --partition=gpu-morgeva
#SBATCH --account=<your_account>
#SBATCH --nodelist=t-100
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
#SBATCH --signal=B:TERM@120

# Get the directory where the script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Derive REPO root (backbones/slurm/ -> root is ../../)
REPO="$(cd "$SCRIPT_DIR/../../" && pwd)"

mkdir -p "$REPO/backbones/slurm/logs"

trap 'echo "[slurm] caught TERM, resubmitting"; sbatch $0; exit 0' TERM

$REPO/backbones/slurm/auto_resume.sh codecslime_vq8k vq8k-300k
