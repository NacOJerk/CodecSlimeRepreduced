#!/bin/bash
#SBATCH --job-name=cs-melt-fsq18k-n210
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
# Derive REPO root
REPO="$(cd "$SCRIPT_DIR/../../" && pwd)"

VENV="${VENV:-$REPO/.venv}"
if [ -d "$VENV" ]; then
    export PATH="$VENV/bin:$PATH"
    export VIRTUAL_ENV="$VENV"
fi

mkdir -p "$REPO/backbones/slurm/logs"

trap 'echo "[slurm] caught TERM, resubmitting"; sbatch $0; exit 0' TERM

$REPO/backbones/slurm/auto_resume_melt_amd.sh codecslime_melt_fsq18k_n210 melt-fsq18k-n210-12500
