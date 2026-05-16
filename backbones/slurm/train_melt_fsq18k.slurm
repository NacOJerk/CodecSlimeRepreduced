#!/bin/bash
#SBATCH --job-name=cs-melt-fsq18k
#SBATCH --output=backbones/slurm/logs/%x_%j.out
#SBATCH --error=backbones/slurm/logs/%x_%j.err
#SBATCH --partition=killable
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --signal=B:TERM@120

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

# Batch 16 fits comfortably in 48GB L40s (half of the 32 used on H100 80GB).
$REPO/backbones/slurm/auto_resume_melt.sh codecslime_melt_fsq18k melt-fsq18k-100k \
    dataset.train.batch_size=16
