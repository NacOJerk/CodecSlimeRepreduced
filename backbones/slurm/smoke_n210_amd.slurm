#!/bin/bash
#SBATCH --job-name=cs-smoke-amd
#SBATCH --output=backbones/slurm/logs/%x_%j.out
#SBATCH --error=backbones/slurm/logs/%x_%j.err
#SBATCH --partition=gpu-morgeva
#SBATCH --account=<your_account>
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --nodelist=n-210
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:30:00

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

BIGCODEC=$REPO/external/BigCodec
export PYTHONPATH=$REPO:$BIGCODEC:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

echo "[smoke] node: $(hostname)"
echo "[smoke] python: $(which python)"
echo "[smoke] cuda visible: ${CUDA_VISIBLE_DEVICES:-unset}, rocm visible: ${HIP_VISIBLE_DEVICES:-unset}, rocr: ${ROCR_VISIBLE_DEVICES:-unset}"

cd "$REPO"
python backbones/scripts/smoke_amd.py vq8k fsq18k
